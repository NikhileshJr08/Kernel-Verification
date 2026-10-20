import math

import torch
import torch.nn.functional as F

from exact_m_dense_gemm import gemm_exact_m_

_NUM_HEADS = 8
_HEAD_DIM = 64
_HIDDEN_SIZE = 512
_QKV_OUT = 3 * _HIDDEN_SIZE
_CHUNK_SIZE = 32
_MAX_BACKWARD = 127
_MAX_FORWARD = 0
_CONTEXT_SIZE = 160
_Q_SCALE = (_HEAD_DIM ** -0.5) / math.log(2.0)
_VIEW_CTX_MIN_PADDED_TOKENS = 34816
_SMALL_REFERENCE_MAX_PADDED_TOKENS = 3776

_CAUSAL_MASK_CACHE = {}
_POS_INDICES_CACHE = {}
_PROJ_WORKSPACES = {}
_PREWARMED_DEVICES = set()

try:
    import torch._dynamo

    torch._dynamo.config.cache_size_limit = max(torch._dynamo.config.cache_size_limit, 16)
except Exception:
    pass


def _as_float(value):
    if isinstance(value, torch.Tensor):
        return float(value.item())
    return float(value)



def _pad_dim1(x: torch.Tensor, pad_left: int, pad_right: int) -> torch.Tensor:
    b, _, *tail_shape = x.shape
    left = x.new_zeros((b, pad_left, *tail_shape))
    right = x.new_zeros((b, pad_right, *tail_shape))
    return torch.cat([left, x, right], dim=1)



def _extract_block_context_token(hs: torch.Tensor) -> torch.Tensor:
    hs = _pad_dim1(hs, _MAX_BACKWARD, _MAX_FORWARD + _CHUNK_SIZE - 1)
    x_unfolded = hs.unfold(dimension=1, size=_CONTEXT_SIZE, step=_CHUNK_SIZE)
    if hs.ndim > 2 and x_unfolded.ndim > 3:
        x_unfolded = torch.movedim(x_unfolded, source=-1, destination=2)
    return x_unfolded.contiguous()



def _extract_context_from_chunks(hs_chunks: torch.Tensor) -> torch.Tensor:
    batch_size, num_blocks = hs_chunks.shape[:2]
    tail_shape = hs_chunks.shape[3:]
    left = hs_chunks.new_zeros((batch_size, 4, _CHUNK_SIZE, *tail_shape))
    right = hs_chunks.new_zeros((batch_size, 1, _CHUNK_SIZE, *tail_shape))
    padded = torch.cat([left, hs_chunks, right], dim=1)
    flat = padded.reshape(batch_size, (num_blocks + 5) * _CHUNK_SIZE, *tail_shape)[:, 1:]
    x_unfolded = flat.unfold(dimension=1, size=_CONTEXT_SIZE, step=_CHUNK_SIZE)
    if flat.ndim > 2 and x_unfolded.ndim > 3:
        x_unfolded = torch.movedim(x_unfolded, source=-1, destination=2)
    return x_unfolded



def _relative_shift(
    term_bd_before_shift: torch.Tensor,
    bs: int,
    nh: int,
    nqb: int,
    qbs: int,
    kcs: int,
    msp1: int,
) -> torch.Tensor:
    pad_amount_last_dim = (kcs + 1) - msp1
    term_bd_padded = F.pad(term_bd_before_shift, (0, pad_amount_last_dim))
    term_bd_reshaped = term_bd_padded.reshape(bs, nh, nqb, qbs * (kcs + 1))
    term_bd_sliced = term_bd_reshaped[:, :, :, : qbs * kcs]
    return term_bd_sliced.reshape(bs, nh, nqb, qbs, kcs)



def _get_local_causal_valid_mask(device: torch.device) -> torch.Tensor:
    key = str(device)
    mask = _CAUSAL_MASK_CACHE.get(key)
    if mask is None:
        lower_causal_mask = torch.tril(
            torch.ones((_CONTEXT_SIZE, _CHUNK_SIZE), dtype=torch.bool, device=device),
            diagonal=0,
        ).T
        upper_causal_mask = torch.tril(
            torch.ones((_CHUNK_SIZE, _CONTEXT_SIZE), dtype=torch.bool, device=device),
            diagonal=_MAX_BACKWARD + _MAX_FORWARD,
        )
        mask = lower_causal_mask & upper_causal_mask
        _CAUSAL_MASK_CACHE[key] = mask
    return mask



def _get_pos_indices(device: torch.device) -> torch.Tensor:
    key = str(device)
    pos_indices = _POS_INDICES_CACHE.get(key)
    if pos_indices is None:
        pos_indices = torch.arange(
            _MAX_BACKWARD,
            -_MAX_FORWARD - 1,
            -1,
            device=device,
            dtype=torch.float32,
        )
        _POS_INDICES_CACHE[key] = pos_indices
    return pos_indices



def _small_reference_compute(
    hidden_states: torch.Tensor,
    mask: torch.Tensor,
    q_proj_weight: torch.Tensor,
    k_proj_weight: torch.Tensor,
    v_proj_weight: torch.Tensor,
    pos_proj_weight: torch.Tensor,
    per_dim_scale: torch.Tensor,
    inv_timescales: torch.Tensor,
    soft_cap: float,
    causal_mask: torch.Tensor,
    pos_indices: torch.Tensor,
) -> torch.Tensor:
    batch_size, q_time, _ = hidden_states.shape
    device = hidden_states.device

    query_states = torch.matmul(hidden_states, q_proj_weight.T)
    key_states = torch.matmul(hidden_states, k_proj_weight.T)
    value_states = torch.matmul(hidden_states, v_proj_weight.T)

    qkv_shape = (batch_size, q_time, _NUM_HEADS, _HEAD_DIM)
    query_states = query_states.reshape(qkv_shape).contiguous()
    key_states = key_states.reshape(qkv_shape).contiguous()
    value_states = value_states.reshape(qkv_shape).contiguous()

    q_scale_vec = (_Q_SCALE * F.softplus(per_dim_scale)).view(1, 1, 1, _HEAD_DIM)
    query_states = query_states * q_scale_vec

    num_blocks = (q_time + _CHUNK_SIZE - 1) // _CHUNK_SIZE
    padded_seq_len = num_blocks * _CHUNK_SIZE
    if padded_seq_len == q_time:
        query_padded = query_states
    else:
        query_padded = _pad_dim1(query_states, 0, padded_seq_len - q_time)
    query_blocks = query_padded.view(batch_size, num_blocks, _CHUNK_SIZE, _NUM_HEADS, _HEAD_DIM)

    key_blocks = _extract_block_context_token(key_states)
    value_blocks = _extract_block_context_token(value_states)

    extracted_valid_mask_blocks = _extract_block_context_token(~mask)
    if extracted_valid_mask_blocks.ndim == 4:
        extracted_valid_mask_blocks = extracted_valid_mask_blocks.reshape(
            batch_size,
            num_blocks,
            _CONTEXT_SIZE,
        )

    final_condition_for_where = torch.logical_and(
        extracted_valid_mask_blocks.unsqueeze(1).unsqueeze(-2),
        causal_mask.unsqueeze(0).unsqueeze(0).unsqueeze(0),
    )

    max_span_plus_1 = pos_indices.shape[1]
    scaled_time = pos_indices.unsqueeze(-1) * inv_timescales.unsqueeze(0).unsqueeze(0)
    timing_signal = torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=-1)
    projected_sin_emb = torch.matmul(timing_signal, pos_proj_weight.T)
    sin_emb = projected_sin_emb.reshape(1, max_span_plus_1, _NUM_HEADS, _HEAD_DIM).squeeze(0)

    queries_p = query_blocks.permute(0, 3, 1, 2, 4)
    keys_p_t = key_blocks.permute(0, 3, 1, 4, 2)
    term_ac = torch.matmul(queries_p, keys_p_t)

    q_reshaped = queries_p.reshape(batch_size, _NUM_HEADS, num_blocks * _CHUNK_SIZE, _HEAD_DIM)
    term_bd_unshifed_matmul = torch.matmul(q_reshaped, sin_emb.permute(1, 2, 0))
    term_bd_unshifed = term_bd_unshifed_matmul.reshape(
        batch_size,
        _NUM_HEADS,
        num_blocks,
        _CHUNK_SIZE,
        max_span_plus_1,
    )
    term_bd_shifted = _relative_shift(
        term_bd_unshifed,
        batch_size,
        _NUM_HEADS,
        num_blocks,
        _CHUNK_SIZE,
        _CONTEXT_SIZE,
        max_span_plus_1,
    )

    logits = term_ac + term_bd_shifted
    softcap_val = torch.tensor(soft_cap, device=device, dtype=torch.float32)
    logits = logits / softcap_val
    logits = torch.tanh(logits)
    logits = logits * softcap_val
    logits = torch.where(final_condition_for_where, logits, torch.finfo(logits.dtype).min)

    probabilities = F.softmax(logits, dim=-1, dtype=torch.float32)
    b_dim, n_dim, u_dim, w_dim, c_dim = probabilities.shape
    h_dim = value_blocks.shape[-1]
    prob_bun = probabilities.permute(0, 2, 1, 3, 4).reshape(-1, w_dim, c_dim)
    v_bun = value_blocks.permute(0, 1, 3, 2, 4).reshape(-1, c_dim, h_dim)
    result_bmm = torch.bmm(prob_bun, v_bun)
    out = result_bmm.reshape(b_dim, u_dim, n_dim, w_dim, h_dim).permute(0, 1, 3, 2, 4)
    return out.reshape(batch_size, padded_seq_len, _NUM_HEADS, _HEAD_DIM)[:, :q_time]


if hasattr(torch, "compile"):
    _compiled_small_reference_compute = torch.compile(
        _small_reference_compute,
        dynamic=False,
        mode="max-autotune-no-cudagraphs",
    )
else:
    _compiled_small_reference_compute = _small_reference_compute


def _medium_token_context_compute(
    query_chunks: torch.Tensor,
    key_chunks: torch.Tensor,
    value_chunks: torch.Tensor,
    valid_mask_chunks: torch.Tensor,
    pos_proj_weight: torch.Tensor,
    inv_timescales: torch.Tensor,
    soft_cap: float,
    causal_mask: torch.Tensor,
    pos_indices: torch.Tensor,
) -> torch.Tensor:
    batch_size, num_blocks = query_chunks.shape[:2]
    device = query_chunks.device
    padded_seq_len = num_blocks * _CHUNK_SIZE

    key_blocks = _extract_block_context_token(
        key_chunks.reshape(batch_size, padded_seq_len, _NUM_HEADS, _HEAD_DIM)
    )
    value_blocks = _extract_block_context_token(
        value_chunks.reshape(batch_size, padded_seq_len, _NUM_HEADS, _HEAD_DIM)
    )
    extracted_valid_mask_blocks = _extract_block_context_token(
        valid_mask_chunks.reshape(batch_size, padded_seq_len)
    )
    if extracted_valid_mask_blocks.ndim == 4:
        extracted_valid_mask_blocks = extracted_valid_mask_blocks.reshape(
            batch_size,
            num_blocks,
            _CONTEXT_SIZE,
        )

    final_condition_for_where = torch.logical_and(
        extracted_valid_mask_blocks.unsqueeze(1).unsqueeze(-2),
        causal_mask.unsqueeze(0).unsqueeze(0).unsqueeze(0),
    )

    max_span_plus_1 = pos_indices.shape[1]
    scaled_time = pos_indices.unsqueeze(-1) * inv_timescales.unsqueeze(0).unsqueeze(0)
    timing_signal = torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=-1)
    projected_sin_emb = torch.matmul(timing_signal, pos_proj_weight.T)
    sin_emb = projected_sin_emb.reshape(1, max_span_plus_1, _NUM_HEADS, _HEAD_DIM).squeeze(0).to(query_chunks.dtype)

    queries_p = query_chunks.permute(0, 3, 1, 2, 4)
    keys_p_t = key_blocks.permute(0, 3, 1, 4, 2)
    term_ac = torch.matmul(queries_p, keys_p_t).float()

    q_reshaped = queries_p.reshape(batch_size, _NUM_HEADS, padded_seq_len, _HEAD_DIM)
    term_bd_unshifed_matmul = torch.matmul(q_reshaped, sin_emb.permute(1, 2, 0)).float()
    term_bd_unshifed = term_bd_unshifed_matmul.reshape(
        batch_size,
        _NUM_HEADS,
        num_blocks,
        _CHUNK_SIZE,
        max_span_plus_1,
    )
    term_bd_shifted = _relative_shift(
        term_bd_unshifed,
        batch_size,
        _NUM_HEADS,
        num_blocks,
        _CHUNK_SIZE,
        _CONTEXT_SIZE,
        max_span_plus_1,
    )

    logits = term_ac + term_bd_shifted
    softcap_val = torch.tensor(soft_cap, device=device, dtype=torch.float32)
    logits = logits / softcap_val
    logits = torch.tanh(logits)
    logits = logits * softcap_val
    logits = torch.where(final_condition_for_where, logits, torch.finfo(logits.dtype).min)

    probabilities = F.softmax(logits, dim=-1, dtype=torch.float32)
    b_dim, n_dim, u_dim, w_dim, c_dim = probabilities.shape
    h_dim = value_blocks.shape[-1]
    prob_bun = probabilities.permute(0, 2, 1, 3, 4).reshape(-1, w_dim, c_dim)
    v_bun = value_blocks.permute(0, 1, 3, 2, 4).reshape(-1, c_dim, h_dim).float()
    result_bmm = torch.bmm(prob_bun, v_bun)
    out = result_bmm.reshape(b_dim, u_dim, n_dim, w_dim, h_dim).permute(0, 1, 3, 2, 4)
    return out.reshape(batch_size, padded_seq_len, _NUM_HEADS, _HEAD_DIM)


if hasattr(torch, "compile"):
    _compiled_medium_token_context_compute = torch.compile(
        _medium_token_context_compute,
        dynamic=False,
        mode="max-autotune-no-cudagraphs",
    )
else:
    _compiled_medium_token_context_compute = _medium_token_context_compute


def _large_view_context_compute(
    query_chunks: torch.Tensor,
    key_chunks: torch.Tensor,
    value_chunks: torch.Tensor,
    valid_mask_chunks: torch.Tensor,
    pos_proj_weight: torch.Tensor,
    inv_timescales: torch.Tensor,
    soft_cap: float,
    causal_mask: torch.Tensor,
    pos_indices: torch.Tensor,
) -> torch.Tensor:
    batch_size, num_blocks = query_chunks.shape[:2]
    device = query_chunks.device
    padded_seq_len = num_blocks * _CHUNK_SIZE

    key_blocks = _extract_context_from_chunks(key_chunks)
    value_blocks = _extract_context_from_chunks(value_chunks)
    extracted_valid_mask_blocks = _extract_context_from_chunks(valid_mask_chunks)
    if extracted_valid_mask_blocks.ndim == 4:
        extracted_valid_mask_blocks = extracted_valid_mask_blocks.reshape(
            batch_size,
            num_blocks,
            _CONTEXT_SIZE,
        )

    final_condition_for_where = torch.logical_and(
        extracted_valid_mask_blocks.unsqueeze(1).unsqueeze(-2),
        causal_mask.unsqueeze(0).unsqueeze(0).unsqueeze(0),
    )

    max_span_plus_1 = pos_indices.shape[1]
    scaled_time = pos_indices.unsqueeze(-1) * inv_timescales.unsqueeze(0).unsqueeze(0)
    timing_signal = torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=-1)
    projected_sin_emb = torch.matmul(timing_signal, pos_proj_weight.T)
    sin_emb = projected_sin_emb.reshape(1, max_span_plus_1, _NUM_HEADS, _HEAD_DIM).squeeze(0).to(query_chunks.dtype)

    queries_p = query_chunks.permute(0, 3, 1, 2, 4)
    keys_p_t = key_blocks.permute(0, 3, 1, 4, 2)
    term_ac = torch.matmul(queries_p, keys_p_t).float()

    q_reshaped = queries_p.reshape(batch_size, _NUM_HEADS, padded_seq_len, _HEAD_DIM)
    term_bd_unshifed_matmul = torch.matmul(q_reshaped, sin_emb.permute(1, 2, 0)).float()
    term_bd_unshifed = term_bd_unshifed_matmul.reshape(
        batch_size,
        _NUM_HEADS,
        num_blocks,
        _CHUNK_SIZE,
        max_span_plus_1,
    )
    term_bd_shifted = _relative_shift(
        term_bd_unshifed,
        batch_size,
        _NUM_HEADS,
        num_blocks,
        _CHUNK_SIZE,
        _CONTEXT_SIZE,
        max_span_plus_1,
    )

    logits = term_ac + term_bd_shifted
    softcap_val = torch.tensor(soft_cap, device=device, dtype=torch.float32)
    logits = logits / softcap_val
    logits = torch.tanh(logits)
    logits = logits * softcap_val
    logits = torch.where(final_condition_for_where, logits, torch.finfo(logits.dtype).min)

    probabilities = F.softmax(logits, dim=-1, dtype=torch.float32)
    out = torch.matmul(
        probabilities.permute(0, 2, 1, 3, 4),
        value_blocks.permute(0, 1, 3, 2, 4).float(),
    ).permute(0, 1, 3, 2, 4)
    return out.reshape(batch_size, padded_seq_len, _NUM_HEADS, _HEAD_DIM)


if hasattr(torch, "compile"):
    _compiled_large_view_context_compute = torch.compile(
        _large_view_context_compute,
        dynamic=False,
        mode="max-autotune-no-cudagraphs",
    )
else:
    _compiled_large_view_context_compute = _large_view_context_compute


def _medium_from_qkv_compute(
    qkv_out: torch.Tensor,
    mask: torch.Tensor,
    per_dim_scale: torch.Tensor,
    query_chunks: torch.Tensor,
    key_chunks: torch.Tensor,
    value_chunks: torch.Tensor,
    valid_mask_chunks: torch.Tensor,
    pos_proj_weight: torch.Tensor,
    inv_timescales: torch.Tensor,
    soft_cap: float,
    causal_mask: torch.Tensor,
    pos_indices: torch.Tensor,
) -> torch.Tensor:
    batch_size, padded_seq_len = qkv_out.shape[:2]
    q_time = mask.shape[1]
    num_blocks = padded_seq_len // _CHUNK_SIZE
    qkv_blocks = qkv_out.view(batch_size, num_blocks, _CHUNK_SIZE, 3, _NUM_HEADS, _HEAD_DIM)
    q_scale_vec = (_Q_SCALE * F.softplus(per_dim_scale)).to(qkv_out.dtype).view(1, 1, 1, 1, _HEAD_DIM)

    torch.mul(qkv_blocks[:, :, :, 0], q_scale_vec, out=query_chunks)
    key_chunks.copy_(qkv_blocks[:, :, :, 1])
    value_chunks.copy_(qkv_blocks[:, :, :, 2])
    if padded_seq_len == q_time:
        valid_mask_chunks.copy_((~mask).view(batch_size, num_blocks, _CHUNK_SIZE))
    else:
        valid_mask_chunks.fill_(False)
        valid_mask_chunks.view(batch_size, padded_seq_len)[:, :q_time].copy_(~mask)

    return _medium_token_context_compute(
        query_chunks,
        key_chunks,
        value_chunks,
        valid_mask_chunks,
        pos_proj_weight,
        inv_timescales,
        soft_cap,
        causal_mask,
        pos_indices,
    )


if hasattr(torch, "compile"):
    _compiled_medium_from_qkv_compute = torch.compile(
        _medium_from_qkv_compute,
        dynamic=False,
        mode="max-autotune-no-cudagraphs",
    )
else:
    _compiled_medium_from_qkv_compute = _medium_from_qkv_compute


def _large_from_qkv_compute(
    qkv_out: torch.Tensor,
    mask: torch.Tensor,
    per_dim_scale: torch.Tensor,
    valid_mask_chunks: torch.Tensor,
    pos_proj_weight: torch.Tensor,
    inv_timescales: torch.Tensor,
    soft_cap: float,
    causal_mask: torch.Tensor,
    pos_indices: torch.Tensor,
) -> torch.Tensor:
    batch_size, padded_seq_len = qkv_out.shape[:2]
    q_time = mask.shape[1]
    num_blocks = padded_seq_len // _CHUNK_SIZE
    qkv_blocks = qkv_out.view(batch_size, num_blocks, _CHUNK_SIZE, 3, _NUM_HEADS, _HEAD_DIM)
    q_scale_vec = (_Q_SCALE * F.softplus(per_dim_scale)).to(qkv_out.dtype).view(1, 1, 1, 1, _HEAD_DIM)

    if padded_seq_len == q_time:
        valid_mask_chunks.copy_((~mask).view(batch_size, num_blocks, _CHUNK_SIZE))
    else:
        valid_mask_chunks.fill_(False)
        valid_mask_chunks.view(batch_size, padded_seq_len)[:, :q_time].copy_(~mask)

    return _large_view_context_compute(
        qkv_blocks[:, :, :, 0] * q_scale_vec,
        qkv_blocks[:, :, :, 1],
        qkv_blocks[:, :, :, 2],
        valid_mask_chunks,
        pos_proj_weight,
        inv_timescales,
        soft_cap,
        causal_mask,
        pos_indices,
    )


if hasattr(torch, "compile"):
    _compiled_large_from_qkv_compute = torch.compile(
        _large_from_qkv_compute,
        dynamic=False,
        mode="max-autotune-no-cudagraphs",
    )
else:
    _compiled_large_from_qkv_compute = _large_from_qkv_compute


def _get_projection_workspace(
    batch_size: int,
    num_blocks: int,
    device: torch.device,
):
    key = (str(device), batch_size, num_blocks)
    workspace = _PROJ_WORKSPACES.get(key)
    if workspace is None:
        padded_seq_len = num_blocks * _CHUNK_SIZE
        workspace = {
            "hidden_bf16": torch.empty(
                (batch_size, padded_seq_len, _HIDDEN_SIZE),
                dtype=torch.bfloat16,
                device=device,
            ),
            "weight_bf16": torch.empty(
                (_QKV_OUT, _HIDDEN_SIZE),
                dtype=torch.bfloat16,
                device=device,
            ),
            "qkv_out": torch.empty(
                (batch_size, padded_seq_len, _QKV_OUT),
                dtype=torch.bfloat16,
                device=device,
            ),
            "query_chunks": torch.empty(
                (batch_size, num_blocks, _CHUNK_SIZE, _NUM_HEADS, _HEAD_DIM),
                dtype=torch.bfloat16,
                device=device,
            ),
            "key_chunks": torch.empty(
                (batch_size, num_blocks, _CHUNK_SIZE, _NUM_HEADS, _HEAD_DIM),
                dtype=torch.bfloat16,
                device=device,
            ),
            "value_chunks": torch.empty(
                (batch_size, num_blocks, _CHUNK_SIZE, _NUM_HEADS, _HEAD_DIM),
                dtype=torch.bfloat16,
                device=device,
            ),
            "valid_mask_chunks": torch.empty(
                (batch_size, num_blocks, _CHUNK_SIZE),
                dtype=torch.bool,
                device=device,
            ),
        }
        _PROJ_WORKSPACES[key] = workspace
    return workspace


@torch.no_grad()
def _prewarm_large_view_path(device: torch.device) -> None:
    device_key = str(device)
    if device_key in _PREWARMED_DEVICES:
        return

    q_scale_vec = torch.ones((1, 1, 1, 1, _HEAD_DIM), dtype=torch.bfloat16, device=device)
    zero_weight = torch.zeros((_QKV_OUT, _HIDDEN_SIZE), dtype=torch.bfloat16, device=device)

    for batch_size, num_blocks in ((32, 34), (64, 37), (64, 54)):
        padded_seq_len = num_blocks * _CHUNK_SIZE
        hidden_bf16 = torch.zeros(
            (batch_size, padded_seq_len, _HIDDEN_SIZE),
            dtype=torch.bfloat16,
            device=device,
        )
        qkv_out = torch.empty(
            (batch_size, padded_seq_len, _QKV_OUT),
            dtype=torch.bfloat16,
            device=device,
        )
        gemm_exact_m_(hidden_bf16.view(-1, _HIDDEN_SIZE), zero_weight, qkv_out.view(-1, _QKV_OUT))

        qkv_blocks = qkv_out.view(batch_size, num_blocks, _CHUNK_SIZE, 3, _NUM_HEADS, _HEAD_DIM)
        qkv_blocks[:, :, :, 0].mul_(q_scale_vec)
        query_chunks = qkv_blocks[:, :, :, 0]

        key_blocks = _extract_context_from_chunks(qkv_blocks[:, :, :, 1])
        value_blocks = _extract_context_from_chunks(qkv_blocks[:, :, :, 2])

        queries_p = query_chunks.permute(0, 3, 1, 2, 4)
        keys_p_t = key_blocks.permute(0, 3, 1, 4, 2)
        torch.matmul(queries_p, keys_p_t).float()

        probabilities = torch.zeros(
            (batch_size, _NUM_HEADS, num_blocks, _CHUNK_SIZE, _CONTEXT_SIZE),
            dtype=torch.float32,
            device=device,
        )
        torch.matmul(
            probabilities.permute(0, 2, 1, 3, 4),
            value_blocks.permute(0, 1, 3, 2, 4).float(),
        )

        del hidden_bf16, qkv_out, query_chunks, key_blocks, value_blocks, queries_p, keys_p_t, probabilities

    torch.cuda.synchronize(device)
    _PREWARMED_DEVICES.add(device_key)


@torch.no_grad()
def _project_qkv_cuda(
    hidden_states: torch.Tensor,
    q_proj_weight: torch.Tensor,
    k_proj_weight: torch.Tensor,
    v_proj_weight: torch.Tensor,
):
    batch_size, q_time, hidden_size = hidden_states.shape
    num_blocks = (q_time + _CHUNK_SIZE - 1) // _CHUNK_SIZE
    padded_seq_len = num_blocks * _CHUNK_SIZE
    workspace = _get_projection_workspace(batch_size, num_blocks, hidden_states.device)

    hidden_bf16 = workspace["hidden_bf16"]
    hidden_bf16[:, :q_time].copy_(hidden_states)
    if padded_seq_len > q_time:
        hidden_bf16[:, q_time:].zero_()

    weight_bf16 = workspace["weight_bf16"]
    weight_bf16[:_HIDDEN_SIZE].copy_(q_proj_weight)
    weight_bf16[_HIDDEN_SIZE : 2 * _HIDDEN_SIZE].copy_(k_proj_weight)
    weight_bf16[2 * _HIDDEN_SIZE :].copy_(v_proj_weight)

    qkv_out = workspace["qkv_out"]
    gemm_exact_m_(hidden_bf16.view(-1, hidden_size), weight_bf16, qkv_out.view(-1, _QKV_OUT))
    return qkv_out, num_blocks, padded_seq_len, workspace


@torch.no_grad()
def _project_qkv_fallback(
    hidden_states: torch.Tensor,
    q_proj_weight: torch.Tensor,
    k_proj_weight: torch.Tensor,
    v_proj_weight: torch.Tensor,
):
    batch_size, q_time, hidden_size = hidden_states.shape
    num_blocks = (q_time + _CHUNK_SIZE - 1) // _CHUNK_SIZE
    padded_seq_len = num_blocks * _CHUNK_SIZE
    if padded_seq_len == q_time:
        hidden_padded = hidden_states
    else:
        hidden_padded = _pad_dim1(hidden_states, 0, padded_seq_len - q_time)
    qkv_weight = torch.cat([q_proj_weight, k_proj_weight, v_proj_weight], dim=0)
    qkv_out = torch.matmul(hidden_padded.reshape(-1, hidden_size), qkv_weight.T)
    return qkv_out.view(batch_size, padded_seq_len, _QKV_OUT), num_blocks, padded_seq_len


@torch.no_grad()
def _run_small_reference_path(
    hidden_states: torch.Tensor,
    mask: torch.Tensor,
    q_proj_weight: torch.Tensor,
    k_proj_weight: torch.Tensor,
    v_proj_weight: torch.Tensor,
    pos_proj_weight: torch.Tensor,
    per_dim_scale: torch.Tensor,
    inv_timescales: torch.Tensor,
    soft_cap: float,
    context_vectors: torch.Tensor,
) -> None:
    device = hidden_states.device
    compute_small = _compiled_small_reference_compute if device.type == "cuda" else _small_reference_compute
    context_vectors.copy_(
        compute_small(
            hidden_states,
            mask,
            q_proj_weight,
            k_proj_weight,
            v_proj_weight,
            pos_proj_weight,
            per_dim_scale,
            inv_timescales,
            soft_cap,
            _get_local_causal_valid_mask(device),
            _get_pos_indices(device).unsqueeze(0),
        )
    )


@torch.no_grad()
def run(
    hidden_states: torch.Tensor,
    mask: torch.Tensor,
    q_proj_weight: torch.Tensor,
    k_proj_weight: torch.Tensor,
    v_proj_weight: torch.Tensor,
    pos_proj_weight: torch.Tensor,
    per_dim_scale: torch.Tensor,
    inv_timescales: torch.Tensor,
    attention_logits_soft_cap,
    context_vectors: torch.Tensor,
):
    batch_size, q_time, hidden_size = hidden_states.shape
    device = hidden_states.device
    soft_cap = _as_float(attention_logits_soft_cap)
    num_blocks = (q_time + _CHUNK_SIZE - 1) // _CHUNK_SIZE
    padded_seq_len = num_blocks * _CHUNK_SIZE
    total_padded_tokens = batch_size * padded_seq_len

    if total_padded_tokens <= _SMALL_REFERENCE_MAX_PADDED_TOKENS:
        _run_small_reference_path(
            hidden_states,
            mask,
            q_proj_weight,
            k_proj_weight,
            v_proj_weight,
            pos_proj_weight,
            per_dim_scale,
            inv_timescales,
            soft_cap,
            context_vectors,
        )
        return

    if device.type == "cuda":
        _prewarm_large_view_path(device)
        qkv_out, num_blocks, padded_seq_len, workspace = _project_qkv_cuda(
            hidden_states,
            q_proj_weight,
            k_proj_weight,
            v_proj_weight,
        )
    else:
        qkv_out, num_blocks, padded_seq_len = _project_qkv_fallback(
            hidden_states,
            q_proj_weight,
            k_proj_weight,
            v_proj_weight,
        )
        workspace = None

    use_view_ctx = total_padded_tokens >= _VIEW_CTX_MIN_PADDED_TOKENS
    qkv_blocks = qkv_out.view(batch_size, num_blocks, _CHUNK_SIZE, 3, _NUM_HEADS, _HEAD_DIM)
    causal_mask = _get_local_causal_valid_mask(device)
    pos_indices = _get_pos_indices(device).unsqueeze(0)

    if workspace is not None:
        if not use_view_ctx:
            context_vectors.copy_(
                _compiled_medium_from_qkv_compute(
                    qkv_out,
                    mask,
                    per_dim_scale,
                    workspace["query_chunks"],
                    workspace["key_chunks"],
                    workspace["value_chunks"],
                    workspace["valid_mask_chunks"],
                    pos_proj_weight,
                    inv_timescales,
                    soft_cap,
                    causal_mask,
                    pos_indices,
                )[:, :q_time]
            )
            return

        context_vectors.copy_(
            _compiled_large_from_qkv_compute(
                qkv_out,
                mask,
                per_dim_scale,
                workspace["valid_mask_chunks"],
                pos_proj_weight,
                inv_timescales,
                soft_cap,
                causal_mask,
                pos_indices,
            )[:, :q_time]
        )
        return

    q_scale_vec = (_Q_SCALE * F.softplus(per_dim_scale)).to(qkv_out.dtype).view(1, 1, 1, 1, _HEAD_DIM)
    if workspace is None:
        if use_view_ctx:
            query_chunks = qkv_blocks[:, :, :, 0] * q_scale_vec
        else:
            query_chunks = (qkv_blocks[:, :, :, 0] * q_scale_vec).contiguous()
        if not use_view_ctx:
            key_chunks = qkv_blocks[:, :, :, 1].contiguous()
            value_chunks = qkv_blocks[:, :, :, 2].contiguous()
        original_valid_mask = ~mask
        if padded_seq_len != q_time:
            original_valid_mask = _pad_dim1(original_valid_mask, 0, padded_seq_len - q_time)
        valid_mask_chunks = original_valid_mask.view(batch_size, num_blocks, _CHUNK_SIZE)

    if not use_view_ctx:
        compute_medium = _compiled_medium_token_context_compute if device.type == "cuda" else _medium_token_context_compute
        context_vectors.copy_(
            compute_medium(
                query_chunks,
                key_chunks,
                value_chunks,
                valid_mask_chunks,
                pos_proj_weight,
                inv_timescales,
                soft_cap,
                causal_mask,
                pos_indices,
            )[:, :q_time]
        )
        return

    compute_large = _compiled_large_view_context_compute if device.type == "cuda" else _large_view_context_compute
    context_vectors.copy_(
        compute_large(
            query_chunks,
            qkv_blocks[:, :, :, 1],
            qkv_blocks[:, :, :, 2],
            valid_mask_chunks,
            pos_proj_weight,
            inv_timescales,
            soft_cap,
            causal_mask,
            pos_indices,
        )[:, :q_time]
    )
