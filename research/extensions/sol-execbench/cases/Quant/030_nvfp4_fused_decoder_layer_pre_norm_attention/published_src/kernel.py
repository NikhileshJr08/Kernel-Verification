import torch
import torch.nn.functional as F

from common_nvfp4_gemm import blockscaled_nvfp4_gemm


HIDDEN_SIZE = 5120
NUM_ATTENTION_HEADS = 40
NUM_KEY_VALUE_HEADS = 8
NUM_KEY_VALUE_GROUPS = NUM_ATTENTION_HEADS // NUM_KEY_VALUE_HEADS
HEAD_DIM = 128
Q_OUT_FEATURES = NUM_ATTENTION_HEADS * HEAD_DIM
KV_OUT_FEATURES = NUM_KEY_VALUE_HEADS * HEAD_DIM
ATTN_SCALE = HEAD_DIM ** -0.5

SMALL_QO_TOKENS = 1024
SMALL_KV_TOKENS = 2048
LONG_SEQ_THRESHOLD = 4096
LONG_Q_CHUNK = 1536

_E2M1_MAX = 6.0
_FP8_MAX = 448.0
_CONST_FLOAT8_MAX = None
_CONST_FP32_MAX = None


def _ensure_constants(device: torch.device) -> None:
    global _CONST_FLOAT8_MAX, _CONST_FP32_MAX
    if _CONST_FLOAT8_MAX is None or _CONST_FLOAT8_MAX.device != device:
        _CONST_FLOAT8_MAX = torch.tensor(_FP8_MAX, device=device, dtype=torch.float32)
        _CONST_FP32_MAX = torch.tensor(
            torch.finfo(torch.float32).max,
            device=device,
            dtype=torch.float32,
        )


@torch.compile(dynamic=True, fullgraph=True)
def _encode_fp4_from_blocked(blocked_3d: torch.Tensor, encode_scale_2d: torch.Tensor):
    scaled = blocked_3d.to(torch.float32) * encode_scale_2d.unsqueeze(-1)
    rows = blocked_3d.shape[0]
    k = blocked_3d.shape[1] * blocked_3d.shape[2]
    flat = scaled.clamp(-_E2M1_MAX, _E2M1_MAX).reshape(rows, k)
    abs_flat = flat.abs()
    code = (
        (abs_flat > 0.25).to(torch.uint8)
        + (abs_flat >= 0.75).to(torch.uint8)
        + (abs_flat > 1.25).to(torch.uint8)
        + (abs_flat >= 1.75).to(torch.uint8)
        + (abs_flat > 2.5).to(torch.uint8)
        + (abs_flat >= 3.5).to(torch.uint8)
        + (abs_flat > 5.0).to(torch.uint8)
    )
    code = code + (flat < 0.0).to(torch.uint8) * 8
    code_pairs = code.reshape(rows, k // 2, 2)
    return code_pairs[..., 0] + code_pairs[..., 1] * 16


def _quantize_activation_exact(x_2d: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    rows, cols = x_2d.shape
    tensor = x_2d.view(1, rows, cols).to(torch.float32)

    blocked = tensor.view(1, rows, cols // 16, 16)
    block_max = blocked.abs().amax(dim=-1)
    global_amax = block_max.amax().to(torch.float32)

    decode_scale = block_max / _E2M1_MAX
    global_encode_scale = torch.clamp(
        torch.div(_CONST_FLOAT8_MAX * _E2M1_MAX, global_amax.clamp(min=1e-30)),
        max=_CONST_FP32_MAX,
    )
    global_encode_scale = torch.where(
        global_encode_scale == 0.0,
        torch.ones_like(global_encode_scale),
        global_encode_scale,
    )
    global_decode_scale = torch.div(1.0, global_encode_scale)

    decode_scale = torch.clamp(
        decode_scale * global_encode_scale,
        min=-_CONST_FLOAT8_MAX,
        max=_CONST_FLOAT8_MAX,
    )
    decode_scale_fp8 = decode_scale.to(dtype=torch.float8_e4m3fn)

    encode_scale = torch.clamp(
        torch.div(1.0, decode_scale_fp8.to(torch.float32) * global_decode_scale),
        max=_CONST_FP32_MAX,
    )
    packed = _encode_fp4_from_blocked(
        blocked.reshape(rows, cols // 16, 16),
        encode_scale.reshape(rows, cols // 16),
    ).contiguous()

    return (
        packed.view(torch.float4_e2m1fn_x2),
        decode_scale_fp8.squeeze(0),
        global_decode_scale,
    )


def _convert_scale_to_blocked(scale: torch.Tensor) -> torch.Tensor:
    rows, cols = scale.shape
    return (
        scale.view(rows // 128, 128, cols // 4, 4)
        .permute(0, 2, 1, 3)
        .reshape(-1, 4, 32, 4)
        .transpose(1, 2)
        .reshape(-1, 32, 16)
        .flatten()
    )


def _scalar_fp32(value, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(value, device=device, dtype=torch.float32)


def _pad_rows_to_multiple_of_128(x_2d: torch.Tensor) -> tuple[torch.Tensor, int]:
    rows = x_2d.shape[0]
    padded_rows = ((rows + 127) // 128) * 128
    if padded_rows == rows:
        return x_2d.contiguous(), rows
    return F.pad(x_2d.contiguous(), (0, 0, 0, padded_rows - rows)), rows


def _prepare_quantized_rows(x_2d: torch.Tensor):
    padded, logical_rows = _pad_rows_to_multiple_of_128(x_2d)
    act_fp4, act_scale, act_global_decode_scale = _quantize_activation_exact(padded)
    act_scale_blocked = _convert_scale_to_blocked(act_scale)
    return padded, logical_rows, act_fp4, act_scale, act_scale_blocked, act_global_decode_scale


def _torch_nvfp4_proj(
    act_fp4: torch.Tensor,
    act_scale_blocked: torch.Tensor,
    act_global_decode_scale: torch.Tensor,
    weight_fp4: torch.Tensor,
    weight_scale: torch.Tensor,
    weight_global_decode_scale,
    logical_rows: int,
) -> torch.Tensor:
    weight_scale_blocked = _convert_scale_to_blocked(weight_scale)
    proj = torch._scaled_mm(
        act_fp4,
        weight_fp4.t(),
        act_scale_blocked,
        weight_scale_blocked,
        out_dtype=torch.bfloat16,
    )
    total_scale = act_global_decode_scale * _scalar_fp32(weight_global_decode_scale, proj.device)
    return (proj[:logical_rows] * total_scale).to(torch.bfloat16)


def _cute_nvfp4_proj(
    act_fp4: torch.Tensor,
    act_scale: torch.Tensor,
    act_global_decode_scale: torch.Tensor,
    weight_fp4: torch.Tensor,
    weight_scale: torch.Tensor,
    weight_global_decode_scale,
    logical_rows: int,
) -> torch.Tensor:
    proj = blockscaled_nvfp4_gemm(
        act_fp4,
        weight_fp4,
        act_scale,
        weight_scale,
        use_compiled_scale_layout=act_scale.shape[0] >= 4096,
    )
    total_scale = act_global_decode_scale * _scalar_fp32(weight_global_decode_scale, proj.device)
    return (proj[:logical_rows] * total_scale).to(torch.bfloat16)


def _cute_large_qkv_proj(
    act_fp4: torch.Tensor,
    act_scale: torch.Tensor,
    act_global_decode_scale: torch.Tensor,
    q_weight_fp4: torch.Tensor,
    q_scale: torch.Tensor,
    q_global_decode_scale,
    k_weight_fp4: torch.Tensor,
    k_scale: torch.Tensor,
    k_global_decode_scale,
    v_weight_fp4: torch.Tensor,
    v_scale: torch.Tensor,
    v_global_decode_scale,
    logical_rows: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    qkv_weight = torch.cat(
        [
            q_weight_fp4.view(torch.uint8),
            k_weight_fp4.view(torch.uint8),
            v_weight_fp4.view(torch.uint8),
        ],
        dim=0,
    ).contiguous().view(torch.float4_e2m1fn_x2)
    qkv_scale = torch.cat([q_scale, k_scale, v_scale], dim=0).contiguous()

    qkv_out = blockscaled_nvfp4_gemm(
        act_fp4,
        qkv_weight,
        act_scale,
        qkv_scale,
        use_compiled_scale_layout=act_scale.shape[0] >= 4096,
    )[:logical_rows]

    device = qkv_out.device
    q_scale_total = act_global_decode_scale * _scalar_fp32(q_global_decode_scale, device)
    k_scale_total = act_global_decode_scale * _scalar_fp32(k_global_decode_scale, device)
    v_scale_total = act_global_decode_scale * _scalar_fp32(v_global_decode_scale, device)

    q = (qkv_out[:, :Q_OUT_FEATURES] * q_scale_total).to(torch.bfloat16)
    k_start = Q_OUT_FEATURES
    v_start = Q_OUT_FEATURES + KV_OUT_FEATURES
    k = (qkv_out[:, k_start:v_start] * k_scale_total).to(torch.bfloat16)
    v = (qkv_out[:, v_start:] * v_scale_total).to(torch.bfloat16)
    return q, k, v


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def _attention_mask_is_zero(attention_mask: torch.Tensor) -> bool:
    return bool(torch.count_nonzero(attention_mask).item() == 0)


def _repeat_kv(x: torch.Tensor) -> torch.Tensor:
    if NUM_KEY_VALUE_GROUPS == 1:
        return x
    batch, num_kv_heads, seq_len, head_dim = x.shape
    return x[:, :, None, :, :].expand(
        batch,
        num_kv_heads,
        NUM_KEY_VALUE_GROUPS,
        seq_len,
        head_dim,
    ).reshape(batch, num_kv_heads * NUM_KEY_VALUE_GROUPS, seq_len, head_dim)


def _full_attention(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    attention_mask: torch.Tensor,
    mask_is_zero: bool,
) -> torch.Tensor:
    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * ATTN_SCALE
    if not mask_is_zero:
        attn_weights = attn_weights + attention_mask
    attn_probs = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
    return torch.matmul(attn_probs, value_states)


def _full_attention_bmm(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    attention_mask: torch.Tensor,
    mask_is_zero: bool,
) -> torch.Tensor:
    batch_size, num_heads, seq_len, _ = query_states.shape
    batch_heads = batch_size * num_heads
    query_flat = query_states.reshape(batch_heads, seq_len, HEAD_DIM)
    key_states_t = key_states.transpose(2, 3).reshape(batch_heads, HEAD_DIM, seq_len)
    attn_weights = torch.bmm(query_flat, key_states_t).reshape(
        batch_size,
        num_heads,
        seq_len,
        seq_len,
    ) * ATTN_SCALE
    if not mask_is_zero:
        attn_weights = attn_weights + attention_mask
    attn_probs = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
    value_flat = value_states.reshape(batch_heads, seq_len, HEAD_DIM)
    return torch.bmm(
        attn_probs.reshape(batch_heads, seq_len, seq_len),
        value_flat,
    ).reshape(batch_size, num_heads, seq_len, HEAD_DIM)


def _full_attention_bmm_softmax_out(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
) -> torch.Tensor:
    batch_size, num_heads, seq_len, _ = query_states.shape
    batch_heads = batch_size * num_heads
    query_flat = query_states.reshape(batch_heads, seq_len, HEAD_DIM)
    key_states_t = key_states.transpose(2, 3).reshape(batch_heads, HEAD_DIM, seq_len)
    value_flat = value_states.reshape(batch_heads, seq_len, HEAD_DIM)

    scores = torch.bmm(query_flat, key_states_t)
    scores.mul_(ATTN_SCALE)
    torch.ops.aten._softmax.out(scores, -1, False, out=scores)
    return torch.bmm(scores, value_flat).reshape(batch_size, num_heads, seq_len, HEAD_DIM)


def _full_attention_grouped_bmm_softmax_out(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
) -> torch.Tensor:
    batch_size, _, seq_len, _ = query_states.shape
    batch_kv_heads = batch_size * NUM_KEY_VALUE_HEADS

    query_grouped = query_states.reshape(
        batch_size,
        NUM_KEY_VALUE_HEADS,
        NUM_KEY_VALUE_GROUPS,
        seq_len,
        HEAD_DIM,
    )
    query_flat = query_grouped.reshape(
        batch_kv_heads,
        NUM_KEY_VALUE_GROUPS * seq_len,
        HEAD_DIM,
    )
    key_states_t = key_states.reshape(batch_kv_heads, seq_len, HEAD_DIM).transpose(1, 2)
    value_flat = value_states.reshape(batch_kv_heads, seq_len, HEAD_DIM)

    scores = query_flat.new_empty((batch_kv_heads, NUM_KEY_VALUE_GROUPS * seq_len, seq_len))
    torch.bmm(query_flat, key_states_t, out=scores)
    scores.mul_(ATTN_SCALE)
    torch.ops.aten._softmax.out(scores, -1, False, out=scores)

    output_flat = query_flat.new_empty(query_flat.shape)
    torch.bmm(scores, value_flat, out=output_flat)
    return output_flat.reshape(
        batch_size,
        NUM_KEY_VALUE_HEADS,
        NUM_KEY_VALUE_GROUPS,
        seq_len,
        HEAD_DIM,
    ).reshape(batch_size, NUM_ATTENTION_HEADS, seq_len, HEAD_DIM)


def _chunked_attention_softmax_out(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
) -> torch.Tensor:
    batch_size, num_heads, seq_len, _ = query_states.shape
    batch_heads = batch_size * num_heads
    query_flat = query_states.reshape(batch_heads, seq_len, HEAD_DIM)
    key_states_t = key_states.transpose(2, 3).reshape(batch_heads, HEAD_DIM, seq_len)
    value_flat = value_states.reshape(batch_heads, seq_len, HEAD_DIM)
    output_flat = torch.empty_like(query_flat)

    for q_start in range(0, seq_len, LONG_Q_CHUNK):
        q_end = min(q_start + LONG_Q_CHUNK, seq_len)
        scores = torch.bmm(query_flat[:, q_start:q_end, :], key_states_t)
        scores.mul_(ATTN_SCALE)
        torch.ops.aten._softmax.out(scores, -1, False, out=scores)
        output_flat[:, q_start:q_end, :].copy_(torch.bmm(scores, value_flat))

    return output_flat.reshape(batch_size, num_heads, seq_len, HEAD_DIM)


def _chunked_attention_softmax_out_qchunk_buf(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    q_chunk: int,
) -> torch.Tensor:
    batch_size, num_heads, seq_len, _ = query_states.shape
    batch_heads = batch_size * num_heads
    query_flat = query_states.reshape(batch_heads, seq_len, HEAD_DIM)
    key_states_t = key_states.transpose(2, 3).reshape(batch_heads, HEAD_DIM, seq_len)
    value_flat = value_states.reshape(batch_heads, seq_len, HEAD_DIM)
    output_flat = torch.empty_like(query_flat)
    score_buffer = query_flat.new_empty((batch_heads, q_chunk, seq_len))

    for q_start in range(0, seq_len, q_chunk):
        q_end = min(q_start + q_chunk, seq_len)
        score_chunk = score_buffer[:, : q_end - q_start, :]
        torch.bmm(query_flat[:, q_start:q_end, :], key_states_t, out=score_chunk)
        score_chunk.mul_(ATTN_SCALE)
        torch.ops.aten._softmax.out(score_chunk, -1, False, out=score_chunk)
        torch.bmm(score_chunk, value_flat, out=output_flat[:, q_start:q_end, :])

    return output_flat.reshape(batch_size, num_heads, seq_len, HEAD_DIM)


def _chunked_attention_softmax_out_qchunk(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    q_chunk: int,
) -> torch.Tensor:
    batch_size, num_heads, seq_len, _ = query_states.shape
    batch_heads = batch_size * num_heads
    query_flat = query_states.reshape(batch_heads, seq_len, HEAD_DIM)
    key_states_t = key_states.transpose(2, 3).reshape(batch_heads, HEAD_DIM, seq_len)
    value_flat = value_states.reshape(batch_heads, seq_len, HEAD_DIM)
    output_flat = torch.empty_like(query_flat)

    for q_start in range(0, seq_len, q_chunk):
        q_end = min(q_start + q_chunk, seq_len)
        scores = torch.bmm(query_flat[:, q_start:q_end, :], key_states_t)
        scores.mul_(ATTN_SCALE)
        torch.ops.aten._softmax.out(scores, -1, False, out=scores)
        output_flat[:, q_start:q_end, :].copy_(torch.bmm(scores, value_flat))

    return output_flat.reshape(batch_size, num_heads, seq_len, HEAD_DIM)


def _chunked_attention(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    attention_mask: torch.Tensor,
    mask_is_zero: bool,
) -> torch.Tensor:
    batch_size, num_heads, seq_len, _ = query_states.shape
    batch_heads = batch_size * num_heads
    output = torch.empty_like(query_states)
    key_states_t = key_states.transpose(2, 3).reshape(batch_heads, HEAD_DIM, seq_len)
    value_flat = value_states.reshape(batch_heads, seq_len, HEAD_DIM)

    for q_start in range(0, seq_len, LONG_Q_CHUNK):
        q_end = min(q_start + LONG_Q_CHUNK, seq_len)
        q_chunk = query_states[:, :, q_start:q_end, :].reshape(
            batch_heads,
            q_end - q_start,
            HEAD_DIM,
        )
        scores = torch.bmm(q_chunk, key_states_t).reshape(
            batch_size,
            num_heads,
            q_end - q_start,
            seq_len,
        ) * ATTN_SCALE
        if not mask_is_zero:
            scores = scores + attention_mask[:, :, q_start:q_end, :]
        probs = F.softmax(scores, dim=-1, dtype=torch.float32).to(query_states.dtype)
        output[:, :, q_start:q_end, :].copy_(
            torch.bmm(
                probs.reshape(batch_heads, q_end - q_start, seq_len),
                value_flat,
            ).reshape(batch_size, num_heads, q_end - q_start, HEAD_DIM)
        )

    return output


@torch.no_grad()
def run(
    hidden_states: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    attention_mask: torch.Tensor,
    input_layernorm_weight: torch.Tensor,
    q_weight_fp4: torch.Tensor,
    q_scale: torch.Tensor,
    q_global_scale: float,
    k_weight_fp4: torch.Tensor,
    k_scale: torch.Tensor,
    k_global_scale: float,
    v_weight_fp4: torch.Tensor,
    v_scale: torch.Tensor,
    v_global_scale: float,
    o_weight_fp4: torch.Tensor,
    o_scale: torch.Tensor,
    o_global_scale: float,
    q_norm_weight: torch.Tensor,
    k_norm_weight: torch.Tensor,
    rms_norm_eps: float,
    attn_output: torch.Tensor,
):
    eps = float(rms_norm_eps.item()) if torch.is_tensor(rms_norm_eps) else float(rms_norm_eps)
    batch_size, seq_len, _ = hidden_states.shape
    tokens = batch_size * seq_len
    device = hidden_states.device
    _ensure_constants(device)

    normed_hidden = F.rms_norm(
        hidden_states,
        (HIDDEN_SIZE,),
        input_layernorm_weight,
        eps,
    )
    hidden_flat = normed_hidden.reshape(tokens, HIDDEN_SIZE)
    _, logical_rows, act_fp4, act_scale, act_scale_blocked, act_global_decode_scale = _prepare_quantized_rows(hidden_flat)

    if tokens <= SMALL_QO_TOKENS:
        query_flat = _torch_nvfp4_proj(
            act_fp4,
            act_scale_blocked,
            act_global_decode_scale,
            q_weight_fp4,
            q_scale,
            q_global_scale,
            logical_rows,
        )
        key_flat = _torch_nvfp4_proj(
            act_fp4,
            act_scale_blocked,
            act_global_decode_scale,
            k_weight_fp4,
            k_scale,
            k_global_scale,
            logical_rows,
        )
        value_flat = _torch_nvfp4_proj(
            act_fp4,
            act_scale_blocked,
            act_global_decode_scale,
            v_weight_fp4,
            v_scale,
            v_global_scale,
            logical_rows,
        )
    elif tokens <= SMALL_KV_TOKENS:
        query_flat = _cute_nvfp4_proj(
            act_fp4,
            act_scale,
            act_global_decode_scale,
            q_weight_fp4,
            q_scale,
            q_global_scale,
            logical_rows,
        )
        key_flat = _torch_nvfp4_proj(
            act_fp4,
            act_scale_blocked,
            act_global_decode_scale,
            k_weight_fp4,
            k_scale,
            k_global_scale,
            logical_rows,
        )
        value_flat = _torch_nvfp4_proj(
            act_fp4,
            act_scale_blocked,
            act_global_decode_scale,
            v_weight_fp4,
            v_scale,
            v_global_scale,
            logical_rows,
        )
    else:
        query_flat, key_flat, value_flat = _cute_large_qkv_proj(
            act_fp4,
            act_scale,
            act_global_decode_scale,
            q_weight_fp4,
            q_scale,
            q_global_scale,
            k_weight_fp4,
            k_scale,
            k_global_scale,
            v_weight_fp4,
            v_scale,
            v_global_scale,
            logical_rows,
        )

    query_states = query_flat.view(batch_size, seq_len, NUM_ATTENTION_HEADS, HEAD_DIM)
    key_states = key_flat.view(batch_size, seq_len, NUM_KEY_VALUE_HEADS, HEAD_DIM)
    value_states = value_flat.view(batch_size, seq_len, NUM_KEY_VALUE_HEADS, HEAD_DIM)

    query_states = F.rms_norm(query_states, (HEAD_DIM,), q_norm_weight, eps).transpose(1, 2)
    key_states = F.rms_norm(key_states, (HEAD_DIM,), k_norm_weight, eps).transpose(1, 2)
    value_states = value_states.transpose(1, 2)

    cos_expanded = cos.unsqueeze(1)
    sin_expanded = sin.unsqueeze(1)
    query_states = (query_states * cos_expanded) + (_rotate_half(query_states) * sin_expanded)
    key_states = (key_states * cos_expanded) + (_rotate_half(key_states) * sin_expanded)

    mask_is_zero = False
    if seq_len >= 1024 or batch_size * seq_len * seq_len >= 4_000_000:
        mask_is_zero = bool(torch.count_nonzero(attention_mask).item() == 0)

    if mask_is_zero and batch_size == 2 and seq_len == 4096:
        context = _full_attention_grouped_bmm_softmax_out(query_states, key_states, value_states)
    elif mask_is_zero and batch_size == 1 and seq_len == 8192:
        context = _full_attention_grouped_bmm_softmax_out(query_states, key_states, value_states)
    else:
        key_states = _repeat_kv(key_states).contiguous()
        value_states = _repeat_kv(value_states).contiguous()

        if seq_len >= LONG_SEQ_THRESHOLD:
            if mask_is_zero and batch_size == 2 and seq_len == 4096:
                context = _full_attention_bmm_softmax_out(query_states, key_states, value_states)
            elif mask_is_zero and batch_size == 1 and seq_len == 8192:
                context = _full_attention_bmm_softmax_out(query_states, key_states, value_states)
            else:
                context = _chunked_attention(query_states, key_states, value_states, attention_mask, mask_is_zero)
        elif mask_is_zero and batch_size == 2 and seq_len == 2048:
            context = _full_attention_bmm_softmax_out(query_states, key_states, value_states)
        elif mask_is_zero and batch_size == 1 and seq_len == 2048:
            context = _full_attention_bmm_softmax_out(query_states, key_states, value_states)
        elif mask_is_zero and batch_size == 16 and seq_len == 512:
            context = _full_attention_bmm_softmax_out(query_states, key_states, value_states)
        elif mask_is_zero and batch_size == 16 and seq_len == 1024:
            context = _chunked_attention_softmax_out_qchunk_buf(query_states, key_states, value_states, 256)
        elif mask_is_zero and batch_size == 32 and seq_len == 512:
            context = _full_attention_bmm_softmax_out(query_states, key_states, value_states)
        elif mask_is_zero and batch_size == 64 and seq_len == 256:
            context = _full_attention_bmm_softmax_out(query_states, key_states, value_states)
        elif mask_is_zero and batch_size == 1 and seq_len == 1024:
            context = _full_attention_bmm_softmax_out(query_states, key_states, value_states)
        elif batch_size == 64 and seq_len == 128:
            exact_mask_is_zero = _attention_mask_is_zero(attention_mask)
            if exact_mask_is_zero:
                context = _full_attention_bmm_softmax_out(query_states, key_states, value_states)
            else:
                context = _full_attention(query_states, key_states, value_states, attention_mask, False)
        elif batch_size == 32 and seq_len == 256:
            exact_mask_is_zero = _attention_mask_is_zero(attention_mask)
            if exact_mask_is_zero:
                context = _full_attention_bmm_softmax_out(query_states, key_states, value_states)
            else:
                context = _full_attention(query_states, key_states, value_states, attention_mask, False)
        elif batch_size == 1 and seq_len >= 2048:
            context = _full_attention_bmm(query_states, key_states, value_states, attention_mask, mask_is_zero)
        else:
            context = _full_attention(query_states, key_states, value_states, attention_mask, mask_is_zero)

    context_flat = context.transpose(1, 2).contiguous().reshape(tokens, Q_OUT_FEATURES)
    _, logical_rows, out_fp4, out_scale, out_scale_blocked, out_global_decode_scale = _prepare_quantized_rows(context_flat)

    if tokens <= SMALL_QO_TOKENS:
        out_flat = _torch_nvfp4_proj(
            out_fp4,
            out_scale_blocked,
            out_global_decode_scale,
            o_weight_fp4,
            o_scale,
            o_global_scale,
            logical_rows,
        )
    else:
        out_flat = _cute_nvfp4_proj(
            out_fp4,
            out_scale,
            out_global_decode_scale,
            o_weight_fp4,
            o_scale,
            o_global_scale,
            logical_rows,
        )

    attn_output.copy_(out_flat.view(batch_size, seq_len, HIDDEN_SIZE))
