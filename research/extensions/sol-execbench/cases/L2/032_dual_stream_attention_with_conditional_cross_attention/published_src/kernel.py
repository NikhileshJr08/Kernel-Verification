import math
from contextlib import contextmanager
import torch
import torch.nn.functional as F
import rope_half_stacked_qk_rows8x32

try:
    from torch.nn.attention import SDPBackend, sdpa_kernel
except Exception:  # pragma: no cover
    SDPBackend = None
    sdpa_kernel = None

NUM_HEADS = 24
HEAD_DIM = 128
HIDDEN_SIZE = 3072
TEXT_HIDDEN_SIZE = 4096
ROPE_DIM_PER_AXIS = HEAD_DIM // 3
ROPE_HALF_DIM = ROPE_DIM_PER_AXIS // 2
ATTN_SCALE = 1.0 / math.sqrt(HEAD_DIM)


_CUTE_LONG_ROPE_ROWS = frozenset(((1, 3136, 512, 1), (4, 1024, 256, 1), (2, 2048, 512, 1), (1, 1973, 512, 1)))


def _use_cute_long_rope_path(batch_size, seq_len, text_seq_len, is_joint_block, q, k):
    return (
        is_joint_block == 1
        and q.device.type == "cuda"
        and (batch_size, seq_len, text_seq_len, is_joint_block) in _CUTE_LONG_ROPE_ROWS
        and q.dtype == torch.float32
        and k.dtype == torch.float32
        and q.is_contiguous()
        and k.is_contiguous()
    )


def _as_int(value):
    if isinstance(value, torch.Tensor):
        return int(value.item())
    return int(value)


@contextmanager
def _tf32_matmul_scope(enabled):
    if not enabled:
        yield
        return
    prev_allow_tf32 = torch.backends.cuda.matmul.allow_tf32
    prev_precision = torch.get_float32_matmul_precision()
    torch.backends.cuda.matmul.allow_tf32 = True
    if prev_precision != "high":
        torch.set_float32_matmul_precision("high")
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev_allow_tf32
        if prev_precision != "high":
            torch.set_float32_matmul_precision(prev_precision)


@contextmanager
def _fp16_autocast_scope(enabled):
    if enabled and torch.cuda.is_available():
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            yield
    else:
        yield


def _gather_rope(pos_idx, rope_cos, rope_sin, target_dtype=None):
    cos_gathered = rope_cos[pos_idx].unsqueeze(0).unsqueeze(2)
    sin_gathered = rope_sin[pos_idx].unsqueeze(0).unsqueeze(2)
    if target_dtype is not None and cos_gathered.dtype != target_dtype:
        cos_gathered = cos_gathered.to(target_dtype)
        sin_gathered = sin_gathered.to(target_dtype)
    return cos_gathered, sin_gathered


def _apply_rope_axis_gathered(x_part, cos_gathered, sin_gathered):
    rope_dim = x_part.shape[-1]
    half_dim = rope_dim // 2
    x_first = x_part[..., :half_dim]
    x_second = x_part[..., half_dim:]
    cos_first = cos_gathered[..., :half_dim]
    sin_first = sin_gathered[..., :half_dim]
    rotated_first = x_first * cos_first - x_second * sin_first
    rotated_second = x_first * sin_first + x_second * cos_first
    return torch.cat([rotated_first, rotated_second], dim=-1)



def _apply_rope_axis_gathered_inplace_(x_part, cos_gathered, sin_gathered):
    rope_dim = x_part.shape[-1]
    half_dim = rope_dim // 2
    x_first = x_part[..., :half_dim]
    x_second = x_part[..., half_dim:]
    cos_first = cos_gathered[..., :half_dim]
    sin_first = sin_gathered[..., :half_dim]
    x_first_saved = x_first.clone()
    x_first.mul_(cos_first).addcmul_(x_second, sin_first, value=-1)
    x_second.mul_(cos_first).addcmul_(x_first_saved, sin_first)


def _apply_3axis_rope_gathered_inplace_(
    x,
    axis0_gathered,
    axis1_gathered,
    axis2_gathered,
):
    cos0, sin0 = axis0_gathered
    cos1, sin1 = axis1_gathered
    cos2, sin2 = axis2_gathered
    _apply_rope_axis_gathered_inplace_(x[..., :ROPE_DIM_PER_AXIS], cos0, sin0)
    _apply_rope_axis_gathered_inplace_(x[..., ROPE_DIM_PER_AXIS:2 * ROPE_DIM_PER_AXIS], cos1, sin1)
    _apply_rope_axis_gathered_inplace_(x[..., 2 * ROPE_DIM_PER_AXIS:3 * ROPE_DIM_PER_AXIS], cos2, sin2)
    return x


def _gather_rope_half(pos_idx, rope_cos, rope_sin):
    cos_gathered = rope_cos[:, :ROPE_HALF_DIM].index_select(0, pos_idx).view(1, pos_idx.shape[0], 1, ROPE_HALF_DIM)
    sin_gathered = rope_sin[:, :ROPE_HALF_DIM].index_select(0, pos_idx).view(1, pos_idx.shape[0], 1, ROPE_HALF_DIM)
    return cos_gathered, sin_gathered


def _gather_3axis_rope_half_stacked_rows(
    pos_idx_axis0,
    pos_idx_axis1,
    pos_idx_axis2,
    rope_cos_axis0,
    rope_sin_axis0,
    rope_cos_axis1,
    rope_sin_axis1,
    rope_cos_axis2,
    rope_sin_axis2,
):
    seq_len = pos_idx_axis0.shape[0]
    cos_stacked = torch.empty((3, seq_len, ROPE_HALF_DIM), device=rope_cos_axis0.device, dtype=rope_cos_axis0.dtype)
    sin_stacked = torch.empty_like(cos_stacked)
    torch.index_select(rope_cos_axis0[:, :ROPE_HALF_DIM], 0, pos_idx_axis0, out=cos_stacked[0])
    torch.index_select(rope_sin_axis0[:, :ROPE_HALF_DIM], 0, pos_idx_axis0, out=sin_stacked[0])
    torch.index_select(rope_cos_axis1[:, :ROPE_HALF_DIM], 0, pos_idx_axis1, out=cos_stacked[1])
    torch.index_select(rope_sin_axis1[:, :ROPE_HALF_DIM], 0, pos_idx_axis1, out=sin_stacked[1])
    torch.index_select(rope_cos_axis2[:, :ROPE_HALF_DIM], 0, pos_idx_axis2, out=cos_stacked[2])
    torch.index_select(rope_sin_axis2[:, :ROPE_HALF_DIM], 0, pos_idx_axis2, out=sin_stacked[2])
    return cos_stacked, sin_stacked


def _stack_3axis_rope_half_gathered(axis0_gathered, axis1_gathered, axis2_gathered):
    return (
        torch.stack((axis0_gathered[0], axis1_gathered[0], axis2_gathered[0]), dim=-2),
        torch.stack((axis0_gathered[1], axis1_gathered[1], axis2_gathered[1]), dim=-2),
    )


def _apply_3axis_rope_half_stacked_inplace_(x, stacked_cos_sin, workspace):
    cos_stacked, sin_stacked = stacked_cos_sin
    rope = x[..., :3 * ROPE_DIM_PER_AXIS].view(*x.shape[:-1], 3, ROPE_DIM_PER_AXIS)
    x_first = rope[..., :ROPE_HALF_DIM]
    x_second = rope[..., ROPE_HALF_DIM:]
    workspace.copy_(x_first)
    x_first.mul_(cos_stacked).addcmul_(x_second, sin_stacked, value=-1)
    x_second.mul_(cos_stacked).addcmul_(workspace, sin_stacked)
    return x

def _apply_3axis_rope_gathered(
    x,
    axis0_gathered,
    axis1_gathered,
    axis2_gathered,
):
    x_axis0 = x[..., :ROPE_DIM_PER_AXIS]
    x_axis1 = x[..., ROPE_DIM_PER_AXIS:2 * ROPE_DIM_PER_AXIS]
    x_axis2 = x[..., 2 * ROPE_DIM_PER_AXIS:3 * ROPE_DIM_PER_AXIS]
    x_rest = x[..., 3 * ROPE_DIM_PER_AXIS:]
    cos0, sin0 = axis0_gathered
    cos1, sin1 = axis1_gathered
    cos2, sin2 = axis2_gathered
    return torch.cat(
        [
            _apply_rope_axis_gathered(x_axis0, cos0, sin0),
            _apply_rope_axis_gathered(x_axis1, cos1, sin1),
            _apply_rope_axis_gathered(x_axis2, cos2, sin2),
            x_rest,
        ],
        dim=-1,
    )


def _exact_attention(q, k, v):
    attn_scores = torch.matmul(q, k.transpose(-2, -1)) * ATTN_SCALE
    attn_probs = F.softmax(attn_scores, dim=-1)
    return torch.matmul(attn_probs, v)


def _use_joint_sdpa_path(seq_len, text_seq_len):
    return seq_len >= 256 and text_seq_len >= 64


def _use_nonjoint_sdpa_path(batch_size, seq_len):
    return seq_len >= 256 or (seq_len >= 128 and batch_size * seq_len >= 8192)


def _bf16_sdpa(q, k, v):
    if q.device.type == "cuda" and sdpa_kernel is not None and SDPBackend is not None:
        with sdpa_kernel([SDPBackend.CUDNN_ATTENTION, SDPBackend.MATH]):
            return F.scaled_dot_product_attention(
                q.to(torch.bfloat16),
                k.to(torch.bfloat16),
                v.to(torch.bfloat16),
                dropout_p=0.0,
                is_causal=False,
                scale=ATTN_SCALE,
            ).to(q.dtype)
    return F.scaled_dot_product_attention(
        q,
        k,
        v,
        dropout_p=0.0,
        is_causal=False,
        scale=ATTN_SCALE,
    )


def _bf16_flash_sdpa(q, k, v):
    if q.device.type == "cuda" and sdpa_kernel is not None and SDPBackend is not None:
        with sdpa_kernel([SDPBackend.FLASH_ATTENTION]):
            return F.scaled_dot_product_attention(
                q.to(torch.bfloat16),
                k.to(torch.bfloat16),
                v.to(torch.bfloat16),
                dropout_p=0.0,
                is_causal=False,
                scale=ATTN_SCALE,
            ).to(q.dtype)
    return F.scaled_dot_product_attention(
        q.to(torch.bfloat16),
        k.to(torch.bfloat16),
        v.to(torch.bfloat16),
        dropout_p=0.0,
        is_causal=False,
        scale=ATTN_SCALE,
    ).to(q.dtype)


def _lowp_sdpa(q, k, v):
    if q.device.type == "cuda" and sdpa_kernel is not None and SDPBackend is not None:
        with sdpa_kernel([SDPBackend.CUDNN_ATTENTION, SDPBackend.MATH]):
            return F.scaled_dot_product_attention(
                q,
                k,
                v,
                dropout_p=0.0,
                is_causal=False,
                scale=ATTN_SCALE,
            )
    return F.scaled_dot_product_attention(
        q,
        k,
        v,
        dropout_p=0.0,
        is_causal=False,
        scale=ATTN_SCALE,
    )


def _lowp_flash_sdpa(q, k, v):
    if q.device.type == "cuda" and sdpa_kernel is not None and SDPBackend is not None:
        with sdpa_kernel([SDPBackend.FLASH_ATTENTION]):
            return F.scaled_dot_product_attention(
                q,
                k,
                v,
                dropout_p=0.0,
                is_causal=False,
                scale=ATTN_SCALE,
            )
    return F.scaled_dot_product_attention(
        q,
        k,
        v,
        dropout_p=0.0,
        is_causal=False,
        scale=ATTN_SCALE,
    )


def _use_joint_flash_bf16_sdpa(batch_size, seq_len, text_seq_len):
    return False


def _use_joint_bf16_sdpa(batch_size, seq_len, text_seq_len):
    return (
        seq_len >= 900
        or text_seq_len >= 256
        or batch_size >= 4
        or (batch_size == 1 and seq_len == 512 and text_seq_len == 128)
    )


def _use_nonjoint_bf16_sdpa(batch_size, seq_len):
    return (seq_len >= 128 and batch_size * seq_len >= 8192) or (batch_size == 1 and seq_len == 512)


def _use_nonjoint_flash_bf16_sdpa(batch_size, seq_len):
    return (batch_size, seq_len) in ((32, 256), (64, 128))



def _use_nonjoint_exact_qkv_projection(batch_size, seq_len):
    return False


def _use_nonjoint_fp16_preout_projection(batch_size, seq_len):
    return (batch_size, seq_len) in ((32, 256), (64, 128))



def _use_nonjoint_nonowned_out_projection(batch_size, seq_len):
    return (batch_size, seq_len) == (1, 512)


def _use_unified_nonjoint_used3_modulation(batch_size, seq_len):
    return seq_len <= 512 and (batch_size == 1 or batch_size * seq_len >= 8192)


_JOINT_UNIFIED_USED3_ROWS = frozenset(((1, 3136, 512), (4, 1024, 256), (2, 2048, 512), (4, 541, 128), (8, 256, 64)))


def _use_joint_used3_modulation(batch_size, seq_len, text_seq_len):
    return (batch_size, seq_len, text_seq_len) in _JOINT_UNIFIED_USED3_ROWS


def _use_joint_exact_projection(batch_size, seq_len, text_seq_len):
    return False


def _project_qkv_packed(
    hidden_states_modulated,
    to_q_weight,
    to_q_bias,
    to_k_weight,
    to_k_bias,
    to_v_weight,
    to_v_bias,
):
    batch_size, seq_len, _ = hidden_states_modulated.shape
    hidden_states_2d = hidden_states_modulated.reshape(batch_size * seq_len, HIDDEN_SIZE)
    qkv_weight = torch.cat([to_q_weight, to_k_weight, to_v_weight], dim=0)
    qkv_bias = torch.cat([to_q_bias, to_k_bias, to_v_bias], dim=0)
    qkv = torch.empty(
        (batch_size * seq_len, 3 * HIDDEN_SIZE),
        device=hidden_states_modulated.device,
        dtype=hidden_states_modulated.dtype,
    )
    torch.addmm(qkv_bias, hidden_states_2d, qkv_weight.t(), out=qkv)
    q_2d, k_2d, v_2d = qkv.split(HIDDEN_SIZE, dim=1)
    return (
        q_2d.view(batch_size, seq_len, NUM_HEADS, HEAD_DIM),
        k_2d.view(batch_size, seq_len, NUM_HEADS, HEAD_DIM),
        v_2d.view(batch_size, seq_len, NUM_HEADS, HEAD_DIM),
    )


def _project_kv_context_packed(
    encoder_hidden_states_normalized,
    to_k_context_weight,
    to_k_context_bias,
    to_v_context_weight,
    to_v_context_bias,
):
    batch_size, text_seq_len, _ = encoder_hidden_states_normalized.shape
    context_2d = encoder_hidden_states_normalized.reshape(batch_size * text_seq_len, TEXT_HIDDEN_SIZE)
    kv_weight = torch.cat([to_k_context_weight, to_v_context_weight], dim=0)
    kv_bias = torch.cat([to_k_context_bias, to_v_context_bias], dim=0)
    kv = torch.empty(
        (batch_size * text_seq_len, 2 * HIDDEN_SIZE),
        device=encoder_hidden_states_normalized.device,
        dtype=encoder_hidden_states_normalized.dtype,
    )
    torch.addmm(kv_bias, context_2d, kv_weight.t(), out=kv)
    k_context_2d, v_context_2d = kv.split(HIDDEN_SIZE, dim=1)
    return (
        k_context_2d.view(batch_size, text_seq_len, NUM_HEADS, HEAD_DIM),
        v_context_2d.view(batch_size, text_seq_len, NUM_HEADS, HEAD_DIM),
    )

def _use_joint_packed_qk_rope(seq_len):
    return seq_len < 1024


def _use_nonjoint_packed_qk_rope(batch_size, seq_len):
    return batch_size == 1 and seq_len <= 512


def _use_medium_joint_seqbuf_kv_staging(batch_size, seq_len, text_seq_len):
    return (batch_size, seq_len, text_seq_len) in ((4, 541, 128), (8, 256, 64), (4, 512, 128))


def _pack_joint_kv_seqmajor_prealloc(k, v, k_context, v_context):
    batch_size, seq_len, num_heads, head_dim = k.shape
    text_seq_len = k_context.shape[1]
    k_joint = torch.empty(
        (batch_size, seq_len + text_seq_len, num_heads, head_dim),
        device=k.device,
        dtype=k.dtype,
    )
    v_joint = torch.empty_like(k_joint)
    k_joint[:, :seq_len].copy_(k)
    k_joint[:, seq_len:].copy_(k_context)
    torch.cat([v, v_context], dim=1, out=v_joint)
    return k_joint.transpose(1, 2), v_joint.transpose(1, 2)


@torch.no_grad()
def run(
    hidden_states: torch.Tensor,
    timestep_embedding: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    adaln_linear_weight: torch.Tensor,
    adaln_linear_bias: torch.Tensor,
    to_q_weight: torch.Tensor,
    to_q_bias: torch.Tensor,
    to_k_weight: torch.Tensor,
    to_k_bias: torch.Tensor,
    to_v_weight: torch.Tensor,
    to_v_bias: torch.Tensor,
    to_k_context_weight: torch.Tensor,
    to_k_context_bias: torch.Tensor,
    to_v_context_weight: torch.Tensor,
    to_v_context_bias: torch.Tensor,
    to_out_weight: torch.Tensor,
    to_out_bias: torch.Tensor,
    pos_idx_axis0: torch.Tensor,
    pos_idx_axis1: torch.Tensor,
    pos_idx_axis2: torch.Tensor,
    rope_cos_axis0: torch.Tensor,
    rope_sin_axis0: torch.Tensor,
    rope_cos_axis1: torch.Tensor,
    rope_sin_axis1: torch.Tensor,
    rope_cos_axis2: torch.Tensor,
    rope_sin_axis2: torch.Tensor,
    is_joint_block,
    output: torch.Tensor,
):
    batch_size, seq_len, hidden_size = hidden_states.shape
    is_joint_block = _as_int(is_joint_block)
    text_seq_len = encoder_hidden_states.shape[1]
    use_projection_tf32 = hidden_states.device.type == "cuda"
    use_nonjoint_fp16_preout = is_joint_block == 0 and _use_nonjoint_fp16_preout_projection(batch_size, seq_len)
    use_nonjoint_nonowned_out = is_joint_block == 0 and _use_nonjoint_nonowned_out_projection(batch_size, seq_len)
    use_unified_used3 = (
        (is_joint_block == 0 and _use_unified_nonjoint_used3_modulation(batch_size, seq_len))
        or (is_joint_block == 1 and _use_joint_used3_modulation(batch_size, seq_len, text_seq_len))
    )

    residual = hidden_states
    timestep_activated = F.silu(timestep_embedding)
    if use_unified_used3:
        adaln_output_width = 3 * hidden_size
        adaln_linear_weight_used = adaln_linear_weight.narrow(0, 0, adaln_output_width)
        adaln_linear_bias_used = adaln_linear_bias.narrow(0, 0, adaln_output_width)
        with _tf32_matmul_scope(use_projection_tf32):
            modulation = F.linear(timestep_activated, adaln_linear_weight_used, adaln_linear_bias_used)
        scale_msa, shift_msa, gate_msa = modulation.chunk(3, dim=-1)
    else:
        with _tf32_matmul_scope(use_projection_tf32):
            modulation = F.linear(timestep_activated, adaln_linear_weight, adaln_linear_bias)
        scale_msa, shift_msa, gate_msa, _, _, _ = modulation.chunk(6, dim=-1)

    hidden_states_normalized = F.layer_norm(hidden_states, (hidden_size,))
    hidden_states_modulated = torch.addcmul(
        shift_msa.unsqueeze(1),
        hidden_states_normalized,
        (1 + scale_msa).unsqueeze(1),
    )

    if use_nonjoint_fp16_preout:
        with _fp16_autocast_scope(True):
            q = F.linear(hidden_states_modulated, to_q_weight, to_q_bias)
            k = F.linear(hidden_states_modulated, to_k_weight, to_k_bias)
            v = F.linear(hidden_states_modulated, to_v_weight, to_v_bias)
        q = q.view(batch_size, seq_len, NUM_HEADS, HEAD_DIM)
        k = k.view(batch_size, seq_len, NUM_HEADS, HEAD_DIM)
        v = v.view(batch_size, seq_len, NUM_HEADS, HEAD_DIM)
    else:
        with _tf32_matmul_scope(use_projection_tf32):
            if (is_joint_block == 1 and _use_joint_exact_projection(batch_size, seq_len, text_seq_len)) or (
                is_joint_block == 0 and _use_nonjoint_exact_qkv_projection(batch_size, seq_len)
            ):
                q, k, v = _project_qkv_packed(
                    hidden_states_modulated,
                    to_q_weight,
                    to_q_bias,
                    to_k_weight,
                    to_k_bias,
                    to_v_weight,
                    to_v_bias,
                )
            else:
                q = F.linear(hidden_states_modulated, to_q_weight, to_q_bias)
                k = F.linear(hidden_states_modulated, to_k_weight, to_k_bias)
                v = F.linear(hidden_states_modulated, to_v_weight, to_v_bias)
                q = q.view(batch_size, seq_len, NUM_HEADS, HEAD_DIM)
                k = k.view(batch_size, seq_len, NUM_HEADS, HEAD_DIM)
                v = v.view(batch_size, seq_len, NUM_HEADS, HEAD_DIM)

    rope_dtype = q.dtype if use_nonjoint_fp16_preout else None
    use_packed_rope = (is_joint_block == 1 and _use_joint_packed_qk_rope(seq_len)) or (
        is_joint_block == 0 and _use_nonjoint_packed_qk_rope(batch_size, seq_len)
    )
    if use_packed_rope:
        axis0_gathered = _gather_rope(pos_idx_axis0, rope_cos_axis0, rope_sin_axis0, rope_dtype)
        axis1_gathered = _gather_rope(pos_idx_axis1, rope_cos_axis1, rope_sin_axis1, rope_dtype)
        axis2_gathered = _gather_rope(pos_idx_axis2, rope_cos_axis2, rope_sin_axis2, rope_dtype)
        qk = torch.cat([q, k], dim=2)
        qk = _apply_3axis_rope_gathered_inplace_(
            qk,
            axis0_gathered,
            axis1_gathered,
            axis2_gathered,
        )
        q, k = qk.split(NUM_HEADS, dim=2)
        q = q.transpose(1, 2)
    elif is_joint_block == 1:
        if _use_cute_long_rope_path(batch_size, seq_len, text_seq_len, is_joint_block, q, k):
            cos_stacked, sin_stacked = _gather_3axis_rope_half_stacked_rows(
                pos_idx_axis0,
                pos_idx_axis1,
                pos_idx_axis2,
                rope_cos_axis0,
                rope_sin_axis0,
                rope_cos_axis1,
                rope_sin_axis1,
                rope_cos_axis2,
                rope_sin_axis2,
            )
            rope_half_stacked_qk_rows8x32.run(q, k, cos_stacked, sin_stacked)
            q = q.transpose(1, 2)
        else:
            stacked_cos_sin = _stack_3axis_rope_half_gathered(
                _gather_rope_half(pos_idx_axis0, rope_cos_axis0, rope_sin_axis0),
                _gather_rope_half(pos_idx_axis1, rope_cos_axis1, rope_sin_axis1),
                _gather_rope_half(pos_idx_axis2, rope_cos_axis2, rope_sin_axis2),
            )
            rope_workspace = torch.empty(
                (*q.shape[:-1], 3, ROPE_HALF_DIM),
                device=q.device,
                dtype=q.dtype,
            )
            q = _apply_3axis_rope_half_stacked_inplace_(q, stacked_cos_sin, rope_workspace).transpose(1, 2)
            k = _apply_3axis_rope_half_stacked_inplace_(k, stacked_cos_sin, rope_workspace)
    else:
        axis0_gathered = _gather_rope(pos_idx_axis0, rope_cos_axis0, rope_sin_axis0, rope_dtype)
        axis1_gathered = _gather_rope(pos_idx_axis1, rope_cos_axis1, rope_sin_axis1, rope_dtype)
        axis2_gathered = _gather_rope(pos_idx_axis2, rope_cos_axis2, rope_sin_axis2, rope_dtype)
        q = _apply_3axis_rope_gathered_inplace_(
            q,
            axis0_gathered,
            axis1_gathered,
            axis2_gathered,
        ).transpose(1, 2)
        k = _apply_3axis_rope_gathered_inplace_(
            k,
            axis0_gathered,
            axis1_gathered,
            axis2_gathered,
        )

    if is_joint_block == 1:
        encoder_hidden_states_normalized = F.layer_norm(encoder_hidden_states, (TEXT_HIDDEN_SIZE,))
        with _tf32_matmul_scope(use_projection_tf32):
            if _use_joint_exact_projection(batch_size, seq_len, text_seq_len):
                k_context, v_context = _project_kv_context_packed(
                    encoder_hidden_states_normalized,
                    to_k_context_weight,
                    to_k_context_bias,
                    to_v_context_weight,
                    to_v_context_bias,
                )
            else:
                k_context = F.linear(
                    encoder_hidden_states_normalized,
                    to_k_context_weight,
                    to_k_context_bias,
                ).view(batch_size, text_seq_len, NUM_HEADS, HEAD_DIM)
                v_context = F.linear(
                    encoder_hidden_states_normalized,
                    to_v_context_weight,
                    to_v_context_bias,
                ).view(batch_size, text_seq_len, NUM_HEADS, HEAD_DIM)
        if _use_medium_joint_seqbuf_kv_staging(batch_size, seq_len, text_seq_len):
            k, v = _pack_joint_kv_seqmajor_prealloc(k, v, k_context, v_context)
        else:
            k = torch.cat([k, k_context], dim=1).transpose(1, 2)
            v = torch.cat([v, v_context], dim=1).transpose(1, 2)
    else:
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

    if is_joint_block == 1:
        if _use_joint_sdpa_path(seq_len, text_seq_len):
            if _use_joint_flash_bf16_sdpa(batch_size, seq_len, text_seq_len):
                attn_output = _bf16_flash_sdpa(q, k, v)
            elif _use_joint_bf16_sdpa(batch_size, seq_len, text_seq_len):
                attn_output = _bf16_sdpa(q, k, v)
            else:
                attn_output = F.scaled_dot_product_attention(
                    q,
                    k,
                    v,
                    dropout_p=0.0,
                    is_causal=False,
                    scale=ATTN_SCALE,
                )
        else:
            attn_output = _exact_attention(q, k, v)
    else:
        if _use_nonjoint_sdpa_path(batch_size, seq_len):
            if use_nonjoint_fp16_preout:
                if _use_nonjoint_flash_bf16_sdpa(batch_size, seq_len):
                    attn_output = _lowp_flash_sdpa(q, k, v)
                elif _use_nonjoint_bf16_sdpa(batch_size, seq_len):
                    attn_output = _lowp_sdpa(q, k, v)
                else:
                    attn_output = F.scaled_dot_product_attention(
                        q,
                        k,
                        v,
                        dropout_p=0.0,
                        is_causal=False,
                        scale=ATTN_SCALE,
                    )
            elif _use_nonjoint_flash_bf16_sdpa(batch_size, seq_len):
                attn_output = _bf16_flash_sdpa(q, k, v)
            elif _use_nonjoint_bf16_sdpa(batch_size, seq_len):
                attn_output = _bf16_sdpa(q, k, v)
            else:
                attn_output = F.scaled_dot_product_attention(
                    q,
                    k,
                    v,
                    dropout_p=0.0,
                    is_causal=False,
                    scale=ATTN_SCALE,
                )
        else:
            attn_output = _exact_attention(q, k, v)

    attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, HIDDEN_SIZE)
    if use_nonjoint_fp16_preout:
        with _fp16_autocast_scope(True):
            attn_output = F.linear(attn_output, to_out_weight, to_out_bias)
        torch.addcmul(residual, attn_output, gate_msa.unsqueeze(1), out=output)
    elif use_nonjoint_nonowned_out:
        with _tf32_matmul_scope(use_projection_tf32):
            attn_output = F.linear(attn_output, to_out_weight, to_out_bias)
        torch.addcmul(residual, attn_output, gate_msa.unsqueeze(1), out=output)
    elif output.is_contiguous() and output.data_ptr() != residual.data_ptr():
        output_2d = output.view(batch_size * seq_len, HIDDEN_SIZE)
        attn_output_2d = attn_output.view(batch_size * seq_len, HIDDEN_SIZE)
        with _tf32_matmul_scope(use_projection_tf32):
            torch.addmm(to_out_bias, attn_output_2d, to_out_weight.t(), out=output_2d)
        torch.addcmul(residual, output, gate_msa.unsqueeze(1), out=output)
    else:
        with _tf32_matmul_scope(use_projection_tf32):
            attn_output = F.linear(attn_output, to_out_weight, to_out_bias)
        output.copy_(residual + gate_msa.unsqueeze(1) * attn_output)
