import torch

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32
from cutlass.cute.runtime import make_ptr

HEAD_DIM = 128
NUM_HEADS = 24
ROPE_DIM_PER_AXIS = HEAD_DIM // 3
ROPE_HALF_DIM = ROPE_DIM_PER_AXIS // 2
ROWS_PER_BLOCK = 8
WARP_THREADS = 32
NUM_THREADS = ROWS_PER_BLOCK * WARP_THREADS

_compiled_kernel = None
_compile_failed = False
_compile_error = None


@cute.kernel
def _rope_half_stacked_qk_rows8x32_kernel(
    g_q: cute.Tensor,
    g_k: cute.Tensor,
    g_cos: cute.Tensor,
    g_sin: cute.Tensor,
    rows: Int32,
    seq_len: Int32,
):
    block_row = cute.arch.block_idx()[0]
    warp = cute.arch.warp_idx()
    lane = cute.arch.lane_idx()
    row = block_row * ROWS_PER_BLOCK + warp

    if row < rows:
        seq_idx = (row // Int32(NUM_HEADS)) % seq_len

        pair_idx = lane
        if pair_idx < 3 * ROPE_HALF_DIM:
            inner = Int32(0)
            dim_base = Int32(0)
            cos_val = Float32(0.0)
            sin_val = Float32(0.0)
            if pair_idx < ROPE_HALF_DIM:
                inner = pair_idx
                dim_base = Int32(0)
                cos_val = Float32(g_cos[0, seq_idx, inner])
                sin_val = Float32(g_sin[0, seq_idx, inner])
            elif pair_idx < 2 * ROPE_HALF_DIM:
                inner = pair_idx - ROPE_HALF_DIM
                dim_base = Int32(ROPE_DIM_PER_AXIS)
                cos_val = Float32(g_cos[1, seq_idx, inner])
                sin_val = Float32(g_sin[1, seq_idx, inner])
            else:
                inner = pair_idx - 2 * ROPE_HALF_DIM
                dim_base = Int32(2 * ROPE_DIM_PER_AXIS)
                cos_val = Float32(g_cos[2, seq_idx, inner])
                sin_val = Float32(g_sin[2, seq_idx, inner])

            first_col = dim_base + inner
            second_col = first_col + ROPE_HALF_DIM

            q_first = Float32(g_q[row, first_col])
            q_second = Float32(g_q[row, second_col])
            g_q[row, first_col] = q_first * cos_val - q_second * sin_val
            g_q[row, second_col] = q_second * cos_val + q_first * sin_val

            k_first = Float32(g_k[row, first_col])
            k_second = Float32(g_k[row, second_col])
            g_k[row, first_col] = k_first * cos_val - k_second * sin_val
            g_k[row, second_col] = k_second * cos_val + k_first * sin_val

        pair_idx_2 = lane + WARP_THREADS
        if pair_idx_2 < 3 * ROPE_HALF_DIM:
            inner = Int32(0)
            dim_base = Int32(0)
            cos_val = Float32(0.0)
            sin_val = Float32(0.0)
            if pair_idx_2 < ROPE_HALF_DIM:
                inner = pair_idx_2
                dim_base = Int32(0)
                cos_val = Float32(g_cos[0, seq_idx, inner])
                sin_val = Float32(g_sin[0, seq_idx, inner])
            elif pair_idx_2 < 2 * ROPE_HALF_DIM:
                inner = pair_idx_2 - ROPE_HALF_DIM
                dim_base = Int32(ROPE_DIM_PER_AXIS)
                cos_val = Float32(g_cos[1, seq_idx, inner])
                sin_val = Float32(g_sin[1, seq_idx, inner])
            else:
                inner = pair_idx_2 - 2 * ROPE_HALF_DIM
                dim_base = Int32(2 * ROPE_DIM_PER_AXIS)
                cos_val = Float32(g_cos[2, seq_idx, inner])
                sin_val = Float32(g_sin[2, seq_idx, inner])

            first_col = dim_base + inner
            second_col = first_col + ROPE_HALF_DIM

            q_first = Float32(g_q[row, first_col])
            q_second = Float32(g_q[row, second_col])
            g_q[row, first_col] = q_first * cos_val - q_second * sin_val
            g_q[row, second_col] = q_second * cos_val + q_first * sin_val

            k_first = Float32(g_k[row, first_col])
            k_second = Float32(g_k[row, second_col])
            g_k[row, first_col] = k_first * cos_val - k_second * sin_val
            g_k[row, second_col] = k_second * cos_val + k_first * sin_val


@cute.jit
def _host_rope_half_stacked_qk_rows8x32(
    q_ptr: cute.Pointer,
    k_ptr: cute.Pointer,
    cos_ptr: cute.Pointer,
    sin_ptr: cute.Pointer,
    rows: Int32,
    seq_len: Int32,
):
    q = cute.make_tensor(
        q_ptr,
        cute.make_layout((rows, HEAD_DIM), stride=(HEAD_DIM, 1)),
    )
    k = cute.make_tensor(
        k_ptr,
        cute.make_layout((rows, HEAD_DIM), stride=(HEAD_DIM, 1)),
    )
    cos = cute.make_tensor(
        cos_ptr,
        cute.make_layout((3, seq_len, ROPE_HALF_DIM), stride=(seq_len * ROPE_HALF_DIM, ROPE_HALF_DIM, 1)),
    )
    sin = cute.make_tensor(
        sin_ptr,
        cute.make_layout((3, seq_len, ROPE_HALF_DIM), stride=(seq_len * ROPE_HALF_DIM, ROPE_HALF_DIM, 1)),
    )

    _rope_half_stacked_qk_rows8x32_kernel(q, k, cos, sin, rows, seq_len).launch(
        grid=[cute.ceil_div(rows, ROWS_PER_BLOCK), 1, 1],
        block=[NUM_THREADS, 1, 1],
    )


def _get_compiled_kernel():
    global _compiled_kernel, _compile_failed, _compile_error
    if _compiled_kernel is None and not _compile_failed:
        try:
            _compiled_kernel = cute.compile(
                _host_rope_half_stacked_qk_rows8x32,
                make_ptr(cutlass.Float32, 16, cute.AddressSpace.gmem, assumed_align=16),
                make_ptr(cutlass.Float32, 16, cute.AddressSpace.gmem, assumed_align=16),
                make_ptr(cutlass.Float32, 16, cute.AddressSpace.gmem, assumed_align=16),
                make_ptr(cutlass.Float32, 16, cute.AddressSpace.gmem, assumed_align=16),
                Int32(1),
                Int32(1),
            )
        except Exception as exc:
            _compile_failed = True
            _compiled_kernel = None
            _compile_error = repr(exc)
    return _compiled_kernel


def _prewarm_compiled_kernel():
    if not torch.cuda.is_available():
        return
    try:
        _get_compiled_kernel()
    except Exception:
        pass


_prewarm_compiled_kernel()


def _torch_rotate_inplace(x: torch.Tensor, cos_stacked: torch.Tensor, sin_stacked: torch.Tensor) -> None:
    seq_len = x.shape[1]
    cos = cos_stacked.view(3, seq_len, ROPE_HALF_DIM).permute(1, 0, 2).view(1, seq_len, 1, 3, ROPE_HALF_DIM)
    sin = sin_stacked.view(3, seq_len, ROPE_HALF_DIM).permute(1, 0, 2).view(1, seq_len, 1, 3, ROPE_HALF_DIM)
    rope = x[..., :3 * ROPE_DIM_PER_AXIS].view(*x.shape[:-1], 3, ROPE_DIM_PER_AXIS)
    first_half = rope[..., :ROPE_HALF_DIM].clone()
    second_half = rope[..., ROPE_HALF_DIM:]
    rope[..., :ROPE_HALF_DIM].copy_(first_half * cos - second_half * sin)
    rope[..., ROPE_HALF_DIM:].copy_(second_half * cos + first_half * sin)


@torch.inference_mode()
def run(q: torch.Tensor, k: torch.Tensor, cos_stacked: torch.Tensor, sin_stacked: torch.Tensor) -> None:
    assert q.ndim == 4 and q.shape[2] == NUM_HEADS and q.shape[3] == HEAD_DIM
    assert k.shape == q.shape
    assert cos_stacked.shape == (3, q.shape[1], ROPE_HALF_DIM)
    assert sin_stacked.shape == cos_stacked.shape
    assert q.dtype == torch.float32
    assert k.dtype == torch.float32
    assert cos_stacked.dtype == torch.float32
    assert sin_stacked.dtype == torch.float32
    assert q.is_contiguous()
    assert k.is_contiguous()
    assert cos_stacked.is_contiguous()
    assert sin_stacked.is_contiguous()

    if q.device.type != 'cuda':
        _torch_rotate_inplace(q, cos_stacked, sin_stacked)
        _torch_rotate_inplace(k, cos_stacked, sin_stacked)
        return

    kernel = _get_compiled_kernel()
    if kernel is None:
        raise RuntimeError(f'CuTe compile failed for rope_half_stacked_qk_rows8x32: {_compile_error}')
    rows = q.shape[0] * q.shape[1] * q.shape[2]
    kernel(
        make_ptr(cutlass.Float32, q.data_ptr(), cute.AddressSpace.gmem, assumed_align=16),
        make_ptr(cutlass.Float32, k.data_ptr(), cute.AddressSpace.gmem, assumed_align=16),
        make_ptr(cutlass.Float32, cos_stacked.data_ptr(), cute.AddressSpace.gmem, assumed_align=16),
        make_ptr(cutlass.Float32, sin_stacked.data_ptr(), cute.AddressSpace.gmem, assumed_align=16),
        Int32(rows),
        Int32(q.shape[1]),
    )
