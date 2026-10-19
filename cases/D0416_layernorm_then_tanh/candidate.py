import torch
import torch.nn as nn
import torch.nn.functional as F

# Try to import Triton. Fallback to PyTorch if unavailable.
try:
    import triton
    import triton.language as tl
    _TRITON_AVAILABLE = True
except Exception:
    _TRITON_AVAILABLE = False


if _TRITON_AVAILABLE:
    @triton.jit
    def layernorm_tanh_kernel(
        x_ptr, y_ptr, w_ptr, b_ptr,
        N,
        stride_x_row, stride_x_col,
        stride_y_row, stride_y_col,
        eps,
        BLOCK_SIZE: tl.constexpr,
    ):
        # One program per row
        row_id = tl.program_id(0)
        offs = tl.arange(0, BLOCK_SIZE)

        # Base pointers for this row
        x_row_ptr = x_ptr + row_id * stride_x_row
        y_row_ptr = y_ptr + row_id * stride_y_row

        # If the whole row fits in one tile, do a single-pass to avoid reloading
        if N <= BLOCK_SIZE:
            idx = offs
            mask = idx < N

            x_vec = tl.load(x_row_ptr + idx * stride_x_col, mask=mask, other=0.0).to(tl.float32)
            # Compute mean/var in FP32
            sum_val = tl.sum(x_vec, axis=0)
            sum_sq_val = tl.sum(x_vec * x_vec, axis=0)

            mean = sum_val / N
            var = sum_sq_val / N - mean * mean
            rstd = 1.0 / tl.sqrt(var + eps)

            # Affine params
            w_vec = tl.load(w_ptr + idx, mask=mask, other=0.0).to(tl.float32)
            b_vec = tl.load(b_ptr + idx, mask=mask, other=0.0).to(tl.float32)

            z = ((x_vec - mean) * rstd) * w_vec + b_vec

            # Numerically stable tanh via exp(-2*|z|)
            az = tl.abs(z)
            u = tl.exp(-2.0 * az)
            t = (1.0 - u) / (1.0 + u)
            out = tl.where(z >= 0, t, -t)

            tl.store(y_row_ptr + idx * stride_y_col, out, mask=mask)
            return

        # General path: two passes for rows larger than BLOCK_SIZE
        # First pass: compute sum and sum of squares
        sum_val = tl.zeros((), dtype=tl.float32)
        sum_sq_val = tl.zeros((), dtype=tl.float32)
        col = 0
        while col < N:
            idx = col + offs
            mask = idx < N
            x = tl.load(x_row_ptr + idx * stride_x_col, mask=mask, other=0.0).to(tl.float32)
            sum_val += tl.sum(x, axis=0)
            sum_sq_val += tl.sum(x * x, axis=0)
            col += BLOCK_SIZE

        mean = sum_val / N
        var = sum_sq_val / N - mean * mean
        rstd = 1.0 / tl.sqrt(var + eps)

        # Second pass: normalize, affine, tanh, and store
        col = 0
        while col < N:
            idx = col + offs
            mask = idx < N

            x = tl.load(x_row_ptr + idx * stride_x_col, mask=mask, other=0.0).to(tl.float32)
            w = tl.load(w_ptr + idx, mask=mask, other=0.0).to(tl.float32)
            b = tl.load(b_ptr + idx, mask=mask, other=0.0).to(tl.float32)

            z = ((x - mean) * rstd) * w + b

            # Numerically stable tanh
            az = tl.abs(z)
            u = tl.exp(-2.0 * az)
            t = (1.0 - u) / (1.0 + u)
            out = tl.where(z >= 0, t, -t)

            tl.store(y_row_ptr + idx * stride_y_col, out, mask=mask)
            col += BLOCK_SIZE


def _next_power_of_two(x: int) -> int:
    if x <= 1:
        return 1
    return 1 << ((x - 1).bit_length())


class ModelNew(nn.Module):
    def __init__(self, input_size: int, normalization_shape: int, eps: float):
        super().__init__()
        self.input_size = input_size
        self.normalization_shape = normalization_shape
        self.eps = eps
        # Match nn.LayerNorm defaults (elementwise affine)
        self.weight = nn.Parameter(torch.ones(normalization_shape))
        self.bias = nn.Parameter(torch.zeros(normalization_shape))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Fallback to PyTorch for CPU, training (autograd), unsupported dtype, or missing Triton
        if (
            (not _TRITON_AVAILABLE)
            or (not x.is_cuda)
            or (x.dim() != 2)
            or (x.requires_grad and torch.is_grad_enabled())
            or (x.dtype not in (torch.float32, torch.float16, torch.bfloat16))
        ):
            return torch.tanh(
                F.layer_norm(
                    x,
                    normalized_shape=(self.normalization_shape,),
                    weight=self.weight,
                    bias=self.bias,
                    eps=self.eps,
                )
            )

        B, N = x.shape
        assert N == self.normalization_shape, (
            f"Expected last-dim size {self.normalization_shape}, got {N}"
        )

        # Compute in FP32 for numerical stability, cast back to input dtype afterward
        y_fp32 = torch.empty((B, N), device=x.device, dtype=torch.float32)

        # Heuristics for tile size and warps
        BLOCK_SIZE = min(_next_power_of_two(N), 1024)
        # For H100, 8 warps is generally good around N up to 512; use 16 for very large tiles
        num_warps = 8 if BLOCK_SIZE <= 512 else 16
        num_stages = 2

        # Strides (element strides)
        stride_x_row, stride_x_col = x.stride(0), x.stride(1)
        stride_y_row, stride_y_col = y_fp32.stride(0), y_fp32.stride(1)

        # Launch one program per row
        grid = (B,)
        layernorm_tanh_kernel[grid](
            x,                 # x_ptr
            y_fp32,            # y_ptr (fp32 buffer)
            self.weight,       # w_ptr
            self.bias,         # b_ptr
            N,
            stride_x_row,
            stride_x_col,
            stride_y_row,
            stride_y_col,
            self.eps,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=num_warps,
            num_stages=num_stages,
        )

        # Cast back to input dtype if needed
        if x.dtype == torch.float32:
            return y_fp32
        else:
            return y_fp32.to(dtype=x.dtype)
