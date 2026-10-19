import torch
import torch.nn as nn
import torch.nn.functional as F

# Try to import Triton
try:
    import triton
    import triton.language as tl
    _TRITON_AVAILABLE = True
except Exception:
    _TRITON_AVAILABLE = False


# Triton kernel: one program per row, scalar running min across columns with fused ReLU
if _TRITON_AVAILABLE:
    @triton.jit
    def fused_relu_row_min_kernel_1d(
        X_ptr, Y_ptr,
        N, D,
        stride_x_row, stride_x_col,
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(0)
        if pid >= N:
            return

        # Base pointer for this row
        row_ptr = X_ptr + pid * stride_x_row

        offs = tl.arange(0, BLOCK_SIZE)

        # Initialize running min from the first block
        start = 0
        mask0 = offs < D
        vals0 = tl.load(row_ptr + offs * stride_x_col, mask=mask0, other=float('inf'))
        vals0 = tl.maximum(vals0, 0)             # ReLU
        run = tl.min(vals0, axis=0)              # scalar running min for this row

        # Process remaining blocks (for D > BLOCK_SIZE)
        start += BLOCK_SIZE
        while start < D:
            offs_k = start + offs
            mask_k = offs_k < D
            vals_k = tl.load(row_ptr + offs_k * stride_x_col, mask=mask_k, other=float('inf'))
            vals_k = tl.maximum(vals_k, 0)       # ReLU
            tile_min = tl.min(vals_k, axis=0)    # scalar min in this block
            run = tl.minimum(run, tile_min)      # update running min
            start += BLOCK_SIZE

        # Store result
        tl.store(Y_ptr + pid, run)


class _FusedReluRowMinFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor):
        # CPU or missing Triton: fallback to PyTorch ops
        if (not _TRITON_AVAILABLE) or (x.device.type != 'cuda'):
            y = x.clamp_min(0).min(dim=1).values
            ctx.save_for_backward(x)
            return y

        assert x.dim() == 2, "Expected x to be [N, D]"
        N, D = x.shape

        # Output allocation
        y = torch.empty((N,), device=x.device, dtype=x.dtype)

        # Launch: one program per row; choose BLOCK_SIZE >= D to minimize loops for small D.
        # Tune num_warps to increase on-chip parallelism (H100 benefits from 4 warps here).
        fused_relu_row_min_kernel_1d[(N,)](
            x, y,
            N, D,
            x.stride(0), x.stride(1),
            BLOCK_SIZE=128,
            num_warps=4,
            num_stages=1,
        )

        # Save input for backward gradient computation
        ctx.save_for_backward(x)
        return y

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        (x,) = ctx.saved_tensors
        # y_i = min_j relu(x_ij)
        relu_x = x.clamp_min(0)
        row_min = relu_x.min(dim=1, keepdim=True).values
        # Only positive x contribute due to ReLU derivative
        pos_mask = (x > 0)
        min_mask = (relu_x == row_min) & pos_mask
        # Count minima per row (among positives)
        denom = min_mask.sum(dim=1, keepdim=True)  # [N, 1]
        denom_safe = denom.clamp_min(1)
        share = grad_out.view(-1, 1) / denom_safe
        grad_x = torch.where(min_mask, share, torch.zeros_like(x))
        return grad_x


def fused_relu_row_min(x: torch.Tensor) -> torch.Tensor:
    return _FusedReluRowMinFn.apply(x)


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        # Mirror the original signature; keep the linear layer but do not use it (original forward doesn't).
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Match original behavior: ReLU then row-wise min
        return fused_relu_row_min(x)
