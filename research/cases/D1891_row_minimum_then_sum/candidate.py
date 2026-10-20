import math
import torch
import torch.nn as nn

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


@triton.jit
def row_min_sum_fused_kernel(
    X_ptr,            # * pointer to input tensor
    Out_ptr,          # * pointer to 1-element output buffer
    N,                # number of rows
    M,                # number of columns
    stride_x_row,     # stride between rows (in elements)
    stride_x_col,     # stride between cols (in elements)
    BLOCK_M: tl.constexpr,
):
    # One program per row
    row = tl.program_id(0)
    if row >= N:
        return

    # Initialize running min as a large fp32 scalar
    BIG = 3.4028234663852886e+38  # float32 max
    acc_min = tl.full((), BIG, dtype=tl.float32)  # scalar shape ()

    # Iterate across columns in tiles of size BLOCK_M
    for m0 in tl.range(0, M, BLOCK_M):
        offs_m = m0 + tl.arange(0, BLOCK_M)
        mask = offs_m < M

        # Compute addresses for this tile
        ptrs = X_ptr + row * stride_x_row + offs_m * stride_x_col

        # Load values; masked lanes load 0 then overwritten with BIG after casting
        x = tl.load(ptrs, mask=mask, other=0)
        x32 = x.to(tl.float32)
        x32 = tl.where(mask, x32, BIG)

        # Compute tile min via -max(-x) to maximize compatibility
        y = -x32
        tile_max = tl.max(y, axis=0)    # scalar shape ()
        tile_min = -tile_max            # scalar shape ()

        # Update running min
        acc_min = tl.minimum(acc_min, tile_min)

    # Atomically add the row-wise min to the global scalar output
    # Ensure both pointer and value are block-shaped for Triton atomic semantics
    offs_out = tl.arange(0, 1)
    val = acc_min + tl.zeros((1,), dtype=tl.float32)
    tl.atomic_add(Out_ptr + offs_out, val)


@triton.jit
def rowwise_min_kernel(
    X_ptr,            # * pointer to input tensor
    Mins_ptr,         # * pointer to output vector (row minima), length N
    N,                # number of rows
    M,                # number of columns
    stride_x_row,     # stride between rows (in elements)
    stride_x_col,     # stride between cols (in elements)
    BLOCK_M: tl.constexpr,
):
    # One program per row
    row = tl.program_id(0)
    if row >= N:
        return

    BIG = 3.4028234663852886e+38
    acc_min = tl.full((), BIG, dtype=tl.float32)

    for m0 in tl.range(0, M, BLOCK_M):
        offs_m = m0 + tl.arange(0, BLOCK_M)
        mask = offs_m < M
        ptrs = X_ptr + row * stride_x_row + offs_m * stride_x_col

        x = tl.load(ptrs, mask=mask, other=0)
        x32 = x.to(tl.float32)
        x32 = tl.where(mask, x32, BIG)

        y = -x32
        tile_max = tl.max(y, axis=0)
        tile_min = -tile_max

        acc_min = tl.minimum(acc_min, tile_min)

    tl.store(Mins_ptr + row, acc_min)


class ModelNew(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        # Keep the parameter for parity with the original model (unused in forward)
        self.weight = nn.Parameter(torch.randn(input_dim, output_dim))

    def _pick_kernel_config(self, M: int):
        # Choose BLOCK_M and num_warps based on M to reduce masked overhead and warp scheduling
        # Clamp BLOCK_M to powers of two in [32, 256], pick fewer warps for small tiles.
        if M <= 32:
            return 32, 1, 1
        elif M <= 64:
            return 64, 2, 1
        elif M <= 128:
            return 128, 4, 2
        else:
            return min(256, 1 << math.ceil(math.log2(M))), 4, 2

    def forward(self, x: torch.Tensor):
        # Fallback if Triton not available or input is on CPU
        if (not TRITON_AVAILABLE) or (not x.is_cuda):
            row_mins = torch.min(x, dim=1).values
            return torch.sum(row_mins, dim=0)

        # Expect x to be 2D [N, M]
        assert x.dim() == 2, "ModelNew expects a 2D tensor input [N, M]"
        N, M = x.shape

        stride_row = x.stride(0)
        stride_col = x.stride(1)

        BLOCK_M, num_warps, num_stages = self._pick_kernel_config(M)

        # Try fused kernel first for performance (avoids writing N minima and an extra reduction)
        out = torch.zeros(1, dtype=torch.float32, device=x.device)
        grid = (N,)
        try:
            row_min_sum_fused_kernel[grid](
                x, out,
                N, M,
                stride_row, stride_col,
                BLOCK_M=BLOCK_M,
                num_warps=num_warps, num_stages=num_stages
            )
            return out[0].to(x.dtype)
        except Exception:
            # Fallback to two-step approach (robust across environments)
            row_mins = torch.empty(N, dtype=torch.float32, device=x.device)
            rowwise_min_kernel[grid](
                x, row_mins,
                N, M,
                stride_row, stride_col,
                BLOCK_M=BLOCK_M,
                num_warps=num_warps, num_stages=num_stages
            )
            return row_mins.sum().to(x.dtype)


# Helpers to match the original interface
def get_inputs():
    return [torch.randn(16, 32)]

def get_init_inputs():
    return [32, 16]
