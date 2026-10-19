import torch
import torch.nn as nn

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


# Fused kernel: y = cumsum(x @ W^T + b, dim=1)
# Processes output columns in groups of size GROUP_N:
# - Accumulates [BLOCK_M x GROUP_N] matmul over K via tl.dot
# - Adds bias
# - Computes per-group row-wise prefix sums via multiply with an upper-triangular ones matrix
# - Adds a per-row carry from previous groups, stores results, updates carry
@triton.jit
def linear_cumsum_groups_kernel(
    x_ptr, w_ptr, b_ptr, y_ptr,
    M, N, K,
    stride_xm, stride_xk,       # x: [M, K]
    stride_wn, stride_wk,       # w: [N, K]
    stride_b,                   # b: [N]
    stride_ym, stride_yn,       # y: [M, N]
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_N: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
    REM_N: tl.constexpr,        # remainder group size (BLOCK_N - NUM_GROUPS*GROUP_N), may be 0
):
    pid_m = tl.program_id(0)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m < M

    # Per-row running cumsum across groups
    carry = tl.zeros((BLOCK_M,), dtype=tl.float32)

    # Loop over full groups
    for g in tl.static_range(NUM_GROUPS):
        # Column offsets for this group
        j0 = g * GROUP_N
        offs_n = j0 + tl.arange(0, GROUP_N)
        mask_n = offs_n < N

        # Accumulator for this group's matmul
        acc_g = tl.zeros((BLOCK_M, GROUP_N), dtype=tl.float32)

        # Reduction over K in tiles
        for k0 in range(0, K, BLOCK_K):
            offs_k = k0 + tl.arange(0, BLOCK_K)
            mask_k = offs_k < K

            # Load X tile: [BLOCK_M, BLOCK_K]
            x_tile = tl.load(
                x_ptr + (offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk),
                mask=mask_m[:, None] & mask_k[None, :],
                other=0.0,
            )

            # Load W sub-tile for this group: [BLOCK_K, GROUP_N]
            w_sub = tl.load(
                w_ptr + (offs_n[None, :] * stride_wn + offs_k[:, None] * stride_wk),
                mask=mask_n[None, :] & mask_k[:, None],
                other=0.0,
            )

            # Accumulate product
            acc_g += tl.dot(x_tile, w_sub)

        # Add bias for this group (broadcast across rows)
        b_sub = tl.load(b_ptr + offs_n * stride_b, mask=mask_n, other=0.0)
        acc_g = acc_g + b_sub[None, :]

        # Build upper-triangular ones matrix U_GROUP: [GROUP_N, GROUP_N]
        col = tl.arange(0, GROUP_N)[None, :]
        row = tl.arange(0, GROUP_N)[:, None]
        U = tl.where(row <= col, 1.0, 0.0).to(tl.float32)

        # Compute per-group row-wise inclusive cumsum
        y_sub_prefix = tl.dot(acc_g, U)  # [BLOCK_M, GROUP_N]
        y_sub = y_sub_prefix + carry[:, None]

        # Store this group's results
        tl.store(
            y_ptr + (offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn),
            y_sub,
            mask=mask_m[:, None] & mask_n[None, :],
        )

        # Update carry: add sum of this group's acc_g across its columns
        carry = carry + tl.sum(acc_g, axis=1)

    # Handle remainder group if any
    if REM_N > 0:
        j0 = NUM_GROUPS * GROUP_N
        offs_n = j0 + tl.arange(0, REM_N)
        mask_n = offs_n < N

        acc_r = tl.zeros((BLOCK_M, REM_N), dtype=tl.float32)

        for k0 in range(0, K, BLOCK_K):
            offs_k = k0 + tl.arange(0, BLOCK_K)
            mask_k = offs_k < K

            x_tile = tl.load(
                x_ptr + (offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk),
                mask=mask_m[:, None] & mask_k[None, :],
                other=0.0,
            )

            w_sub = tl.load(
                w_ptr + (offs_n[None, :] * stride_wn + offs_k[:, None] * stride_wk),
                mask=mask_n[None, :] & mask_k[:, None],
                other=0.0,
            )

            acc_r += tl.dot(x_tile, w_sub)

        b_sub = tl.load(b_ptr + offs_n * stride_b, mask=mask_n, other=0.0)
        acc_r = acc_r + b_sub[None, :]

        # Triangular ones for remainder size
        col_r = tl.arange(0, REM_N)[None, :]
        row_r = tl.arange(0, REM_N)[:, None]
        U_r = tl.where(row_r <= col_r, 1.0, 0.0).to(tl.float32)

        y_r_prefix = tl.dot(acc_r, U_r)
        y_r = y_r_prefix + carry[:, None]

        tl.store(
            y_ptr + (offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn),
            y_r,
            mask=mask_m[:, None] & mask_n[None, :],
        )

        carry = carry + tl.sum(acc_r, axis=1)


class ModelNew(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        # Keep nn.Linear params and initialization
        self.linear = nn.Linear(input_size, output_size)
        # Tunable meta-parameters
        self.group_n = 32  # group size across N

    def forward(self, x: torch.Tensor):
        # Fallback for CPU or missing Triton
        if (not TRITON_AVAILABLE) or (not x.is_cuda):
            y = self.linear(x)
            return torch.cumsum(y, dim=1)

        # Shapes
        M, K = x.shape
        N = self.linear.out_features

        # Cap N to a manageable BLOCK_N to keep triangular multiply inexpensive
        MAX_BLOCK_N = 128
        if N > MAX_BLOCK_N:
            # For very large N, two-step PyTorch path avoids O(N^2) per-tile cost
            y = self.linear(x)
            return torch.cumsum(y, dim=1)

        BLOCK_N = N  # specialize kernel to exact N
        GROUP_N = min(self.group_n, BLOCK_N)
        NUM_GROUPS = BLOCK_N // GROUP_N
        REM_N = BLOCK_N - NUM_GROUPS * GROUP_N

        # BLOCK_K selection
        BLOCK_K = 128 if K >= 128 else 64
        # BLOCK_M and warps based on M
        if M <= 16:
            BLOCK_M = 16
            num_warps = 2
        elif M <= 32:
            BLOCK_M = 32
            num_warps = 4
        else:
            BLOCK_M = 64
            num_warps = 4
        num_stages = 2

        # Prepare tensors and cast to float32 for stable accumulation
        x_ = x.contiguous().to(torch.float32)                     # [M, K]
        w_ = self.linear.weight.contiguous().to(torch.float32)    # [N, K]
        b_ = self.linear.bias.contiguous().to(torch.float32)      # [N]

        # Output buffer (float32)
        y_ = torch.empty((M, N), device=x.device, dtype=torch.float32)

        # Strides (elements)
        stride_xm, stride_xk = x_.stride(0), x_.stride(1)
        stride_wn, stride_wk = w_.stride(0), w_.stride(1)
        stride_b = b_.stride(0)
        stride_ym, stride_yn = y_.stride(0), y_.stride(1)

        # Grid over row tiles
        grid = (triton.cdiv(M, BLOCK_M),)

        # Launch fused kernel
        linear_cumsum_groups_kernel[grid](
            x_, w_, b_, y_,
            M, N, K,
            stride_xm, stride_xk,
            stride_wn, stride_wk,
            stride_b,
            stride_ym, stride_yn,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            BLOCK_K=BLOCK_K,
            GROUP_N=GROUP_N,
            NUM_GROUPS=NUM_GROUPS,
            REM_N=REM_N,
            num_warps=num_warps,
            num_stages=num_stages,
        )

        return y_


# Mirrors original helpers
input_size = 128
output_size = 64

def get_inputs():
    return [torch.randn(16, input_size)]

def get_init_inputs():
    return [input_size, output_size]
