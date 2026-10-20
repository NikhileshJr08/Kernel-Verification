import torch

# Try to import Triton; fall back gracefully if not available.
try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


def _torch_dtype_to_triton(dtype):
    if dtype == torch.float32:
        return tl.float32
    if dtype == torch.float16:
        return tl.float16
    if dtype == torch.bfloat16:
        return tl.bfloat16
    return tl.float32


@triton.jit
def mean_log_softmax_streaming_kernel_vecn(
    X_ptr,               # *T, shape (B, M, N)
    Out_ptr,             # *T, shape (M, N): holds mean then final result
    B: tl.constexpr,     # compile-time constant for unrolling
    M, N,                # runtime ints
    stride_b, stride_m, stride_n,
    out_stride_m, out_stride_n,
    OUT_DTYPE: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # One program handles a block of N columns
    pid = tl.program_id(0)
    offs_n = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < N

    # Streaming logsumexp state per lane
    m_max = tl.full([BLOCK_N], -float("inf"), dtype=tl.float32)
    s_acc = tl.zeros([BLOCK_N], dtype=tl.float32)

    recip_B = 1.0 / float(B)

    # First pass over M: compute mean and update streaming LSE; store mean to Out (temporary)
    m = 0
    while m < M:
        sum_b = tl.zeros([BLOCK_N], dtype=tl.float32)

        # Unroll small B for speed
        if B == 1:
            x_ptrs0 = X_ptr + 0 * stride_b + m * stride_m + offs_n * stride_n
            vals0 = tl.load(x_ptrs0, mask=mask_n, other=0.0)
            sum_b += vals0.to(tl.float32)
        elif B == 2:
            x_ptrs0 = X_ptr + 0 * stride_b + m * stride_m + offs_n * stride_n
            x_ptrs1 = X_ptr + 1 * stride_b + m * stride_m + offs_n * stride_n
            vals0 = tl.load(x_ptrs0, mask=mask_n, other=0.0)
            vals1 = tl.load(x_ptrs1, mask=mask_n, other=0.0)
            sum_b += vals0.to(tl.float32) + vals1.to(tl.float32)
        elif B == 3:
            x_ptrs0 = X_ptr + 0 * stride_b + m * stride_m + offs_n * stride_n
            x_ptrs1 = X_ptr + 1 * stride_b + m * stride_m + offs_n * stride_n
            x_ptrs2 = X_ptr + 2 * stride_b + m * stride_m + offs_n * stride_n
            vals0 = tl.load(x_ptrs0, mask=mask_n, other=0.0)
            vals1 = tl.load(x_ptrs1, mask=mask_n, other=0.0)
            vals2 = tl.load(x_ptrs2, mask=mask_n, other=0.0)
            sum_b += vals0.to(tl.float32) + vals1.to(tl.float32) + vals2.to(tl.float32)
        else:
            b = 0
            while b < B:
                x_ptrs = X_ptr + b * stride_b + m * stride_m + offs_n * stride_n
                vals = tl.load(x_ptrs, mask=mask_n, other=0.0)
                sum_b += vals.to(tl.float32)
                b += 1

        mean_mn = sum_b * recip_B

        # Store mean to Out (temporary)
        out_ptrs = Out_ptr + m * out_stride_m + offs_n * out_stride_n
        tl.store(out_ptrs, mean_mn.to(OUT_DTYPE), mask=mask_n)

        # Streaming logsumexp update with a single exp per lane
        g = (mean_mn > m_max) & mask_n
        e = tl.exp(mean_mn - m_max)  # single exp
        # If not greater: s_acc += e
        # If greater: s_acc = s_acc / e + 1; m_max = mean_mn
        s_acc = tl.where(g, s_acc / e + 1.0, tl.where(mask_n, s_acc + e, s_acc))
        m_max = tl.where(g, mean_mn, m_max)

        m += 1

    # Final lse
    lse = m_max + tl.log(s_acc + 1e-20)

    # Second pass: finalize outputs = mean - lse
    m = 0
    while m < M:
        out_ptrs = Out_ptr + m * out_stride_m + offs_n * out_stride_n
        mean_mn = tl.load(out_ptrs, mask=mask_n, other=0.0).to(tl.float32)
        res = mean_mn - lse
        tl.store(out_ptrs, res.to(OUT_DTYPE), mask=mask_n)
        m += 1


def _choose_block_n_and_warps(N: int):
    # Simple heuristic for coalesced loads and occupancy
    if N >= 256:
        return 256, 8
    elif N >= 128:
        return 128, 4
    elif N >= 64:
        return 64, 2
    elif N >= 32:
        return 32, 2
    else:
        return 16, 1


class ModelNew(torch.nn.Module):
    """
    Triton-optimized fused mean(dim=0) + log_softmax(dim=0) for input [B, M, N].
    Two-pass design using streaming logsumexp with one exp per element.
    Vectorizes across N (contiguous) for coalesced memory access and unrolls small B.
    """
    def __init__(self):
        super().__init__()
        if not TRITON_AVAILABLE:
            pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(f"Expected 3D tensor [B, M, N], got shape {tuple(x.shape)}")
        B, M, N = x.shape

        # Fallback for non-CUDA or missing Triton
        if (not TRITON_AVAILABLE) or (not x.is_cuda):
            y = torch.mean(x, dim=0)
            y = torch.log_softmax(y, dim=0)
            return y

        out = torch.empty((M, N), dtype=x.dtype, device=x.device)

        stride_b, stride_m, stride_n = x.stride()
        out_stride_m, out_stride_n = out.stride()

        out_dtype_tl = _torch_dtype_to_triton(out.dtype)
        BLOCK_N, num_warps = _choose_block_n_and_warps(N)
        grid = (triton.cdiv(N, BLOCK_N),)

        mean_log_softmax_streaming_kernel_vecn[grid](
            x, out,
            B,
            M, N,
            stride_b, stride_m, stride_n,
            out_stride_m, out_stride_n,
            OUT_DTYPE=out_dtype_tl,
            BLOCK_N=BLOCK_N,
            num_warps=num_warps,
            num_stages=1,  # minimal pipeline depth for tiny shapes
        )

        return out


# Reference model for fallback/testing
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x):
        x = torch.mean(x, dim=0)
        x = torch.log_softmax(x, dim=0)
        return x


def get_inputs():
    return [torch.randn(3, 5, 10)]


def get_init_inputs():
    return []
