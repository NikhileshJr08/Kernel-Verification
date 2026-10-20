import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_C': 16}, num_warps=4),
        triton.Config({'BLOCK_C': 32}, num_warps=4),
        triton.Config({'BLOCK_C': 64}, num_warps=4),
        triton.Config({'BLOCK_C': 128}, num_warps=4),
        triton.Config({'BLOCK_C': 256}, num_warps=8),
        triton.Config({'BLOCK_C': 512}, num_warps=8),
        triton.Config({'BLOCK_C': 1024}, num_warps=8),
    ],
    key=['C'],
)
@triton.jit
def fused_softmax_logsumexp_tanh_kernel(
    x_ptr,           # *T
    out_ptr,         # *T_out
    N: tl.constexpr,
    C: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    sN: tl.constexpr,
    sC: tl.constexpr,
    sH: tl.constexpr,
    sW: tl.constexpr,
    out_sN: tl.constexpr,
    out_sH: tl.constexpr,
    out_sW: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    # One program per (n, h, w)
    pid = tl.program_id(axis=0)
    total = N * H * W
    if pid >= total:
        return

    # Compute coordinates
    n = pid // (H * W)
    hw = pid % (H * W)
    h = hw // W
    w = hw % W

    # Base pointer offset for x and out (strides are in elements)
    base_x = n * sN + h * sH + w * sW
    base_out = n * out_sN + h * out_sH + w * out_sW

    # Load the C elements for this (n, h, w) position.
    offs = tl.arange(0, BLOCK_C)
    mask = offs < C
    # Load logits; cast to fp32 for numerics. For masked lanes, use -inf so they don't affect max/sum.
    x_vals = tl.load(x_ptr + base_x + offs * sC, mask=mask, other=-float('inf')).to(tl.float32)

    # Stable softmax over the C-vector
    max_v = tl.max(x_vals, axis=0)
    exp_v = tl.exp(x_vals - max_v)
    denom = tl.sum(exp_v, axis=0)
    inv_denom = 1.0 / denom

    # Compute S = sum(exp(p)) directly where p = exp_v / denom
    scaled = exp_v * inv_denom
    sum_exp_probs = tl.sum(tl.exp(scaled), axis=0)

    # tanh(log(S)) via identity: (S^2 - 1) / (S^2 + 1)
    s2 = sum_exp_probs * sum_exp_probs
    out_val = (s2 - 1.0) / (s2 + 1.0)

    # Store; Triton will cast to the pointer's dtype if needed
    tl.store(out_ptr + base_out, out_val)


class ModelNew(nn.Module):
    def __init__(self, max_block_c: int = 1024, fallback_if_too_many_classes: bool = True):
        super().__init__()
        self.max_block_c = int(max_block_c)
        self.fallback_if_too_many_classes = bool(fallback_if_too_many_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input: [N, C, H, W]
        # Output: [N, H, W]
        if not x.is_cuda:
            # CPU fallback: original ops
            x_soft = F.softmax(x, dim=1)
            x_lse = torch.logsumexp(x_soft, dim=1)
            return torch.tanh(x_lse)

        N, C, H, W = x.shape
        # If C exceeds supported BLOCK_C max and we want a safe fallback, use PyTorch
        if self.fallback_if_too_many_classes and C > self.max_block_c:
            x_soft = F.softmax(x, dim=1)
            x_lse = torch.logsumexp(x_soft, dim=1)
            return torch.tanh(x_lse)

        # Allocate output [N, H, W], same dtype/device as x
        out = torch.empty((N, H, W), dtype=x.dtype, device=x.device)

        # Get strides (in elements)
        sN, sC, sH, sW = x.stride()
        out_sN, out_sH, out_sW = out.stride()

        # Launch kernel: one program per (n, h, w)
        grid = (N * H * W,)
        fused_softmax_logsumexp_tanh_kernel[grid](
            x,
            out,
            N, C, H, W,
            sN, sC, sH, sW,
            out_sN, out_sH, out_sW,
        )
        return out


# Reference model for validation (optional)
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = F.softmax(x, dim=1)
        x = torch.logsumexp(x, dim=1)
        x = torch.tanh(x)
        return x


# Optional quick test
if __name__ == "__main__":
    batch_size = 32
    num_classes = 10
    height = 32
    width = 32

    x = torch.randn(batch_size, num_classes, height, width, device="cuda", dtype=torch.float32)
    model_ref = Model().cuda()
    model_new = ModelNew()

    with torch.no_grad():
        y_ref = model_ref(x)
        y_new = model_new(x)
        diff = (y_ref - y_new).abs().max().item()
        print("Max abs diff:", diff)

    # Simple timing
    import time
    for _ in range(10):
        _ = model_new(x)
        _ = model_ref(x)
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(100):
        _ = model_new(x)
    torch.cuda.synchronize()
    t1 = time.time()
    for _ in range(100):
        _ = model_ref(x)
    torch.cuda.synchronize()
    t2 = time.time()
    print(f"Triton fused: {t1 - t0:.4f}s, PyTorch baseline: {t2 - t1:.4f}s")
