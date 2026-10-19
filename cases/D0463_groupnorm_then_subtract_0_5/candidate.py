import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


@triton.jit
def groupnorm_fused_kernel(
    X_ptr,          # *dtype
    Y_ptr,          # *dtype
    Weight_ptr,     # *float32
    Bias_ptr,       # *float32 (already fused with -0.5 on host)
    B, C, HW, G,    # ints
    eps,            # float
    OUT_DTYPE: tl.constexpr,
    BLOCK_HW: tl.constexpr,
):
    pid_n = tl.program_id(0)  # batch index
    pid_g = tl.program_id(1)  # group index

    group_size = C // G
    c_start = pid_g * group_size

    # First pass: compute sum and sumsq over all channels in the group
    sum_ = tl.zeros([1], dtype=tl.float32)
    sumsq_ = tl.zeros([1], dtype=tl.float32)

    ch = 0
    while ch < group_size:
        c = c_start + ch
        base = ((pid_n * C + c) * HW)
        off = 0
        while off < HW:
            idx = off + tl.arange(0, BLOCK_HW)
            mask = idx < HW
            x = tl.load(X_ptr + base + idx, mask=mask, other=0.0)
            xf = tl.cast(x, tl.float32)
            sum_ += tl.sum(xf, axis=0)
            sumsq_ += tl.sum(xf * xf, axis=0)
            off += BLOCK_HW
        ch += 1

    denom = group_size * HW
    mean = sum_ / denom
    var = (sumsq_ / denom) - (mean * mean)
    inv_std = 1.0 / tl.sqrt(var + eps)

    # Second pass: normalize and apply affine using precomputed scalars s and t
    ch = 0
    while ch < group_size:
        c = c_start + ch
        base = ((pid_n * C + c) * HW)

        # Load per-channel parameters once
        w = tl.load(Weight_ptr + c)
        b = tl.load(Bias_ptr + c)  # already fused with -0.5
        wf = tl.cast(w, tl.float32)
        bf = tl.cast(b, tl.float32)

        # Precompute scale and shift: y = x * s + t
        s = wf * inv_std
        t = bf - mean * s

        off = 0
        while off < HW:
            idx = off + tl.arange(0, BLOCK_HW)
            mask = idx < HW
            x = tl.load(X_ptr + base + idx, mask=mask, other=0.0)
            xf = tl.cast(x, tl.float32)
            y = xf * s + t
            y_out = tl.cast(y, OUT_DTYPE)
            tl.store(Y_ptr + base + idx, y_out, mask=mask)
            off += BLOCK_HW
        ch += 1


def _torch_dtype_to_triton(dtype: torch.dtype):
    if dtype == torch.float32:
        return tl.float32
    elif dtype == torch.float16:
        return tl.float16
    elif dtype == torch.bfloat16:
        return tl.bfloat16
    else:
        raise ValueError(f"Unsupported dtype for Triton GroupNorm kernel: {dtype}")


class ModelNew(nn.Module):
    def __init__(self, num_channels, num_groups, eps):
        super().__init__()
        assert num_channels % num_groups == 0, "num_channels must be divisible by num_groups"
        self.num_channels = num_channels
        self.num_groups = num_groups
        self.eps = float(eps)

        # Learnable affine parameters per channel (match nn.GroupNorm with affine=True)
        self.weight = nn.Parameter(torch.ones(num_channels, dtype=torch.float32))
        self.bias = nn.Parameter(torch.zeros(num_channels, dtype=torch.float32))

        # Fallback module for CPU / Triton unavailable
        self._fallback = nn.GroupNorm(num_groups, num_channels, eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Fallback path
        if (not x.is_cuda) or (not TRITON_AVAILABLE):
            y = F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
            return y - 0.5

        # Ensure contiguous NCHW layout
        x = x.contiguous()
        B, C, H, W = x.shape
        assert C == self.num_channels and C % self.num_groups == 0
        HW = H * W

        # Output tensor
        y = torch.empty_like(x)

        # Device-local parameters in float32
        weight_dev = self.weight.to(device=x.device, dtype=torch.float32)
        # Fuse constant subtraction into bias once on host/device
        bias_dev_fused = (self.bias - 0.5).to(device=x.device, dtype=torch.float32)

        OUT_DTYPE = _torch_dtype_to_triton(x.dtype)

        # Heuristic tiling and launch configuration for H100
        # Use larger tiles to reduce loop iterations and increase bandwidth utilization.
        BLOCK_HW = 2048
        num_warps = 8
        num_stages = 4

        grid = (B, self.num_groups)

        groupnorm_fused_kernel[grid](
            x, y, weight_dev, bias_dev_fused,
            B, C, HW, self.num_groups,
            self.eps,
            OUT_DTYPE=OUT_DTYPE,
            BLOCK_HW=BLOCK_HW,
            num_warps=num_warps,
            num_stages=num_stages,
        )
        return y
