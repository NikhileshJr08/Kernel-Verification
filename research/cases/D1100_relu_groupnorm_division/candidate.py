import torch
import torch.nn as nn

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


@triton.jit
def _gn_relu_group_kernel(
    x_ptr,            # *x
    y_ptr,            # *y
    weight_ptr,       # *weight[C] (float32)
    bias_ptr,         # *bias[C] (float32)
    n_stride, c_stride,
    N, C, HW, G,
    eps,              # float
    BLOCK_HW: tl.constexpr,
):
    # One program per (n, g)
    pid_n = tl.program_id(0)
    pid_g = tl.program_id(1)

    channels_per_group = C // G
    c0 = pid_g * channels_per_group

    # Accumulate sum and sumsq over the group after ReLU
    s = tl.zeros((), dtype=tl.float32)
    s2 = tl.zeros((), dtype=tl.float32)

    # First pass: compute stats
    for dc in range(0, channels_per_group):
        c = c0 + dc
        base = x_ptr + pid_n * n_stride + c * c_stride

        off = 0
        while off < HW:
            offs = off + tl.arange(0, BLOCK_HW)
            mask = offs < HW
            ptrs = base + offs  # contiguous addressing within the (n, c) plane

            x = tl.load(ptrs, mask=mask, other=0.0)
            x32 = x.to(tl.float32)
            xr = tl.maximum(x32, 0.0)  # ReLU

            s += tl.sum(xr, axis=0)
            s2 += tl.sum(xr * xr, axis=0)

            off += BLOCK_HW

    E = channels_per_group * HW
    inv_E = 1.0 / E
    mean = s * inv_E
    var = s2 * inv_E - mean * mean
    # Numerical guard to avoid tiny negative variances due to FP rounding
    var = tl.maximum(var, 0.0)
    inv_std = 1.0 / tl.sqrt(var + eps)

    # Second pass: normalize, affine, divide-by-2, write output
    for dc in range(0, channels_per_group):
        c = c0 + dc
        base_x = x_ptr + pid_n * n_stride + c * c_stride
        base_y = y_ptr + pid_n * n_stride + c * c_stride

        # load affine parameters once per channel; fuse division-by-2 by scaling
        w = tl.load(weight_ptr + c).to(tl.float32) * 0.5
        b = tl.load(bias_ptr + c).to(tl.float32) * 0.5

        off = 0
        while off < HW:
            offs = off + tl.arange(0, BLOCK_HW)
            mask = offs < HW
            x_ptrs = base_x + offs
            y_ptrs = base_y + offs

            x = tl.load(x_ptrs, mask=mask, other=0.0)
            x32 = x.to(tl.float32)
            xr = tl.maximum(x32, 0.0)  # ReLU before normalization

            y32 = (xr - mean) * inv_std * w + b
            y = y32.to(x.dtype)
            tl.store(y_ptrs, y, mask=mask)

            off += BLOCK_HW


def _pick_launch_params(HW: int):
    # Heuristic: larger block when HW is large to reduce loop iterations;
    # also tune num_warps and num_stages for pipelining.
    if HW >= 2048:
        return 1024, 8, 4
    elif HW >= 1024:
        return 1024, 8, 4
    elif HW >= 512:
        return 512, 8, 4
    elif HW >= 256:
        return 256, 4, 3
    else:
        return 128, 2, 2


class ModelNew(nn.Module):
    """
    Triton-optimized version of:
      y = GroupNorm(ReLU(x)) / 2.0

    - Uses a single Triton kernel to compute per-group stats and normalize, fusing ReLU and division by 2.0.
    - Falls back to PyTorch ops for training or unsupported conditions.
    """
    def __init__(self, num_features, num_groups, eps: float = 1e-5):
        super().__init__()
        self.relu = nn.ReLU()
        self.group_norm = nn.GroupNorm(num_groups=num_groups, num_channels=num_features, eps=eps)
        self.num_features = num_features
        self.num_groups = num_groups
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Validate shape
        assert x.dim() == 4, f"Expected input [N, C, H, W], got {x.shape}"
        N, C, H, W = x.shape
        assert C == self.num_features, f"num_features ({self.num_features}) != C ({C})"
        assert C % self.num_groups == 0, f"Channels ({C}) must be divisible by num_groups ({self.num_groups})"

        use_triton = (
            TRITON_AVAILABLE
            and x.is_cuda
            and not x.requires_grad  # forward-only fast path; training uses fallback
            and x.is_contiguous()    # contiguous gives best coalescing; non-contiguous falls back
        )

        if not use_triton:
            # Fallback path preserves autograd and uses the same modules as the original model
            x = self.relu(x)
            x = self.group_norm(x)
            x = x / 2.0
            return x

        device = x.device

        # Affine parameters from GroupNorm (match PyTorch semantics)
        if self.group_norm.affine:
            weight = self.group_norm.weight.detach().to(torch.float32).to(device)
            bias = self.group_norm.bias.detach().to(torch.float32).to(device)
        else:
            weight = torch.ones(C, dtype=torch.float32, device=device)
            bias = torch.zeros(C, dtype=torch.float32, device=device)

        # Output tensor
        y = torch.empty_like(x)

        # Extract strides in elements (contiguous path ensures simple addressing)
        n_stride, c_stride, h_stride, w_stride = x.stride()
        # HW product
        HW = H * W

        # Choose block size and launch params adaptively
        BLOCK_HW, num_warps, num_stages = _pick_launch_params(HW)

        # Launch single fused kernel: grid over (N, G)
        grid = (N, self.num_groups)
        _gn_relu_group_kernel[grid](
            x, y, weight, bias,
            n_stride, c_stride,
            N, C, HW, self.num_groups,
            self.eps,
            BLOCK_HW=BLOCK_HW,
            num_warps=num_warps,
            num_stages=num_stages,
        )

        return y


# Helper functions retained for compatibility with the evaluation harness
def get_inputs():
    batch_size = 128
    channels = 3
    height, width = 32, 32
    return [torch.randn(batch_size, channels, height, width)]

def get_init_inputs():
    channels = 3
    num_groups = 3
    return [channels, num_groups]
