import torch
import torch.nn as nn

try:
    import triton
    import triton.language as tl
    _TRITON_AVAILABLE = True
except Exception:
    _TRITON_AVAILABLE = False


if _TRITON_AVAILABLE:
    @triton.jit
    def _global_mean_div_atomic_kernel(
        x_ptr, y_ptr,
        stride_x_n, stride_x_c, stride_x_h, stride_x_w,
        stride_y_n, stride_y_c,
        N, C, H, W,
        scale,  # 1.0 / (H*W*div_factor)
        BLOCK: tl.constexpr,
    ):
        # 2D grid: pid_nc over (N*C), pid_tile over tiles of H*W
        pid_nc = tl.program_id(0)
        pid_tile = tl.program_id(1)

        n = pid_nc // C
        c = pid_nc % C

        x_base = x_ptr + n * stride_x_n + c * stride_x_c
        y_base = y_ptr + n * stride_y_n + c * stride_y_c

        HW = H * W

        start = pid_tile * BLOCK
        offs = start + tl.arange(0, BLOCK)
        mask = offs < HW

        h = offs // W
        w = offs % W

        ptrs = x_base + h * stride_x_h + w * stride_x_w
        vals = tl.load(ptrs, mask=mask, other=0.0)

        # Accumulate in fp32
        partial_sum = tl.sum(vals.to(tl.float32), axis=0)
        contrib = partial_sum * scale  # scale is fp32

        # Atomically accumulate into y[n, c, 0, 0]
        tl.atomic_add(y_base, contrib)


class GlobalMeanDivFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, kernel_size, stride, padding, div_factor):
        # Fallback to PyTorch on CPU or if Triton unavailable
        if (not _TRITON_AVAILABLE) or (not x.is_cuda):
            y = torch.nn.functional.adaptive_avg_pool2d(x, (1, 1))
            y = torch.nn.functional.max_pool2d(y, kernel_size=kernel_size, stride=stride, padding=padding)
            y = y / div_factor
            ctx.H = x.shape[-2]
            ctx.W = x.shape[-1]
            ctx.div_factor = float(div_factor)
            return y

        N, C, H, W = x.shape
        # Initialize output to zeros for atomic accumulation
        y = torch.zeros((N, C, 1, 1), dtype=x.dtype, device=x.device)

        sxn, sxc, sxh, sxw = x.stride()
        syn, syc, syh, syw = y.stride()

        HW = H * W
        # Choose BLOCK; for 64x64, 1024 works well (4 tiles)
        BLOCK = 1024 if HW >= 1024 else (1 << (HW.bit_length() - 1)) if HW > 0 else 1
        num_tiles = (HW + BLOCK - 1) // BLOCK

        # Precompute scaling factor (fp32)
        scale = 1.0 / (HW * float(div_factor))

        # 2D grid: (N*C, num_tiles)
        grid = (N * C, num_tiles)

        _global_mean_div_atomic_kernel[grid](
            x, y,
            sxn, sxc, sxh, sxw,
            syn, syc,
            N, C, H, W,
            scale,
            BLOCK=BLOCK,
            num_warps=4,
            num_stages=2,
        )

        ctx.H = H
        ctx.W = W
        ctx.div_factor = float(div_factor)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        H = ctx.H
        W = ctx.W
        div = ctx.div_factor
        coeff = 1.0 / (H * W * div)
        grad_input = grad_output * coeff
        grad_input = grad_input.expand(-1, -1, H, W).contiguous()
        return grad_input, None, None, None, None


class ModelNew(nn.Module):
    def __init__(self, in_channels, kernel_size, stride, padding, div_factor, use_triton=True):
        super().__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.div_factor = div_factor
        self.use_triton = use_triton and _TRITON_AVAILABLE

        # Fallback modules for CPU or when Triton disabled/unavailable
        self._adaptive_avg_pool_2d = nn.AdaptiveAvgPool2d((1, 1))
        self._max = nn.MaxPool2d(kernel_size, stride=stride, padding=padding)

    def forward(self, x):
        if self.use_triton and x.is_cuda:
            return GlobalMeanDivFn.apply(x, self.kernel_size, self.stride, self.padding, self.div_factor)
        # Fallback: original behavior
        y = self._adaptive_avg_pool_2d(x)
        y = self._max(y)
        y = y / self.div_factor
        return y
