import torch
import torch.nn as nn

try:
    import triton
    import triton.language as tl
    _TRITON_AVAILABLE = True
except Exception:
    _TRITON_AVAILABLE = False


# In-place fused bias + clamp for contiguous NCHW tensors
@triton.jit
def _bias_clamp_inplace_contig(
    x_ptr,           # input/output pointer (in-place)
    b_ptr,           # bias [C] contiguous
    N, C, H, W,
    HW,              # H * W
    min_ptr,         # pointer to 1-element tensor of same dtype as x
    BLOCK: tl.constexpr,
):
    pid_nc = tl.program_id(0)
    pid_blk = tl.program_id(1)

    n = pid_nc // C
    c = pid_nc % C

    offs = pid_blk * BLOCK + tl.arange(0, BLOCK)
    mask_hw = offs < HW

    base = ((n * C + c) * HW)
    idx = base + offs

    x = tl.load(x_ptr + idx, mask=mask_hw, other=0)
    b = tl.load(b_ptr + c)
    z = x + b

    mv = tl.load(min_ptr)
    z = tl.maximum(z, mv)

    tl.store(x_ptr + idx, z, mask=mask_hw)


# In-place fused bias + clamp for arbitrary-strided NCHW tensors
@triton.jit
def _bias_clamp_inplace_strided(
    x_ptr,  # input/output pointer (in-place)
    b_ptr,  # bias [C] with stride_bc
    N, C, H, W,
    stride_xn, stride_xc, stride_xh, stride_xw,
    stride_bc,
    HW,
    min_ptr,  # pointer to 1-element tensor of same dtype as x
    BLOCK: tl.constexpr,
):
    pid_nc = tl.program_id(0)
    pid_blk = tl.program_id(1)

    n = pid_nc // C
    c = pid_nc % C

    offs = pid_blk * BLOCK + tl.arange(0, BLOCK)
    mask_hw = offs < HW

    w = offs % W
    h = offs // W

    x_ix = n * stride_xn + c * stride_xc + h * stride_xh + w * stride_xw

    x = tl.load(x_ptr + x_ix, mask=mask_hw, other=0)
    b = tl.load(b_ptr + c * stride_bc)
    z = x + b

    mv = tl.load(min_ptr)
    z = tl.maximum(z, mv)

    tl.store(x_ptr + x_ix, z, mask=mask_hw)


class _BiasClampFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, bias, min_value):
        # Fallback if Triton unavailable or tensor not on CUDA
        if (not _TRITON_AVAILABLE) or (x.device.type != "cuda"):
            y = x + bias
            y = torch.clamp(y, min=min_value)
            ctx.save_for_backward(y)
            ctx.min_value = float(min_value)
            return y

        # Bias shape must be (C,1,1)
        assert bias.dim() == 3 and bias.shape[1] == 1 and bias.shape[2] == 1, \
            f"Expected bias of shape (C,1,1), got {tuple(bias.shape)}"
        N, C, H, W = x.shape
        HW = H * W

        # Prepare bias as [C] contiguous vector
        b_flat = bias.reshape(C).contiguous()

        # Min value buffer (1-element tensor) matching dtype/device
        min_buf = torch.tensor([min_value], dtype=x.dtype, device=x.device)

        # Decide path based on contiguity
        is_contig = x.is_contiguous() and b_flat.is_contiguous()

        # Choose a fixed configuration tuned for H100
        BLOCK = 1024
        num_warps = 8

        if is_contig:
            grid = (N * C, triton.cdiv(HW, BLOCK))
            _bias_clamp_inplace_contig[grid](
                x, b_flat,
                N, C, H, W,
                HW,
                min_buf,
                BLOCK=BLOCK,
                num_warps=num_warps,
            )
        else:
            stride_xn, stride_xc, stride_xh, stride_xw = x.stride()
            stride_bc = b_flat.stride(0)

            grid = (N * C, triton.cdiv(HW, BLOCK))
            _bias_clamp_inplace_strided[grid](
                x, b_flat,
                N, C, H, W,
                stride_xn, stride_xc, stride_xh, stride_xw,
                stride_bc,
                HW,
                min_buf,
                BLOCK=BLOCK,
                num_warps=num_warps,
            )

        # Save clamped output for backward mask computation
        ctx.save_for_backward(x)
        ctx.min_value = float(min_value)
        return x

    @staticmethod
    def backward(ctx, grad_out):
        (y,) = ctx.saved_tensors
        min_value = ctx.min_value

        # Gradient for clamp(min): 1 where y > min_value, 0 otherwise
        mask = y > min_value

        grad_x = grad_out * mask
        grad_b = grad_x.sum(dim=(0, 2, 3), keepdim=True)

        return grad_x, grad_b, None


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape, min_value):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding
        )
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.min_value = float(min_value)

    def forward(self, x):
        # ConvTranspose2d runs in cuDNN
        x = self.conv_transpose(x)
        # Fuse bias + clamp in-place with Triton
        x = _BiasClampFunction.apply(x, self.bias, self.min_value)
        return x


# Original scaffolding for compatibility
batch_size = 128
in_channels = 3
out_channels = 16
height, width = 32, 32
kernel_size = 3
stride = 2
padding = 1
output_padding = 1
bias_shape = (out_channels, 1, 1)
min_value = 0.0


def get_inputs():
    return [torch.randn(batch_size, in_channels, height, width)]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape, min_value]
