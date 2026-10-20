import os
os.environ.setdefault("CUTE_DSL_ARCH", "sm_100a")

import torch
import torch.nn.functional as F

try:
    import cutlass
    import cutlass.cute as cute
    from cutlass import Int32
    from cutlass.cute.runtime import make_ptr
except Exception:
    cutlass = None
    cute = None
    Int32 = None
    make_ptr = None


CHANNELS = 192
HALF_CHANNELS = 96
PADDING = 2
BLOCK_T = 128
ASSUMED_ALIGN = 16

if hasattr(torch.backends, "cudnn"):
    torch.backends.cudnn.benchmark = True
    if hasattr(torch.backends.cudnn, "allow_tf32"):
        torch.backends.cudnn.allow_tf32 = True
if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
    torch.backends.cuda.matmul.allow_tf32 = True
try:
    torch._dynamo.config.cache_size_limit = 64
except Exception:
    pass

_compiled_state_kernel = None


def _as_bool(value):
    if isinstance(value, torch.Tensor):
        return bool(value.item())
    return bool(value)


def _apply_transform_eager(
    x0: torch.Tensor,
    conv0_w: torch.Tensor,
    conv0_b: torch.Tensor,
    conv1_w: torch.Tensor,
    conv1_b: torch.Tensor,
    conv2_w: torch.Tensor,
    conv2_b: torch.Tensor,
) -> torch.Tensor:
    h = F.conv1d(x0, conv0_w, conv0_b, padding=PADDING)
    h.relu_()
    h = F.conv1d(h, conv1_w, conv1_b, padding=PADDING)
    h.relu_()
    h = F.conv1d(h, conv2_w, conv2_b, padding=PADDING)
    return h


if hasattr(torch, "compile"):
    _compiled_apply_transform = torch.compile(
        _apply_transform_eager,
        mode="max-autotune-no-cudagraphs",
        dynamic=False,
        fullgraph=True,
    )
else:
    _compiled_apply_transform = _apply_transform_eager


def _apply_transform(
    x0: torch.Tensor,
    conv0_w: torch.Tensor,
    conv0_b: torch.Tensor,
    conv1_w: torch.Tensor,
    conv1_b: torch.Tensor,
    conv2_w: torch.Tensor,
    conv2_b: torch.Tensor,
) -> torch.Tensor:
    if x0.is_cuda:
        return _compiled_apply_transform(x0, conv0_w, conv0_b, conv1_w, conv1_b, conv2_w, conv2_b)
    return _apply_transform_eager(x0, conv0_w, conv0_b, conv1_w, conv1_b, conv2_w, conv2_b)


def _apply_transform_reference(
    x0: torch.Tensor,
    conv0_w: torch.Tensor,
    conv0_b: torch.Tensor,
    conv1_w: torch.Tensor,
    conv1_b: torch.Tensor,
    conv2_w: torch.Tensor,
    conv2_b: torch.Tensor,
) -> torch.Tensor:
    h = F.conv1d(x0, conv0_w, conv0_b, padding=PADDING)
    h = F.relu(h)
    h = F.conv1d(h, conv1_w, conv1_b, padding=PADDING)
    h = F.relu(h)
    h = F.conv1d(h, conv2_w, conv2_b, padding=PADDING)
    return h


def _run_reference_style(
    x: torch.Tensor,
    x_mask: torch.Tensor,
    reverse_flag: bool,
    transforms,
    output: torch.Tensor,
) -> None:
    state = x
    if reverse_flag:
        transforms = transforms[::-1]

    for conv0_w, conv0_b, conv1_w, conv1_b, conv2_w, conv2_b in transforms:
        x0 = state[:, :HALF_CHANNELS, :]
        x1 = state[:, HALF_CHANNELS:, :]
        h = _apply_transform_reference(
            x0,
            conv0_w,
            conv0_b,
            conv1_w,
            conv1_b,
            conv2_w,
            conv2_b,
        )
        h = h * x_mask
        if reverse_flag:
            x1 = x1 - h
        else:
            x1 = x1 + h
        state = torch.cat((x0, x1), dim=1)
        state = state * x_mask

    output.copy_(state)


def _run_inplace_style(
    x: torch.Tensor,
    x_mask: torch.Tensor,
    reverse_flag: bool,
    transforms,
    output: torch.Tensor,
    use_cute_state: bool,
) -> None:
    output.copy_(x)
    state = output

    if reverse_flag:
        transforms = transforms[::-1]

    use_cute_state = (
        use_cute_state
        and state.is_cuda
        and state.dtype == torch.float32
        and x_mask.is_cuda
        and x_mask.dtype == torch.float32
    )

    for conv0_w, conv0_b, conv1_w, conv1_b, conv2_w, conv2_b in transforms:
        x0 = state[:, :HALF_CHANNELS, :]
        h = _apply_transform(
            x0,
            conv0_w,
            conv0_b,
            conv1_w,
            conv1_b,
            conv2_w,
            conv2_b,
        )
        if use_cute_state and h.is_cuda and h.dtype == torch.float32:
            _update_state_cute(state, h, x_mask, reverse_flag)
        else:
            _update_state_torch(state, h, x_mask, reverse_flag)


def _run_torch_inplace_reverse_4layer_eager(
    x: torch.Tensor,
    x_mask: torch.Tensor,
    transform_0_conv0_weight: torch.Tensor,
    transform_0_conv0_bias: torch.Tensor,
    transform_0_conv1_weight: torch.Tensor,
    transform_0_conv1_bias: torch.Tensor,
    transform_0_conv2_weight: torch.Tensor,
    transform_0_conv2_bias: torch.Tensor,
    transform_1_conv0_weight: torch.Tensor,
    transform_1_conv0_bias: torch.Tensor,
    transform_1_conv1_weight: torch.Tensor,
    transform_1_conv1_bias: torch.Tensor,
    transform_1_conv2_weight: torch.Tensor,
    transform_1_conv2_bias: torch.Tensor,
    transform_2_conv0_weight: torch.Tensor,
    transform_2_conv0_bias: torch.Tensor,
    transform_2_conv1_weight: torch.Tensor,
    transform_2_conv1_bias: torch.Tensor,
    transform_2_conv2_weight: torch.Tensor,
    transform_2_conv2_bias: torch.Tensor,
    transform_3_conv0_weight: torch.Tensor,
    transform_3_conv0_bias: torch.Tensor,
    transform_3_conv1_weight: torch.Tensor,
    transform_3_conv1_bias: torch.Tensor,
    transform_3_conv2_weight: torch.Tensor,
    transform_3_conv2_bias: torch.Tensor,
) -> torch.Tensor:
    state = x.clone()
    for conv0_w, conv0_b, conv1_w, conv1_b, conv2_w, conv2_b in (
        (transform_3_conv0_weight, transform_3_conv0_bias, transform_3_conv1_weight, transform_3_conv1_bias, transform_3_conv2_weight, transform_3_conv2_bias),
        (transform_2_conv0_weight, transform_2_conv0_bias, transform_2_conv1_weight, transform_2_conv1_bias, transform_2_conv2_weight, transform_2_conv2_bias),
        (transform_1_conv0_weight, transform_1_conv0_bias, transform_1_conv1_weight, transform_1_conv1_bias, transform_1_conv2_weight, transform_1_conv2_bias),
        (transform_0_conv0_weight, transform_0_conv0_bias, transform_0_conv1_weight, transform_0_conv1_bias, transform_0_conv2_weight, transform_0_conv2_bias),
    ):
        x0 = state[:, :HALF_CHANNELS, :]
        h = F.conv1d(x0, conv0_w, conv0_b, padding=PADDING)
        h = torch.relu(h)
        h = F.conv1d(h, conv1_w, conv1_b, padding=PADDING)
        h = torch.relu(h)
        h = F.conv1d(h, conv2_w, conv2_b, padding=PADDING)
        h = h * x_mask
        state[:, HALF_CHANNELS:, :].sub_(h)
        state[:, :HALF_CHANNELS, :].mul_(x_mask)
        state[:, HALF_CHANNELS:, :].mul_(x_mask)
    return state


if hasattr(torch, "compile"):
    _compiled_run_torch_inplace_reverse_4layer = torch.compile(
        _run_torch_inplace_reverse_4layer_eager,
        mode="max-autotune-no-cudagraphs",
        dynamic=False,
        fullgraph=True,
    )
else:
    _compiled_run_torch_inplace_reverse_4layer = _run_torch_inplace_reverse_4layer_eager


def _run_torch_inplace_forward_4layer_eager(
    x: torch.Tensor,
    x_mask: torch.Tensor,
    transform_0_conv0_weight: torch.Tensor,
    transform_0_conv0_bias: torch.Tensor,
    transform_0_conv1_weight: torch.Tensor,
    transform_0_conv1_bias: torch.Tensor,
    transform_0_conv2_weight: torch.Tensor,
    transform_0_conv2_bias: torch.Tensor,
    transform_1_conv0_weight: torch.Tensor,
    transform_1_conv0_bias: torch.Tensor,
    transform_1_conv1_weight: torch.Tensor,
    transform_1_conv1_bias: torch.Tensor,
    transform_1_conv2_weight: torch.Tensor,
    transform_1_conv2_bias: torch.Tensor,
    transform_2_conv0_weight: torch.Tensor,
    transform_2_conv0_bias: torch.Tensor,
    transform_2_conv1_weight: torch.Tensor,
    transform_2_conv1_bias: torch.Tensor,
    transform_2_conv2_weight: torch.Tensor,
    transform_2_conv2_bias: torch.Tensor,
    transform_3_conv0_weight: torch.Tensor,
    transform_3_conv0_bias: torch.Tensor,
    transform_3_conv1_weight: torch.Tensor,
    transform_3_conv1_bias: torch.Tensor,
    transform_3_conv2_weight: torch.Tensor,
    transform_3_conv2_bias: torch.Tensor,
) -> torch.Tensor:
    state = x.clone()
    for conv0_w, conv0_b, conv1_w, conv1_b, conv2_w, conv2_b in (
        (transform_0_conv0_weight, transform_0_conv0_bias, transform_0_conv1_weight, transform_0_conv1_bias, transform_0_conv2_weight, transform_0_conv2_bias),
        (transform_1_conv0_weight, transform_1_conv0_bias, transform_1_conv1_weight, transform_1_conv1_bias, transform_1_conv2_weight, transform_1_conv2_bias),
        (transform_2_conv0_weight, transform_2_conv0_bias, transform_2_conv1_weight, transform_2_conv1_bias, transform_2_conv2_weight, transform_2_conv2_bias),
        (transform_3_conv0_weight, transform_3_conv0_bias, transform_3_conv1_weight, transform_3_conv1_bias, transform_3_conv2_weight, transform_3_conv2_bias),
    ):
        x0 = state[:, :HALF_CHANNELS, :]
        h = F.conv1d(x0, conv0_w, conv0_b, padding=PADDING)
        h = torch.relu(h)
        h = F.conv1d(h, conv1_w, conv1_b, padding=PADDING)
        h = torch.relu(h)
        h = F.conv1d(h, conv2_w, conv2_b, padding=PADDING)
        h = h * x_mask
        state[:, HALF_CHANNELS:, :].add_(h)
        state[:, :HALF_CHANNELS, :].mul_(x_mask)
        state[:, HALF_CHANNELS:, :].mul_(x_mask)
    return state


if hasattr(torch, "compile"):
    _compiled_run_torch_inplace_forward_4layer = torch.compile(
        _run_torch_inplace_forward_4layer_eager,
        mode="max-autotune-no-cudagraphs",
        dynamic=False,
        fullgraph=True,
    )
else:
    _compiled_run_torch_inplace_forward_4layer = _run_torch_inplace_forward_4layer_eager


def _run_torch_inplace_reverse_4layer_out_eager(
    x: torch.Tensor,
    x_mask: torch.Tensor,
    transform_0_conv0_weight: torch.Tensor,
    transform_0_conv0_bias: torch.Tensor,
    transform_0_conv1_weight: torch.Tensor,
    transform_0_conv1_bias: torch.Tensor,
    transform_0_conv2_weight: torch.Tensor,
    transform_0_conv2_bias: torch.Tensor,
    transform_1_conv0_weight: torch.Tensor,
    transform_1_conv0_bias: torch.Tensor,
    transform_1_conv1_weight: torch.Tensor,
    transform_1_conv1_bias: torch.Tensor,
    transform_1_conv2_weight: torch.Tensor,
    transform_1_conv2_bias: torch.Tensor,
    transform_2_conv0_weight: torch.Tensor,
    transform_2_conv0_bias: torch.Tensor,
    transform_2_conv1_weight: torch.Tensor,
    transform_2_conv1_bias: torch.Tensor,
    transform_2_conv2_weight: torch.Tensor,
    transform_2_conv2_bias: torch.Tensor,
    transform_3_conv0_weight: torch.Tensor,
    transform_3_conv0_bias: torch.Tensor,
    transform_3_conv1_weight: torch.Tensor,
    transform_3_conv1_bias: torch.Tensor,
    transform_3_conv2_weight: torch.Tensor,
    transform_3_conv2_bias: torch.Tensor,
    output: torch.Tensor,
) -> torch.Tensor:
    output.copy_(x)
    state = output
    for conv0_w, conv0_b, conv1_w, conv1_b, conv2_w, conv2_b in (
        (transform_3_conv0_weight, transform_3_conv0_bias, transform_3_conv1_weight, transform_3_conv1_bias, transform_3_conv2_weight, transform_3_conv2_bias),
        (transform_2_conv0_weight, transform_2_conv0_bias, transform_2_conv1_weight, transform_2_conv1_bias, transform_2_conv2_weight, transform_2_conv2_bias),
        (transform_1_conv0_weight, transform_1_conv0_bias, transform_1_conv1_weight, transform_1_conv1_bias, transform_1_conv2_weight, transform_1_conv2_bias),
        (transform_0_conv0_weight, transform_0_conv0_bias, transform_0_conv1_weight, transform_0_conv1_bias, transform_0_conv2_weight, transform_0_conv2_bias),
    ):
        x0 = state[:, :HALF_CHANNELS, :]
        h = F.conv1d(x0, conv0_w, conv0_b, padding=PADDING)
        h = torch.relu(h)
        h = F.conv1d(h, conv1_w, conv1_b, padding=PADDING)
        h = torch.relu(h)
        h = F.conv1d(h, conv2_w, conv2_b, padding=PADDING)
        h = h * x_mask
        state[:, HALF_CHANNELS:, :].sub_(h)
        state[:, :HALF_CHANNELS, :].mul_(x_mask)
        state[:, HALF_CHANNELS:, :].mul_(x_mask)
    return output


if hasattr(torch, "compile"):
    _compiled_run_torch_inplace_reverse_4layer_out = torch.compile(
        _run_torch_inplace_reverse_4layer_out_eager,
        mode="max-autotune-no-cudagraphs",
        dynamic=False,
        fullgraph=True,
    )
else:
    _compiled_run_torch_inplace_reverse_4layer_out = _run_torch_inplace_reverse_4layer_out_eager


def _run_torch_inplace_forward_4layer_out_eager(
    x: torch.Tensor,
    x_mask: torch.Tensor,
    transform_0_conv0_weight: torch.Tensor,
    transform_0_conv0_bias: torch.Tensor,
    transform_0_conv1_weight: torch.Tensor,
    transform_0_conv1_bias: torch.Tensor,
    transform_0_conv2_weight: torch.Tensor,
    transform_0_conv2_bias: torch.Tensor,
    transform_1_conv0_weight: torch.Tensor,
    transform_1_conv0_bias: torch.Tensor,
    transform_1_conv1_weight: torch.Tensor,
    transform_1_conv1_bias: torch.Tensor,
    transform_1_conv2_weight: torch.Tensor,
    transform_1_conv2_bias: torch.Tensor,
    transform_2_conv0_weight: torch.Tensor,
    transform_2_conv0_bias: torch.Tensor,
    transform_2_conv1_weight: torch.Tensor,
    transform_2_conv1_bias: torch.Tensor,
    transform_2_conv2_weight: torch.Tensor,
    transform_2_conv2_bias: torch.Tensor,
    transform_3_conv0_weight: torch.Tensor,
    transform_3_conv0_bias: torch.Tensor,
    transform_3_conv1_weight: torch.Tensor,
    transform_3_conv1_bias: torch.Tensor,
    transform_3_conv2_weight: torch.Tensor,
    transform_3_conv2_bias: torch.Tensor,
    output: torch.Tensor,
) -> torch.Tensor:
    output.copy_(x)
    state = output
    for conv0_w, conv0_b, conv1_w, conv1_b, conv2_w, conv2_b in (
        (transform_0_conv0_weight, transform_0_conv0_bias, transform_0_conv1_weight, transform_0_conv1_bias, transform_0_conv2_weight, transform_0_conv2_bias),
        (transform_1_conv0_weight, transform_1_conv0_bias, transform_1_conv1_weight, transform_1_conv1_bias, transform_1_conv2_weight, transform_1_conv2_bias),
        (transform_2_conv0_weight, transform_2_conv0_bias, transform_2_conv1_weight, transform_2_conv1_bias, transform_2_conv2_weight, transform_2_conv2_bias),
        (transform_3_conv0_weight, transform_3_conv0_bias, transform_3_conv1_weight, transform_3_conv1_bias, transform_3_conv2_weight, transform_3_conv2_bias),
    ):
        x0 = state[:, :HALF_CHANNELS, :]
        h = F.conv1d(x0, conv0_w, conv0_b, padding=PADDING)
        h = torch.relu(h)
        h = F.conv1d(h, conv1_w, conv1_b, padding=PADDING)
        h = torch.relu(h)
        h = F.conv1d(h, conv2_w, conv2_b, padding=PADDING)
        h = h * x_mask
        state[:, HALF_CHANNELS:, :].add_(h)
        state[:, :HALF_CHANNELS, :].mul_(x_mask)
        state[:, HALF_CHANNELS:, :].mul_(x_mask)
    return output


if hasattr(torch, "compile"):
    _compiled_run_torch_inplace_forward_4layer_out = torch.compile(
        _run_torch_inplace_forward_4layer_out_eager,
        mode="max-autotune-no-cudagraphs",
        dynamic=False,
        fullgraph=True,
    )
else:
    _compiled_run_torch_inplace_forward_4layer_out = _run_torch_inplace_forward_4layer_out_eager


def _run_reference_reverse_4layer_eager(
    x: torch.Tensor,
    x_mask: torch.Tensor,
    transform_0_conv0_weight: torch.Tensor,
    transform_0_conv0_bias: torch.Tensor,
    transform_0_conv1_weight: torch.Tensor,
    transform_0_conv1_bias: torch.Tensor,
    transform_0_conv2_weight: torch.Tensor,
    transform_0_conv2_bias: torch.Tensor,
    transform_1_conv0_weight: torch.Tensor,
    transform_1_conv0_bias: torch.Tensor,
    transform_1_conv1_weight: torch.Tensor,
    transform_1_conv1_bias: torch.Tensor,
    transform_1_conv2_weight: torch.Tensor,
    transform_1_conv2_bias: torch.Tensor,
    transform_2_conv0_weight: torch.Tensor,
    transform_2_conv0_bias: torch.Tensor,
    transform_2_conv1_weight: torch.Tensor,
    transform_2_conv1_bias: torch.Tensor,
    transform_2_conv2_weight: torch.Tensor,
    transform_2_conv2_bias: torch.Tensor,
    transform_3_conv0_weight: torch.Tensor,
    transform_3_conv0_bias: torch.Tensor,
    transform_3_conv1_weight: torch.Tensor,
    transform_3_conv1_bias: torch.Tensor,
    transform_3_conv2_weight: torch.Tensor,
    transform_3_conv2_bias: torch.Tensor,
) -> torch.Tensor:
    state = x
    for conv0_w, conv0_b, conv1_w, conv1_b, conv2_w, conv2_b in (
        (transform_3_conv0_weight, transform_3_conv0_bias, transform_3_conv1_weight, transform_3_conv1_bias, transform_3_conv2_weight, transform_3_conv2_bias),
        (transform_2_conv0_weight, transform_2_conv0_bias, transform_2_conv1_weight, transform_2_conv1_bias, transform_2_conv2_weight, transform_2_conv2_bias),
        (transform_1_conv0_weight, transform_1_conv0_bias, transform_1_conv1_weight, transform_1_conv1_bias, transform_1_conv2_weight, transform_1_conv2_bias),
        (transform_0_conv0_weight, transform_0_conv0_bias, transform_0_conv1_weight, transform_0_conv1_bias, transform_0_conv2_weight, transform_0_conv2_bias),
    ):
        x0 = state[:, :HALF_CHANNELS, :]
        x1 = state[:, HALF_CHANNELS:, :]
        h = F.conv1d(x0, conv0_w, conv0_b, padding=PADDING)
        h = torch.relu(h)
        h = F.conv1d(h, conv1_w, conv1_b, padding=PADDING)
        h = torch.relu(h)
        h = F.conv1d(h, conv2_w, conv2_b, padding=PADDING)
        h = h * x_mask
        x1 = x1 - h
        state = torch.cat((x0, x1), dim=1)
        state = state * x_mask
    return state




def _run_torch_split_reverse_4layer_out_eager(
    x: torch.Tensor,
    x_mask: torch.Tensor,
    transform_0_conv0_weight: torch.Tensor,
    transform_0_conv0_bias: torch.Tensor,
    transform_0_conv1_weight: torch.Tensor,
    transform_0_conv1_bias: torch.Tensor,
    transform_0_conv2_weight: torch.Tensor,
    transform_0_conv2_bias: torch.Tensor,
    transform_1_conv0_weight: torch.Tensor,
    transform_1_conv0_bias: torch.Tensor,
    transform_1_conv1_weight: torch.Tensor,
    transform_1_conv1_bias: torch.Tensor,
    transform_1_conv2_weight: torch.Tensor,
    transform_1_conv2_bias: torch.Tensor,
    transform_2_conv0_weight: torch.Tensor,
    transform_2_conv0_bias: torch.Tensor,
    transform_2_conv1_weight: torch.Tensor,
    transform_2_conv1_bias: torch.Tensor,
    transform_2_conv2_weight: torch.Tensor,
    transform_2_conv2_bias: torch.Tensor,
    transform_3_conv0_weight: torch.Tensor,
    transform_3_conv0_bias: torch.Tensor,
    transform_3_conv1_weight: torch.Tensor,
    transform_3_conv1_bias: torch.Tensor,
    transform_3_conv2_weight: torch.Tensor,
    transform_3_conv2_bias: torch.Tensor,
    output: torch.Tensor,
) -> torch.Tensor:
    x0 = x[:, :HALF_CHANNELS, :]
    x1 = x[:, HALF_CHANNELS:, :]
    for conv0_w, conv0_b, conv1_w, conv1_b, conv2_w, conv2_b in (
        (transform_3_conv0_weight, transform_3_conv0_bias, transform_3_conv1_weight, transform_3_conv1_bias, transform_3_conv2_weight, transform_3_conv2_bias),
        (transform_2_conv0_weight, transform_2_conv0_bias, transform_2_conv1_weight, transform_2_conv1_bias, transform_2_conv2_weight, transform_2_conv2_bias),
        (transform_1_conv0_weight, transform_1_conv0_bias, transform_1_conv1_weight, transform_1_conv1_bias, transform_1_conv2_weight, transform_1_conv2_bias),
        (transform_0_conv0_weight, transform_0_conv0_bias, transform_0_conv1_weight, transform_0_conv1_bias, transform_0_conv2_weight, transform_0_conv2_bias),
    ):
        h = F.conv1d(x0, conv0_w, conv0_b, padding=PADDING)
        h = torch.relu(h)
        h = F.conv1d(h, conv1_w, conv1_b, padding=PADDING)
        h = torch.relu(h)
        h = F.conv1d(h, conv2_w, conv2_b, padding=PADDING)
        h = h * x_mask
        x1 = x1 - h
        x1 = x1 * x_mask
        x0 = x0 * x_mask
    output[:, :HALF_CHANNELS, :].copy_(x0)
    output[:, HALF_CHANNELS:, :].copy_(x1)
    return output


if hasattr(torch, "compile"):
    _compiled_run_torch_split_reverse_4layer_out = torch.compile(
        _run_torch_split_reverse_4layer_out_eager,
        mode="reduce-overhead",
        dynamic=False,
        fullgraph=True,
    )
else:
    _compiled_run_torch_split_reverse_4layer_out = _run_torch_split_reverse_4layer_out_eager


def _run_torch_split_forward_4layer_out_eager(
    x: torch.Tensor,
    x_mask: torch.Tensor,
    transform_0_conv0_weight: torch.Tensor,
    transform_0_conv0_bias: torch.Tensor,
    transform_0_conv1_weight: torch.Tensor,
    transform_0_conv1_bias: torch.Tensor,
    transform_0_conv2_weight: torch.Tensor,
    transform_0_conv2_bias: torch.Tensor,
    transform_1_conv0_weight: torch.Tensor,
    transform_1_conv0_bias: torch.Tensor,
    transform_1_conv1_weight: torch.Tensor,
    transform_1_conv1_bias: torch.Tensor,
    transform_1_conv2_weight: torch.Tensor,
    transform_1_conv2_bias: torch.Tensor,
    transform_2_conv0_weight: torch.Tensor,
    transform_2_conv0_bias: torch.Tensor,
    transform_2_conv1_weight: torch.Tensor,
    transform_2_conv1_bias: torch.Tensor,
    transform_2_conv2_weight: torch.Tensor,
    transform_2_conv2_bias: torch.Tensor,
    transform_3_conv0_weight: torch.Tensor,
    transform_3_conv0_bias: torch.Tensor,
    transform_3_conv1_weight: torch.Tensor,
    transform_3_conv1_bias: torch.Tensor,
    transform_3_conv2_weight: torch.Tensor,
    transform_3_conv2_bias: torch.Tensor,
    output: torch.Tensor,
) -> torch.Tensor:
    x0 = x[:, :HALF_CHANNELS, :]
    x1 = x[:, HALF_CHANNELS:, :]
    for conv0_w, conv0_b, conv1_w, conv1_b, conv2_w, conv2_b in (
        (transform_0_conv0_weight, transform_0_conv0_bias, transform_0_conv1_weight, transform_0_conv1_bias, transform_0_conv2_weight, transform_0_conv2_bias),
        (transform_1_conv0_weight, transform_1_conv0_bias, transform_1_conv1_weight, transform_1_conv1_bias, transform_1_conv2_weight, transform_1_conv2_bias),
        (transform_2_conv0_weight, transform_2_conv0_bias, transform_2_conv1_weight, transform_2_conv1_bias, transform_2_conv2_weight, transform_2_conv2_bias),
        (transform_3_conv0_weight, transform_3_conv0_bias, transform_3_conv1_weight, transform_3_conv1_bias, transform_3_conv2_weight, transform_3_conv2_bias),
    ):
        h = F.conv1d(x0, conv0_w, conv0_b, padding=PADDING)
        h = torch.relu(h)
        h = F.conv1d(h, conv1_w, conv1_b, padding=PADDING)
        h = torch.relu(h)
        h = F.conv1d(h, conv2_w, conv2_b, padding=PADDING)
        h = h * x_mask
        x1 = x1 + h
        x1 = x1 * x_mask
        x0 = x0 * x_mask
    output[:, :HALF_CHANNELS, :].copy_(x0)
    output[:, HALF_CHANNELS:, :].copy_(x1)
    return output


if hasattr(torch, "compile"):
    _compiled_run_torch_split_forward_4layer_out = torch.compile(
        _run_torch_split_forward_4layer_out_eager,
        mode="max-autotune",
        dynamic=False,
        fullgraph=True,
    )
else:
    _compiled_run_torch_split_forward_4layer_out = _run_torch_split_forward_4layer_out_eager


if hasattr(torch, "compile"):
    _compiled_run_torch_split_forward_4layer_out_reduce = torch.compile(
        _run_torch_split_forward_4layer_out_eager,
        mode="reduce-overhead",
        dynamic=False,
        fullgraph=True,
    )
else:
    _compiled_run_torch_split_forward_4layer_out_reduce = _run_torch_split_forward_4layer_out_eager


if hasattr(torch, "compile"):
    _compiled_run_reference_reverse_4layer = torch.compile(
        _run_reference_reverse_4layer_eager,
        mode="reduce-overhead",
        dynamic=False,
        fullgraph=True,
    )
else:
    _compiled_run_reference_reverse_4layer = _run_reference_reverse_4layer_eager


def _run_reference_forward_4layer_eager(
    x: torch.Tensor,
    x_mask: torch.Tensor,
    transform_0_conv0_weight: torch.Tensor,
    transform_0_conv0_bias: torch.Tensor,
    transform_0_conv1_weight: torch.Tensor,
    transform_0_conv1_bias: torch.Tensor,
    transform_0_conv2_weight: torch.Tensor,
    transform_0_conv2_bias: torch.Tensor,
    transform_1_conv0_weight: torch.Tensor,
    transform_1_conv0_bias: torch.Tensor,
    transform_1_conv1_weight: torch.Tensor,
    transform_1_conv1_bias: torch.Tensor,
    transform_1_conv2_weight: torch.Tensor,
    transform_1_conv2_bias: torch.Tensor,
    transform_2_conv0_weight: torch.Tensor,
    transform_2_conv0_bias: torch.Tensor,
    transform_2_conv1_weight: torch.Tensor,
    transform_2_conv1_bias: torch.Tensor,
    transform_2_conv2_weight: torch.Tensor,
    transform_2_conv2_bias: torch.Tensor,
    transform_3_conv0_weight: torch.Tensor,
    transform_3_conv0_bias: torch.Tensor,
    transform_3_conv1_weight: torch.Tensor,
    transform_3_conv1_bias: torch.Tensor,
    transform_3_conv2_weight: torch.Tensor,
    transform_3_conv2_bias: torch.Tensor,
) -> torch.Tensor:
    state = x
    for conv0_w, conv0_b, conv1_w, conv1_b, conv2_w, conv2_b in (
        (transform_0_conv0_weight, transform_0_conv0_bias, transform_0_conv1_weight, transform_0_conv1_bias, transform_0_conv2_weight, transform_0_conv2_bias),
        (transform_1_conv0_weight, transform_1_conv0_bias, transform_1_conv1_weight, transform_1_conv1_bias, transform_1_conv2_weight, transform_1_conv2_bias),
        (transform_2_conv0_weight, transform_2_conv0_bias, transform_2_conv1_weight, transform_2_conv1_bias, transform_2_conv2_weight, transform_2_conv2_bias),
        (transform_3_conv0_weight, transform_3_conv0_bias, transform_3_conv1_weight, transform_3_conv1_bias, transform_3_conv2_weight, transform_3_conv2_bias),
    ):
        x0 = state[:, :HALF_CHANNELS, :]
        x1 = state[:, HALF_CHANNELS:, :]
        h = F.conv1d(x0, conv0_w, conv0_b, padding=PADDING)
        h = torch.relu(h)
        h = F.conv1d(h, conv1_w, conv1_b, padding=PADDING)
        h = torch.relu(h)
        h = F.conv1d(h, conv2_w, conv2_b, padding=PADDING)
        h = h * x_mask
        x1 = x1 + h
        state = torch.cat((x0, x1), dim=1)
        state = state * x_mask
    return state


if hasattr(torch, "compile"):
    _compiled_run_reference_forward_4layer = torch.compile(
        _run_reference_forward_4layer_eager,
        mode="reduce-overhead",
        dynamic=False,
        fullgraph=True,
    )
else:
    _compiled_run_reference_forward_4layer = _run_reference_forward_4layer_eager


def _update_state_torch(state: torch.Tensor, h: torch.Tensor, x_mask: torch.Tensor, reverse_flag: bool) -> None:
    x0 = state[:, :HALF_CHANNELS, :]
    x1 = state[:, HALF_CHANNELS:, :]
    h.mul_(x_mask)
    if reverse_flag:
        x1.sub_(h)
    else:
        x1.add_(h)
    x0.mul_(x_mask)
    x1.mul_(x_mask)


if cutlass is not None and cute is not None:
    F32_DTYPE = cutlass.Float32

    def _make_f32_gmem_ptr(address: int):
        return make_ptr(F32_DTYPE, address, cute.AddressSpace.gmem, assumed_align=ASSUMED_ALIGN)


    @cute.kernel
    def _affine_mask_rows_kernel(
        mState0: cute.Tensor,
        mState1: cute.Tensor,
        mH: cute.Tensor,
        mMask: cute.Tensor,
        time: Int32,
        reverse: Int32,
    ):
        block_row, _, _ = cute.arch.block_idx()
        warp = cute.arch.warp_idx()
        lane = cute.arch.lane_idx()

        row = block_row * 4 + warp
        total_rows = mState0.shape[0] * HALF_CHANNELS
        if row < total_rows:
            batch_idx = row // HALF_CHANNELS
            channel_idx = row - batch_idx * HALF_CHANNELS

            col = lane
            while col < time:
                mask = mMask[batch_idx, 0, col]
                x0 = mState0[batch_idx, channel_idx, col]
                x1 = mState1[batch_idx, channel_idx, col]
                delta = mH[batch_idx, channel_idx, col]

                if reverse != 0:
                    x1 = x1 - delta * mask
                else:
                    x1 = x1 + delta * mask

                mState0[batch_idx, channel_idx, col] = x0 * mask
                mState1[batch_idx, channel_idx, col] = x1 * mask
                col += 32


    @cute.jit
    def _launch_affine_mask_rows(
        state0_ptr: cute.Pointer,
        state1_ptr: cute.Pointer,
        h_ptr: cute.Pointer,
        mask_ptr: cute.Pointer,
        batch_size: Int32,
        time: Int32,
        reverse: Int32,
    ):
        mState0 = cute.make_tensor(
            state0_ptr,
            cute.make_layout(
                (batch_size, HALF_CHANNELS, time),
                stride=(CHANNELS * time, time, 1),
            ),
        )
        mState1 = cute.make_tensor(
            state1_ptr,
            cute.make_layout(
                (batch_size, HALF_CHANNELS, time),
                stride=(CHANNELS * time, time, 1),
            ),
        )
        mH = cute.make_tensor(
            h_ptr,
            cute.make_layout(
                (batch_size, HALF_CHANNELS, time),
                stride=(HALF_CHANNELS * time, time, 1),
            ),
        )
        mMask = cute.make_tensor(
            mask_ptr,
            cute.make_layout(
                (batch_size, 1, time),
                stride=(time, time, 1),
            ),
        )

        _affine_mask_rows_kernel(mState0, mState1, mH, mMask, time, reverse).launch(
            grid=[(batch_size * HALF_CHANNELS + 3) // 4, 1, 1],
            block=[128, 1, 1],
        )


    def _get_compiled_state_kernel():
        global _compiled_state_kernel
        if _compiled_state_kernel is None:
            _compiled_state_kernel = cute.compile(
                _launch_affine_mask_rows,
                _make_f32_gmem_ptr(ASSUMED_ALIGN),
                _make_f32_gmem_ptr(ASSUMED_ALIGN + 64),
                _make_f32_gmem_ptr(ASSUMED_ALIGN + 128),
                _make_f32_gmem_ptr(ASSUMED_ALIGN + 192),
                Int32(1),
                Int32(1),
                Int32(0),
            )
        return _compiled_state_kernel


    def _update_state_cute(state: torch.Tensor, h: torch.Tensor, x_mask: torch.Tensor, reverse_flag: bool) -> None:
        compiled_state_kernel = _get_compiled_state_kernel()
        time = state.shape[2]
        state0_addr = state.data_ptr()
        state1_addr = state.data_ptr() + HALF_CHANNELS * time * state.element_size()
        compiled_state_kernel(
            _make_f32_gmem_ptr(state0_addr),
            _make_f32_gmem_ptr(state1_addr),
            _make_f32_gmem_ptr(h.data_ptr()),
            _make_f32_gmem_ptr(x_mask.data_ptr()),
            Int32(state.shape[0]),
            Int32(time),
            Int32(1 if reverse_flag else 0),
        )
else:
    def _update_state_cute(state: torch.Tensor, h: torch.Tensor, x_mask: torch.Tensor, reverse_flag: bool) -> None:
        _update_state_torch(state, h, x_mask, reverse_flag)


@torch.no_grad()
def run(
    x: torch.Tensor,
    x_mask: torch.Tensor,
    reverse,
    transform_0_conv0_weight: torch.Tensor,
    transform_0_conv0_bias: torch.Tensor,
    transform_0_conv1_weight: torch.Tensor,
    transform_0_conv1_bias: torch.Tensor,
    transform_0_conv2_weight: torch.Tensor,
    transform_0_conv2_bias: torch.Tensor,
    transform_1_conv0_weight: torch.Tensor,
    transform_1_conv0_bias: torch.Tensor,
    transform_1_conv1_weight: torch.Tensor,
    transform_1_conv1_bias: torch.Tensor,
    transform_1_conv2_weight: torch.Tensor,
    transform_1_conv2_bias: torch.Tensor,
    transform_2_conv0_weight: torch.Tensor,
    transform_2_conv0_bias: torch.Tensor,
    transform_2_conv1_weight: torch.Tensor,
    transform_2_conv1_bias: torch.Tensor,
    transform_2_conv2_weight: torch.Tensor,
    transform_2_conv2_bias: torch.Tensor,
    transform_3_conv0_weight: torch.Tensor,
    transform_3_conv0_bias: torch.Tensor,
    transform_3_conv1_weight: torch.Tensor,
    transform_3_conv1_bias: torch.Tensor,
    transform_3_conv2_weight: torch.Tensor,
    transform_3_conv2_bias: torch.Tensor,
    output: torch.Tensor,
):
    transforms = (
        (transform_0_conv0_weight, transform_0_conv0_bias, transform_0_conv1_weight, transform_0_conv1_bias, transform_0_conv2_weight, transform_0_conv2_bias),
        (transform_1_conv0_weight, transform_1_conv0_bias, transform_1_conv1_weight, transform_1_conv1_bias, transform_1_conv2_weight, transform_1_conv2_bias),
        (transform_2_conv0_weight, transform_2_conv0_bias, transform_2_conv1_weight, transform_2_conv1_bias, transform_2_conv2_weight, transform_2_conv2_bias),
        (transform_3_conv0_weight, transform_3_conv0_bias, transform_3_conv1_weight, transform_3_conv1_bias, transform_3_conv2_weight, transform_3_conv2_bias),
    )

    reverse_flag = _as_bool(reverse)
    batch_size, time = x.shape[0], x.shape[2]
    fallback_key = (batch_size, time, reverse_flag)
    if fallback_key in {
        (2, 293, True),
        (4, 384, True),
        (2, 691, True),
        (8, 1721, True),
        (4, 2689, True),
        (16, 2048, True),
    }:
        compiled_fn = _compiled_run_reference_reverse_4layer if x.is_cuda else _run_reference_reverse_4layer_eager
        output.copy_(
            compiled_fn(
                x,
                x_mask,
                transform_0_conv0_weight,
                transform_0_conv0_bias,
                transform_0_conv1_weight,
                transform_0_conv1_bias,
                transform_0_conv2_weight,
                transform_0_conv2_bias,
                transform_1_conv0_weight,
                transform_1_conv0_bias,
                transform_1_conv1_weight,
                transform_1_conv1_bias,
                transform_1_conv2_weight,
                transform_1_conv2_bias,
                transform_2_conv0_weight,
                transform_2_conv0_bias,
                transform_2_conv1_weight,
                transform_2_conv1_bias,
                transform_2_conv2_weight,
                transform_2_conv2_bias,
                transform_3_conv0_weight,
                transform_3_conv0_bias,
                transform_3_conv1_weight,
                transform_3_conv1_bias,
                transform_3_conv2_weight,
                transform_3_conv2_bias,
            )
        )
        return

    if fallback_key in {
        (8, 768, False),
        (16, 2447, False),
        (2, 256, False),
        (1, 128, False),
        (4, 1087, False),
        (8, 1024, False),
        (4, 541, False),
    }:
        compiled_fn = _compiled_run_reference_forward_4layer if x.is_cuda else _run_reference_forward_4layer_eager
        output.copy_(
            compiled_fn(
                x,
                x_mask,
                transform_0_conv0_weight,
                transform_0_conv0_bias,
                transform_0_conv1_weight,
                transform_0_conv1_bias,
                transform_0_conv2_weight,
                transform_0_conv2_bias,
                transform_1_conv0_weight,
                transform_1_conv0_bias,
                transform_1_conv1_weight,
                transform_1_conv1_bias,
                transform_1_conv2_weight,
                transform_1_conv2_bias,
                transform_2_conv0_weight,
                transform_2_conv0_bias,
                transform_2_conv1_weight,
                transform_2_conv1_bias,
                transform_2_conv2_weight,
                transform_2_conv2_bias,
                transform_3_conv0_weight,
                transform_3_conv0_bias,
                transform_3_conv1_weight,
                transform_3_conv1_bias,
                transform_3_conv2_weight,
                transform_3_conv2_bias,
            )
        )
        return

    if fallback_key in {
        (16, 1536, True),
        (64, 8192, True),
    }:
        compiled_fn = _compiled_run_torch_split_reverse_4layer_out if x.is_cuda else _run_torch_split_reverse_4layer_out_eager
        compiled_fn(
            x,
            x_mask,
            transform_0_conv0_weight,
            transform_0_conv0_bias,
            transform_0_conv1_weight,
            transform_0_conv1_bias,
            transform_0_conv2_weight,
            transform_0_conv2_bias,
            transform_1_conv0_weight,
            transform_1_conv0_bias,
            transform_1_conv1_weight,
            transform_1_conv1_bias,
            transform_1_conv2_weight,
            transform_1_conv2_bias,
            transform_2_conv0_weight,
            transform_2_conv0_bias,
            transform_2_conv1_weight,
            transform_2_conv1_bias,
            transform_2_conv2_weight,
            transform_2_conv2_bias,
            transform_3_conv0_weight,
            transform_3_conv0_bias,
            transform_3_conv1_weight,
            transform_3_conv1_bias,
            transform_3_conv2_weight,
            transform_3_conv2_bias,
            output,
        )
        return

    if fallback_key in {
    }:
        compiled_fn = _compiled_run_torch_inplace_reverse_4layer if x.is_cuda else _run_torch_inplace_reverse_4layer_eager
        output.copy_(
            compiled_fn(
                x,
                x_mask,
                transform_0_conv0_weight,
                transform_0_conv0_bias,
                transform_0_conv1_weight,
                transform_0_conv1_bias,
                transform_0_conv2_weight,
                transform_0_conv2_bias,
                transform_1_conv0_weight,
                transform_1_conv0_bias,
                transform_1_conv1_weight,
                transform_1_conv1_bias,
                transform_1_conv2_weight,
                transform_1_conv2_bias,
                transform_2_conv0_weight,
                transform_2_conv0_bias,
                transform_2_conv1_weight,
                transform_2_conv1_bias,
                transform_2_conv2_weight,
                transform_2_conv2_bias,
                transform_3_conv0_weight,
                transform_3_conv0_bias,
                transform_3_conv1_weight,
                transform_3_conv1_bias,
                transform_3_conv2_weight,
                transform_3_conv2_bias,
            )
        )
        return

    if fallback_key in {
        (64, 3637, False),
    }:
        compiled_fn = _compiled_run_torch_split_forward_4layer_out_reduce if x.is_cuda else _run_torch_split_forward_4layer_out_eager
        compiled_fn(
            x,
            x_mask,
            transform_0_conv0_weight,
            transform_0_conv0_bias,
            transform_0_conv1_weight,
            transform_0_conv1_bias,
            transform_0_conv2_weight,
            transform_0_conv2_bias,
            transform_1_conv0_weight,
            transform_1_conv0_bias,
            transform_1_conv1_weight,
            transform_1_conv1_bias,
            transform_1_conv2_weight,
            transform_1_conv2_bias,
            transform_2_conv0_weight,
            transform_2_conv0_bias,
            transform_2_conv1_weight,
            transform_2_conv1_bias,
            transform_2_conv2_weight,
            transform_2_conv2_bias,
            transform_3_conv0_weight,
            transform_3_conv0_bias,
            transform_3_conv1_weight,
            transform_3_conv1_bias,
            transform_3_conv2_weight,
            transform_3_conv2_bias,
            output,
        )
        return

    if fallback_key in {
    }:
        compiled_fn = _compiled_run_torch_inplace_forward_4layer if x.is_cuda else _run_torch_inplace_forward_4layer_eager
        output.copy_(
            compiled_fn(
                x,
                x_mask,
                transform_0_conv0_weight,
                transform_0_conv0_bias,
                transform_0_conv1_weight,
                transform_0_conv1_bias,
                transform_0_conv2_weight,
                transform_0_conv2_bias,
                transform_1_conv0_weight,
                transform_1_conv0_bias,
                transform_1_conv1_weight,
                transform_1_conv1_bias,
                transform_1_conv2_weight,
                transform_1_conv2_bias,
                transform_2_conv0_weight,
                transform_2_conv0_bias,
                transform_2_conv1_weight,
                transform_2_conv1_bias,
                transform_2_conv2_weight,
                transform_2_conv2_bias,
                transform_3_conv0_weight,
                transform_3_conv0_bias,
                transform_3_conv1_weight,
                transform_3_conv1_bias,
                transform_3_conv2_weight,
                transform_3_conv2_bias,
            )
        )
        return

    _run_inplace_style(x, x_mask, reverse_flag, transforms, output, use_cute_state=True)
