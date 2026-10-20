import os
os.environ.setdefault("CUTE_DSL_ARCH", "sm_100a")

import torch
import cutlass
import cutlass.cute as cute
import cutlass.utils as cutlass_utils
from cutlass.cute.runtime import from_dlpack

from blockwise_gemm_impl import BlockwiseGemmKernel


_E4M3_MAX = 448.0
_BLOCK_K = 128
_BLOCK_N = 128
_MMA_TILER_MN = (128, 128)
_CLUSTER_SHAPE_MN = (1, 1)
_COMPILED_GEMM = {}
_MAX_ACTIVE_CLUSTERS = None
_OPT_OPTIONS = None


def _get_opt_options() -> str:
    global _OPT_OPTIONS
    if _OPT_OPTIONS is None:
        try:
            from cutlass import CUDA_VERSION

            opt_level = 3 if CUDA_VERSION.major < 13 or (
                CUDA_VERSION.major == 13 and CUDA_VERSION.minor < 1
            ) else 2
        except ImportError:
            opt_level = 3
        _OPT_OPTIONS = f"--opt-level {opt_level}"
    return _OPT_OPTIONS



def _view_with_unit_batch(tensor_2d: torch.Tensor) -> torch.Tensor:
    rows, cols = tensor_2d.shape
    return torch.as_strided(
        tensor_2d,
        size=(rows, cols, 1),
        stride=(tensor_2d.stride(0), tensor_2d.stride(1), rows * cols),
    )



def _to_cute_dynamic_m(tensor: torch.Tensor) -> cute.Tensor:
    return from_dlpack(
        tensor,
        assumed_align=16,
        use_32bit_stride=True,
    ).mark_layout_dynamic(leading_dim=1)



def _to_cute_static(tensor: torch.Tensor) -> cute.Tensor:
    return from_dlpack(tensor, assumed_align=16)



def _to_cute_fp8_dynamic_m(tensor_bits: torch.Tensor) -> cute.Tensor:
    tensor = from_dlpack(
        tensor_bits,
        assumed_align=16,
        use_32bit_stride=True,
    )
    tensor.element_type = cutlass.Float8E4M3FN
    return tensor.mark_layout_dynamic(leading_dim=1)



def _to_cute_fp8_static(tensor_bits: torch.Tensor) -> cute.Tensor:
    tensor = from_dlpack(tensor_bits, assumed_align=16)
    tensor.element_type = cutlass.Float8E4M3FN
    return tensor



def _check_k_multiple(k: int):
    if k % _BLOCK_K != 0:
        raise ValueError(f"Expected K to be a multiple of {_BLOCK_K}, got {k}")



def _check_nk_multiples(n: int, k: int):
    if k % _BLOCK_K != 0 or n % _BLOCK_N != 0:
        raise ValueError(
            f"Expected N and K to be multiples of {_BLOCK_N}/{_BLOCK_K}, got N={n}, K={k}"
        )



def _cutlass_output_dtype(dtype: torch.dtype):
    if dtype == torch.bfloat16:
        return cutlass.BFloat16
    if dtype == torch.float32:
        return cutlass.Float32
    raise TypeError(f"Unsupported output dtype: {dtype}")



@torch.no_grad()
def quantize_activation(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    m, k = x.shape
    _check_k_multiple(k)
    num_k_blocks = k // _BLOCK_K
    x_fp32 = x.to(torch.float32)
    x_blocks = x_fp32.reshape(m, num_k_blocks, _BLOCK_K)
    scale_x = (x_blocks.abs().amax(dim=2) / _E4M3_MAX).clamp(min=1e-12)
    qx = (
        (x_blocks / scale_x.unsqueeze(2))
        .clamp(-_E4M3_MAX, _E4M3_MAX)
        .to(torch.float8_e4m3fn)
        .reshape(m, k)
    )
    return qx, scale_x



@torch.no_grad()
def quantize_weight_runtime(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    n, k = weight.shape
    _check_nk_multiples(n, k)
    num_k_blocks = k // _BLOCK_K
    num_n_blocks = n // _BLOCK_N
    weight_fp32 = weight.t().to(torch.float32).contiguous()
    weight_blocks = weight_fp32.reshape(num_k_blocks, _BLOCK_K, num_n_blocks, _BLOCK_N)
    scale_kn = (weight_blocks.abs().amax(dim=(1, 3)) / _E4M3_MAX).clamp(min=1e-12)
    qw = (
        (weight_blocks / scale_kn[:, None, :, None])
        .clamp(-_E4M3_MAX, _E4M3_MAX)
        .reshape(k, n)
        .t()
        .contiguous()
        .to(torch.float8_e4m3fn)
    )
    return qw, scale_kn.t().contiguous()



@torch.no_grad()
def quantize_weight_from_scales_kn(
    weight: torch.Tensor,
    scale_kn: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    n, k = weight.shape
    _check_nk_multiples(n, k)
    num_k_blocks = k // _BLOCK_K
    num_n_blocks = n // _BLOCK_N
    expected = (num_k_blocks, num_n_blocks)
    if tuple(scale_kn.shape) != expected:
        raise ValueError(f"scale_kn shape must be {expected}, got {tuple(scale_kn.shape)}")
    weight_fp32 = weight.t().to(torch.float32).contiguous()
    weight_blocks = weight_fp32.reshape(num_k_blocks, _BLOCK_K, num_n_blocks, _BLOCK_N)
    scale_kn_f32 = scale_kn.to(torch.float32).contiguous()
    qw = (
        (weight_blocks / scale_kn_f32[:, None, :, None])
        .clamp(-_E4M3_MAX, _E4M3_MAX)
        .reshape(k, n)
        .t()
        .contiguous()
        .to(torch.float8_e4m3fn)
    )
    return qw, scale_kn_f32.t().contiguous()



def _dequantize_activation(qx: torch.Tensor, scale_x: torch.Tensor) -> torch.Tensor:
    m, k = qx.shape
    return (
        qx.to(torch.float32).reshape(m, k // _BLOCK_K, _BLOCK_K) * scale_x.to(torch.float32).unsqueeze(2)
    ).reshape(m, k)



def _dequantize_weight(qw: torch.Tensor, scale_w_t: torch.Tensor) -> torch.Tensor:
    n, k = qw.shape
    num_k_blocks = k // _BLOCK_K
    num_n_blocks = n // _BLOCK_N
    scale_kn = scale_w_t.to(torch.float32).t().contiguous()
    weight_t = (
        qw.to(torch.float32)
        .t()
        .reshape(num_k_blocks, _BLOCK_K, num_n_blocks, _BLOCK_N)
        * scale_kn[:, None, :, None]
    )
    return weight_t.reshape(k, n).t().contiguous()



@torch.no_grad()
def reference_fp8_blockwise_mm_explicit(
    qx: torch.Tensor,
    scale_x: torch.Tensor,
    qw: torch.Tensor,
    scale_w_t: torch.Tensor,
    bias: torch.Tensor | None = None,
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    y = _dequantize_activation(qx, scale_x) @ _dequantize_weight(qw, scale_w_t).t()
    if bias is not None and bias.numel():
        y = y + bias.to(torch.float32)
    return y.to(out_dtype)



def _get_max_active_clusters() -> int:
    global _MAX_ACTIVE_CLUSTERS
    if _MAX_ACTIVE_CLUSTERS is None:
        hardware_info = cutlass_utils.HardwareInfo()
        _MAX_ACTIVE_CLUSTERS = hardware_info.get_max_active_clusters(
            _CLUSTER_SHAPE_MN[0] * _CLUSTER_SHAPE_MN[1]
        )
    return _MAX_ACTIVE_CLUSTERS



def _compile_gemm(
    a_bits_3d: torch.Tensor,
    b_bits_3d: torch.Tensor,
    c_3d: torch.Tensor,
    sfa_3d: torch.Tensor,
    sfb_3d: torch.Tensor,
):
    key = (b_bits_3d.shape[0], a_bits_3d.shape[1], c_3d.dtype)
    compiled = _COMPILED_GEMM.get(key)
    if compiled is not None:
        return compiled

    gemm = BlockwiseGemmKernel(
        acc_dtype=cutlass.Float32,
        use_2cta_instrs=False,
        mma_tiler_mn=_MMA_TILER_MN,
        cluster_shape_mn=_CLUSTER_SHAPE_MN,
    )
    compiled = cute.compile(
        gemm,
        _to_cute_fp8_dynamic_m(a_bits_3d),
        _to_cute_fp8_static(b_bits_3d),
        _to_cute_dynamic_m(c_3d),
        _to_cute_dynamic_m(sfa_3d),
        _to_cute_static(sfb_3d),
        _get_max_active_clusters(),
        options=_get_opt_options(),
    )
    _COMPILED_GEMM[key] = compiled
    return compiled



@torch.no_grad()
def fp8_blockwise_mm_explicit(
    qx: torch.Tensor,
    scale_x: torch.Tensor,
    qw: torch.Tensor,
    scale_w_t: torch.Tensor,
    bias: torch.Tensor | None = None,
    out_dtype: torch.dtype = torch.bfloat16,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    if not qx.is_cuda or not qw.is_cuda:
        result = reference_fp8_blockwise_mm_explicit(
            qx,
            scale_x,
            qw,
            scale_w_t,
            bias=bias,
            out_dtype=out_dtype,
        )
        if out is not None:
            out.copy_(result)
            return out
        return result

    m, k = qx.shape
    n, k_weight = qw.shape
    if k != k_weight:
        raise ValueError(f"K dimension mismatch: {k} vs {k_weight}")
    if qx.dtype != torch.float8_e4m3fn or qw.dtype != torch.float8_e4m3fn:
        raise TypeError("fp8_blockwise_mm_explicit expects FP8 e4m3fn tensors")
    expected_scale_x = (m, k // _BLOCK_K)
    expected_scale_w_t = (n // _BLOCK_N, k // _BLOCK_K)
    if tuple(scale_x.shape) != expected_scale_x:
        raise ValueError(f"scale_x shape must be {expected_scale_x}, got {tuple(scale_x.shape)}")
    if tuple(scale_w_t.shape) != expected_scale_w_t:
        raise ValueError(
            f"scale_w_t shape must be {expected_scale_w_t}, got {tuple(scale_w_t.shape)}"
        )

    c_cutlass_dtype = _cutlass_output_dtype(out_dtype)
    if not BlockwiseGemmKernel.can_implement(
        cutlass.Float8E4M3FN,
        cutlass.Float32,
        c_cutlass_dtype,
        False,
        _MMA_TILER_MN,
        _CLUSTER_SHAPE_MN,
        m,
        n,
        k,
        1,
        "k",
        "k",
        "n",
    ):
        raise RuntimeError(
            f"Unsupported configuration for BlockwiseGemmKernel: M={m}, N={n}, K={k}"
        )

    qx = qx.contiguous()
    qw = qw.contiguous()
    scale_x = scale_x.to(torch.float32).contiguous()
    scale_w_t = scale_w_t.to(torch.float32).contiguous()
    if out is None:
        output = torch.empty((m, n), dtype=out_dtype, device=qx.device)
    else:
        output = out
        if output.shape != (m, n) or output.dtype != out_dtype or output.device != qx.device:
            raise ValueError(
                f"out must have shape {(m, n)}, dtype {out_dtype}, and device {qx.device}, "
                f"got shape={output.shape}, dtype={output.dtype}, device={output.device}"
            )

    a_bits_3d = _view_with_unit_batch(qx.view(torch.int8))
    b_bits_3d = _view_with_unit_batch(qw.view(torch.int8))
    c_3d = _view_with_unit_batch(output)
    sfa_3d = _view_with_unit_batch(scale_x)
    sfb_3d = _view_with_unit_batch(scale_w_t)

    compiled_gemm = _compile_gemm(a_bits_3d, b_bits_3d, c_3d, sfa_3d, sfb_3d)
    compiled_gemm(
        _to_cute_fp8_dynamic_m(a_bits_3d),
        _to_cute_fp8_static(b_bits_3d),
        _to_cute_dynamic_m(c_3d),
        _to_cute_dynamic_m(sfa_3d),
        _to_cute_static(sfb_3d),
    )
    if bias is not None and bias.numel():
        output.add_(bias)
    return output



@torch.no_grad()
def fp8_blockwise_linear_runtime(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    if not x.is_cuda or not weight.is_cuda:
        qx, scale_x = quantize_activation(x)
        qw, scale_w_t = quantize_weight_runtime(weight)
        return reference_fp8_blockwise_mm_explicit(
            qx,
            scale_x,
            qw,
            scale_w_t,
            bias=bias,
            out_dtype=out_dtype,
        )

    qx, scale_x = quantize_activation(x)
    qw, scale_w_t = quantize_weight_runtime(weight)
    return fp8_blockwise_mm_explicit(
        qx,
        scale_x,
        qw,
        scale_w_t,
        bias=bias,
        out_dtype=out_dtype,
    )



@torch.no_grad()
def fp8_blockwise_linear_weight_scales(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_scales_kn: torch.Tensor,
    bias: torch.Tensor | None = None,
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    qx, scale_x = quantize_activation(x)
    qw, scale_w_t = quantize_weight_from_scales_kn(weight, weight_scales_kn)
    return fp8_blockwise_mm_explicit(
        qx,
        scale_x,
        qw,
        scale_w_t,
        bias=bias,
        out_dtype=out_dtype,
    )
