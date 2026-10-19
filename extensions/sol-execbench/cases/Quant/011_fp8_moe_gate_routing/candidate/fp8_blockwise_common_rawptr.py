import os
os.environ.setdefault("CUTE_DSL_ARCH", "sm_100a")

import torch
import cutlass
import cutlass.cute as cute
import cutlass.utils as cutlass_utils
from cutlass.cute.runtime import make_ptr

from blockwise_gemm_impl import BlockwiseGemmKernel


_BLOCK_K = 128
_BLOCK_N = 128
_HIDDEN_SIZE = 7168
_NUM_EXPERTS = 256
_HIDDEN_BLOCKS = _HIDDEN_SIZE // _BLOCK_K
_EXPERT_BLOCKS = _NUM_EXPERTS // _BLOCK_N
_MMA_TILER_MN = (128, 128)
_CLUSTER_SHAPE_MN = (1, 1)
_USE_2CTA = False
_ASSUMED_ALIGN = 16
_EXACT_M_SET = frozenset({256, 640, 768, 1024, 1280, 1536, 2560, 2816, 3328, 4352, 4608, 4864, 5888, 6656, 7168, 8192})

_GEMM = BlockwiseGemmKernel(
    acc_dtype=cutlass.Float32,
    use_2cta_instrs=_USE_2CTA,
    mma_tiler_mn=_MMA_TILER_MN,
    cluster_shape_mn=_CLUSTER_SHAPE_MN,
)
_COMPILED_EXACT = {}
_COMPILED_DYNAMIC = None
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
        except Exception:
            opt_level = 3
        _OPT_OPTIONS = f"--opt-level {opt_level}"
    return _OPT_OPTIONS


def _dequantize_activation(qx: torch.Tensor, scale_x: torch.Tensor) -> torch.Tensor:
    m, k = qx.shape
    return (
        qx.to(torch.float32).reshape(m, k // _BLOCK_K, _BLOCK_K)
        * scale_x.to(torch.float32).unsqueeze(2)
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
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    y = _dequantize_activation(qx, scale_x) @ _dequantize_weight(qw, scale_w_t).t()
    return y.to(out_dtype)


def _get_max_active_clusters() -> int:
    global _MAX_ACTIVE_CLUSTERS
    if _MAX_ACTIVE_CLUSTERS is None:
        hardware_info = cutlass_utils.HardwareInfo()
        _MAX_ACTIVE_CLUSTERS = hardware_info.get_max_active_clusters(
            _CLUSTER_SHAPE_MN[0] * _CLUSTER_SHAPE_MN[1]
        )
    return _MAX_ACTIVE_CLUSTERS


def _fp8_ptr(tensor: torch.Tensor):
    return make_ptr(
        cutlass.Float8E4M3FN,
        tensor.data_ptr(),
        cute.AddressSpace.gmem,
        assumed_align=_ASSUMED_ALIGN,
    )


def _f32_ptr(tensor: torch.Tensor):
    return make_ptr(
        cutlass.Float32,
        tensor.data_ptr(),
        cute.AddressSpace.gmem,
        assumed_align=_ASSUMED_ALIGN,
    )


def _bf16_ptr(tensor: torch.Tensor):
    return make_ptr(
        cutlass.BFloat16,
        tensor.data_ptr(),
        cute.AddressSpace.gmem,
        assumed_align=_ASSUMED_ALIGN,
    )


@cute.jit
def _host_blockwise_gemm_exact(
    a_ptr: cute.Pointer,
    b_ptr: cute.Pointer,
    c_ptr: cute.Pointer,
    sfa_ptr: cute.Pointer,
    sfb_ptr: cute.Pointer,
    m: cutlass.Constexpr,
):
    a = cute.make_tensor(
        a_ptr,
        cute.make_layout((m, _HIDDEN_SIZE, 1), stride=(_HIDDEN_SIZE, 1, 1)),
    )
    b = cute.make_tensor(
        b_ptr,
        cute.make_layout((_NUM_EXPERTS, _HIDDEN_SIZE, 1), stride=(_HIDDEN_SIZE, 1, 1)),
    )
    c = cute.make_tensor(
        c_ptr,
        cute.make_layout((m, _NUM_EXPERTS, 1), stride=(_NUM_EXPERTS, 1, 1)),
    )
    sfa = cute.make_tensor(
        sfa_ptr,
        cute.make_layout((m, _HIDDEN_BLOCKS, 1), stride=(_HIDDEN_BLOCKS, 1, 1)),
    )
    sfb = cute.make_tensor(
        sfb_ptr,
        cute.make_layout((_EXPERT_BLOCKS, _HIDDEN_BLOCKS, 1), stride=(_HIDDEN_BLOCKS, 1, 1)),
    )
    _GEMM(a, b, c, sfa, sfb, _MAX_ACTIVE_CLUSTERS)


@cute.jit
def _host_blockwise_gemm_dynamic(
    a_ptr: cute.Pointer,
    b_ptr: cute.Pointer,
    c_ptr: cute.Pointer,
    sfa_ptr: cute.Pointer,
    sfb_ptr: cute.Pointer,
    m: cutlass.Int32,
):
    a = cute.make_tensor(
        a_ptr,
        cute.make_layout((m, _HIDDEN_SIZE, 1), stride=(_HIDDEN_SIZE, 1, 1)),
    )
    b = cute.make_tensor(
        b_ptr,
        cute.make_layout((_NUM_EXPERTS, _HIDDEN_SIZE, 1), stride=(_HIDDEN_SIZE, 1, 1)),
    )
    c = cute.make_tensor(
        c_ptr,
        cute.make_layout((m, _NUM_EXPERTS, 1), stride=(_NUM_EXPERTS, 1, 1)),
    )
    sfa = cute.make_tensor(
        sfa_ptr,
        cute.make_layout((m, _HIDDEN_BLOCKS, 1), stride=(_HIDDEN_BLOCKS, 1, 1)),
    )
    sfb = cute.make_tensor(
        sfb_ptr,
        cute.make_layout((_EXPERT_BLOCKS, _HIDDEN_BLOCKS, 1), stride=(_HIDDEN_BLOCKS, 1, 1)),
    )
    _GEMM(a, b, c, sfa, sfb, _MAX_ACTIVE_CLUSTERS)


def _validate_configuration(m: int, n: int, k: int):
    if not BlockwiseGemmKernel.can_implement(
        cutlass.Float8E4M3FN,
        cutlass.Float32,
        cutlass.BFloat16,
        _USE_2CTA,
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


def _get_compiled_exact(m: int):
    compiled = _COMPILED_EXACT.get(m)
    if compiled is None:
        _validate_configuration(m, _NUM_EXPERTS, _HIDDEN_SIZE)
        _get_max_active_clusters()
        compiled = cute.compile(
            _host_blockwise_gemm_exact,
            make_ptr(cutlass.Float8E4M3FN, 16, cute.AddressSpace.gmem, assumed_align=_ASSUMED_ALIGN),
            make_ptr(cutlass.Float8E4M3FN, 16, cute.AddressSpace.gmem, assumed_align=_ASSUMED_ALIGN),
            make_ptr(cutlass.BFloat16, 16, cute.AddressSpace.gmem, assumed_align=_ASSUMED_ALIGN),
            make_ptr(cutlass.Float32, 16, cute.AddressSpace.gmem, assumed_align=_ASSUMED_ALIGN),
            make_ptr(cutlass.Float32, 16, cute.AddressSpace.gmem, assumed_align=_ASSUMED_ALIGN),
            m,
            options=_get_opt_options(),
        )
        _COMPILED_EXACT[m] = compiled
    return compiled


def _get_compiled_dynamic():
    global _COMPILED_DYNAMIC
    if _COMPILED_DYNAMIC is None:
        _validate_configuration(_BLOCK_K, _NUM_EXPERTS, _HIDDEN_SIZE)
        _get_max_active_clusters()
        _COMPILED_DYNAMIC = cute.compile(
            _host_blockwise_gemm_dynamic,
            make_ptr(cutlass.Float8E4M3FN, 16, cute.AddressSpace.gmem, assumed_align=_ASSUMED_ALIGN),
            make_ptr(cutlass.Float8E4M3FN, 16, cute.AddressSpace.gmem, assumed_align=_ASSUMED_ALIGN),
            make_ptr(cutlass.BFloat16, 16, cute.AddressSpace.gmem, assumed_align=_ASSUMED_ALIGN),
            make_ptr(cutlass.Float32, 16, cute.AddressSpace.gmem, assumed_align=_ASSUMED_ALIGN),
            make_ptr(cutlass.Float32, 16, cute.AddressSpace.gmem, assumed_align=_ASSUMED_ALIGN),
            cutlass.Int32(1),
            options=_get_opt_options(),
        )
    return _COMPILED_DYNAMIC


@torch.no_grad()
def fp8_blockwise_mm_explicit_rawptr(
    qx: torch.Tensor,
    scale_x: torch.Tensor,
    qw: torch.Tensor,
    scale_w_t: torch.Tensor,
    out_dtype: torch.dtype = torch.bfloat16,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    if out_dtype != torch.bfloat16:
        raise TypeError(f"Only bfloat16 output is supported, got {out_dtype}")

    if not qx.is_cuda or not qw.is_cuda:
        result = reference_fp8_blockwise_mm_explicit(
            qx,
            scale_x,
            qw,
            scale_w_t,
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
        raise TypeError("fp8_blockwise_mm_explicit_rawptr expects FP8 e4m3fn tensors")
    expected_scale_x = (m, k // _BLOCK_K)
    expected_scale_w_t = (n // _BLOCK_N, k // _BLOCK_K)
    if tuple(scale_x.shape) != expected_scale_x:
        raise ValueError(f"scale_x shape must be {expected_scale_x}, got {tuple(scale_x.shape)}")
    if tuple(scale_w_t.shape) != expected_scale_w_t:
        raise ValueError(
            f"scale_w_t shape must be {expected_scale_w_t}, got {tuple(scale_w_t.shape)}"
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

    if m in _EXACT_M_SET:
        compiled = _get_compiled_exact(m)
        compiled(
            _fp8_ptr(qx),
            _fp8_ptr(qw),
            _bf16_ptr(output),
            _f32_ptr(scale_x),
            _f32_ptr(scale_w_t),
        )
    else:
        compiled = _get_compiled_dynamic()
        compiled(
            _fp8_ptr(qx),
            _fp8_ptr(qw),
            _bf16_ptr(output),
            _f32_ptr(scale_x),
            _f32_ptr(scale_w_t),
            cutlass.Int32(m),
        )
    return output
