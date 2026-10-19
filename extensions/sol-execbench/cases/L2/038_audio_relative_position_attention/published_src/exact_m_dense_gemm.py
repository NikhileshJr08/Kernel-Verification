import os

import torch
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import make_ptr

from dense_gemm_kernel import DenseGemmKernel

os.environ.setdefault("CUTE_DSL_ARCH", "sm_100a")

_K_DIM = 512
_N_DIM = 1536
_ASSUME_ALIGN = 16
_COMPILED_GEMMS = {}


def _get_opt_level() -> int:
    try:
        from cutlass import CUDA_VERSION

        if CUDA_VERSION.major < 13 or (
            CUDA_VERSION.major == 13 and CUDA_VERSION.minor < 1
        ):
            return 3
        return 2
    except Exception:
        return 3


_OPT_LEVEL = _get_opt_level()


def _cutlass_dtype(dtype: torch.dtype):
    if dtype == torch.bfloat16:
        return cutlass.BFloat16
    if dtype == torch.float32:
        return cutlass.Float32
    raise TypeError(f"Unsupported output dtype for exact-M CuTe GEMM: {dtype}")


def _select_config(m: int):
    if m >= 100000:
        return True, (256, 256), (2, 1)
    if m >= 50000:
        return True, (256, 128), (2, 1)
    return False, (128, 256), (1, 1)


def _make_host_gemm(
    m: int,
    out_torch_dtype: torch.dtype,
    use_2cta_instrs: bool,
    mma_tiler_mn,
    cluster_shape_mn,
):
    gemm = DenseGemmKernel(
        acc_dtype=cutlass.Float32,
        use_2cta_instrs=use_2cta_instrs,
        mma_tiler_mn=mma_tiler_mn,
        cluster_shape_mn=cluster_shape_mn,
        use_tma_store=True,
    )

    @cute.jit
    def _host_gemm(
        a_ptr: cute.Pointer,
        b_ptr: cute.Pointer,
        c_ptr: cute.Pointer,
    ):
        a = cute.make_tensor(
            a_ptr,
            cute.make_layout((m, _K_DIM, 1), stride=(_K_DIM, 1, 1)),
        )
        b = cute.make_tensor(
            b_ptr,
            cute.make_layout((_N_DIM, _K_DIM, 1), stride=(_K_DIM, 1, 1)),
        )
        c = cute.make_tensor(
            c_ptr,
            cute.make_layout((m, _N_DIM, 1), stride=(_N_DIM, 1, 1)),
        )
        gemm(a, b, c)

    return _host_gemm


def _get_compiled_gemm(device: torch.device, m: int, out_dtype: torch.dtype):
    use_2cta_instrs, mma_tiler_mn, cluster_shape_mn = _select_config(m)
    key = (
        str(device),
        m,
        out_dtype,
        use_2cta_instrs,
        mma_tiler_mn,
        cluster_shape_mn,
    )
    compiled = _COMPILED_GEMMS.get(key)
    if compiled is None:
        out_cutlass_dtype = _cutlass_dtype(out_dtype)
        compiled = cute.compile(
            _make_host_gemm(
                m,
                out_dtype,
                use_2cta_instrs,
                mma_tiler_mn,
                cluster_shape_mn,
            ),
            make_ptr(cutlass.BFloat16, 16, cute.AddressSpace.gmem, assumed_align=_ASSUME_ALIGN),
            make_ptr(cutlass.BFloat16, 16, cute.AddressSpace.gmem, assumed_align=_ASSUME_ALIGN),
            make_ptr(out_cutlass_dtype, 16, cute.AddressSpace.gmem, assumed_align=_ASSUME_ALIGN),
            options=f"--opt-level {_OPT_LEVEL}",
        )
        _COMPILED_GEMMS[key] = compiled
    return compiled


@torch.no_grad()
def gemm_exact_m_(a: torch.Tensor, b: torch.Tensor, out: torch.Tensor) -> None:
    if a.ndim != 2 or b.ndim != 2 or out.ndim != 2:
        raise ValueError(
            f"gemm_exact_m_ expects rank-2 tensors, got a={a.ndim}D b={b.ndim}D out={out.ndim}D"
        )
    if a.shape[1] != _K_DIM or b.shape != (_N_DIM, _K_DIM):
        raise ValueError(
            f"Expected A=[M,{_K_DIM}] and B=[{_N_DIM},{_K_DIM}], got A={tuple(a.shape)} B={tuple(b.shape)}"
        )
    expected = (a.shape[0], _N_DIM)
    if tuple(out.shape) != expected:
        raise ValueError(f"Output shape {tuple(out.shape)} does not match expected {expected}")
    if not a.is_contiguous() or not b.is_contiguous() or not out.is_contiguous():
        raise ValueError("gemm_exact_m_ expects contiguous tensors")
    if a.dtype != torch.bfloat16 or b.dtype != torch.bfloat16:
        raise TypeError(f"gemm_exact_m_ expects BF16 A/B tensors, got {a.dtype} and {b.dtype}")
    if out.dtype not in (torch.bfloat16, torch.float32):
        raise TypeError(f"gemm_exact_m_ expects BF16 or FP32 output, got {out.dtype}")

    compiled = _get_compiled_gemm(a.device, a.shape[0], out.dtype)
    out_cutlass_dtype = _cutlass_dtype(out.dtype)
    compiled(
        make_ptr(cutlass.BFloat16, a.data_ptr(), cute.AddressSpace.gmem, assumed_align=_ASSUME_ALIGN),
        make_ptr(cutlass.BFloat16, b.data_ptr(), cute.AddressSpace.gmem, assumed_align=_ASSUME_ALIGN),
        make_ptr(out_cutlass_dtype, out.data_ptr(), cute.AddressSpace.gmem, assumed_align=_ASSUME_ALIGN),
    )
