import os

os.environ.setdefault("CUTE_DSL_ARCH", "sm_100a")

import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

from dense_gemm_helper import DenseGemmKernel

_GEMM_IMPLS = {
    "default": DenseGemmKernel(
        acc_dtype=cutlass.Float32,
        use_2cta_instrs=False,
        mma_tiler_mn=(128, 64),
        cluster_shape_mn=(1, 1),
        use_tma_store=True,
    ),
    "small": DenseGemmKernel(
        acc_dtype=cutlass.Float32,
        use_2cta_instrs=False,
        mma_tiler_mn=(64, 32),
        cluster_shape_mn=(1, 1),
        use_tma_store=True,
    ),
    "qkv_large": DenseGemmKernel(
        acc_dtype=cutlass.Float32,
        use_2cta_instrs=True,
        mma_tiler_mn=(256, 128),
        cluster_shape_mn=(2, 1),
        use_tma_store=True,
    ),
}


def _to_batched_2d(tensor):
    if tensor.ndim == 2:
        return tensor.unsqueeze(2)
    return tensor


def _as_row_major_cute(tensor):
    return from_dlpack(
        tensor,
        assumed_align=32,
        use_32bit_stride=True,
    ).mark_layout_dynamic(leading_dim=1)


def compile_gemm(a, b, c, *, variant="default"):
    a_batched = _to_batched_2d(a)
    b_batched = _to_batched_2d(b)
    c_batched = _to_batched_2d(c)
    gemm = _GEMM_IMPLS[variant]
    a_cute = _as_row_major_cute(a_batched)
    b_cute = _as_row_major_cute(b_batched)
    c_cute = _as_row_major_cute(c_batched)
    if not gemm.can_implement(a_cute, b_cute, c_cute):
        raise RuntimeError(
            f"DenseGemmKernel[{variant}] cannot implement shape a={tuple(a_batched.shape)} b={tuple(b_batched.shape)} c={tuple(c_batched.shape)}"
        )
    return cute.compile(gemm, a_cute, b_cute, c_cute)


def launch_gemm(compiled_gemm, a, b, c):
    a_batched = _to_batched_2d(a)
    b_batched = _to_batched_2d(b)
    c_batched = _to_batched_2d(c)
    compiled_gemm(
        _as_row_major_cute(a_batched),
        _as_row_major_cute(b_batched),
        _as_row_major_cute(c_batched),
    )
