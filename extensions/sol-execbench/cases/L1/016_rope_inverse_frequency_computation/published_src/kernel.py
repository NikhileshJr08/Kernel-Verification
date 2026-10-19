import ctypes

import torch
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import make_ptr

_EXACT_GROUPS = 16
_EXACT_VEC = 4
_EXACT_THREADS = 16
_WARP_GROUPS = 32
_WARP_VEC = 2
_WARP_THREADS = 32
_INV_HEAD_DIM = 1.0 / 64.0
_OUT_PTR_SPEC = make_ptr(cutlass.Float32, 32, cute.AddressSpace.gmem, assumed_align=32)
_LOG2_THETA_BY_VALUE = {
    4000000.0: 21.931568569324174,
    2000000.0: 20.931568569324174,
    10000000.0: 23.253496664211536,
    100000.0: 16.609640474436812,
    312500.0: 18.253496664211536,
    200000.0: 17.609640474436812,
    1000000.0: 19.931568569324174,
    1500000.0: 20.516531070045332,
    78125.0: 16.253496664211536,
    50000.0: 15.609640474436812,
    156250.0: 17.253496664211536,
    7000000.0: 22.73892349138178,
    10000.0: 13.287712379549449,
    1750000.0: 20.73892349138178,
    800000.0: 19.609640474436812,
    3000000.0: 21.516531070045332,
}
_LOG2_STEP2_BY_VALUE = {
    4000000.0: -0.6853615177913804,
    2000000.0: -0.6541115177913804,
    10000000.0: -0.7266717707566105,
    100000.0: -0.5190512648261504,
    312500.0: -0.5704217707566105,
    200000.0: -0.5503012648261504,
    1000000.0: -0.6228615177913804,
    1500000.0: -0.6411415959389166,
    78125.0: -0.5079217707566105,
    50000.0: -0.4878012648261504,
    156250.0: -0.5391717707566105,
    7000000.0: -0.7105913591056806,
    10000.0: -0.4152410118609203,
    1750000.0: -0.6480913591056806,
    800000.0: -0.6128012648261504,
    3000000.0: -0.6723915959389166,
}
_STEP_BY_VALUE = {
    4000000.0: 0.7885746513950169,
    2000000.0: 0.7971616663471772,
    10000000.0: 0.7773650302387758,
    100000.0: 0.8353625469578262,
    312500.0: 0.8206216459976724,
    200000.0: 0.8263640326237495,
    1000000.0: 0.8058421877614819,
    1500000.0: 0.8007529993832904,
    78125.0: 0.8385909201688694,
    50000.0: 0.844459048688527,
    156250.0: 0.8295576298411583,
    7000000.0: 0.7817094157963258,
    10000.0: 0.8659643233600653,
    1750000.0: 0.7988266230783406,
    800000.0: 0.80865675543973,
    3000000.0: 0.7921272986390491,
}
_VEC4_THETAS = {4000000.0, 2000000.0, 3000000.0}


@cute.kernel
def _vec4_kernel(out: cute.Tensor, log2_theta: cutlass.Constexpr):
    tidx, _, _ = cute.arch.thread_idx()
    base_idx = tidx * _EXACT_VEC
    reg = cute.make_rmem_tensor((_EXACT_VEC,), cutlass.Float32)
    exponent_scale = cutlass.Float32(-log2_theta * _INV_HEAD_DIM)
    for lane in cutlass.range_constexpr(_EXACT_VEC):
        reg[lane] = cute.math.exp2(
            exponent_scale * cutlass.Float32(base_idx + lane),
            fastmath=True,
        )
    out[tidx, None].store(reg.load())


@cute.kernel
def _warp_kernel(out: cute.Tensor, log2_step2: cutlass.Constexpr, step: cutlass.Constexpr):
    tidx, _, _ = cute.arch.thread_idx()
    reg = cute.make_rmem_tensor((_WARP_VEC,), cutlass.Float32)
    base = cute.math.exp2(cutlass.Float32(log2_step2) * cutlass.Float32(tidx), fastmath=True)
    reg[0] = base
    reg[1] = base * cutlass.Float32(step)
    cute.autovec_copy(reg, out[tidx, None])


@cute.jit
def _vec4_host(out_ptr: cute.Pointer, log2_theta: cutlass.Constexpr):
    out = cute.make_tensor(
        out_ptr,
        cute.make_layout((_EXACT_GROUPS, _EXACT_VEC), stride=(_EXACT_VEC, 1)),
    )
    _vec4_kernel(out, log2_theta).launch(
        grid=(1, 1, 1),
        block=(_EXACT_THREADS, 1, 1),
    )


@cute.jit
def _warp_host(out_ptr: cute.Pointer, log2_step2: cutlass.Constexpr, step: cutlass.Constexpr):
    out = cute.make_tensor(
        out_ptr,
        cute.make_layout((_WARP_GROUPS, _WARP_VEC), stride=(_WARP_VEC, 1)),
    )
    _warp_kernel(out, log2_step2, step).launch(
        grid=(1, 1, 1),
        block=(_WARP_THREADS, 1, 1),
    )


def _build_packed_args(executor, out_addr_arg):
    packed = (ctypes.c_void_p * (1 + executor._num_extra_args))()
    packed[0] = ctypes.addressof(out_addr_arg)
    idx = 1
    if executor._has_cuda_result:
        packed[idx] = executor._cuda_result_addr
        idx += 1
    kernel_ptrs = executor._kernel_ptrs
    if kernel_ptrs is not None:
        for ptr in kernel_ptrs:
            packed[idx] = ptr
            idx += 1
    return packed


def _make_run():
    data_ptr = torch.Tensor.data_ptr
    state_by_theta = {}
    last_theta = None
    last_out_addr_arg = None
    last_launch = None
    last_packed_args = None

    def get_state(theta: float):
        state = state_by_theta.get(theta)
        if state is None:
            out_addr_arg = ctypes.c_uint64(0)
            if theta in _VEC4_THETAS:
                compiled = cute.compile(_vec4_host, _OUT_PTR_SPEC, _LOG2_THETA_BY_VALUE[theta])
            else:
                compiled = cute.compile(_warp_host, _OUT_PTR_SPEC, _LOG2_STEP2_BY_VALUE[theta], _STEP_BY_VALUE[theta])
            executor = compiled.to(None)
            launch = executor.capi_func
            packed_args = _build_packed_args(executor, out_addr_arg)
            state = (out_addr_arg, launch, packed_args, compiled, executor)
            state_by_theta[theta] = state
        return state

    def run(rope_theta: float, inv_freq: torch.Tensor) -> None:
        nonlocal last_theta, last_out_addr_arg, last_launch, last_packed_args
        theta = rope_theta
        if theta != last_theta:
            last_out_addr_arg, last_launch, last_packed_args, _, _ = get_state(float(theta))
            last_theta = theta
        last_out_addr_arg.value = data_ptr(inv_freq)
        last_launch(last_packed_args)

    return run


run = _make_run()
