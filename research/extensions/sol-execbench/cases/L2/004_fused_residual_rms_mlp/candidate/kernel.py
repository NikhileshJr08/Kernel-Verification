import gc
import weakref

import torch
import torch.nn.functional as F

import cutlass
import cutlass.cute as cute
import cutlass.cute.math as cute_math
from cutlass.cute.runtime import from_dlpack

from dynamic_persistent_gemm_impl import DynamicPersistentDenseGemmKernel


_GEMM_INTERMEDIATE_DEFAULT = DynamicPersistentDenseGemmKernel(
    acc_dtype=cutlass.Float32,
    use_2cta_instrs=False,
    mma_tiler_mn=(128, 256),
    cluster_shape_mn=(1, 1),
    use_tma_store=True,
)

_GEMM_OUTPUT_DEFAULT = DynamicPersistentDenseGemmKernel(
    acc_dtype=cutlass.Float32,
    use_2cta_instrs=False,
    mma_tiler_mn=(128, 256),
    cluster_shape_mn=(1, 1),
    use_tma_store=True,
)



_GEMM_INTERMEDIATE_TOKEN256 = DynamicPersistentDenseGemmKernel(
    acc_dtype=cutlass.Float32,
    use_2cta_instrs=True,
    mma_tiler_mn=(256, 128),
    cluster_shape_mn=(2, 1),
    use_tma_store=True,
)

_GEMM_OUTPUT_TOKEN256 = DynamicPersistentDenseGemmKernel(
    acc_dtype=cutlass.Float32,
    use_2cta_instrs=True,
    mma_tiler_mn=(256, 128),
    cluster_shape_mn=(2, 1),
    use_tma_store=True,
)

_GEMM_INTERMEDIATE_TOKEN8192 = DynamicPersistentDenseGemmKernel(
    acc_dtype=cutlass.Float32,
    use_2cta_instrs=False,
    mma_tiler_mn=(128, 256),
    cluster_shape_mn=(1, 2),
    use_tma_store=True,
)

_GEMM_OUTPUT_TOKEN8192 = DynamicPersistentDenseGemmKernel(
    acc_dtype=cutlass.Float32,
    use_2cta_instrs=False,
    mma_tiler_mn=(128, 256),
    cluster_shape_mn=(1, 2),
    use_tma_store=True,
    num_c_stage_override=1,
)

_COMPILED_GEMMS = {}
_TENSOR_CACHE = {}
_WORKSPACE_STATE = None
_DYNAMIC_M_STRIDE_ORDER = (2, 0, 1)
_RUNTIME_CACHE_KEY = None
_COMPILE_CACHE_KEY = None
_COMPILED_FUSED_PREPASS_W4 = None
_COMPILED_FUSED_PREPASS_W4_PREWARMED = False


def _silu_epilogue(x):
    one = cute.full_like(x, 1.0)
    return (x / (one + cute_math.exp(-x, fastmath=True))).to(cutlass.BFloat16)


def _ensure_row_major_2d(t: torch.Tensor) -> torch.Tensor:
    if t.dim() != 2:
        raise ValueError(f"expected a 2D tensor, got shape {tuple(t.shape)}")
    if t.is_contiguous() and t.stride(-1) == 1:
        return t
    return t.contiguous()


def _batched_view_l1(t: torch.Tensor) -> torch.Tensor:
    t = _ensure_row_major_2d(t)
    m, n = t.shape
    batch_stride = t.stride(0) * m
    return t.as_strided((m, n, 1), (t.stride(0), t.stride(1), batch_stride))


def _tensor_meta(t: torch.Tensor):
    return (
        tuple(t.shape),
        tuple(t.stride()),
        t.storage_offset(),
        t.dtype,
        t.device,
    )


def _install_tensor_cache_entry(key, tensor: torch.Tensor, meta, tensor_3d: torch.Tensor, cute_tensor):
    def _cleanup(_ref, _key=key):
        _TENSOR_CACHE.pop(_key, None)

    _TENSOR_CACHE[key] = (weakref.ref(tensor, _cleanup), meta, tensor_3d, cute_tensor)


def _to_cute_batched(t: torch.Tensor, dynamic_m: bool):
    t2d = _ensure_row_major_2d(t)
    if t2d is not t:
        cute_tensor = from_dlpack(
            _batched_view_l1(t2d),
            assumed_align=16,
            use_32bit_stride=True,
        )
        if dynamic_m:
            cute_tensor = cute_tensor.mark_compact_shape_dynamic(
                mode=0,
                stride_order=_DYNAMIC_M_STRIDE_ORDER,
            )
        return cute_tensor

    key = (id(t), dynamic_m)
    meta = (_tensor_meta(t), dynamic_m)
    cached = _TENSOR_CACHE.get(key)
    if cached is not None:
        ref, old_meta, _tensor_3d, cute_tensor = cached
        if ref() is t and old_meta == meta:
            return cute_tensor

    tensor_3d = _batched_view_l1(t)
    cute_tensor = from_dlpack(
        tensor_3d,
        assumed_align=16,
        use_32bit_stride=True,
    )
    if dynamic_m:
        cute_tensor = cute_tensor.mark_compact_shape_dynamic(
            mode=0,
            stride_order=_DYNAMIC_M_STRIDE_ORDER,
        )
    _install_tensor_cache_entry(key, t, meta, tensor_3d, cute_tensor)
    return cute_tensor


def _to_cute_batched_static(t: torch.Tensor):
    t2d = _ensure_row_major_2d(t)
    if t2d is not t:
        return from_dlpack(
            _batched_view_l1(t2d),
            assumed_align=16,
            use_32bit_stride=True,
        )

    key = (id(t), "static")
    meta = (_tensor_meta(t), "static")
    cached = _TENSOR_CACHE.get(key)
    if cached is not None:
        ref, old_meta, _tensor_3d, cute_tensor = cached
        if ref() is t and old_meta == meta:
            return cute_tensor

    tensor_3d = _batched_view_l1(t)
    cute_tensor = from_dlpack(
        tensor_3d,
        assumed_align=16,
        use_32bit_stride=True,
    )
    _install_tensor_cache_entry(key, t, meta, tensor_3d, cute_tensor)
    return cute_tensor


def _get_compiled_gemm(
    key: str,
    gemm: DynamicPersistentDenseGemmKernel,
    a,
    b,
    c,
    epilogue_op=None,
):
    compiled = _COMPILED_GEMMS.get(key)
    if compiled is None:
        compile_args = [gemm, a, b, c]
        if epilogue_op is not None:
            compile_args.append(epilogue_op)
        compiled = cute.compile(*compile_args)
        _COMPILED_GEMMS[key] = compiled
    return compiled


def _run_gemm(
    key: str,
    gemm: DynamicPersistentDenseGemmKernel,
    a_2d: torch.Tensor,
    b_2d: torch.Tensor,
    c_2d: torch.Tensor,
    epilogue_op=None,
    static_m: bool = False,
) -> None:
    if static_m:
        a = _to_cute_batched_static(a_2d)
        c = _to_cute_batched_static(c_2d)
    else:
        a = _to_cute_batched(a_2d, True)
        c = _to_cute_batched(c_2d, True)
    b = _to_cute_batched(b_2d, False)
    compiled = _get_compiled_gemm(key, gemm, a, b, c, epilogue_op)
    compiled(a, b, c)


def _get_workspace(
    token_count: int,
    intermediate_size: int,
    dtype: torch.dtype,
    device: torch.device,
):
    global _WORKSPACE_STATE
    meta_key = (intermediate_size, dtype, device)
    state = _WORKSPACE_STATE
    if (
        state is None
        or state["meta_key"] != meta_key
        or state["capacity_token_count"] < token_count
    ):
        gate_output_full = torch.empty(
            (token_count, intermediate_size),
            dtype=dtype,
            device=device,
        )
        up_output_full = torch.empty_like(gate_output_full)
        _WORKSPACE_STATE = {
            "meta_key": meta_key,
            "capacity_token_count": token_count,
            "view_token_count": token_count,
            "gate_output_full": gate_output_full,
            "up_output_full": up_output_full,
            "gate_output": gate_output_full,
            "up_output": up_output_full,
        }
        return _WORKSPACE_STATE

    if state["view_token_count"] != token_count:
        state["view_token_count"] = token_count
        state["gate_output"] = state["gate_output_full"].narrow(0, 0, token_count)
        state["up_output"] = state["up_output_full"].narrow(0, 0, token_count)
    return state


def _make_compile_cache_key(
    hidden_states: torch.Tensor,
    gate_proj_weight: torch.Tensor,
    up_proj_weight: torch.Tensor,
    down_proj_weight: torch.Tensor,
):
    return (
        hidden_states.dtype,
        hidden_states.device,
        tuple(gate_proj_weight.shape),
        gate_proj_weight.dtype,
        gate_proj_weight.device,
        tuple(up_proj_weight.shape),
        up_proj_weight.dtype,
        up_proj_weight.device,
        tuple(down_proj_weight.shape),
        down_proj_weight.dtype,
        down_proj_weight.device,
    )


def _maybe_reset_runtime_caches(runtime_key, compile_key) -> None:
    global _COMPILE_CACHE_KEY, _RUNTIME_CACHE_KEY, _WORKSPACE_STATE

    if _COMPILE_CACHE_KEY != compile_key:
        _COMPILED_GEMMS.clear()
        _WORKSPACE_STATE = None
        _COMPILE_CACHE_KEY = compile_key

    if _RUNTIME_CACHE_KEY == runtime_key:
        return

    had_ephemeral_state = bool(_TENSOR_CACHE)
    _TENSOR_CACHE.clear()
    _RUNTIME_CACHE_KEY = runtime_key
    if had_ephemeral_state:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


_FRMSNORM_TOKEN_THRESHOLD = 4096
_GATE_SILU_EPILOGUE_TOKEN_COUNT = 8192


def _should_use_compiled_fused_prepass_w4(
    batch_size: int,
    seq_len: int,
    token_count: int,
) -> bool:
    return token_count == 8192 and batch_size == 64 and seq_len == 128


def _compiled_fused_prepass_3d(
    hidden_states: torch.Tensor,
    residual: torch.Tensor,
    norm_weight: torch.Tensor,
    eps,
):
    hidden_states = hidden_states + residual
    return F.rms_norm(hidden_states, (hidden_states.shape[-1],), norm_weight, eps)


def _get_compiled_fused_prepass_w4():
    global _COMPILED_FUSED_PREPASS_W4
    if _COMPILED_FUSED_PREPASS_W4 is None:
        _COMPILED_FUSED_PREPASS_W4 = torch.compile(
            _compiled_fused_prepass_3d,
            dynamic=False,
            mode="reduce-overhead",
        )
    return _COMPILED_FUSED_PREPASS_W4


def _prewarm_compiled_fused_prepass_w4() -> None:
    global _COMPILED_FUSED_PREPASS_W4_PREWARMED
    if _COMPILED_FUSED_PREPASS_W4_PREWARMED:
        return
    if not torch.cuda.is_available():
        return

    device = torch.device("cuda")
    hidden_size = 16384
    with torch.no_grad():
        hidden_states = torch.zeros((64, 128, hidden_size), dtype=torch.bfloat16, device=device)
        residual = torch.zeros_like(hidden_states)
        norm_weight = torch.ones((hidden_size,), dtype=torch.bfloat16, device=device)
        _get_compiled_fused_prepass_w4()(hidden_states, residual, norm_weight, 1.0e-5)
        torch.cuda.synchronize()
    del hidden_states, residual, norm_weight
    _COMPILED_FUSED_PREPASS_W4_PREWARMED = True


def _should_use_gate_silu_epilogue(batch_size: int, seq_len: int, token_count: int) -> bool:
    return token_count == _GATE_SILU_EPILOGUE_TOKEN_COUNT or (
        token_count == 4096 and batch_size == 1 and seq_len == 4096
    )


def _rmsnorm(hidden_states: torch.Tensor, norm_weight: torch.Tensor, eps, token_count: int) -> torch.Tensor:
    if token_count <= _FRMSNORM_TOKEN_THRESHOLD:
        return F.rms_norm(hidden_states, (hidden_states.shape[-1],), norm_weight, eps)

    input_dtype = hidden_states.dtype
    hidden_states_f32 = hidden_states.to(torch.float32)
    variance = hidden_states_f32.square().mean(dim=-1, keepdim=True)
    return (
        hidden_states_f32 * torch.rsqrt(variance + eps) * norm_weight.to(torch.float32)
    ).to(input_dtype)


@torch.no_grad()
def run(
    hidden_states: torch.Tensor,
    residual: torch.Tensor,
    norm_weight: torch.Tensor,
    gate_proj_weight: torch.Tensor,
    up_proj_weight: torch.Tensor,
    down_proj_weight: torch.Tensor,
    eps,
    output: torch.Tensor,
) -> None:
    batch_size, seq_len, hidden_size = hidden_states.shape
    token_count = hidden_states.numel() // hidden_size
    compile_key = _make_compile_cache_key(
        hidden_states,
        gate_proj_weight,
        up_proj_weight,
        down_proj_weight,
    )
    _maybe_reset_runtime_caches(
        (token_count, compile_key),
        compile_key,
    )
    if _should_use_compiled_fused_prepass_w4(batch_size, seq_len, token_count):
        hidden_states_norm = _get_compiled_fused_prepass_w4()(
            hidden_states,
            residual,
            norm_weight,
            eps,
        )
    else:
        hidden_states = hidden_states + residual
        hidden_states_norm = _rmsnorm(hidden_states, norm_weight, eps, token_count)
    input_dtype = hidden_states_norm.dtype

    intermediate_size = gate_proj_weight.shape[0]
    use_token256 = token_count == 256
    use_token8192 = token_count == 8192
    intermediate_gemm = (
        _GEMM_INTERMEDIATE_TOKEN256
        if use_token256
        else _GEMM_INTERMEDIATE_TOKEN8192 if use_token8192 else _GEMM_INTERMEDIATE_DEFAULT
    )
    output_gemm = (
        _GEMM_OUTPUT_TOKEN256
        if use_token256
        else _GEMM_OUTPUT_TOKEN8192 if use_token8192 else _GEMM_OUTPUT_DEFAULT
    )
    intermediate_key = (
        "intermediate_token256"
        if use_token256
        else "intermediate_token8192" if use_token8192 else "intermediate_default"
    )
    output_key = (
        "output_token256"
        if use_token256
        else "output_token8192" if use_token8192 else "output_default"
    )
    use_gate_silu_epilogue = _should_use_gate_silu_epilogue(batch_size, seq_len, token_count)
    gate_key = (
        f"{intermediate_key}_gate_silu"
        if use_gate_silu_epilogue
        else intermediate_key
    )
    use_static_token8192 = use_token8192

    hidden_states_2d = hidden_states_norm.reshape(token_count, hidden_size)
    workspace = _get_workspace(
        token_count,
        intermediate_size,
        input_dtype,
        hidden_states.device,
    )
    gate_output = workspace["gate_output"]
    up_output = workspace["up_output"]

    _run_gemm(
        gate_key,
        intermediate_gemm,
        hidden_states_2d,
        gate_proj_weight,
        gate_output,
        _silu_epilogue if use_gate_silu_epilogue else None,
        static_m=use_static_token8192,
    )
    _run_gemm(
        intermediate_key,
        intermediate_gemm,
        hidden_states_2d,
        up_proj_weight,
        up_output,
        static_m=use_static_token8192,
    )

    if not use_gate_silu_epilogue:
        F.silu(gate_output, inplace=True)
    gate_output.mul_(up_output)

    output_2d = output.reshape(token_count, hidden_size)

    _run_gemm(
        output_key,
        output_gemm,
        gate_output,
        down_proj_weight,
        output_2d,
        static_m=use_static_token8192,
    )
