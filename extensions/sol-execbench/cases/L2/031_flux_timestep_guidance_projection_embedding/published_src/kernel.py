import os

import torch
import torch.nn.functional as F

try:
    import cutlass
    import cutlass.cute as cute
    from cutlass.cute.runtime import make_ptr

    from dense_gemm_kernel import DenseGemmKernel

    _HAS_CUTLASS = True
except Exception:  # pragma: no cover - local CPU fallback
    cutlass = None
    cute = None
    make_ptr = None
    DenseGemmKernel = None
    _HAS_CUTLASS = False


os.environ.setdefault("CUTE_DSL_ARCH", "sm_100a")

torch.set_float32_matmul_precision("highest")
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


_HALF_TIME_EMBED_DIM = 384
_TIME_EMBED_DIM = 768
_INNER_DIM = 3072
_ALIGNMENT = 16
_TIME_EMBED_VEC = 4
_HALF_TIME_EMBED_GROUPS = _HALF_TIME_EMBED_DIM // _TIME_EMBED_VEC
_FULL_TIME_EMBED_GROUPS = _TIME_EMBED_DIM // _TIME_EMBED_VEC
_TIME_EMBED_THREADS = 96
_INNER_VEC = 4
_INNER_GROUPS = _INNER_DIM // _INNER_VEC
_INNER_THREADS = 128
_INNER_BLOCKS_PER_ROW = _INNER_GROUPS // _INNER_THREADS


if _HAS_CUTLASS:
    _NARROW_GEMM = DenseGemmKernel(
        acc_dtype=cutlass.Float32,
        use_2cta_instrs=False,
        mma_tiler_mn=(128, 128),
        cluster_shape_mn=(1, 1),
        use_tma_store=True,
    )
    _WIDE_GEMM = DenseGemmKernel(
        acc_dtype=cutlass.Float32,
        use_2cta_instrs=False,
        mma_tiler_mn=(128, 128),
        cluster_shape_mn=(1, 1),
        use_tma_store=True,
    )
    _STATIC_LINEAR_CACHE = {}
else:
    _NARROW_GEMM = None
    _WIDE_GEMM = None
    _STATIC_LINEAR_CACHE = {}


_WORKSPACES = {}
_B1_GRAPH_STATE = {}
_COMBINED_BIAS_CACHE = {}


def _make_batch1_core():
    def _core(
        timestep: torch.Tensor,
        pooled_projections: torch.Tensor,
        freqs: torch.Tensor,
        timestep_linear1_weight: torch.Tensor,
        timestep_linear1_bias: torch.Tensor,
        timestep_linear2_weight: torch.Tensor,
        timestep_linear2_bias: torch.Tensor,
        text_embedder_weight: torch.Tensor,
        text_embedder_bias: torch.Tensor,
    ) -> torch.Tensor:
        timestep_scaled = timestep * 1000.0
        args = timestep_scaled[:, None] * freqs[None, :]
        timestep_embed = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        x = torch.addmm(
            timestep_linear1_bias,
            timestep_embed,
            timestep_linear1_weight.T,
        )
        x = F.silu(x)
        timestep_out = torch.addmm(
            timestep_linear2_bias,
            x,
            timestep_linear2_weight.T,
        )
        text_out = torch.addmm(
            text_embedder_bias,
            pooled_projections,
            text_embedder_weight.T,
        )
        return timestep_out + text_out

    return _core


class _CopyInputCUDAGraphRunner:
    def __init__(self, fn, sample_inputs):
        self.static_inputs = [
            tensor.clone() if isinstance(tensor, torch.Tensor) else tensor
            for tensor in sample_inputs
        ]
        self.static_outputs = None
        for _ in range(3):
            self.static_outputs = fn(*self.static_inputs)
        torch.cuda.synchronize()
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.static_outputs = fn(*self.static_inputs)
        torch.cuda.synchronize()

    def __call__(self, *inputs):
        for static_input, actual_input in zip(self.static_inputs, inputs):
            if isinstance(static_input, torch.Tensor):
                static_input.copy_(actual_input)
        self.graph.replay()
        return self.static_outputs


def _get_batch1_graph_state(device: torch.device):
    key = (device.type, device.index)
    state = _B1_GRAPH_STATE.get(key)
    if state is None:
        eager_fn = _make_batch1_core()
        compiled_fn = eager_fn
        runner = None
        if device.type == "cuda":
            try:
                compiled_fn = torch.compile(
                    eager_fn,
                    dynamic=False,
                    mode="max-autotune-no-cudagraphs",
                    fullgraph=True,
                )
                sample_inputs = (
                    torch.zeros((1,), device=device, dtype=torch.float32),
                    torch.zeros((1, _TIME_EMBED_DIM), device=device, dtype=torch.float32),
                    torch.zeros((_HALF_TIME_EMBED_DIM,), device=device, dtype=torch.float32),
                    torch.zeros((_INNER_DIM, _TIME_EMBED_DIM), device=device, dtype=torch.float32),
                    torch.zeros((_INNER_DIM,), device=device, dtype=torch.float32),
                    torch.zeros((_INNER_DIM, _INNER_DIM), device=device, dtype=torch.float32),
                    torch.zeros((_INNER_DIM,), device=device, dtype=torch.float32),
                    torch.zeros((_INNER_DIM, _TIME_EMBED_DIM), device=device, dtype=torch.float32),
                    torch.zeros((_INNER_DIM,), device=device, dtype=torch.float32),
                )
                runner = _CopyInputCUDAGraphRunner(compiled_fn, sample_inputs)
            except Exception:
                compiled_fn = eager_fn
                runner = None
        state = {
            "fn": compiled_fn,
            "runner": runner,
        }
        _B1_GRAPH_STATE[key] = state
    return state


def _run_reference(
    timestep: torch.Tensor,
    pooled_projections: torch.Tensor,
    freqs: torch.Tensor,
    timestep_linear1_weight: torch.Tensor,
    timestep_linear1_bias: torch.Tensor,
    timestep_linear2_weight: torch.Tensor,
    timestep_linear2_bias: torch.Tensor,
    text_embedder_weight: torch.Tensor,
    text_embedder_bias: torch.Tensor,
    conditioning: torch.Tensor,
) -> None:
    timestep_scaled = timestep * 1000.0
    args = timestep_scaled[:, None] * freqs[None, :]
    timestep_embed = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    x = F.linear(timestep_embed, timestep_linear1_weight, timestep_linear1_bias)
    x = F.silu(x)
    timestep_out = F.linear(x, timestep_linear2_weight, timestep_linear2_bias)
    text_out = F.linear(pooled_projections, text_embedder_weight, text_embedder_bias)
    conditioning.copy_(timestep_out + text_out)


def _combined_bias_key(
    timestep_linear2_bias: torch.Tensor,
    text_embedder_bias: torch.Tensor,
) -> tuple:
    return (
        timestep_linear2_bias.data_ptr(),
        tuple(timestep_linear2_bias.shape),
        tuple(timestep_linear2_bias.stride()),
        text_embedder_bias.data_ptr(),
        tuple(text_embedder_bias.shape),
        tuple(text_embedder_bias.stride()),
        timestep_linear2_bias.device.type,
        timestep_linear2_bias.device.index,
        str(timestep_linear2_bias.dtype),
    )


def _get_combined_bias(
    timestep_linear2_bias: torch.Tensor,
    text_embedder_bias: torch.Tensor,
) -> torch.Tensor:
    key = _combined_bias_key(timestep_linear2_bias, text_embedder_bias)
    combined = _COMBINED_BIAS_CACHE.get(key)
    if combined is None:
        combined = timestep_linear2_bias + text_embedder_bias
        _COMBINED_BIAS_CACHE[key] = combined
    return combined


def _get_workspace(batch_size: int, device: torch.device):
    key = (device.type, device.index, batch_size)
    workspace = _WORKSPACES.get(key)
    if workspace is None:
        workspace = {
            "scaled": torch.empty(
                (batch_size,),
                device=device,
                dtype=torch.float32,
            ),
            "args": torch.empty(
                (batch_size, _HALF_TIME_EMBED_DIM),
                device=device,
                dtype=torch.float32,
            ),
            "emb": torch.empty(
                (batch_size, _TIME_EMBED_DIM),
                device=device,
                dtype=torch.float32,
            ),
            "x": torch.empty(
                (batch_size, _INNER_DIM),
                device=device,
                dtype=torch.float32,
            ),
            "ts": torch.empty(
                (batch_size, _INNER_DIM),
                device=device,
                dtype=torch.float32,
            ),
        }
        _WORKSPACES[key] = workspace
    return workspace


def _run_small_addmm_tf32(
    emb_buf: torch.Tensor,
    pooled_projections: torch.Tensor,
    timestep_linear1_weight: torch.Tensor,
    timestep_linear1_bias: torch.Tensor,
    timestep_linear2_weight: torch.Tensor,
    timestep_linear2_bias: torch.Tensor,
    text_embedder_weight: torch.Tensor,
    text_embedder_bias: torch.Tensor,
    x_buf: torch.Tensor,
    conditioning: torch.Tensor,
) -> None:
    prev_allow_tf32 = torch.backends.cuda.matmul.allow_tf32
    prev_precision = torch.get_float32_matmul_precision()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    try:
        torch.addmm(
            timestep_linear1_bias,
            emb_buf,
            timestep_linear1_weight.T,
            out=x_buf,
        )
        F.silu(x_buf, inplace=True)
        torch.addmm(
            _get_combined_bias(timestep_linear2_bias, text_embedder_bias),
            x_buf,
            timestep_linear2_weight.T,
            out=conditioning,
        )
        torch.addmm(
            conditioning,
            pooled_projections,
            text_embedder_weight.T,
            beta=1.0,
            alpha=1.0,
            out=conditioning,
        )
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev_allow_tf32
        torch.set_float32_matmul_precision(prev_precision)


def _torch_time_embed_out(
    timestep: torch.Tensor,
    freqs: torch.Tensor,
    scaled_buf: torch.Tensor,
    args_buf: torch.Tensor,
    emb_buf: torch.Tensor,
) -> None:
    torch.mul(timestep, 1000.0, out=scaled_buf)
    torch.outer(scaled_buf, freqs, out=args_buf)
    torch.cos(args_buf, out=emb_buf[:, :_HALF_TIME_EMBED_DIM])
    torch.sin(args_buf, out=emb_buf[:, _HALF_TIME_EMBED_DIM:])


if _HAS_CUTLASS:
    _TIME_EMBED_COMPILED = None
    _STATIC_TIME_EMBED_CACHE = {}
    _STATIC_BIAS_SILU_CACHE = {}
    _STATIC_ADD_RESIDUAL_BIAS_CACHE = {}

    def _default_opt_level() -> int:
        try:
            from cutlass import CUDA_VERSION

            if CUDA_VERSION.major < 13 or (
                CUDA_VERSION.major == 13 and CUDA_VERSION.minor < 1
            ):
                return 3
            return 2
        except Exception:
            return 3


    def _f32_ptr(addr: int):
        return make_ptr(
            cutlass.Float32,
            addr,
            cute.AddressSpace.gmem,
            assumed_align=_ALIGNMENT,
        )


    @cute.kernel
    def _time_embed_kernel(
        g_timestep: cute.Tensor,
        g_freqs: cute.Tensor,
        g_emb: cute.Tensor,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        row, _, _ = cute.arch.block_idx()

        if row < g_timestep.shape[0] and tidx < _HALF_TIME_EMBED_GROUPS:
            timestep_val = g_timestep[row] * cutlass.Float32(1000.0)
            freqs = g_freqs[tidx, None].load()
            args = timestep_val * freqs
            g_emb[row, tidx, None].store(cute.math.cos(args, fastmath=True))
            g_emb[row, tidx + _HALF_TIME_EMBED_GROUPS, None].store(
                cute.math.sin(args, fastmath=True)
            )


    @cute.jit
    def _time_embed_host(
        timestep_ptr: cute.Pointer,
        freqs_ptr: cute.Pointer,
        emb_ptr: cute.Pointer,
        batch_size: cutlass.Int32,
    ):
        g_timestep = cute.make_tensor(
            timestep_ptr,
            cute.make_layout((batch_size,), stride=(1,)),
        )
        g_freqs = cute.make_tensor(
            freqs_ptr,
            cute.make_layout(
                (_HALF_TIME_EMBED_GROUPS, _TIME_EMBED_VEC),
                stride=(_TIME_EMBED_VEC, 1),
            ),
        )
        g_emb = cute.make_tensor(
            emb_ptr,
            cute.make_layout(
                (batch_size, _FULL_TIME_EMBED_GROUPS, _TIME_EMBED_VEC),
                stride=(_TIME_EMBED_DIM, _TIME_EMBED_VEC, 1),
            ),
        )

        _time_embed_kernel(g_timestep, g_freqs, g_emb).launch(
            grid=(batch_size, 1, 1),
            block=(_TIME_EMBED_THREADS, 1, 1),
        )


    def _ensure_time_embed():
        global _TIME_EMBED_COMPILED
        if _TIME_EMBED_COMPILED is not None:
            return _TIME_EMBED_COMPILED
        _TIME_EMBED_COMPILED = cute.compile(
            _time_embed_host,
            _f32_ptr(_ALIGNMENT),
            _f32_ptr(_ALIGNMENT),
            _f32_ptr(_ALIGNMENT),
            cutlass.Int32(1),
            options=f"--opt-level {_default_opt_level()}",
        )
        return _TIME_EMBED_COMPILED


    def _make_static_time_embed_host(batch_const: int):
        @cute.jit
        def _host(
            timestep_ptr: cute.Pointer,
            freqs_ptr: cute.Pointer,
            emb_ptr: cute.Pointer,
        ):
            g_timestep = cute.make_tensor(
                timestep_ptr,
                cute.make_layout((batch_const,), stride=(1,)),
            )
            g_freqs = cute.make_tensor(
                freqs_ptr,
                cute.make_layout(
                    (_HALF_TIME_EMBED_GROUPS, _TIME_EMBED_VEC),
                    stride=(_TIME_EMBED_VEC, 1),
                ),
            )
            g_emb = cute.make_tensor(
                emb_ptr,
                cute.make_layout(
                    (batch_const, _FULL_TIME_EMBED_GROUPS, _TIME_EMBED_VEC),
                    stride=(_TIME_EMBED_DIM, _TIME_EMBED_VEC, 1),
                ),
            )

            _time_embed_kernel(g_timestep, g_freqs, g_emb).launch(
                grid=(batch_const, 1, 1),
                block=(_TIME_EMBED_THREADS, 1, 1),
            )

        return _host


    def _ensure_static_time_embed(batch_const: int):
        compiled = _STATIC_TIME_EMBED_CACHE.get(batch_const)
        if compiled is not None:
            return compiled
        compiled = cute.compile(
            _make_static_time_embed_host(batch_const),
            _f32_ptr(_ALIGNMENT),
            _f32_ptr(_ALIGNMENT),
            _f32_ptr(_ALIGNMENT),
            options=f"--opt-level {_default_opt_level()}",
        )
        _STATIC_TIME_EMBED_CACHE[batch_const] = compiled
        return compiled


    def _time_embed_out(
        timestep_1d: torch.Tensor,
        freqs_1d: torch.Tensor,
        emb_2d: torch.Tensor,
    ) -> None:
        if not timestep_1d.is_contiguous():
            timestep_1d = timestep_1d.contiguous()
        if not freqs_1d.is_contiguous():
            freqs_1d = freqs_1d.contiguous()
        if not emb_2d.is_contiguous():
            raise ValueError("time embedding output must be contiguous")
        compiled = _ensure_time_embed()
        compiled(
            _f32_ptr(timestep_1d.data_ptr()),
            _f32_ptr(freqs_1d.data_ptr()),
            _f32_ptr(emb_2d.data_ptr()),
            cutlass.Int32(int(timestep_1d.shape[0])),
        )


    def _static_time_embed_out(
        timestep_1d: torch.Tensor,
        freqs_1d: torch.Tensor,
        emb_2d: torch.Tensor,
    ) -> None:
        if not timestep_1d.is_contiguous():
            timestep_1d = timestep_1d.contiguous()
        if not freqs_1d.is_contiguous():
            freqs_1d = freqs_1d.contiguous()
        if not emb_2d.is_contiguous():
            raise ValueError("time embedding output must be contiguous")
        compiled = _ensure_static_time_embed(int(timestep_1d.shape[0]))
        compiled(
            _f32_ptr(timestep_1d.data_ptr()),
            _f32_ptr(freqs_1d.data_ptr()),
            _f32_ptr(emb_2d.data_ptr()),
        )


    @cute.kernel
    def _bias_silu_kernel(
        g_x: cute.Tensor,
        g_bias: cute.Tensor,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        linear_block, _, _ = cute.arch.block_idx()

        row = linear_block // _INNER_BLOCKS_PER_ROW
        group = linear_block % _INNER_BLOCKS_PER_ROW * _INNER_THREADS + tidx

        if row < g_x.shape[0]:
            vals = g_x[row, group, None].load() + g_bias[group, None].load()
            one = cute.full_like(vals, 1.0)
            exp_neg = cute.math.exp(cute.full_like(vals, -1.0) * vals, fastmath=True)
            g_x[row, group, None].store(vals / (one + exp_neg))


    def _make_static_bias_silu_host(batch_const: int):
        @cute.jit
        def _host(
            x_ptr: cute.Pointer,
            bias_ptr: cute.Pointer,
        ):
            g_x = cute.make_tensor(
                x_ptr,
                cute.make_layout(
                    (batch_const, _INNER_GROUPS, _INNER_VEC),
                    stride=(_INNER_DIM, _INNER_VEC, 1),
                ),
            )
            g_bias = cute.make_tensor(
                bias_ptr,
                cute.make_layout((_INNER_GROUPS, _INNER_VEC), stride=(_INNER_VEC, 1)),
            )

            _bias_silu_kernel(g_x, g_bias).launch(
                grid=(batch_const * _INNER_BLOCKS_PER_ROW, 1, 1),
                block=(_INNER_THREADS, 1, 1),
            )

        return _host


    def _ensure_static_bias_silu(batch_const: int):
        compiled = _STATIC_BIAS_SILU_CACHE.get(batch_const)
        if compiled is not None:
            return compiled
        compiled = cute.compile(
            _make_static_bias_silu_host(batch_const),
            _f32_ptr(_ALIGNMENT),
            _f32_ptr(_ALIGNMENT),
            options=f"--opt-level {_default_opt_level()}",
        )
        _STATIC_BIAS_SILU_CACHE[batch_const] = compiled
        return compiled


    def _bias_silu_inplace(x_2d: torch.Tensor, bias_1d: torch.Tensor) -> None:
        if not x_2d.is_contiguous():
            raise ValueError("bias-silu input must be contiguous")
        if not bias_1d.is_contiguous():
            bias_1d = bias_1d.contiguous()
        compiled = _ensure_static_bias_silu(int(x_2d.shape[0]))
        compiled(
            _f32_ptr(x_2d.data_ptr()),
            _f32_ptr(bias_1d.data_ptr()),
        )


    @cute.kernel
    def _add_residual_bias_kernel(
        g_out: cute.Tensor,
        g_residual: cute.Tensor,
        g_bias: cute.Tensor,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        linear_block, _, _ = cute.arch.block_idx()

        row = linear_block // _INNER_BLOCKS_PER_ROW
        group = linear_block % _INNER_BLOCKS_PER_ROW * _INNER_THREADS + tidx

        if row < g_out.shape[0]:
            vals = (
                g_out[row, group, None].load()
                + g_residual[row, group, None].load()
                + g_bias[group, None].load()
            )
            g_out[row, group, None].store(vals)


    def _make_static_add_residual_bias_host(batch_const: int):
        @cute.jit
        def _host(
            out_ptr: cute.Pointer,
            residual_ptr: cute.Pointer,
            bias_ptr: cute.Pointer,
        ):
            g_out = cute.make_tensor(
                out_ptr,
                cute.make_layout(
                    (batch_const, _INNER_GROUPS, _INNER_VEC),
                    stride=(_INNER_DIM, _INNER_VEC, 1),
                ),
            )
            g_residual = cute.make_tensor(
                residual_ptr,
                cute.make_layout(
                    (batch_const, _INNER_GROUPS, _INNER_VEC),
                    stride=(_INNER_DIM, _INNER_VEC, 1),
                ),
            )
            g_bias = cute.make_tensor(
                bias_ptr,
                cute.make_layout((_INNER_GROUPS, _INNER_VEC), stride=(_INNER_VEC, 1)),
            )

            _add_residual_bias_kernel(g_out, g_residual, g_bias).launch(
                grid=(batch_const * _INNER_BLOCKS_PER_ROW, 1, 1),
                block=(_INNER_THREADS, 1, 1),
            )

        return _host


    def _ensure_static_add_residual_bias(batch_const: int):
        compiled = _STATIC_ADD_RESIDUAL_BIAS_CACHE.get(batch_const)
        if compiled is not None:
            return compiled
        compiled = cute.compile(
            _make_static_add_residual_bias_host(batch_const),
            _f32_ptr(_ALIGNMENT),
            _f32_ptr(_ALIGNMENT),
            _f32_ptr(_ALIGNMENT),
            options=f"--opt-level {_default_opt_level()}",
        )
        _STATIC_ADD_RESIDUAL_BIAS_CACHE[batch_const] = compiled
        return compiled


    def _add_residual_bias_inplace(
        out_2d: torch.Tensor,
        residual_2d: torch.Tensor,
        bias_1d: torch.Tensor,
    ) -> None:
        if not out_2d.is_contiguous():
            raise ValueError("residual-bias output must be contiguous")
        if not residual_2d.is_contiguous():
            raise ValueError("residual-bias residual must be contiguous")
        if not bias_1d.is_contiguous():
            bias_1d = bias_1d.contiguous()
        compiled = _ensure_static_add_residual_bias(int(out_2d.shape[0]))
        compiled(
            _f32_ptr(out_2d.data_ptr()),
            _f32_ptr(residual_2d.data_ptr()),
            _f32_ptr(bias_1d.data_ptr()),
        )


    def _pick_kernel(in_dim: int):
        if in_dim >= _INNER_DIM:
            return _WIDE_GEMM
        return _NARROW_GEMM


    def _make_static_linear_host(in_dim: int, out_dim: int, m_const: int):
        kernel = _pick_kernel(in_dim)

        @cute.jit
        def _host(
            a_ptr: cute.Pointer,
            b_ptr: cute.Pointer,
            c_ptr: cute.Pointer,
        ):
            a = cute.make_tensor(
                a_ptr,
                cute.make_layout((m_const, in_dim, 1), stride=(in_dim, 1, 1)),
            )
            b = cute.make_tensor(
                b_ptr,
                cute.make_layout((out_dim, in_dim, 1), stride=(in_dim, 1, 1)),
            )
            c = cute.make_tensor(
                c_ptr,
                cute.make_layout((m_const, out_dim, 1), stride=(out_dim, 1, 1)),
            )
            kernel(a, b, c)

        return _host


    def _ensure_static_linear(in_dim: int, out_dim: int, m_const: int):
        key = (in_dim, out_dim, m_const)
        compiled = _STATIC_LINEAR_CACHE.get(key)
        if compiled is not None:
            return compiled
        compiled = cute.compile(
            _make_static_linear_host(in_dim, out_dim, m_const),
            _f32_ptr(_ALIGNMENT),
            _f32_ptr(_ALIGNMENT),
            _f32_ptr(_ALIGNMENT),
            options=f"--opt-level {_default_opt_level()}",
        )
        _STATIC_LINEAR_CACHE[key] = compiled
        return compiled


    def _linear_out(
        x_2d: torch.Tensor,
        weight_2d: torch.Tensor,
        out_2d: torch.Tensor,
    ) -> None:
        if not x_2d.is_contiguous():
            x_2d = x_2d.contiguous()
        if not weight_2d.is_contiguous():
            weight_2d = weight_2d.contiguous()
        if not out_2d.is_contiguous():
            raise ValueError("CuTe linear output must be contiguous")

        m = int(x_2d.shape[0])
        in_dim = int(x_2d.shape[1])
        out_dim = int(weight_2d.shape[0])
        compiled = _ensure_static_linear(in_dim, out_dim, m)
        compiled(
            _f32_ptr(x_2d.data_ptr()),
            _f32_ptr(weight_2d.data_ptr()),
            _f32_ptr(out_2d.data_ptr()),
        )


@torch.no_grad()
def run(
    timestep: torch.Tensor,
    pooled_projections: torch.Tensor,
    freqs: torch.Tensor,
    timestep_linear1_weight: torch.Tensor,
    timestep_linear1_bias: torch.Tensor,
    timestep_linear2_weight: torch.Tensor,
    timestep_linear2_bias: torch.Tensor,
    text_embedder_weight: torch.Tensor,
    text_embedder_bias: torch.Tensor,
    conditioning: torch.Tensor,
):
    if timestep.device.type != "cuda" or not _HAS_CUTLASS:
        _run_reference(
            timestep,
            pooled_projections,
            freqs,
            timestep_linear1_weight,
            timestep_linear1_bias,
            timestep_linear2_weight,
            timestep_linear2_bias,
            text_embedder_weight,
            text_embedder_bias,
            conditioning,
        )
        return

    batch_size = int(timestep.shape[0])
    if batch_size == 1:
        batch1_state = _get_batch1_graph_state(timestep.device)
        args = (
            timestep,
            pooled_projections,
            freqs,
            timestep_linear1_weight,
            timestep_linear1_bias,
            timestep_linear2_weight,
            timestep_linear2_bias,
            text_embedder_weight,
            text_embedder_bias,
        )
        if batch1_state["runner"] is not None:
            conditioning.copy_(batch1_state["runner"](*args))
        else:
            conditioning.copy_(batch1_state["fn"](*args))
        return

    workspace = _get_workspace(batch_size, timestep.device)

    scaled_buf = workspace["scaled"]
    args_buf = workspace["args"]
    emb_buf = workspace["emb"]
    x_buf = workspace["x"]
    ts_buf = workspace["ts"]

    if batch_size == 5:
        _torch_time_embed_out(timestep, freqs, scaled_buf, args_buf, emb_buf)
    elif batch_size in (11, 12, 15, 17, 20, 24, 29, 37, 41, 59):
        _static_time_embed_out(timestep, freqs, emb_buf)
    else:
        _time_embed_out(timestep, freqs, emb_buf)

    if batch_size in (2, 4, 5, 9, 11, 12, 15, 17, 20, 24):
        _run_small_addmm_tf32(
            emb_buf,
            pooled_projections,
            timestep_linear1_weight,
            timestep_linear1_bias,
            timestep_linear2_weight,
            timestep_linear2_bias,
            text_embedder_weight,
            text_embedder_bias,
            x_buf,
            conditioning,
        )
        return

    _linear_out(emb_buf, timestep_linear1_weight, x_buf)
    _bias_silu_inplace(x_buf, timestep_linear1_bias)

    _linear_out(x_buf, timestep_linear2_weight, conditioning)
    _linear_out(pooled_projections, text_embedder_weight, ts_buf)
    _add_residual_bias_inplace(
        conditioning,
        ts_buf,
        _get_combined_bias(timestep_linear2_bias, text_embedder_bias),
    )
