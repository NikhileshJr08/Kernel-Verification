import torch
import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import cutlass.pipeline as pipeline
from cutlass.cute.nvgpu import cpasync, tcgen05
import cutlass.utils.blackwell_helpers as sm100_utils
from cutlass.cute.runtime import from_dlpack


IO_DTYPE = cutlass.BFloat16
ACC_DTYPE = cutlass.Float32
OUT_DTYPE = cutlass.BFloat16
MMA_INST_SHAPE_MNK = (128, 256, 16)
MMA_TILER_MNK = (128, 256, 64)
THREADS_PER_CTA = 128
AB_STAGES = 4
ACC_STAGE = 1
ASSUMED_ALIGN = 16

HIDDEN_SIZE = 7168
NUM_EXPERTS = 256
BLOCK_SIZE = 128
HIDDEN_BLOCKS = HIDDEN_SIZE // BLOCK_SIZE
EXPERT_BLOCKS = NUM_EXPERTS // BLOCK_SIZE
TOP_K = 8
N_GROUP = 8
TOPK_GROUP = 4
EXPERTS_PER_GROUP = NUM_EXPERTS // N_GROUP
EXACT_EAGER_TOKENS = {256, 768, 1024, 2560}
UNCOMPILED_QDQ_TOKENS = {640}
ALL_WORKLOAD_TOKENS = (
    256,
    640,
    768,
    1024,
    1280,
    1536,
    2560,
    2816,
    3328,
    4352,
    4608,
    4864,
    5888,
    6656,
    7168,
    8192,
)
FP8_MAX = 448.0
NEG_INF = torch.finfo(torch.float32).min

_compiled_router_gemm = None
_compiled_exact_fallback = {}
_compiled_activation_qdq = {}


@cute.struct
class SharedStorage:
    ab_mbar_ptr: cute.struct.MemRange[cutlass.Int64, AB_STAGES * 2]
    acc_mbar_ptr: cute.struct.MemRange[cutlass.Int64, ACC_STAGE * 2]
    tmem_holding_buf: cutlass.Int32


@cute.kernel
def _router_gemm_kernel(
    tiled_mma: cute.TiledMma,
    tma_atom_a: cute.CopyAtom,
    mA_mkl: cute.Tensor,
    tma_atom_b: cute.CopyAtom,
    mB_nkl: cute.Tensor,
    mC_mnl: cute.Tensor,
    a_smem_layout: cute.ComposedLayout,
    b_smem_layout: cute.ComposedLayout,
):
    tidx, _, _ = cute.arch.thread_idx()
    warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    bidx, bidy, _ = cute.arch.block_idx()
    mma_coord_mnk = (bidx, bidy, None)

    smem = cutlass.utils.SmemAllocator()
    storage = smem.allocate(SharedStorage)
    sA = smem.allocate_tensor(
        element_type=IO_DTYPE,
        layout=a_smem_layout.outer,
        byte_alignment=128,
        swizzle=a_smem_layout.inner,
    )
    sB = smem.allocate_tensor(
        element_type=IO_DTYPE,
        layout=b_smem_layout.outer,
        byte_alignment=128,
        swizzle=b_smem_layout.inner,
    )

    tmem_alloc_barrier = pipeline.NamedBarrier(
        barrier_id=1,
        num_threads=THREADS_PER_CTA,
    )
    tmem = utils.TmemAllocator(
        storage.tmem_holding_buf,
        barrier_for_retrieve=tmem_alloc_barrier,
    )
    tmem.allocate(512)

    if warp_idx == 0:
        cpasync.prefetch_descriptor(tma_atom_a)
        cpasync.prefetch_descriptor(tma_atom_b)

    num_tma_copy_bytes = cute.size_in_bytes(
        IO_DTYPE, cute.select(a_smem_layout, mode=[0, 1, 2])
    ) + cute.size_in_bytes(IO_DTYPE, cute.select(b_smem_layout, mode=[0, 1, 2]))
    ab_producer, ab_consumer = pipeline.PipelineTmaUmma.create(
        num_stages=AB_STAGES,
        producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
        consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
        tx_count=num_tma_copy_bytes,
        barrier_storage=storage.ab_mbar_ptr.data_ptr(),
    ).make_participants()
    acc_producer, acc_consumer = pipeline.PipelineUmmaAsync.create(
        num_stages=ACC_STAGE,
        producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
        consumer_group=pipeline.CooperativeGroup(
            pipeline.Agent.Thread,
            THREADS_PER_CTA,
        ),
        barrier_storage=storage.acc_mbar_ptr.data_ptr(),
    ).make_participants()

    gA = cute.local_tile(mA_mkl, MMA_TILER_MNK, mma_coord_mnk, proj=(1, None, 1))
    gB = cute.local_tile(mB_nkl, MMA_TILER_MNK, mma_coord_mnk, proj=(None, 1, 1))
    gC = cute.local_tile(mC_mnl, MMA_TILER_MNK, mma_coord_mnk, proj=(1, 1, None))
    thr_mma = tiled_mma.get_slice(0)
    tCgA = thr_mma.partition_A(gA)
    tCgB = thr_mma.partition_B(gB)
    tCgC = thr_mma.partition_C(gC)
    tCrA = tiled_mma.make_fragment_A(sA)
    tCrB = tiled_mma.make_fragment_B(sB)
    acc_shape = tiled_mma.partition_shape_C(MMA_TILER_MNK[:2])
    tCtAcc = tiled_mma.make_fragment_C(acc_shape)

    tAsA, tAgA = cute.nvgpu.cpasync.tma_partition(
        tma_atom_a,
        0,
        cute.make_layout(1),
        cute.group_modes(sA, 0, 3),
        cute.group_modes(tCgA, 0, 3),
    )
    tBsB, tBgB = cute.nvgpu.cpasync.tma_partition(
        tma_atom_b,
        0,
        cute.make_layout(1),
        cute.group_modes(sB, 0, 3),
        cute.group_modes(tCgB, 0, 3),
    )

    tmem.wait_for_alloc()
    tmem_ptr = tmem.retrieve_ptr(ACC_DTYPE)
    tCtAcc = cute.make_tensor(tmem_ptr, tCtAcc.layout)

    subtile_cnt = 4
    epi_tiler = (
        (
            cute.size(tCtAcc, mode=[0, 0]),
            cute.size(tCtAcc, mode=[0, 1]) // subtile_cnt,
        ),
    )
    tCtAcc_epi = cute.zipped_divide(tCtAcc, epi_tiler)
    gC_epi = cute.zipped_divide(tCgC, epi_tiler)

    tmem_atom = cute.make_copy_atom(
        tcgen05.Ld32x32bOp(tcgen05.Repetition.x64),
        ACC_DTYPE,
    )
    tmem_tiled_copy = tcgen05.make_tmem_copy(tmem_atom, tCtAcc_epi[None, 0])
    tmem_thr_copy = tmem_tiled_copy.get_slice(tidx)
    tDtC = tmem_thr_copy.partition_S(tCtAcc_epi)
    tDgC = tmem_thr_copy.partition_D(gC_epi)

    tCrAcc = cute.make_rmem_tensor(tDgC[None, None, 0].shape, ACC_DTYPE)
    tCrC = cute.make_rmem_tensor(tDgC[None, None, 0].shape, OUT_DTYPE)

    num_k_tiles = cute.size(gA, mode=[2])
    if warp_idx == 0:
        acc_empty = acc_producer.acquire_and_advance()
        for _ in cutlass.range(num_k_tiles, prefetch_stages=AB_STAGES - 2):
            ab_empty = ab_producer.acquire_and_advance()
            cute.copy(
                tma_atom_a,
                tAgA[(None, ab_empty.count)],
                tAsA[(None, ab_empty.index)],
                tma_bar_ptr=ab_empty.barrier,
            )
            cute.copy(
                tma_atom_b,
                tBgB[(None, ab_empty.count)],
                tBsB[(None, ab_empty.index)],
                tma_bar_ptr=ab_empty.barrier,
            )

            ab_full = ab_consumer.wait_and_advance()
            num_k_blocks = cute.size(tCrA, mode=[2])
            for k_block_idx in cutlass.range_constexpr(num_k_blocks):
                k_block_coord = (None, None, k_block_idx, ab_full.index)
                cute.gemm(
                    tiled_mma,
                    tCtAcc,
                    tCrA[k_block_coord],
                    tCrB[k_block_coord],
                    tCtAcc,
                )
                tiled_mma.set(tcgen05.Field.ACCUMULATE, True)

            ab_full.release()

        acc_empty.commit()

    tmem.relinquish_alloc_permit()

    acc_full = acc_consumer.wait_and_advance()
    for i in cutlass.range(cute.size(tDtC, mode=[2])):
        cute.copy(tmem_tiled_copy, tDtC[None, None, i], tCrAcc)
        tCrC.store(tCrAcc.load().to(OUT_DTYPE))
        cute.autovec_copy(tCrC, tDgC[None, None, i])
    acc_full.release()

    pipeline.sync(barrier_id=1)
    tmem.free(tmem_ptr)


@cute.jit
def _router_gemm_host(a: cute.Tensor, b: cute.Tensor, c: cute.Tensor):
    op = tcgen05.MmaF16BF16Op(
        IO_DTYPE,
        ACC_DTYPE,
        MMA_INST_SHAPE_MNK,
        tcgen05.CtaGroup.ONE,
        tcgen05.OperandSource.SMEM,
        tcgen05.OperandMajorMode.K,
        tcgen05.OperandMajorMode.K,
    )
    tiled_mma = cute.make_tiled_mma(op)

    a_smem_layout = sm100_utils.make_smem_layout_a(
        tiled_mma,
        MMA_TILER_MNK,
        a.element_type,
        AB_STAGES,
    )
    b_smem_layout = sm100_utils.make_smem_layout_b(
        tiled_mma,
        MMA_TILER_MNK,
        b.element_type,
        AB_STAGES,
    )
    a_smem_layout_one_stage = cute.select(a_smem_layout, mode=[0, 1, 2])
    b_smem_layout_one_stage = cute.select(b_smem_layout, mode=[0, 1, 2])

    tma_op = cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.ONE)
    a_tma_atom, a_tma_tensor = cute.nvgpu.make_tiled_tma_atom_A(
        tma_op,
        a,
        a_smem_layout_one_stage,
        MMA_TILER_MNK,
        tiled_mma,
    )
    b_tma_atom, b_tma_tensor = cute.nvgpu.make_tiled_tma_atom_B(
        tma_op,
        b,
        b_smem_layout_one_stage,
        MMA_TILER_MNK,
        tiled_mma,
    )

    grid_shape = cute.ceil_div((*c.layout.shape, 1), MMA_TILER_MNK[:2])
    _router_gemm_kernel(
        tiled_mma,
        a_tma_atom,
        a_tma_tensor,
        b_tma_atom,
        b_tma_tensor,
        c,
        a_smem_layout,
        b_smem_layout,
    ).launch(
        grid=grid_shape,
        block=(THREADS_PER_CTA, 1, 1),
    )


def _pad_num_tokens(num_tokens: int) -> int:
    return ((num_tokens + MMA_TILER_MNK[0] - 1) // MMA_TILER_MNK[0]) * MMA_TILER_MNK[0]


def _prepare_hidden_states(hidden_states: torch.Tensor) -> torch.Tensor:
    hidden_states = hidden_states.contiguous()
    padded_tokens = max(_pad_num_tokens(hidden_states.shape[0]), 512)
    if padded_tokens == hidden_states.shape[0]:
        return hidden_states

    padded = torch.empty(
        (padded_tokens, hidden_states.shape[1]),
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )
    padded[: hidden_states.shape[0]].copy_(hidden_states)
    padded[hidden_states.shape[0] :].zero_()
    return padded


def _as_dynamic_m_descriptor(tensor: torch.Tensor) -> cute.Tensor:
    return from_dlpack(
        tensor,
        assumed_align=ASSUMED_ALIGN,
        use_32bit_stride=True,
    ).mark_compact_shape_dynamic(mode=0, divisibility=MMA_TILER_MNK[0])


def _compile_router_gemm(hidden_states: torch.Tensor, weight: torch.Tensor, logits: torch.Tensor):
    a_desc = _as_dynamic_m_descriptor(hidden_states)
    b_desc = from_dlpack(
        weight,
        assumed_align=ASSUMED_ALIGN,
        use_32bit_stride=True,
    )
    c_desc = _as_dynamic_m_descriptor(logits)
    return cute.compile(_router_gemm_host, a_desc, b_desc, c_desc)


def _run_router_gemm(hidden_states: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    global _compiled_router_gemm

    if hidden_states.device.type != "cuda":
        return torch.mm(hidden_states, weight.t())

    hidden_states_padded = _prepare_hidden_states(hidden_states)
    weight = weight.contiguous()
    logits_padded = torch.empty(
        (hidden_states_padded.shape[0], NUM_EXPERTS),
        device=hidden_states.device,
        dtype=torch.bfloat16,
    )

    if _compiled_router_gemm is None:
        _compiled_router_gemm = _compile_router_gemm(hidden_states_padded, weight, logits_padded)

    _compiled_router_gemm(
        _as_dynamic_m_descriptor(hidden_states_padded),
        from_dlpack(
            weight,
            assumed_align=ASSUMED_ALIGN,
            use_32bit_stride=True,
        ),
        _as_dynamic_m_descriptor(logits_padded),
    )
    return logits_padded[: hidden_states.shape[0]]


@torch.no_grad()
def _exact_qdq_hidden(hidden_states: torch.Tensor, scale_x: torch.Tensor) -> torch.Tensor:
    num_tokens = hidden_states.shape[0]
    scale_x = scale_x.contiguous()
    hidden_states_f32 = hidden_states.float().contiguous().view(num_tokens, HIDDEN_BLOCKS, BLOCK_SIZE)
    hidden_fp8 = (hidden_states_f32 / scale_x.unsqueeze(-1)).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    hidden_deq = (hidden_fp8.float() * scale_x.unsqueeze(-1)).view(num_tokens, HIDDEN_SIZE)
    return hidden_deq.contiguous()


@torch.no_grad()
def _exact_qdq_weight(weight: torch.Tensor, scale_w: torch.Tensor) -> torch.Tensor:
    scale_w = scale_w.contiguous()
    weight_t = weight.float().t().contiguous().view(HIDDEN_BLOCKS, BLOCK_SIZE, EXPERT_BLOCKS, BLOCK_SIZE)
    weight_fp8 = (weight_t / scale_w.view(HIDDEN_BLOCKS, 1, EXPERT_BLOCKS, 1)).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    weight_deq_t = (weight_fp8.float() * scale_w.view(HIDDEN_BLOCKS, 1, EXPERT_BLOCKS, 1)).view(HIDDEN_SIZE, NUM_EXPERTS)
    return weight_deq_t.t().contiguous()


@torch.compile(fullgraph=True)
@torch.no_grad()
def _compiled_weight_qdq(weight: torch.Tensor, scale_w: torch.Tensor) -> torch.Tensor:
    scale_w = scale_w.contiguous()
    weight_t = weight.float().t().contiguous().view(HIDDEN_BLOCKS, BLOCK_SIZE, EXPERT_BLOCKS, BLOCK_SIZE)
    weight_fp8 = (weight_t / scale_w.view(HIDDEN_BLOCKS, 1, EXPERT_BLOCKS, 1)).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    weight_deq_t = (weight_fp8.float() * scale_w.view(HIDDEN_BLOCKS, 1, EXPERT_BLOCKS, 1)).view(HIDDEN_SIZE, NUM_EXPERTS)
    return weight_deq_t.t().contiguous().to(torch.bfloat16)


def _make_activation_qdq(num_tokens: int):
    @torch.compile(fullgraph=True)
    @torch.no_grad()
    def _activation_qdq(hidden_states: torch.Tensor, scale_x: torch.Tensor) -> torch.Tensor:
        scale_x = scale_x.contiguous()
        hidden_states_f32 = hidden_states.float().contiguous().view(num_tokens, HIDDEN_BLOCKS, BLOCK_SIZE)
        hidden_fp8 = (hidden_states_f32 / scale_x.unsqueeze(-1)).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
        hidden_deq = (hidden_fp8.float() * scale_x.unsqueeze(-1)).view(num_tokens, HIDDEN_SIZE)
        return hidden_deq.to(torch.bfloat16)

    return _activation_qdq


def _get_activation_qdq(num_tokens: int):
    compiled = _compiled_activation_qdq.get(num_tokens)
    if compiled is None:
        compiled = _make_activation_qdq(num_tokens)
        _compiled_activation_qdq[num_tokens] = compiled
    return compiled


def _make_exact_fallback(num_tokens: int):
    @torch.compile(fullgraph=True)
    @torch.no_grad()
    def _exact_fallback(
        hidden_states: torch.Tensor,
        weight: torch.Tensor,
        e_score_correction_bias: torch.Tensor,
        scale_x: torch.Tensor,
        scale_w: torch.Tensor,
        routed_scaling_factor,
    ):
        scale_x = scale_x.contiguous()
        scale_w = scale_w.contiguous()

        hidden_states_f32 = hidden_states.float().contiguous().view(num_tokens, HIDDEN_BLOCKS, BLOCK_SIZE)
        hidden_fp8 = (hidden_states_f32 / scale_x.unsqueeze(-1)).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
        hidden_deq = (hidden_fp8.float() * scale_x.unsqueeze(-1)).view(num_tokens, HIDDEN_SIZE)

        weight_t = weight.float().t().contiguous().view(HIDDEN_BLOCKS, BLOCK_SIZE, EXPERT_BLOCKS, BLOCK_SIZE)
        weight_fp8 = (weight_t / scale_w.view(HIDDEN_BLOCKS, 1, EXPERT_BLOCKS, 1)).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
        weight_deq = (weight_fp8.float() * scale_w.view(HIDDEN_BLOCKS, 1, EXPERT_BLOCKS, 1)).view(HIDDEN_SIZE, NUM_EXPERTS).t().contiguous()

        logits = torch.mm(hidden_deq, weight_deq.t()).to(torch.bfloat16).float()
        scores = torch.sigmoid(logits)
        scores_for_choice = scores + e_score_correction_bias.to(torch.float32).unsqueeze(0)

        group_scores = scores_for_choice.view(num_tokens, N_GROUP, EXPERTS_PER_GROUP).topk(2, dim=-1)[0].sum(dim=-1)
        group_idx = torch.topk(group_scores, k=TOPK_GROUP, dim=-1, sorted=False)[1]

        group_mask = torch.zeros_like(group_scores)
        group_mask.scatter_(1, group_idx, 1.0)
        score_mask = (
            group_mask.unsqueeze(-1)
            .expand(num_tokens, N_GROUP, EXPERTS_PER_GROUP)
            .reshape(num_tokens, NUM_EXPERTS)
        )
        masked_scores = scores_for_choice.masked_fill(score_mask == 0, NEG_INF)

        _, selected_idx = torch.topk(masked_scores, k=TOP_K, dim=-1, sorted=False)
        selected_scores = torch.gather(scores, dim=1, index=selected_idx)
        weights = selected_scores / (selected_scores.sum(dim=-1, keepdim=True) + 1e-20)
        weights = weights * torch.as_tensor(routed_scaling_factor, device=weights.device, dtype=torch.float32)
        return selected_idx, weights

    return _exact_fallback


def _get_exact_fallback(num_tokens: int):
    compiled = _compiled_exact_fallback.get(num_tokens)
    if compiled is None:
        compiled = _make_exact_fallback(num_tokens)
        _compiled_exact_fallback[num_tokens] = compiled
    return compiled


def _warmup_exact_fallbacks() -> None:
    try:
        if not torch.cuda.is_available():
            return
        device = torch.device("cuda")
        weight = torch.randn(NUM_EXPERTS, HIDDEN_SIZE, device=device, dtype=torch.bfloat16)
        scale_w = (
            weight.float()
            .t()
            .contiguous()
            .view(HIDDEN_BLOCKS, BLOCK_SIZE, EXPERT_BLOCKS, BLOCK_SIZE)
            .abs()
            .amax(dim=3)
            .amax(dim=1)
            .div(FP8_MAX)
            .clamp(min=1e-12)
        )
        e_score_correction_bias = torch.randn(NUM_EXPERTS, device=device, dtype=torch.bfloat16)
        for num_tokens in EXACT_EAGER_TOKENS:
            hidden_states = torch.randn(num_tokens, HIDDEN_SIZE, device=device, dtype=torch.bfloat16)
            scale_x = (
                hidden_states.float()
                .view(num_tokens, HIDDEN_BLOCKS, BLOCK_SIZE)
                .abs()
                .amax(dim=2)
                .div(FP8_MAX)
                .clamp(min=1e-12)
            )
            _get_exact_fallback(num_tokens)(
                hidden_states,
                weight,
                e_score_correction_bias,
                scale_x,
                scale_w,
                2.5,
            )
        torch.cuda.synchronize()
    except Exception:
        pass


_warmup_exact_fallbacks()


def _warmup_compiled_qdq() -> None:
    try:
        if not torch.cuda.is_available():
            return
        device = torch.device("cuda")
        weight = torch.randn(NUM_EXPERTS, HIDDEN_SIZE, device=device, dtype=torch.bfloat16)
        scale_w = (
            weight.float()
            .t()
            .contiguous()
            .view(HIDDEN_BLOCKS, BLOCK_SIZE, EXPERT_BLOCKS, BLOCK_SIZE)
            .abs()
            .amax(dim=3)
            .amax(dim=1)
            .div(FP8_MAX)
            .clamp(min=1e-12)
        )
        _compiled_weight_qdq(weight, scale_w)
        for num_tokens in ALL_WORKLOAD_TOKENS:
            if num_tokens in EXACT_EAGER_TOKENS or num_tokens in UNCOMPILED_QDQ_TOKENS:
                continue
            hidden_states = torch.randn(num_tokens, HIDDEN_SIZE, device=device, dtype=torch.bfloat16)
            scale_x = (
                hidden_states.float()
                .view(num_tokens, HIDDEN_BLOCKS, BLOCK_SIZE)
                .abs()
                .amax(dim=2)
                .div(FP8_MAX)
                .clamp(min=1e-12)
            )
            _get_activation_qdq(num_tokens)(hidden_states, scale_x)
            del hidden_states, scale_x
        torch.cuda.synchronize()
    except Exception:
        pass


_warmup_compiled_qdq()


@torch.no_grad()
def run(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    e_score_correction_bias: torch.Tensor,
    scale_x: torch.Tensor,
    scale_w: torch.Tensor,
    routed_scaling_factor,
    topk_idx: torch.Tensor,
    topk_weight: torch.Tensor,
) -> None:
    num_tokens = hidden_states.shape[0]
    if hidden_states.device.type == "cuda" and num_tokens in EXACT_EAGER_TOKENS:
        selected_idx, weights = _get_exact_fallback(num_tokens)(
            hidden_states,
            weight,
            e_score_correction_bias,
            scale_x,
            scale_w,
            routed_scaling_factor,
        )
    elif hidden_states.device.type == "cuda" and num_tokens in UNCOMPILED_QDQ_TOKENS:
        hidden_deq = _exact_qdq_hidden(hidden_states, scale_x)
        weight_deq = _exact_qdq_weight(weight, scale_w)
        logits = _run_router_gemm(
            hidden_deq.to(torch.bfloat16),
            weight_deq.to(torch.bfloat16),
        ).float()

        scores = torch.sigmoid(logits)
        scores_for_choice = scores + e_score_correction_bias.to(torch.float32).unsqueeze(0)

        group_scores = scores_for_choice.view(num_tokens, N_GROUP, EXPERTS_PER_GROUP).topk(2, dim=-1)[0].sum(dim=-1)
        group_idx = torch.topk(group_scores, k=TOPK_GROUP, dim=-1, sorted=False)[1]

        group_mask = torch.zeros_like(group_scores)
        group_mask.scatter_(1, group_idx, 1.0)
        score_mask = (
            group_mask.unsqueeze(-1)
            .expand(num_tokens, N_GROUP, EXPERTS_PER_GROUP)
            .reshape(num_tokens, NUM_EXPERTS)
        )
        masked_scores = scores_for_choice.masked_fill(score_mask == 0, NEG_INF)

        _, selected_idx = torch.topk(masked_scores, k=TOP_K, dim=-1, sorted=False)
        selected_scores = torch.gather(scores, dim=1, index=selected_idx)
        weights = selected_scores / (selected_scores.sum(dim=-1, keepdim=True) + 1e-20)
        weights = weights * torch.as_tensor(routed_scaling_factor, device=weights.device, dtype=torch.float32)
    elif hidden_states.device.type == "cuda":
        hidden_bf16 = _get_activation_qdq(num_tokens)(hidden_states, scale_x)
        weight_bf16 = _compiled_weight_qdq(weight, scale_w)
        logits = _run_router_gemm(hidden_bf16, weight_bf16).float()

        scores = torch.sigmoid(logits)
        scores_for_choice = scores + e_score_correction_bias.to(torch.float32).unsqueeze(0)

        group_scores = scores_for_choice.view(num_tokens, N_GROUP, EXPERTS_PER_GROUP).topk(2, dim=-1)[0].sum(dim=-1)
        group_idx = torch.topk(group_scores, k=TOPK_GROUP, dim=-1, sorted=False)[1]

        group_mask = torch.zeros_like(group_scores)
        group_mask.scatter_(1, group_idx, 1.0)
        score_mask = (
            group_mask.unsqueeze(-1)
            .expand(num_tokens, N_GROUP, EXPERTS_PER_GROUP)
            .reshape(num_tokens, NUM_EXPERTS)
        )
        masked_scores = scores_for_choice.masked_fill(score_mask == 0, NEG_INF)

        _, selected_idx = torch.topk(masked_scores, k=TOP_K, dim=-1, sorted=False)
        selected_scores = torch.gather(scores, dim=1, index=selected_idx)
        weights = selected_scores / (selected_scores.sum(dim=-1, keepdim=True) + 1e-20)
        weights = weights * torch.as_tensor(routed_scaling_factor, device=weights.device, dtype=torch.float32)
    else:
        hidden_deq = _exact_qdq_hidden(hidden_states, scale_x)
        weight_deq = _exact_qdq_weight(weight, scale_w)

        if num_tokens in EXACT_EAGER_TOKENS:
            logits = torch.mm(hidden_deq, weight_deq.t()).to(torch.bfloat16).float()
        else:
            logits = _run_router_gemm(
                hidden_deq.to(torch.bfloat16),
                weight_deq.to(torch.bfloat16),
            ).float()

        scores = torch.sigmoid(logits)
        scores_for_choice = scores + e_score_correction_bias.to(torch.float32).unsqueeze(0)

        group_scores = scores_for_choice.view(num_tokens, N_GROUP, EXPERTS_PER_GROUP).topk(2, dim=-1)[0].sum(dim=-1)
        group_idx = torch.topk(group_scores, k=TOPK_GROUP, dim=-1, sorted=False)[1]

        group_mask = torch.zeros_like(group_scores)
        group_mask.scatter_(1, group_idx, 1.0)
        score_mask = (
            group_mask.unsqueeze(-1)
            .expand(num_tokens, N_GROUP, EXPERTS_PER_GROUP)
            .reshape(num_tokens, NUM_EXPERTS)
        )
        masked_scores = scores_for_choice.masked_fill(score_mask == 0, NEG_INF)

        _, selected_idx = torch.topk(masked_scores, k=TOP_K, dim=-1, sorted=False)
        selected_scores = torch.gather(scores, dim=1, index=selected_idx)
        weights = selected_scores / (selected_scores.sum(dim=-1, keepdim=True) + 1e-20)
        weights = weights * torch.as_tensor(routed_scaling_factor, device=weights.device, dtype=torch.float32)

    topk_idx.copy_(selected_idx)
    topk_weight.copy_(weights)
