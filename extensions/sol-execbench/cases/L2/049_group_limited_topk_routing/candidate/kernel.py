import torch
import torch._dynamo

import cutlass
import cutlass.cute as cute
import cutlass.cute.math as cmath
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass.cute.runtime import from_dlpack


NUM_EXPERTS = 256
NUM_EXPERTS_PER_TOK = 8
N_GROUP = 8
TOPK_GROUP = 4
GATING_DIM = 4096
EXPERTS_PER_GROUP = NUM_EXPERTS // N_GROUP
COMPACT_EXPERTS = TOPK_GROUP * EXPERTS_PER_GROUP

GEMM_IO_DTYPE = cutlass.BFloat16
GEMM_ACC_DTYPE = cutlass.Float32
GEMM_OUT_DTYPE = cutlass.Float32
MMA_INST_SHAPE_MNK = (128, 256, 16)
MMA_TILER_MNK = (128, 256, 64)
THREADS_PER_CTA = 128
AB_STAGES = 4
ACC_STAGE = 1

torch._dynamo.config.cache_size_limit = 128


def _use_exact_router_frontend(padded_tokens: int) -> bool:
    return padded_tokens < 6144 or padded_tokens == 8192


def _use_exact_route_compact_frontend(padded_tokens: int) -> bool:
    return padded_tokens < 6144

_compiled_router_gemm_dynamic = None
_compiled_route_compact_dynamic = None
_compiled_router_gemm_exact = {}
_compiled_route_compact_exact = {}
_workspace_cache = {}
_compiled_tail_fns = {}


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
    warp_idx = cute.arch.warp_idx()
    warp_idx = cute.arch.make_warp_uniform(warp_idx)
    bidx, bidy, _ = cute.arch.block_idx()
    mma_coord_mnk = (bidx, bidy, None)

    smem = utils.SmemAllocator()
    storage = smem.allocate(SharedStorage)
    sA = smem.allocate_tensor(
        element_type=GEMM_IO_DTYPE,
        layout=a_smem_layout.outer,
        byte_alignment=128,
        swizzle=a_smem_layout.inner,
    )
    sB = smem.allocate_tensor(
        element_type=GEMM_IO_DTYPE,
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
    num_tmem_cols = 512
    tmem.allocate(num_tmem_cols)

    if warp_idx == 0:
        cpasync.prefetch_descriptor(tma_atom_a)
        cpasync.prefetch_descriptor(tma_atom_b)

    num_tma_copy_bytes = cute.size_in_bytes(
        GEMM_IO_DTYPE, cute.select(a_smem_layout, mode=[0, 1, 2])
    ) + cute.size_in_bytes(GEMM_IO_DTYPE, cute.select(b_smem_layout, mode=[0, 1, 2]))
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
    tmem_ptr = tmem.retrieve_ptr(GEMM_ACC_DTYPE)
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
        GEMM_ACC_DTYPE,
    )
    tmem_tiled_copy = tcgen05.make_tmem_copy(tmem_atom, tCtAcc_epi[None, 0])
    tmem_thr_copy = tmem_tiled_copy.get_slice(tidx)

    tDtC = tmem_thr_copy.partition_S(tCtAcc_epi)
    tDgC = tmem_thr_copy.partition_D(gC_epi)

    tCrAcc = cute.make_rmem_tensor(tDgC[None, None, 0].shape, GEMM_ACC_DTYPE)
    tCrC = cute.make_rmem_tensor(tDgC[None, None, 0].shape, GEMM_OUT_DTYPE)

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
        tCrC.store(tCrAcc.load().to(GEMM_OUT_DTYPE))
        cute.autovec_copy(tCrC, tDgC[None, None, i])
    acc_full.release()

    pipeline.sync(barrier_id=1)
    tmem.free(tmem_ptr)


@cute.jit
def _launch_router_gemm(a: cute.Tensor, b: cute.Tensor, c: cute.Tensor):
    op = tcgen05.MmaF16BF16Op(
        GEMM_IO_DTYPE,
        GEMM_ACC_DTYPE,
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

    op = cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.ONE)
    a_tma_atom, a_tma_tensor = cute.nvgpu.make_tiled_tma_atom_A(
        op,
        a,
        a_smem_layout_one_stage,
        MMA_TILER_MNK,
        tiled_mma,
    )
    b_tma_atom, b_tma_tensor = cute.nvgpu.make_tiled_tma_atom_B(
        op,
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


@cute.kernel
def _route_compact_kernel(
    m_logits: cute.Tensor,
    m_bias: cute.Tensor,
    m_selected_groups: cute.Tensor,
    m_candidate_scores_for_routing: cute.Tensor,
):
    tidx, _, _ = cute.arch.thread_idx()
    row, _, _ = cute.arch.block_idx()
    warp_idx = cute.arch.warp_idx()
    lane_idx = cute.arch.lane_idx()

    smem = utils.SmemAllocator()
    s_biased = smem.allocate_tensor(
        cutlass.Float32,
        cute.make_layout((NUM_EXPERTS,), stride=(1,)),
        byte_alignment=16,
    )
    s_group_scores = smem.allocate_tensor(
        cutlass.Float32,
        cute.make_layout((N_GROUP,), stride=(1,)),
        byte_alignment=4,
    )
    s_selected_groups = smem.allocate_tensor(
        cutlass.Int32,
        cute.make_layout((TOPK_GROUP,), stride=(1,)),
        byte_alignment=4,
    )

    x = m_logits[row, tidx]
    one = cutlass.Float32(1.0)
    biased = one / (one + cmath.exp(-x)) + cutlass.Float32(m_bias[tidx])
    s_biased[tidx] = biased

    best1 = cute.arch.warp_reduction_max(biased)
    is_best1 = cutlass.Int32(0)
    if biased == best1:
        is_best1 = cutlass.Int32(1)
    max_count = cute.arch.warp_reduction_sum(is_best1)

    best2_input = biased
    if max_count < 2 and biased == best1:
        best2_input = cutlass.Float32(-float("inf"))
    best2 = best1
    if max_count < 2:
        best2 = cute.arch.warp_reduction_max(best2_input)

    if lane_idx == 0:
        s_group_scores[warp_idx] = best1 + best2

    cute.arch.sync_threads()

    if tidx == 0:
        top_groups = cute.make_rmem_tensor((TOPK_GROUP,), cutlass.Int32)
        top_scores = cute.make_rmem_tensor((TOPK_GROUP,), cutlass.Float32)

        for i in cutlass.range_constexpr(TOPK_GROUP):
            top_groups[i] = cutlass.Int32(-1)
            top_scores[i] = cutlass.Float32(-float("inf"))

        for group_idx in cutlass.range_constexpr(N_GROUP):
            group_score = s_group_scores[group_idx]
            group_idx_i32 = cutlass.Int32(group_idx)
            if group_score > top_scores[0]:
                top_scores[3] = top_scores[2]
                top_groups[3] = top_groups[2]
                top_scores[2] = top_scores[1]
                top_groups[2] = top_groups[1]
                top_scores[1] = top_scores[0]
                top_groups[1] = top_groups[0]
                top_scores[0] = group_score
                top_groups[0] = group_idx_i32
            elif group_score > top_scores[1]:
                top_scores[3] = top_scores[2]
                top_groups[3] = top_groups[2]
                top_scores[2] = top_scores[1]
                top_groups[2] = top_groups[1]
                top_scores[1] = group_score
                top_groups[1] = group_idx_i32
            elif group_score > top_scores[2]:
                top_scores[3] = top_scores[2]
                top_groups[3] = top_groups[2]
                top_scores[2] = group_score
                top_groups[2] = group_idx_i32
            elif group_score > top_scores[3]:
                top_scores[3] = group_score
                top_groups[3] = group_idx_i32

        if top_groups[1] < top_groups[0]:
            tmp_group = top_groups[0]
            top_groups[0] = top_groups[1]
            top_groups[1] = tmp_group
        if top_groups[3] < top_groups[2]:
            tmp_group = top_groups[2]
            top_groups[2] = top_groups[3]
            top_groups[3] = tmp_group
        if top_groups[2] < top_groups[0]:
            tmp_group = top_groups[0]
            top_groups[0] = top_groups[2]
            top_groups[2] = tmp_group
        if top_groups[3] < top_groups[1]:
            tmp_group = top_groups[1]
            top_groups[1] = top_groups[3]
            top_groups[3] = tmp_group
        if top_groups[2] < top_groups[1]:
            tmp_group = top_groups[1]
            top_groups[1] = top_groups[2]
            top_groups[2] = tmp_group

        for slot in cutlass.range_constexpr(TOPK_GROUP):
            s_selected_groups[slot] = top_groups[slot]

    cute.arch.sync_threads()

    if tidx < TOPK_GROUP:
        m_selected_groups[row, tidx] = s_selected_groups[tidx]

    if tidx < COMPACT_EXPERTS:
        slot = cute.arch.make_warp_uniform(tidx >> 5)
        expert_offset = tidx & (EXPERTS_PER_GROUP - 1)
        group_idx_i32 = s_selected_groups[slot]
        col = group_idx_i32 * EXPERTS_PER_GROUP + expert_offset
        m_candidate_scores_for_routing[row, tidx] = s_biased[col]


@cute.jit
def _launch_route_compact(
    m_logits: cute.Tensor,
    m_bias: cute.Tensor,
    m_selected_groups: cute.Tensor,
    m_candidate_scores_for_routing: cute.Tensor,
):
    _route_compact_kernel(
        m_logits,
        m_bias,
        m_selected_groups,
        m_candidate_scores_for_routing,
    ).launch(
        grid=[m_logits.shape[0], 1, 1],
        block=[NUM_EXPERTS, 1, 1],
    )


def _matrix_to_cute_dynamic(tensor: torch.Tensor, width_divisibility: int) -> cute.Tensor:
    return (
        from_dlpack(tensor, assumed_align=32, use_32bit_stride=True)
        .mark_layout_dynamic(leading_dim=1)
        .mark_compact_shape_dynamic(mode=1, divisibility=width_divisibility)
    )


def _matrix_to_cute_exact(tensor: torch.Tensor) -> cute.Tensor:
    return from_dlpack(tensor, assumed_align=32, use_32bit_stride=True)


def _vector_to_cute(tensor: torch.Tensor) -> cute.Tensor:
    return from_dlpack(tensor, assumed_align=32, use_32bit_stride=True)


def _workspace_key(device: torch.device, m_pad: int):
    if device.index is not None:
        device_idx = device.index
    else:
        device_idx = torch.cuda.current_device()
    return (device_idx, m_pad)


def _tail_key(device: torch.device, num_tokens: int):
    if device.index is not None:
        device_idx = device.index
    else:
        device_idx = torch.cuda.current_device()
    return (device_idx, num_tokens)


def _make_tail():
    def _tail(selected_groups, candidate_scores_for_routing, expert_bias, routed_scaling_factor):
        selected_scores_for_routing, top8_compact = torch.topk(
            candidate_scores_for_routing,
            k=NUM_EXPERTS_PER_TOK,
            dim=-1,
            sorted=False,
        )
        top8_group_slot = torch.bitwise_right_shift(top8_compact, 5)
        top8_within = torch.bitwise_and(top8_compact, EXPERTS_PER_GROUP - 1)
        top_indices = selected_groups.gather(1, top8_group_slot).to(torch.int64) * EXPERTS_PER_GROUP + top8_within
        selected_bias = torch.gather(
            expert_bias.view(1, NUM_EXPERTS).expand(top_indices.shape[0], NUM_EXPERTS),
            1,
            top_indices,
        )
        selected_scores = selected_scores_for_routing - selected_bias
        weights = selected_scores / (selected_scores.sum(dim=-1, keepdim=True) + 1e-20)
        weights = weights * routed_scaling_factor
        return top_indices, weights

    return torch.compile(_tail, mode='max-autotune-no-cudagraphs', dynamic=False)


def _get_workspace(device: torch.device, m_pad: int):
    key = _workspace_key(device, m_pad)
    workspace = _workspace_cache.get(key)
    if workspace is None:
        hidden_pad = torch.empty((m_pad, GATING_DIM), dtype=torch.bfloat16, device=device)
        logits = torch.empty((m_pad, NUM_EXPERTS), dtype=torch.float32, device=device)
        selected_groups = torch.empty((m_pad, TOPK_GROUP), dtype=torch.int32, device=device)
        candidate_scores_for_routing = torch.empty((m_pad, COMPACT_EXPERTS), dtype=torch.float32, device=device)
        workspace = {
            "hidden_pad": hidden_pad,
            "hidden_pad_cute_dynamic": _matrix_to_cute_dynamic(hidden_pad, GATING_DIM),
            "hidden_pad_cute_exact": _matrix_to_cute_exact(hidden_pad),
            "logits": logits,
            "logits_cute_dynamic": _matrix_to_cute_dynamic(logits, NUM_EXPERTS),
            "logits_cute_exact": _matrix_to_cute_exact(logits),
            "selected_groups": selected_groups,
            "selected_groups_cute_dynamic": _matrix_to_cute_dynamic(selected_groups, TOPK_GROUP),
            "selected_groups_cute_exact": _matrix_to_cute_exact(selected_groups),
            "candidate_scores_for_routing": candidate_scores_for_routing,
            "candidate_scores_for_routing_cute_dynamic": _matrix_to_cute_dynamic(
                candidate_scores_for_routing, COMPACT_EXPERTS
            ),
            "candidate_scores_for_routing_cute_exact": _matrix_to_cute_exact(
                candidate_scores_for_routing
            ),
        }
        _workspace_cache[key] = workspace
    return workspace


def _ensure_compiled_router_dynamic(
    sample_hidden: cute.Tensor,
    sample_weight: cute.Tensor,
    sample_logits: cute.Tensor,
):
    global _compiled_router_gemm_dynamic

    if _compiled_router_gemm_dynamic is None:
        _compiled_router_gemm_dynamic = cute.compile(
            _launch_router_gemm,
            sample_hidden,
            sample_weight,
            sample_logits,
        )

    return _compiled_router_gemm_dynamic


def _ensure_compiled_route_compact_dynamic(
    sample_logits: cute.Tensor,
    sample_bias: cute.Tensor,
    sample_selected_groups: cute.Tensor,
    sample_candidate_scores_for_routing: cute.Tensor,
):
    global _compiled_route_compact_dynamic

    if _compiled_route_compact_dynamic is None:
        _compiled_route_compact_dynamic = cute.compile(
            _launch_route_compact,
            sample_logits,
            sample_bias,
            sample_selected_groups,
            sample_candidate_scores_for_routing,
        )

    return _compiled_route_compact_dynamic


def _ensure_compiled_router_exact(
    compile_key,
    sample_hidden: cute.Tensor,
    sample_weight: cute.Tensor,
    sample_logits: cute.Tensor,
):
    compiled_router_gemm = _compiled_router_gemm_exact.get(compile_key)
    if compiled_router_gemm is None:
        compiled_router_gemm = cute.compile(
            _launch_router_gemm,
            sample_hidden,
            sample_weight,
            sample_logits,
        )
        _compiled_router_gemm_exact[compile_key] = compiled_router_gemm

    return compiled_router_gemm


def _ensure_compiled_route_compact_exact(
    compile_key,
    sample_logits: cute.Tensor,
    sample_bias: cute.Tensor,
    sample_selected_groups: cute.Tensor,
    sample_candidate_scores_for_routing: cute.Tensor,
):
    compiled_route_compact = _compiled_route_compact_exact.get(compile_key)
    if compiled_route_compact is None:
        compiled_route_compact = cute.compile(
            _launch_route_compact,
            sample_logits,
            sample_bias,
            sample_selected_groups,
            sample_candidate_scores_for_routing,
        )
        _compiled_route_compact_exact[compile_key] = compiled_route_compact

    return compiled_route_compact


@torch.no_grad()
def run(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    expert_bias: torch.Tensor,
    routed_scaling_factor,
    topk_idx: torch.Tensor,
    topk_weight: torch.Tensor,
) -> None:
    num_tokens = hidden_states.shape[0]
    padded_tokens = ((num_tokens + MMA_TILER_MNK[0] - 1) // MMA_TILER_MNK[0]) * MMA_TILER_MNK[0]
    workspace = _get_workspace(hidden_states.device, padded_tokens)
    compile_key = _workspace_key(hidden_states.device, padded_tokens)
    use_exact_router = _use_exact_router_frontend(padded_tokens)
    use_exact_route_compact = _use_exact_route_compact_frontend(padded_tokens)

    if padded_tokens == num_tokens:
        if use_exact_router:
            hidden_work_cute_router = _matrix_to_cute_exact(hidden_states)
        else:
            hidden_work_cute_router = _matrix_to_cute_dynamic(hidden_states, GATING_DIM)
    else:
        hidden_pad = workspace["hidden_pad"]
        hidden_pad[:num_tokens].copy_(hidden_states)
        hidden_pad[num_tokens:].zero_()
        if use_exact_router:
            hidden_work_cute_router = workspace["hidden_pad_cute_exact"]
        else:
            hidden_work_cute_router = workspace["hidden_pad_cute_dynamic"]

    if use_exact_router:
        weight_cute_router = _matrix_to_cute_exact(weight)
        logits_cute_router = workspace["logits_cute_exact"]
    else:
        weight_cute_router = _matrix_to_cute_dynamic(weight, GATING_DIM)
        logits_cute_router = workspace["logits_cute_dynamic"]

    if use_exact_route_compact:
        logits_cute_route_compact = workspace["logits_cute_exact"]
        selected_groups_cute = workspace["selected_groups_cute_exact"]
        candidate_scores_cute = workspace["candidate_scores_for_routing_cute_exact"]
    else:
        logits_cute_route_compact = workspace["logits_cute_dynamic"]
        selected_groups_cute = workspace["selected_groups_cute_dynamic"]
        candidate_scores_cute = workspace["candidate_scores_for_routing_cute_dynamic"]
    bias_cute = _vector_to_cute(expert_bias)

    if use_exact_router:
        compiled_router_gemm = _ensure_compiled_router_exact(
            compile_key,
            hidden_work_cute_router,
            weight_cute_router,
            logits_cute_router,
        )
    else:
        compiled_router_gemm = _ensure_compiled_router_dynamic(
            hidden_work_cute_router,
            weight_cute_router,
            logits_cute_router,
        )

    if use_exact_route_compact:
        compiled_route_compact = _ensure_compiled_route_compact_exact(
            compile_key,
            logits_cute_route_compact,
            bias_cute,
            selected_groups_cute,
            candidate_scores_cute,
        )
    else:
        compiled_route_compact = _ensure_compiled_route_compact_dynamic(
            logits_cute_route_compact,
            bias_cute,
            selected_groups_cute,
            candidate_scores_cute,
        )

    compiled_router_gemm(hidden_work_cute_router, weight_cute_router, logits_cute_router)
    compiled_route_compact(
        logits_cute_route_compact,
        bias_cute,
        selected_groups_cute,
        candidate_scores_cute,
    )

    selected_groups = workspace["selected_groups"][:num_tokens]
    candidate_scores_for_routing = workspace["candidate_scores_for_routing"][:num_tokens]

    tail_key = _tail_key(hidden_states.device, num_tokens)
    compiled_tail = _compiled_tail_fns.get(tail_key)
    if compiled_tail is None:
        compiled_tail = _make_tail()
        _compiled_tail_fns[tail_key] = compiled_tail

    if isinstance(routed_scaling_factor, torch.Tensor):
        scale = routed_scaling_factor.to(
            device=candidate_scores_for_routing.device,
            dtype=candidate_scores_for_routing.dtype,
        )
    else:
        scale = routed_scaling_factor
    top_indices, weights = compiled_tail(
        selected_groups,
        candidate_scores_for_routing,
        expert_bias,
        scale,
    )

    topk_idx.copy_(top_indices)
    topk_weight.copy_(weights)
