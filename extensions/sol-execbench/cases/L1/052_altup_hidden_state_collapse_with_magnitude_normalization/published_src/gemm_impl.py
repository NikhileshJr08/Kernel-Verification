import torch

import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass.cute.runtime import make_ptr

AB_DTYPE = cutlass.BFloat16
ACC_DTYPE = cutlass.Float32
MMA_INST_SHAPE_MNK = (128, 256, 16)
MMA_TILER_MNK = (128, 256, 64)
THREADS_PER_CTA = 128
AB_STAGES = 4
ACC_STAGE = 1
M_TILE = MMA_TILER_MNK[0]
N_TILE = MMA_TILER_MNK[1]
K_TILE = MMA_TILER_MNK[2]


@cute.struct
class SharedStorage:
    ab_mbar_ptr: cute.struct.MemRange[cutlass.Int64, AB_STAGES * 2]
    acc_mbar_ptr: cute.struct.MemRange[cutlass.Int64, ACC_STAGE * 2]
    tmem_holding_buf: cutlass.Int32


@cute.kernel
def _gemm_kernel(
    tiled_mma: cute.TiledMma,
    tma_atom_a: cute.CopyAtom,
    mA_mkl: cute.Tensor,
    tma_atom_b: cute.CopyAtom,
    mB_nkl: cute.Tensor,
    mC_mnl: cute.Tensor,
    a_smem_layout: cute.ComposedLayout,
    b_smem_layout: cute.ComposedLayout,
    out_dtype: cutlass.Constexpr,
):
    tidx, _, _ = cute.arch.thread_idx()
    warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    bidx, bidy, _ = cute.arch.block_idx()
    mma_coord_mnk = (bidx, bidy, None)

    smem = utils.SmemAllocator()
    storage = smem.allocate(SharedStorage)
    sA = smem.allocate_tensor(
        element_type=AB_DTYPE,
        layout=a_smem_layout.outer,
        byte_alignment=128,
        swizzle=a_smem_layout.inner,
    )
    sB = smem.allocate_tensor(
        element_type=AB_DTYPE,
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
    # Matches tutorial_gemm/fp16_gemm_0.py; intentionally over-allocates for simplicity.
    tmem.allocate(512)

    if warp_idx == 0:
        cpasync.prefetch_descriptor(tma_atom_a)
        cpasync.prefetch_descriptor(tma_atom_b)

    num_tma_copy_bytes = cute.size_in_bytes(
        AB_DTYPE, cute.select(a_smem_layout, mode=[0, 1, 2])
    ) + cute.size_in_bytes(AB_DTYPE, cute.select(b_smem_layout, mode=[0, 1, 2]))
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
        (cute.size(tCtAcc, mode=[0, 0]), cute.size(tCtAcc, mode=[0, 1]) // subtile_cnt),
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
    tCrC = cute.make_rmem_tensor(tDgC[None, None, 0].shape, out_dtype)

    num_k_tiles = cute.size(gA, mode=[2])
    if warp_idx == 0:
        acc_empty = acc_producer.acquire_and_advance()
        for _k_tile_idx in cutlass.range(num_k_tiles, prefetch_stages=AB_STAGES - 2):
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
        tCrC.store(tCrAcc.load().to(out_dtype))
        cute.autovec_copy(tCrC, tDgC[None, None, i])
    acc_full.release()

    pipeline.sync(barrier_id=1)
    tmem.free(tmem_ptr)


@cute.jit
def _gemm_host(a: cute.Tensor, b: cute.Tensor, c: cute.Tensor, out_dtype: cutlass.Constexpr):
    op = tcgen05.MmaF16BF16Op(
        AB_DTYPE,
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

    copy_op = cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.ONE)
    a_tma_atom, a_tma_tensor = cute.nvgpu.make_tiled_tma_atom_A(
        copy_op,
        a,
        a_smem_layout_one_stage,
        MMA_TILER_MNK,
        tiled_mma,
    )
    b_tma_atom, b_tma_tensor = cute.nvgpu.make_tiled_tma_atom_B(
        copy_op,
        b,
        b_smem_layout_one_stage,
        MMA_TILER_MNK,
        tiled_mma,
    )

    grid_shape = cute.ceil_div((*c.layout.shape, 1), MMA_TILER_MNK[:2])
    _gemm_kernel(
        tiled_mma,
        a_tma_atom,
        a_tma_tensor,
        b_tma_atom,
        b_tma_tensor,
        c,
        a_smem_layout,
        b_smem_layout,
        out_dtype,
    ).launch(
        grid=grid_shape,
        block=(THREADS_PER_CTA, 1, 1),
    )


_compiled_gemm = {}
_compiled_gemm_static = {}
_KNOWN_MS = {
    128,
    131,
    256,
    422,
    613,
    2048,
    2984,
    4096,
    7184,
    8192,
    16384,
    17312,
}


def get_gemm_executor(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor):
    k_dim = int(a.shape[1])
    n_dim = int(b.shape[0])
    key = (k_dim, n_dim, c.dtype)
    executor = _compiled_gemm.get(key)
    if executor is None:
        c_type = cutlass.BFloat16 if c.dtype == torch.bfloat16 else cutlass.Float32

        @cute.jit
        def host_ptr_function(
            a_ptr: cute.Pointer,
            b_ptr: cute.Pointer,
            c_ptr: cute.Pointer,
            m: cutlass.Int32,
        ):
            a_tensor = cute.make_tensor(
                a_ptr,
                cute.make_layout((m, k_dim), stride=(k_dim, 1)),
            )
            b_tensor = cute.make_tensor(
                b_ptr,
                cute.make_layout((n_dim, k_dim), stride=(k_dim, 1)),
            )
            c_tensor = cute.make_tensor(
                c_ptr,
                cute.make_layout((m, n_dim), stride=(n_dim, 1)),
            )
            _gemm_host(a_tensor, b_tensor, c_tensor, c_type)

        executor = cute.compile(
            host_ptr_function,
            make_ptr(AB_DTYPE, 16, cute.AddressSpace.gmem, assumed_align=16),
            make_ptr(AB_DTYPE, 16, cute.AddressSpace.gmem, assumed_align=16),
            make_ptr(c_type, 16, cute.AddressSpace.gmem, assumed_align=16),
            cutlass.Int32(1),
        )
        _compiled_gemm[key] = executor
    return executor


def get_gemm_executor_static(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor):
    m_dim = int(a.shape[0])
    k_dim = int(a.shape[1])
    n_dim = int(b.shape[0])
    key = (m_dim, k_dim, n_dim, c.dtype)
    executor = _compiled_gemm_static.get(key)
    if executor is None:
        c_type = cutlass.BFloat16 if c.dtype == torch.bfloat16 else cutlass.Float32

        @cute.jit
        def host_ptr_function_static(
            a_ptr: cute.Pointer,
            b_ptr: cute.Pointer,
            c_ptr: cute.Pointer,
        ):
            a_tensor = cute.make_tensor(
                a_ptr,
                cute.make_layout((m_dim, k_dim), stride=(k_dim, 1)),
            )
            b_tensor = cute.make_tensor(
                b_ptr,
                cute.make_layout((n_dim, k_dim), stride=(k_dim, 1)),
            )
            c_tensor = cute.make_tensor(
                c_ptr,
                cute.make_layout((m_dim, n_dim), stride=(n_dim, 1)),
            )
            _gemm_host(a_tensor, b_tensor, c_tensor, c_type)

        executor = cute.compile(
            host_ptr_function_static,
            make_ptr(AB_DTYPE, 16, cute.AddressSpace.gmem, assumed_align=16),
            make_ptr(AB_DTYPE, 16, cute.AddressSpace.gmem, assumed_align=16),
            make_ptr(c_type, 16, cute.AddressSpace.gmem, assumed_align=16),
        )
        _compiled_gemm_static[key] = executor
    return executor


def gemm_nt(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> None:
    a_ptr = make_ptr(AB_DTYPE, a.data_ptr(), cute.AddressSpace.gmem, assumed_align=16)
    b_ptr = make_ptr(AB_DTYPE, b.data_ptr(), cute.AddressSpace.gmem, assumed_align=16)
    c_type = cutlass.BFloat16 if c.dtype == torch.bfloat16 else cutlass.Float32
    c_ptr = make_ptr(c_type, c.data_ptr(), cute.AddressSpace.gmem, assumed_align=16)
    m = int(a.shape[0])
    if m in _KNOWN_MS:
        executor = get_gemm_executor_static(a, b, c)
        executor(a_ptr, b_ptr, c_ptr)
    else:
        executor = get_gemm_executor(a, b, c)
        executor(a_ptr, b_ptr, c_ptr, cutlass.Int32(m))


def linear_bf16_input(a_2d: torch.Tensor, weight_nk: torch.Tensor, out_2d: torch.Tensor) -> None:
    assert a_2d.ndim == 2
    assert weight_nk.ndim == 2
    assert out_2d.ndim == 2
    m, k = a_2d.shape
    n, k_weight = weight_nk.shape
    if k != k_weight:
        raise ValueError(f"K mismatch: {k} vs {k_weight}")
    if out_2d.shape != (m, n):
        raise ValueError(f"Output shape mismatch: got {tuple(out_2d.shape)}, expected {(m, n)}")

    gemm_nt(a_2d, weight_nk, out_2d)
