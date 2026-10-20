from typing import Optional, Type, Tuple, Union

import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import cutlass.pipeline as pipeline
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait
from cutlass.cute.nvgpu import cpasync, tcgen05
import cutlass.utils.blackwell_helpers as sm100_utils

class DenseGemmKernel:

    def __init__(self, acc_dtype: Type[cutlass.Numeric], use_2cta_instrs: bool, mma_tiler_mn: Tuple[int, int], cluster_shape_mn: Tuple[int, int], use_tma_store: bool):
        self.acc_dtype: Type[cutlass.Numeric] = acc_dtype
        self.use_2cta_instrs = use_2cta_instrs
        self.cluster_shape_mn = cluster_shape_mn
        self.mma_tiler_mn = mma_tiler_mn
        self.mma_tiler = (*mma_tiler_mn, 1)
        self.use_tma_store = use_tma_store
        self.cta_group = tcgen05.CtaGroup.TWO if self.use_2cta_instrs else tcgen05.CtaGroup.ONE
        self.occupancy = 1
        self.threads_per_cta = 128
        self.smem_capacity = utils.get_smem_capacity_in_bytes('sm_100')

    def _setup_attributes(self):
        tiled_mma = sm100_utils.make_trivial_tiled_mma(self.a_dtype, self.a_major_mode, self.b_major_mode, self.acc_dtype, self.cta_group, self.mma_tiler[:2])
        mma_inst_shape_k = cute.size(tiled_mma.shape_mnk, mode=[2])
        mma_inst_tile_k = 4
        self.mma_tiler = (self.mma_tiler[0], self.mma_tiler[1], mma_inst_shape_k * mma_inst_tile_k)
        self.cta_tile_shape_mnk = (self.mma_tiler[0] // cute.size(tiled_mma.thr_id.shape), self.mma_tiler[1], self.mma_tiler[2])
        self.cluster_layout_vmnk = cute.tiled_divide(cute.make_layout((*self.cluster_shape_mn, 1)), (tiled_mma.thr_id.shape,))
        self.num_mcast_ctas_a = cute.size(self.cluster_layout_vmnk.shape[2])
        self.num_mcast_ctas_b = cute.size(self.cluster_layout_vmnk.shape[1])
        self.is_a_mcast = self.num_mcast_ctas_a > 1
        self.is_b_mcast = self.num_mcast_ctas_b > 1
        if cutlass.const_expr(self.use_tma_store):
            self.epi_tile = sm100_utils.compute_epilogue_tile_shape(self.cta_tile_shape_mnk, self.use_2cta_instrs, self.c_layout, self.c_dtype)
        else:
            self.epi_tile = self.cta_tile_shape_mnk[:2]
        (self.num_acc_stage, self.num_ab_stage, self.num_c_stage) = self._compute_stages(tiled_mma, self.mma_tiler, self.a_dtype, self.b_dtype, self.epi_tile, self.c_dtype, self.c_layout, self.smem_capacity, self.occupancy, self.use_tma_store)
        self.a_smem_layout_staged = sm100_utils.make_smem_layout_a(tiled_mma, self.mma_tiler, self.a_dtype, self.num_ab_stage)
        self.b_smem_layout_staged = sm100_utils.make_smem_layout_b(tiled_mma, self.mma_tiler, self.b_dtype, self.num_ab_stage)
        self.c_smem_layout_staged = sm100_utils.make_smem_layout_epi(self.c_dtype, self.c_layout, self.epi_tile, self.num_c_stage) if self.use_tma_store else None
        self.num_tmem_alloc_cols = self._compute_num_tmem_alloc_cols(tiled_mma, self.mma_tiler)

    @cute.jit
    def __call__(self, a: cute.Tensor, b: cute.Tensor, c: cute.Tensor, epilogue_op: cutlass.Constexpr=lambda x: x):
        self.a_dtype: Type[cutlass.Numeric] = a.element_type
        self.b_dtype: Type[cutlass.Numeric] = b.element_type
        self.c_dtype: Type[cutlass.Numeric] = c.element_type
        self.a_major_mode = utils.LayoutEnum.from_tensor(a).mma_major_mode()
        self.b_major_mode = utils.LayoutEnum.from_tensor(b).mma_major_mode()
        self.c_layout = utils.LayoutEnum.from_tensor(c)
        if cutlass.const_expr(self.a_dtype != self.b_dtype):
            raise TypeError(f'Type must match: {self.a_dtype} != {self.b_dtype}')
        self._setup_attributes()
        tiled_mma = sm100_utils.make_trivial_tiled_mma(self.a_dtype, self.a_major_mode, self.b_major_mode, self.acc_dtype, self.cta_group, self.mma_tiler[:2])
        atom_thr_size = cute.size(tiled_mma.thr_id.shape)
        a_op = sm100_utils.cluster_shape_to_tma_atom_A(self.cluster_shape_mn, tiled_mma.thr_id)
        a_smem_layout = cute.slice_(self.a_smem_layout_staged, (None, None, None, 0))
        (tma_atom_a, tma_tensor_a) = cute.nvgpu.make_tiled_tma_atom_A(a_op, a, a_smem_layout, self.mma_tiler, tiled_mma, self.cluster_layout_vmnk.shape, internal_type=cutlass.TFloat32 if a.element_type is cutlass.Float32 else None)
        b_op = sm100_utils.cluster_shape_to_tma_atom_B(self.cluster_shape_mn, tiled_mma.thr_id)
        b_smem_layout = cute.slice_(self.b_smem_layout_staged, (None, None, None, 0))
        (tma_atom_b, tma_tensor_b) = cute.nvgpu.make_tiled_tma_atom_B(b_op, b, b_smem_layout, self.mma_tiler, tiled_mma, self.cluster_layout_vmnk.shape, internal_type=cutlass.TFloat32 if b.element_type is cutlass.Float32 else None)
        a_copy_size = cute.size_in_bytes(self.a_dtype, a_smem_layout)
        b_copy_size = cute.size_in_bytes(self.b_dtype, b_smem_layout)
        self.num_tma_load_bytes = (a_copy_size + b_copy_size) * atom_thr_size
        tma_atom_c = None
        tma_tensor_c = None
        if cutlass.const_expr(self.use_tma_store):
            epi_smem_layout = cute.slice_(self.c_smem_layout_staged, (None, None, 0))
            (tma_atom_c, tma_tensor_c) = cpasync.make_tiled_tma_atom(cpasync.CopyBulkTensorTileS2GOp(), c, epi_smem_layout, self.epi_tile)
        grid = self._compute_grid(c, self.cta_tile_shape_mnk, self.cluster_shape_mn)
        self.kernel(tiled_mma, tma_atom_a, tma_tensor_a, tma_atom_b, tma_tensor_b, tma_atom_c, tma_tensor_c if self.use_tma_store else c, self.cluster_layout_vmnk, self.a_smem_layout_staged, self.b_smem_layout_staged, self.c_smem_layout_staged, self.epi_tile, epilogue_op).launch(grid=grid, block=[self.threads_per_cta, 1, 1], cluster=(*self.cluster_shape_mn, 1))
        return

    @cute.kernel
    def kernel(self, tiled_mma: cute.TiledMma, tma_atom_a: cute.CopyAtom, mA_mkl: cute.Tensor, tma_atom_b: cute.CopyAtom, mB_nkl: cute.Tensor, tma_atom_c: Optional[cute.CopyAtom], mC_mnl: cute.Tensor, cluster_layout_vmnk: cute.Layout, a_smem_layout_staged: cute.ComposedLayout, b_smem_layout_staged: cute.ComposedLayout, c_smem_layout_staged: Union[cute.Layout, cute.ComposedLayout, None], epi_tile: cute.Tile, epilogue_op: cutlass.Constexpr):
        warp_idx = cute.arch.warp_idx()
        warp_idx = cute.arch.make_warp_uniform(warp_idx)
        if warp_idx == 0:
            cpasync.prefetch_descriptor(tma_atom_a)
            cpasync.prefetch_descriptor(tma_atom_b)
            if cutlass.const_expr(self.use_tma_store):
                cpasync.prefetch_descriptor(tma_atom_c)
        use_2cta_instrs = cute.size(tiled_mma.thr_id.shape) == 2
        (bidx, bidy, bidz) = cute.arch.block_idx()
        mma_tile_coord_v = bidx % cute.size(tiled_mma.thr_id.shape)
        is_leader_cta = mma_tile_coord_v == 0
        cta_rank_in_cluster = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        block_in_cluster_coord_vmnk = cluster_layout_vmnk.get_flat_coord(cta_rank_in_cluster)
        cta_coord = (bidx, bidy, bidz)
        mma_tile_coord_mnl = (cta_coord[0] // cute.size(tiled_mma.thr_id.shape), cta_coord[1], cta_coord[2])
        (tidx, _, _) = cute.arch.thread_idx()

        @cute.struct
        class SharedStorage:
            ab_full_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_ab_stage * 2]
            acc_full_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_acc_stage * 2]
            tmem_dealloc_mbar_ptr: cutlass.Int64
            tmem_holding_buf: cutlass.Int32
        smem = utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)
        ab_pipeline_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        num_tma_producer = self.num_mcast_ctas_a + self.num_mcast_ctas_b - 1
        ab_pipeline_consumer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, num_tma_producer)
        (ab_producer, ab_consumer) = pipeline.PipelineTmaUmma.create(barrier_storage=storage.ab_full_mbar_ptr.data_ptr(), num_stages=self.num_ab_stage, producer_group=ab_pipeline_producer_group, consumer_group=ab_pipeline_consumer_group, tx_count=self.num_tma_load_bytes, cta_layout_vmnk=cluster_layout_vmnk, defer_sync=True).make_participants()
        acc_pipeline_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        acc_pipeline_consumer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, self.threads_per_cta)
        acc_pipeline = pipeline.PipelineUmmaAsync.create(barrier_storage=storage.acc_full_mbar_ptr.data_ptr(), num_stages=self.num_acc_stage, producer_group=acc_pipeline_producer_group, consumer_group=acc_pipeline_consumer_group, cta_layout_vmnk=cluster_layout_vmnk, defer_sync=True)
        acc_producer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.num_acc_stage)
        acc_consumer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.num_acc_stage)
        tmem_alloc_barrier = pipeline.NamedBarrier(barrier_id=0, num_threads=self.threads_per_cta)
        tmem = utils.TmemAllocator(storage.tmem_holding_buf, barrier_for_retrieve=tmem_alloc_barrier, is_two_cta=use_2cta_instrs, two_cta_tmem_dealloc_mbar_ptr=storage.tmem_dealloc_mbar_ptr)
        pipeline_init_arrive(cluster_shape_mn=self.cluster_shape_mn, is_relaxed=True)
        sC = None
        if cutlass.const_expr(self.use_tma_store):
            sC = smem.allocate_tensor(element_type=self.c_dtype, layout=c_smem_layout_staged.outer, byte_alignment=128, swizzle=c_smem_layout_staged.inner)
        sA = smem.allocate_tensor(element_type=self.a_dtype, layout=a_smem_layout_staged.outer, byte_alignment=128, swizzle=a_smem_layout_staged.inner)
        sB = smem.allocate_tensor(element_type=self.b_dtype, layout=b_smem_layout_staged.outer, byte_alignment=128, swizzle=b_smem_layout_staged.inner)
        a_full_mcast_mask = None
        b_full_mcast_mask = None
        if cutlass.const_expr(self.is_a_mcast or self.is_b_mcast or use_2cta_instrs):
            a_full_mcast_mask = cpasync.create_tma_multicast_mask(cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=2)
            b_full_mcast_mask = cpasync.create_tma_multicast_mask(cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=1)
        gA_mkl = cute.local_tile(mA_mkl, cute.slice_(self.mma_tiler, (None, 0, None)), (None, None, None))
        gB_nkl = cute.local_tile(mB_nkl, cute.slice_(self.mma_tiler, (0, None, None)), (None, None, None))
        gC_mnl = cute.local_tile(mC_mnl, cute.slice_(self.mma_tiler, (None, None, 0)), (None, None, None))
        k_tile_cnt = cute.size(gA_mkl, mode=[3])
        thr_mma = tiled_mma.get_slice(mma_tile_coord_v)
        tCgA = thr_mma.partition_A(gA_mkl)
        tCgB = thr_mma.partition_B(gB_nkl)
        tCgC = thr_mma.partition_C(gC_mnl)
        a_cta_layout = cute.make_layout(cute.slice_(cluster_layout_vmnk, (0, 0, None, 0)).shape)
        (tAsA, tAgA) = cpasync.tma_partition(tma_atom_a, block_in_cluster_coord_vmnk[2], a_cta_layout, cute.group_modes(sA, 0, 3), cute.group_modes(tCgA, 0, 3))
        b_cta_layout = cute.make_layout(cute.slice_(cluster_layout_vmnk, (0, None, 0, 0)).shape)
        (tBsB, tBgB) = cpasync.tma_partition(tma_atom_b, block_in_cluster_coord_vmnk[1], b_cta_layout, cute.group_modes(sB, 0, 3), cute.group_modes(tCgB, 0, 3))
        tCrA = tiled_mma.make_fragment_A(sA)
        tCrB = tiled_mma.make_fragment_B(sB)
        acc_shape = tiled_mma.partition_shape_C(self.mma_tiler[:2])
        tCtAcc_fake = tiled_mma.make_fragment_C(acc_shape)
        pipeline_init_wait(cluster_shape_mn=self.cluster_shape_mn)
        tmem.allocate(self.num_tmem_alloc_cols)
        tmem.wait_for_alloc()
        tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
        tCtAcc = cute.make_tensor(tmem_ptr, tCtAcc_fake.layout)
        tAgA = tAgA[None, mma_tile_coord_mnl[0], None, mma_tile_coord_mnl[2]]
        tBgB = tBgB[None, mma_tile_coord_mnl[1], None, mma_tile_coord_mnl[2]]
        if warp_idx == 0:
            for k_tile in cutlass.range(k_tile_cnt, prefetch_stages=self.num_ab_stage - 2):
                producer_handle = ab_producer.acquire_and_advance()
                cute.copy(tma_atom_a, tAgA[None, producer_handle.count], tAsA[None, producer_handle.index], tma_bar_ptr=producer_handle.barrier, mcast_mask=a_full_mcast_mask)
                cute.copy(tma_atom_b, tBgB[None, producer_handle.count], tBsB[None, producer_handle.index], tma_bar_ptr=producer_handle.barrier, mcast_mask=b_full_mcast_mask)
                if is_leader_cta:
                    consumer_handle = ab_consumer.wait_and_advance()
                    num_kblks = cute.size(tCrA, mode=[2])
                    for kblk_idx in cutlass.range(num_kblks, unroll_full=True):
                        kblk_crd = (None, None, kblk_idx, consumer_handle.index)
                        cute.gemm(tiled_mma, tCtAcc, tCrA[kblk_crd], tCrB[kblk_crd], tCtAcc)
                        tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                    consumer_handle.release()
            if is_leader_cta:
                acc_pipeline.producer_commit(acc_producer_state)
        tmem.relinquish_alloc_permit()
        acc_pipeline.consumer_wait(acc_consumer_state)
        if cutlass.const_expr(self.use_tma_store):
            assert tma_atom_c is not None and sC is not None
            self.epilogue_tma_store(tidx, warp_idx, mma_tile_coord_mnl, tma_atom_c, tCtAcc, sC, tCgC, epi_tile, epilogue_op)
        else:
            self.epilogue(tidx, mma_tile_coord_mnl, tCtAcc, tCgC, epi_tile, epilogue_op)
        pipeline.sync(barrier_id=1)
        tmem.free(tmem_ptr)
        if warp_idx == 0:
            ab_producer.tail()
        return

    def epilog_tmem_copy_and_partition(self, tidx: cutlass.Int32, tAcc: cute.Tensor, gC_mnl: cute.Tensor, epi_tile: cute.Tile, use_2cta_instrs: Union[cutlass.Boolean, bool]) -> Tuple[cute.TiledCopy, cute.Tensor, cute.Tensor]:
        copy_atom_t2r = sm100_utils.get_tmem_load_op(self.cta_tile_shape_mnk, self.c_layout, self.c_dtype, self.acc_dtype, epi_tile, use_2cta_instrs)
        tAcc_epi = cute.flat_divide(tAcc[(None, None), 0, 0], epi_tile)
        tiled_copy_t2r = tcgen05.make_tmem_copy(copy_atom_t2r, tAcc_epi[None, None, 0, 0])
        thr_copy_t2r = tiled_copy_t2r.get_slice(tidx)
        tTR_tAcc = thr_copy_t2r.partition_S(tAcc_epi)
        gC_mnl_epi = cute.flat_divide(gC_mnl[(None, None), 0, 0, None, None, None], epi_tile)
        tTR_gC = thr_copy_t2r.partition_D(gC_mnl_epi)
        tTR_rAcc = cute.make_rmem_tensor(tTR_gC[None, None, None, 0, 0, 0, 0, 0].shape, self.acc_dtype)
        return (tiled_copy_t2r, tTR_tAcc, tTR_rAcc)

    def epilog_smem_copy_and_partition(self, tiled_copy_t2r: cute.TiledCopy, tTR_rC: cute.Tensor, tidx: cutlass.Int32, sC: cute.Tensor) -> Tuple[cute.TiledCopy, cute.Tensor, cute.Tensor]:
        copy_atom_r2s = sm100_utils.get_smem_store_op(self.c_layout, self.c_dtype, self.acc_dtype, tiled_copy_t2r)
        tiled_copy_r2s = cute.make_tiled_copy_D(copy_atom_r2s, tiled_copy_t2r)
        thr_copy_r2s = tiled_copy_r2s.get_slice(tidx)
        tRS_sC = thr_copy_r2s.partition_D(sC)
        tRS_rC = tiled_copy_r2s.retile(tTR_rC)
        return (tiled_copy_r2s, tRS_rC, tRS_sC)

    @cute.jit
    def epilogue_tma_store(self, epi_tidx: cutlass.Int32, warp_idx: cutlass.Int32, mma_tile_coord_mnl: Tuple[cutlass.Int32, cutlass.Int32, cutlass.Int32], tma_atom_c: cute.CopyAtom, tCtAcc: cute.Tensor, sC: cute.Tensor, tCgC: cute.Tensor, epi_tile: cute.Tile, epilogue_op: cutlass.Constexpr) -> None:
        (tiled_copy_t2r, tTR_tAcc, tTR_rAcc) = self.epilog_tmem_copy_and_partition(epi_tidx, tCtAcc, tCgC, epi_tile, self.use_2cta_instrs)
        tTR_tAcc = cute.group_modes(tTR_tAcc, 3, cute.rank(tTR_tAcc))
        tTR_rC = cute.make_rmem_tensor(tTR_rAcc.shape, self.c_dtype)
        (tiled_copy_r2s, tRS_rC, tRS_sC) = self.epilog_smem_copy_and_partition(tiled_copy_t2r, tTR_rC, epi_tidx, sC)
        tCgC_epi = cute.flat_divide(tCgC[(None, None), 0, 0, None, None, None], epi_tile)
        (bSG_sC, bSG_gC) = cpasync.tma_partition(tma_atom_c, 0, cute.make_layout(1), cute.group_modes(sC, 0, 2), cute.group_modes(tCgC_epi, 0, 2))
        bSG_gC = bSG_gC[(None, None, None, *mma_tile_coord_mnl)]
        bSG_gC = cute.group_modes(bSG_gC, 1, cute.rank(bSG_gC))
        c_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, self.threads_per_cta)
        c_pipeline = pipeline.PipelineTmaStore.create(num_stages=self.num_c_stage, producer_group=c_producer_group)
        subtile_cnt = cute.size(tTR_tAcc.shape, mode=[3])
        for subtile_idx in cutlass.range(subtile_cnt):
            tTR_tAcc_mn = tTR_tAcc[None, None, None, subtile_idx]
            cute.copy(tiled_copy_t2r, tTR_tAcc_mn, tTR_rAcc)
            acc_vec = tiled_copy_r2s.retile(tTR_rAcc).load()
            acc_vec = epilogue_op(acc_vec.to(self.c_dtype))
            tRS_rC.store(acc_vec)
            c_buffer = subtile_idx % self.num_c_stage
            cute.copy(tiled_copy_r2s, tRS_rC, tRS_sC[None, None, None, c_buffer])
            cute.arch.fence_proxy('async.shared', space='cta')
            pipeline.sync(barrier_id=1)
            if warp_idx == 0:
                cute.copy(tma_atom_c, bSG_sC[None, c_buffer], bSG_gC[None, subtile_idx])
                c_pipeline.producer_commit()
                c_pipeline.producer_acquire()
            pipeline.sync(barrier_id=1)
        c_pipeline.producer_tail()

    @cute.jit
    def epilogue(self, epi_tidx: cutlass.Int32, mma_tile_coord_mnl: Tuple[cutlass.Int32, cutlass.Int32, cutlass.Int32], tCtAcc: cute.Tensor, tCgC: cute.Tensor, epi_tile: cute.Tile, epilogue_op: cutlass.Constexpr) -> None:
        (tiled_copy_t2r, tTR_tAcc, tTR_rAcc) = self.epilog_tmem_copy_and_partition(epi_tidx, tCtAcc, tCgC, epi_tile, self.use_2cta_instrs)
        tTR_tAcc = cute.group_modes(tTR_tAcc, 3, cute.rank(tTR_tAcc))
        tCgC_epi = cute.flat_divide(tCgC[(None, None), 0, 0, None, None, None], epi_tile)
        thr_copy_t2r = tiled_copy_t2r.get_slice(epi_tidx)
        tTR_gC = thr_copy_t2r.partition_D(tCgC_epi)
        tTR_rC = cute.make_rmem_tensor(tTR_gC[None, None, None, 0, 0, 0, 0, 0].shape, self.c_dtype)
        simt_atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), self.c_dtype)
        tTR_gC = tTR_gC[(None, None, None, None, None, *mma_tile_coord_mnl)]
        tTR_gC = cute.group_modes(tTR_gC, 3, cute.rank(tTR_gC))
        subtile_cnt = cute.size(tTR_tAcc.shape, mode=[3])
        for subtile_idx in cutlass.range(subtile_cnt):
            tTR_tAcc_mn = tTR_tAcc[None, None, None, subtile_idx]
            cute.copy(tiled_copy_t2r, tTR_tAcc_mn, tTR_rAcc)
            acc_vec = tTR_rAcc.load()
            acc_vec = epilogue_op(acc_vec.to(self.c_dtype))
            tTR_rC.store(acc_vec)
            cute.copy(simt_atom, tTR_rC, tTR_gC[None, None, None, subtile_idx])

    @staticmethod
    def _compute_stages(tiled_mma: cute.TiledMma, mma_tiler_mnk: Tuple[int, int, int], a_dtype: Type[cutlass.Numeric], b_dtype: Type[cutlass.Numeric], epi_tile: cute.Tile, c_dtype: Type[cutlass.Numeric], c_layout: utils.LayoutEnum, smem_capacity: int, occupancy: int, use_tma_store: bool) -> Tuple[int, int, int]:
        num_acc_stage = 1
        num_c_stage = 2 if use_tma_store else 0
        a_smem_layout_stage_one = sm100_utils.make_smem_layout_a(tiled_mma, mma_tiler_mnk, a_dtype, 1)
        b_smem_layout_staged_one = sm100_utils.make_smem_layout_b(tiled_mma, mma_tiler_mnk, b_dtype, 1)
        c_smem_layout_staged_one = sm100_utils.make_smem_layout_epi(c_dtype, c_layout, epi_tile, 1) if use_tma_store else None
        ab_bytes_per_stage = cute.size_in_bytes(a_dtype, a_smem_layout_stage_one) + cute.size_in_bytes(b_dtype, b_smem_layout_staged_one)
        mbar_helpers_bytes = 1024
        c_bytes_per_stage = cute.size_in_bytes(c_dtype, c_smem_layout_staged_one) if use_tma_store else 0
        c_bytes = c_bytes_per_stage * num_c_stage
        num_ab_stage = (smem_capacity - (occupancy + 1) * (mbar_helpers_bytes + c_bytes)) // ab_bytes_per_stage
        if use_tma_store:
            num_c_stage += (smem_capacity - ab_bytes_per_stage * num_ab_stage - (occupancy + 1) * (mbar_helpers_bytes + c_bytes)) // ((occupancy + 1) * c_bytes_per_stage)
        return (num_acc_stage, num_ab_stage, num_c_stage)

    @staticmethod
    def _compute_grid(c: cute.Tensor, cta_tile_shape_mnk: Tuple[int, int, int], cluster_shape_mn: Tuple[int, int]) -> Tuple[int, int, int]:
        cluster_shape_mnl = (*cluster_shape_mn, 1)
        grid = cute.round_up((cute.ceil_div(c.layout.shape[0], cta_tile_shape_mnk[0]), cute.ceil_div(c.layout.shape[1], cta_tile_shape_mnk[1]), c.layout.shape[2]), cluster_shape_mnl)
        return grid

    @staticmethod
    def _compute_num_tmem_alloc_cols(tiled_mma: cute.TiledMma, mma_tiler: Tuple[int, int, int]) -> int:
        acc_shape = tiled_mma.partition_shape_C(mma_tiler[:2])
        tCtAcc_fake = tiled_mma.make_fragment_C(acc_shape)
        return utils.get_num_tmem_alloc_cols(tCtAcc_fake)

    def is_valid_dtypes(self, ab_dtype: Type[cutlass.Numeric], c_dtype: Type[cutlass.Numeric]) -> bool:
        valid_ab_dtypes = {cutlass.Float16, cutlass.BFloat16, cutlass.TFloat32, cutlass.Uint8, cutlass.Int8, cutlass.Float8E4M3FN, cutlass.Float8E5M2}
        if ab_dtype not in valid_ab_dtypes:
            return False
        if self.acc_dtype not in {cutlass.Float32, cutlass.Float16, cutlass.Int32}:
            return False
        acc_ab_compatibility = {cutlass.Float32: {cutlass.Float16, cutlass.BFloat16, cutlass.TFloat32, cutlass.Float8E4M3FN, cutlass.Float8E5M2}, cutlass.Float16: {cutlass.Float16, cutlass.Float8E4M3FN, cutlass.Float8E5M2}, cutlass.Int32: {cutlass.Uint8, cutlass.Int8}}
        if ab_dtype not in acc_ab_compatibility[self.acc_dtype]:
            return False
        acc_c_compatibility = {cutlass.Float32: {cutlass.Float32, cutlass.Float16, cutlass.BFloat16, cutlass.Float8E4M3FN, cutlass.Float8E5M2, cutlass.Int32, cutlass.Int8, cutlass.Uint8}, cutlass.Float16: {cutlass.BFloat16, cutlass.Float16}, cutlass.Int32: {cutlass.BFloat16, cutlass.Float16, cutlass.Float32, cutlass.Int32, cutlass.Int8, cutlass.Uint8}}
        if c_dtype not in acc_c_compatibility[self.acc_dtype]:
            return False
        return True

    def is_valid_mma_tiler_and_cluster_shape(self) -> bool:
        is_valid = True
        if not (not self.use_2cta_instrs and self.mma_tiler_mn[0] in [64, 128] or (self.use_2cta_instrs and self.mma_tiler_mn[0] in [128, 256])):
            is_valid = False
        if self.mma_tiler_mn[1] not in range(32, 257, 32):
            is_valid = False
        if self.cluster_shape_mn[0] % (2 if self.use_2cta_instrs else 1) != 0:
            is_valid = False
        is_power_of_2 = lambda x: x > 0 and x & x - 1 == 0
        if self.cluster_shape_mn[0] * self.cluster_shape_mn[1] > 16 or self.cluster_shape_mn[0] <= 0 or self.cluster_shape_mn[1] <= 0 or (not is_power_of_2(self.cluster_shape_mn[0])) or (not is_power_of_2(self.cluster_shape_mn[1])):
            is_valid = False
        return is_valid

    def is_valid_tensor_alignment(self, m: int, n: int, k: int, l: int, ab_dtype: Type[cutlass.Numeric], c_dtype: Type[cutlass.Numeric], a_major: str, b_major: str, c_major: str) -> bool:
        is_valid = True

        def check_contiguous_16B_alignment(dtype, is_mode0_major, tensor_shape):
            major_mode_idx = 0 if is_mode0_major else 1
            num_major_elements = tensor_shape[major_mode_idx]
            num_contiguous_elements = 16 * 8 // dtype.width
            return num_major_elements % num_contiguous_elements == 0
        if not check_contiguous_16B_alignment(ab_dtype, a_major == 'm', (m, k, l)) or not check_contiguous_16B_alignment(ab_dtype, b_major == 'n', (n, k, l)) or (not check_contiguous_16B_alignment(c_dtype, c_major == 'm', (m, n, l))):
            is_valid = False
        return is_valid

    def is_valid_epilog_store_option(self, m: int, n: int) -> bool:
        is_valid = True
        cta_tile_shape_mn = (self.mma_tiler_mn[0] // (2 if self.use_2cta_instrs else 1), self.mma_tiler_mn[1])
        if not self.use_tma_store:
            if not (m % cta_tile_shape_mn[0] == 0 and n % cta_tile_shape_mn[1] == 0):
                is_valid = False
        return is_valid

    def can_implement(self, a: cute.Tensor, b: cute.Tensor, c: cute.Tensor) -> bool:
        (m, n, k, l) = (a.shape[0], b.shape[0], a.shape[1], a.shape[2])
        is_m_major_a = utils.LayoutEnum.from_tensor(a).is_m_major_a()
        is_n_major_b = utils.LayoutEnum.from_tensor(b).is_n_major_b()
        is_m_major_c = utils.LayoutEnum.from_tensor(c).is_m_major_c()
        a_major = 'm' if is_m_major_a else 'k'
        b_major = 'n' if is_n_major_b else 'k'
        c_major = 'm' if is_m_major_c else 'n'
        can_implement = True
        if not self.is_valid_dtypes(a.element_type, c.element_type):
            can_implement = False
        if not self.is_valid_mma_tiler_and_cluster_shape():
            can_implement = False
        if not self.is_valid_tensor_alignment(m, n, k, l, a.element_type, c.element_type, a_major, b_major, c_major):
            can_implement = False
        if not self.is_valid_epilog_store_option(m, n):
            can_implement = False
        return can_implement
