#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cfloat>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <vector>

namespace {

using bf16 = nv_bfloat16;

constexpr int HIDDEN = 2048;
constexpr int EXPERTS = 512;
constexpr int TOPK = 10;
constexpr int BLOCK_M_TC = 128;
constexpr int BLOCK_N_TC = 128;
constexpr int BLOCK_K_TC = 256;
constexpr int NUM_STAGES_TC = 2;
constexpr int SMALL_TOPK_TOKENS = 256;
constexpr int K_BLOCKS_SCALAR = HIDDEN / 16;
constexpr int PACKED_PER_BLOCK_SCALAR = 8;
constexpr float FP4_MAX_SCALAR = 6.0f;
constexpr float INV_FP4_MAX_SCALAR = 1.0f / 6.0f;
constexpr float FP8_E4M3_MAX_SCALAR = 448.0f;
constexpr float GLOBAL_SCALE_MULT_SCALAR = FP8_E4M3_MAX_SCALAR * FP4_MAX_SCALAR;

__device__ __constant__ float kFp4DecodeScalar[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f};

constexpr int WARP_SIZE = 32;
constexpr int MMA_K = 64;
constexpr uint64_t EVICT_FIRST = 0x12F0000000000000ULL;
constexpr uint64_t EVICT_LAST = 0x14F0000000000000ULL;

__device__ inline constexpr uint64_t desc_encode(uint64_t x) {
  return (x & 0x3'FFFFULL) >> 4ULL;
}

__device__ inline uint32_t elect_sync() {
  uint32_t pred = 0;
  asm volatile(
      "{\n\t"
      ".reg .pred %%px;\n\t"
      "elect.sync _|%%px, %1;\n\t"
      "@%%px mov.s32 %0, 1;\n\t"
      "}"
      : "+r"(pred)
      : "r"(0xFFFFFFFF));
  return pred;
}

__device__ inline void mbarrier_init(int mbar_addr, int count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(mbar_addr), "r"(count));
}

__device__ inline void mbarrier_wait(int mbar_addr, int phase) {
  uint32_t ticks = 0x989680;
  asm volatile(
      "{\n\t"
      ".reg .pred P1;\n\t"
      "LAB_WAIT_%=: \n\t"
      "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 P1, [%0], %1, %2;\n\t"
      "@P1 bra.uni DONE_%=;\n\t"
      "bra.uni LAB_WAIT_%=;\n\t"
      "DONE_%=: \n\t"
      "}"
      :: "r"(mbar_addr), "r"(phase), "r"(ticks));
}

__device__ inline void tma_gmem2smem(int dst, const void* src, int size, int mbar_addr,
                                     uint64_t cache_policy) {
  asm volatile(
      "cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint "
      "[%0], [%1], %2, [%3], %4;"
      :: "r"(dst), "l"(src), "r"(size), "r"(mbar_addr), "l"(cache_policy)
      : "memory");
}

__device__ inline void tma_3d_gmem2smem(int dst, const void* tmap_ptr, int x, int y, int z,
                                        int mbar_addr, uint64_t cache_policy) {
  asm volatile(
      "cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes.cta_group::1"
      ".L2::cache_hint [%0], [%1, {%2, %3, %4}], [%5], %6;"
      :: "r"(dst), "l"(tmap_ptr), "r"(x), "r"(y), "r"(z), "r"(mbar_addr),
         "l"(cache_policy)
      : "memory");
}

__device__ inline void tcgen05_cp_nvfp4(int taddr, uint64_t s_desc) {
  asm volatile("tcgen05.cp.cta_group::1.32x128b.warpx4 [%0], %1;" :: "r"(taddr), "l"(s_desc));
}

__device__ inline void tcgen05_mma_nvfp4(uint64_t a_desc, uint64_t b_desc,
                                         uint32_t i_desc, int scale_A_tmem,
                                         int scale_B_tmem, int enable_input_d) {
  asm volatile(
      "{\n\t"
      ".reg .pred p;\n\t"
      "setp.ne.b32 p, %6, 0;\n\t"
      "tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.block16 "
      "[%0], %1, %2, %3, [%4], [%5], p;\n\t"
      "}"
      :: "r"(0), "l"(a_desc), "l"(b_desc), "r"(i_desc), "r"(scale_A_tmem),
         "r"(scale_B_tmem), "r"(enable_input_d));
}

struct SHAPE {
  static constexpr char _16x256b[] = ".16x256b";
};
constexpr char SHAPE::_16x256b[];

struct NUM {
  static constexpr char x8[] = ".x8";
  static constexpr char x16[] = ".x16";
};
constexpr char NUM::x8[];
constexpr char NUM::x16[];

template <const char* SHAPE_STR, const char* NUM_STR>
__device__ inline void tcgen05_ld_32regs(float* tmp, int addr) {
  asm volatile(
      "tcgen05.ld.sync.aligned%33%34.b32 "
      "{ %0,  %1,  %2,  %3,  %4,  %5,  %6,  %7, "
      "  %8,  %9, %10, %11, %12, %13, %14, %15, "
      " %16, %17, %18, %19, %20, %21, %22, %23, "
      " %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
      : "=f"(tmp[0]), "=f"(tmp[1]), "=f"(tmp[2]), "=f"(tmp[3]), "=f"(tmp[4]),
        "=f"(tmp[5]), "=f"(tmp[6]), "=f"(tmp[7]), "=f"(tmp[8]), "=f"(tmp[9]),
        "=f"(tmp[10]), "=f"(tmp[11]), "=f"(tmp[12]), "=f"(tmp[13]),
        "=f"(tmp[14]), "=f"(tmp[15]), "=f"(tmp[16]), "=f"(tmp[17]),
        "=f"(tmp[18]), "=f"(tmp[19]), "=f"(tmp[20]), "=f"(tmp[21]),
        "=f"(tmp[22]), "=f"(tmp[23]), "=f"(tmp[24]), "=f"(tmp[25]),
        "=f"(tmp[26]), "=f"(tmp[27]), "=f"(tmp[28]), "=f"(tmp[29]),
        "=f"(tmp[30]), "=f"(tmp[31])
      : "r"(addr), "C"(SHAPE_STR), "C"(NUM_STR));
}

template <const char* SHAPE_STR, const char* NUM_STR>
__device__ inline void tcgen05_ld_64regs(float* tmp, int addr) {
  asm volatile(
      "tcgen05.ld.sync.aligned%65%66.b32 "
      "{ %0,  %1,  %2,  %3,  %4,  %5,  %6,  %7, "
      "  %8,  %9, %10, %11, %12, %13, %14, %15, "
      " %16, %17, %18, %19, %20, %21, %22, %23, "
      " %24, %25, %26, %27, %28, %29, %30, %31, "
      " %32, %33, %34, %35, %36, %37, %38, %39, "
      " %40, %41, %42, %43, %44, %45, %46, %47, "
      " %48, %49, %50, %51, %52, %53, %54, %55, "
      " %56, %57, %58, %59, %60, %61, %62, %63}, [%64];"
      : "=f"(tmp[0]), "=f"(tmp[1]), "=f"(tmp[2]), "=f"(tmp[3]), "=f"(tmp[4]),
        "=f"(tmp[5]), "=f"(tmp[6]), "=f"(tmp[7]), "=f"(tmp[8]), "=f"(tmp[9]),
        "=f"(tmp[10]), "=f"(tmp[11]), "=f"(tmp[12]), "=f"(tmp[13]),
        "=f"(tmp[14]), "=f"(tmp[15]), "=f"(tmp[16]), "=f"(tmp[17]),
        "=f"(tmp[18]), "=f"(tmp[19]), "=f"(tmp[20]), "=f"(tmp[21]),
        "=f"(tmp[22]), "=f"(tmp[23]), "=f"(tmp[24]), "=f"(tmp[25]),
        "=f"(tmp[26]), "=f"(tmp[27]), "=f"(tmp[28]), "=f"(tmp[29]),
        "=f"(tmp[30]), "=f"(tmp[31]), "=f"(tmp[32]), "=f"(tmp[33]),
        "=f"(tmp[34]), "=f"(tmp[35]), "=f"(tmp[36]), "=f"(tmp[37]),
        "=f"(tmp[38]), "=f"(tmp[39]), "=f"(tmp[40]), "=f"(tmp[41]),
        "=f"(tmp[42]), "=f"(tmp[43]), "=f"(tmp[44]), "=f"(tmp[45]),
        "=f"(tmp[46]), "=f"(tmp[47]), "=f"(tmp[48]), "=f"(tmp[49]),
        "=f"(tmp[50]), "=f"(tmp[51]), "=f"(tmp[52]), "=f"(tmp[53]),
        "=f"(tmp[54]), "=f"(tmp[55]), "=f"(tmp[56]), "=f"(tmp[57]),
        "=f"(tmp[58]), "=f"(tmp[59]), "=f"(tmp[60]), "=f"(tmp[61]),
        "=f"(tmp[62]), "=f"(tmp[63])
      : "r"(addr), "C"(SHAPE_STR), "C"(NUM_STR));
}

__device__ inline void tcgen05_ld_16x256bx8(float* tmp, int addr) {
  tcgen05_ld_32regs<SHAPE::_16x256b, NUM::x8>(tmp, addr);
}

__device__ inline void tcgen05_ld_16x256bx16(float* tmp, int addr) {
  tcgen05_ld_64regs<SHAPE::_16x256b, NUM::x16>(tmp, addr);
}

__device__ inline float warp_reduce_max(float value) {
  for (int offset = 16; offset > 0; offset >>= 1) {
    value = fmaxf(value, __shfl_down_sync(0xFFFFFFFF, value, offset));
  }
  return value;
}

__device__ inline float block_reduce_max(float value) {
  __shared__ float warp_max[8];
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  value = warp_reduce_max(value);
  if (lane == 0) {
    warp_max[warp] = value;
  }
  __syncthreads();
  value = (threadIdx.x < (blockDim.x >> 5)) ? warp_max[lane] : 0.0f;
  if (warp == 0) {
    value = warp_reduce_max(value);
  }
  __syncthreads();
  return value;
}

__device__ inline uint8_t quantize_fp4_e2m1(float x) {
  if (x >= 0.0f) {
    if (x <= 0.25f) return 0;
    if (x < 0.75f) return 1;
    if (x <= 1.25f) return 2;
    if (x < 1.75f) return 3;
    if (x <= 2.5f) return 4;
    if (x < 3.5f) return 5;
    if (x <= 5.0f) return 6;
    return 7;
  }
  if (x >= -0.25f) return 8;
  if (x > -0.75f) return 9;
  if (x >= -1.25f) return 10;
  if (x > -1.75f) return 11;
  if (x >= -2.5f) return 12;
  if (x > -3.5f) return 13;
  if (x >= -5.0f) return 14;
  return 15;
}

__global__ void global_amax_atomic_kernel(const bf16* x, float* global_amax, int64_t count) {
  float local_max = 0.0f;
  for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; idx < count;
       idx += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    local_max = fmaxf(local_max, fabsf(__bfloat162float(x[idx])));
  }
  const float reduced = block_reduce_max(local_max);
  if (threadIdx.x == 0) {
    atomicMax(reinterpret_cast<unsigned int*>(global_amax), __float_as_uint(reduced));
  }
}

__global__ void reorder_scales_kernel(const uint8_t* in, uint8_t* out, int rows, int sf_k) {
  const int total = rows * sf_k;
  const int rest_k = (sf_k + 3) / 4;
  for (int linear = blockIdx.x * blockDim.x + threadIdx.x; linear < total;
       linear += blockDim.x * gridDim.x) {
    const int row = linear / sf_k;
    const int k = linear - row * sf_k;
    const int row_tile = row / 128;
    const int row_in_tile = row % 128;
    const int mm32 = row_in_tile & 31;
    const int mm4 = row_in_tile >> 5;
    const int k_tile = k >> 2;
    const int k_sub = k & 3;
    const int out_idx = ((((row_tile * rest_k + k_tile) * 32 + mm32) * 4 + mm4) * 4 + k_sub);
    out[out_idx] = in[linear];
  }
}

__global__ void quantize_pack_kernel(const bf16* x, uint8_t* packed, uint8_t* scales,
                                     const float* global_amax_ptr, float* combined_scale_out,
                                     float gate_global_scale, int M, int K) {
  const int row = blockIdx.x;
  if (row >= M) {
    return;
  }

  const float global_amax = global_amax_ptr[0];
  float global_encode = (global_amax > 0.0f) ? (2688.0f / global_amax) : 1.0f;
  if (!isfinite(global_encode) || global_encode == 0.0f) {
    global_encode = 1.0f;
  }
  const float global_decode = 1.0f / global_encode;
  const float global_encode_over_six = global_encode * (1.0f / 6.0f);
  if (row == 0 && threadIdx.x == 0) {
    combined_scale_out[0] = global_decode * gate_global_scale;
  }

  const bf16* row_ptr = x + static_cast<int64_t>(row) * K;
  uint8_t* packed_row = packed + static_cast<int64_t>(row) * (K / 2);

  const int rest_k = K / 64;
  const int row_tile = row / 128;
  const int row_in_tile = row % 128;
  const int mm32 = row_in_tile & 31;
  const int mm4 = row_in_tile >> 5;

  for (int block16 = threadIdx.x; block16 < K / 16; block16 += blockDim.x) {
    float vals[16];
    float block_max = 0.0f;
#pragma unroll
    for (int i = 0; i < 16; ++i) {
      const float v = __bfloat162float(row_ptr[block16 * 16 + i]);
      vals[i] = v;
      block_max = fmaxf(block_max, fabsf(v));
    }

    const int k_tile = block16 >> 2;
    const int k_sub = block16 & 3;
    const int scale_idx = ((((row_tile * rest_k + k_tile) * 32 + mm32) * 4 + mm4) * 4 + k_sub);

    if (block_max == 0.0f) {
      scales[scale_idx] = 0;
      *reinterpret_cast<uint64_t*>(packed_row + block16 * 8) = 0ULL;
      continue;
    }

    float scaled_decode = block_max * global_encode_over_six;
    scaled_decode = fminf(scaled_decode, 448.0f);
    const __nv_fp8_storage_t scale_fp8 =
        __nv_cvt_float_to_fp8(scaled_decode, __NV_SATFINITE, __NV_E4M3);
    scales[scale_idx] = static_cast<uint8_t>(scale_fp8);

    union HalfConverter {
      __half_raw raw;
      __half half;
    } converter;
    converter.raw = __nv_cvt_fp8_to_halfraw(scale_fp8, __NV_E4M3);
    const float rounded_decode = __half2float(converter.half) * global_decode;
    const float encode_scale = (rounded_decode > 0.0f) ? (1.0f / rounded_decode) : FLT_MAX;

    uint64_t packed_bytes = 0ULL;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      float a = vals[2 * i] * encode_scale;
      float b = vals[2 * i + 1] * encode_scale;
      a = fminf(fmaxf(a, -6.0f), 6.0f);
      b = fminf(fmaxf(b, -6.0f), 6.0f);
      const uint8_t byte = static_cast<uint8_t>(quantize_fp4_e2m1(a) | (quantize_fp4_e2m1(b) << 4));
      packed_bytes |= static_cast<uint64_t>(byte) << (i * 8);
    }
    *reinterpret_cast<uint64_t*>(packed_row + block16 * 8) = packed_bytes;
  }
}

struct TopK10 {
  float values[TOPK];
  int indices[TOPK];

  __device__ __forceinline__ void init() {
#pragma unroll
    for (int i = 0; i < TOPK; ++i) {
      values[i] = -INFINITY;
      indices[i] = 0;
    }
  }

  __device__ __forceinline__ void consider(float v, int idx) {
    if (v <= values[TOPK - 1]) {
      return;
    }
    int pos = TOPK - 1;
    while (pos > 0 && v > values[pos - 1]) {
      values[pos] = values[pos - 1];
      indices[pos] = indices[pos - 1];
      --pos;
    }
    values[pos] = v;
    indices[pos] = idx;
  }
};

struct TopK4Vec {
  float values[4];
  int indices[4];

  __device__ __forceinline__ void init() {
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      values[i] = -INFINITY;
      indices[i] = 0;
    }
  }

  __device__ __forceinline__ void consider(float v, int idx) {
    if (v <= values[3]) {
      return;
    }
    int pos = 3;
    while (pos > 0 && v > values[pos - 1]) {
      values[pos] = values[pos - 1];
      indices[pos] = indices[pos - 1];
      --pos;
    }
    values[pos] = v;
    indices[pos] = idx;
  }
};

struct TopK4 {
  float values[4];
  int indices[4];

  __device__ __forceinline__ void init() {
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      values[i] = -INFINITY;
      indices[i] = 0;
    }
  }

  __device__ __forceinline__ void consider(float v, int idx) {
    if (v <= values[3]) {
      return;
    }
    int pos = 3;
    while (pos > 0 && v > values[pos - 1]) {
      values[pos] = values[pos - 1];
      indices[pos] = indices[pos - 1];
      --pos;
    }
    values[pos] = v;
    indices[pos] = idx;
  }
};

__global__ void topk_from_logits_block_kernel(const float* router_logits, bf16* routing_weights,
                                              int64_t* selected_experts, int num_tokens) {
  const int token = blockIdx.x;
  if (token >= num_tokens) {
    return;
  }

  constexpr int BLOCK_THREADS = 128;
  constexpr int BLOCK_WARPS = BLOCK_THREADS / WARP_SIZE;
  constexpr int EXPERTS_PER_WARP = EXPERTS / BLOCK_WARPS;
  constexpr int LOCAL_CANDIDATES = 4;
  static_assert(BLOCK_WARPS == 4, "expected 4 warps");
  static_assert(EXPERTS_PER_WARP == 128, "expected quarter-row partition");

  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const float* row = router_logits + static_cast<int64_t>(token) * EXPERTS;
  const int warp_base = warp * EXPERTS_PER_WARP;
  const float4 v4 = reinterpret_cast<const float4*>(row + warp_base)[lane];

  TopK4Vec local_topk;
  local_topk.init();
  local_topk.consider(v4.x, warp_base + lane * 4 + 0);
  local_topk.consider(v4.y, warp_base + lane * 4 + 1);
  local_topk.consider(v4.z, warp_base + lane * 4 + 2);
  local_topk.consider(v4.w, warp_base + lane * 4 + 3);

  __shared__ float sh_thread_vals[BLOCK_WARPS][WARP_SIZE][LOCAL_CANDIDATES];
  __shared__ int sh_thread_idx[BLOCK_WARPS][WARP_SIZE][LOCAL_CANDIDATES];
  __shared__ float sh_warp_vals[BLOCK_WARPS][TOPK];
  __shared__ int sh_warp_idx[BLOCK_WARPS][TOPK];

#pragma unroll
  for (int i = 0; i < LOCAL_CANDIDATES; ++i) {
    sh_thread_vals[warp][lane][i] = local_topk.values[i];
    sh_thread_idx[warp][lane][i] = local_topk.indices[i];
  }
  __syncthreads();

  if (lane == 0) {
    TopK10 warp_topk;
    warp_topk.init();
#pragma unroll
    for (int t = 0; t < WARP_SIZE; ++t) {
#pragma unroll
      for (int i = 0; i < LOCAL_CANDIDATES; ++i) {
        warp_topk.consider(sh_thread_vals[warp][t][i], sh_thread_idx[warp][t][i]);
      }
    }
#pragma unroll
    for (int i = 0; i < TOPK; ++i) {
      sh_warp_vals[warp][i] = warp_topk.values[i];
      sh_warp_idx[warp][i] = warp_topk.indices[i];
    }
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    TopK10 final_topk;
    final_topk.init();
#pragma unroll
    for (int w = 0; w < BLOCK_WARPS; ++w) {
#pragma unroll
      for (int i = 0; i < TOPK; ++i) {
        final_topk.consider(sh_warp_vals[w][i], sh_warp_idx[w][i]);
      }
    }
    const float max_v = final_topk.values[0];
    float probs[TOPK];
    float denom = 0.0f;
#pragma unroll
    for (int i = 0; i < TOPK; ++i) {
      probs[i] = expf(final_topk.values[i] - max_v);
      denom += probs[i];
    }
    const float inv_denom = 1.0f / denom;
#pragma unroll
    for (int i = 0; i < TOPK; ++i) {
      routing_weights[token * TOPK + i] = __float2bfloat16_rn(probs[i] * inv_denom);
      selected_experts[token * TOPK + i] = static_cast<int64_t>(final_topk.indices[i]);
    }
  }
}

__global__ void scale_output_bf16_kernel(bf16* out, const float* act_global_decode_ptr,
                                         float weight_global_decode, int64_t count) {
  const float global_scale = act_global_decode_ptr[0] * weight_global_decode;
  for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; idx < count;
       idx += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    const float v = __bfloat162float(out[idx]);
    out[idx] = __float2bfloat16_rn(v * global_scale);
  }
}

__global__ void topk_from_logits_scalar_kernel(const float* router_logits, bf16* routing_weights,
                                        int64_t* selected_experts, int num_tokens) {
  const int token = blockIdx.x;
  if (token >= num_tokens || threadIdx.x != 0) {
    return;
  }
  const float* row = router_logits + static_cast<int64_t>(token) * EXPERTS;
  TopK10 topk;
  topk.init();
#pragma unroll 1
  for (int expert = 0; expert < EXPERTS; ++expert) {
    topk.consider(row[expert], expert);
  }
  const float max_v = topk.values[0];
  float probs[TOPK];
  float denom = 0.0f;
#pragma unroll
  for (int i = 0; i < TOPK; ++i) {
    probs[i] = expf(topk.values[i] - max_v);
    denom += probs[i];
  }
  const float inv_denom = 1.0f / denom;
#pragma unroll
  for (int i = 0; i < TOPK; ++i) {
    routing_weights[token * TOPK + i] = __float2bfloat16_rn(probs[i] * inv_denom);
    selected_experts[token * TOPK + i] = static_cast<int64_t>(topk.indices[i]);
  }
}

void check_cuda(cudaError_t err);

union HalfConverterScalarFP8 {
  __half_raw raw;
  __half half;
};

__device__ __forceinline__ float fp8_e4m3_to_float_scalar(uint8_t code) {
  HalfConverterScalarFP8 converter;
  converter.raw = __nv_cvt_fp8_to_halfraw(static_cast<__nv_fp8_storage_t>(code), __NV_E4M3);
  return __half2float(converter.half);
}

__device__ __forceinline__ float fp8_e4m3_round_scalar(float x) {
  if (!(x > 0.0f)) return 0.0f;
  x = fminf(x, FP8_E4M3_MAX_SCALAR);
  const __nv_fp8_storage_t fp8 = __nv_cvt_float_to_fp8(x, __NV_SATFINITE, __NV_E4M3);
  return fp8_e4m3_to_float_scalar(static_cast<uint8_t>(fp8));
}

__device__ __forceinline__ float dot8bytes_scalar(uint64_t a64, uint64_t b64) {
  float acc = 0.0f;
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    const uint8_t a = static_cast<uint8_t>((a64 >> (i * 8)) & 0xffu);
    const uint8_t b = static_cast<uint8_t>((b64 >> (i * 8)) & 0xffu);
    acc += kFp4DecodeScalar[a & 0x0f] * kFp4DecodeScalar[b & 0x0f];
    acc += kFp4DecodeScalar[(a >> 4) & 0x0f] * kFp4DecodeScalar[(b >> 4) & 0x0f];
  }
  return acc;
}

__global__ void prepare_weights_scalar_hwfp8_kernel(
    const uint8_t* __restrict__ gate_weight_fp4,
    const uint8_t* __restrict__ gate_scale_bytes,
    float gate_global_scale,
    uint8_t* __restrict__ weight_blocks,
    float* __restrict__ scale_blocks) {
  const int kblk = blockIdx.x;
  const int tid = threadIdx.x;
  for (int expert = tid; expert < EXPERTS; expert += blockDim.x) {
    const int src_off = expert * (HIDDEN / 2) + kblk * PACKED_PER_BLOCK_SCALAR;
    const int dst_off = (kblk * EXPERTS + expert) * PACKED_PER_BLOCK_SCALAR;
    reinterpret_cast<uint64_t*>(weight_blocks + dst_off)[0] =
        reinterpret_cast<const uint64_t*>(gate_weight_fp4 + src_off)[0];
    const uint8_t scale_code = gate_scale_bytes[expert * K_BLOCKS_SCALAR + kblk];
    scale_blocks[kblk * EXPERTS + expert] = fp8_e4m3_to_float_scalar(scale_code) * gate_global_scale;
  }
}

__global__ void routing_scalar_hwfp8_kernel(
    const bf16* __restrict__ hidden_states,
    const uint8_t* __restrict__ weight_blocks,
    const float* __restrict__ scale_blocks,
    float* __restrict__ router_logits,
    const float* __restrict__ global_amax_ptr) {
  const int token = blockIdx.x;
  const int tid = threadIdx.x;

  __shared__ uint64_t sh_hidden_pack[K_BLOCKS_SCALAR];
  __shared__ float sh_hidden_scale[K_BLOCKS_SCALAR];

  const bf16* row = hidden_states + static_cast<int64_t>(token) * HIDDEN;

  const float global_amax = global_amax_ptr[0];
  float global_encode = (global_amax > 0.0f) ? (GLOBAL_SCALE_MULT_SCALAR / global_amax) : 1.0f;
  if (!isfinite(global_encode) || global_encode == 0.0f) {
    global_encode = 1.0f;
  }
  const float global_decode = 1.0f / global_encode;

  if (tid < K_BLOCKS_SCALAR) {
    const int base = tid * 16;
    float vals[16];
    float block_max = 0.0f;
#pragma unroll
    for (int j = 0; j < 16; ++j) {
      const float v = __bfloat162float(row[base + j]);
      vals[j] = v;
      block_max = fmaxf(block_max, fabsf(v));
    }

    const float rounded_scale = fp8_e4m3_round_scalar(block_max * INV_FP4_MAX_SCALAR * global_encode);
    const float rounded_decode = rounded_scale * global_decode;
    sh_hidden_scale[tid] = rounded_decode;

    uint64_t packed = 0ULL;
    if (rounded_decode > 0.0f) {
      const float encode_scale = 1.0f / rounded_decode;
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        const uint8_t lo = quantize_fp4_e2m1(fminf(fmaxf(vals[2 * j] * encode_scale, -FP4_MAX_SCALAR), FP4_MAX_SCALAR));
        const uint8_t hi = quantize_fp4_e2m1(fminf(fmaxf(vals[2 * j + 1] * encode_scale, -FP4_MAX_SCALAR), FP4_MAX_SCALAR));
        packed |= (static_cast<uint64_t>(lo | (hi << 4)) << (j * 8));
      }
    }
    sh_hidden_pack[tid] = packed;
  }
  __syncthreads();

  const int expert0 = tid;
  const int expert1 = tid + 256;
  float acc0 = 0.0f;
  float acc1 = 0.0f;

  if (expert0 < EXPERTS || expert1 < EXPERTS) {
#pragma unroll 4
    for (int kblk = 0; kblk < K_BLOCKS_SCALAR; ++kblk) {
      const float hs = sh_hidden_scale[kblk];
      if (hs == 0.0f) continue;
      const uint64_t a64 = sh_hidden_pack[kblk];
      if (expert0 < EXPERTS) {
        const int base = (kblk * EXPERTS + expert0) * PACKED_PER_BLOCK_SCALAR;
        const uint64_t b64 = reinterpret_cast<const uint64_t*>(weight_blocks + base)[0];
        acc0 += dot8bytes_scalar(a64, b64) * hs * scale_blocks[kblk * EXPERTS + expert0];
      }
      if (expert1 < EXPERTS) {
        const int base = (kblk * EXPERTS + expert1) * PACKED_PER_BLOCK_SCALAR;
        const uint64_t b64 = reinterpret_cast<const uint64_t*>(weight_blocks + base)[0];
        acc1 += dot8bytes_scalar(a64, b64) * hs * scale_blocks[kblk * EXPERTS + expert1];
      }
    }
  }

  if (expert0 < EXPERTS) {
    router_logits[static_cast<int64_t>(token) * EXPERTS + expert0] = acc0;
  }
  if (expert1 < EXPERTS) {
    router_logits[static_cast<int64_t>(token) * EXPERTS + expert1] = acc1;
  }
}

void launch_large_scalar_hwfp8_fallback(const torch::Tensor& hidden_states,
                                        const torch::Tensor& gate_weight_fp4,
                                        const torch::Tensor& gate_scale,
                                        float gate_global_scale_f,
                                        const torch::Tensor& routing_weights,
                                        const torch::Tensor& selected_experts,
                                        const torch::Tensor& router_logits,
                                        int M_int,
                                        int K_int,
                                        cudaStream_t stream) {
  auto byte_opts = torch::TensorOptions().device(hidden_states.device()).dtype(torch::kUInt8);
  auto float_opts = torch::TensorOptions().device(hidden_states.device()).dtype(torch::kFloat32);
  auto weight_blocks = torch::empty({K_BLOCKS_SCALAR * EXPERTS * PACKED_PER_BLOCK_SCALAR}, byte_opts);
  auto scale_blocks = torch::empty({K_BLOCKS_SCALAR, EXPERTS}, float_opts);
  auto global_amax = torch::zeros({1}, float_opts);

  prepare_weights_scalar_hwfp8_kernel<<<K_BLOCKS_SCALAR, 256, 0, stream>>>(
      reinterpret_cast<const uint8_t*>(gate_weight_fp4.data_ptr()),
      reinterpret_cast<const uint8_t*>(gate_scale.data_ptr()),
      gate_global_scale_f,
      weight_blocks.data_ptr<uint8_t>(),
      scale_blocks.data_ptr<float>());
  check_cuda(cudaGetLastError());

  constexpr int THREADS = 256;
  const int64_t elem_count = static_cast<int64_t>(M_int) * K_int;
  const int reduce_grid = std::min<int64_t>((elem_count + THREADS - 1) / THREADS, 4096);
  global_amax_atomic_kernel<<<reduce_grid, THREADS, 0, stream>>>(
      reinterpret_cast<const bf16*>(hidden_states.data_ptr<c10::BFloat16>()),
      global_amax.data_ptr<float>(), elem_count);
  check_cuda(cudaGetLastError());

  routing_scalar_hwfp8_kernel<<<M_int, 256, 0, stream>>>(
      reinterpret_cast<const bf16*>(hidden_states.data_ptr<c10::BFloat16>()),
      weight_blocks.data_ptr<uint8_t>(),
      scale_blocks.data_ptr<float>(),
      router_logits.data_ptr<float>(),
      global_amax.data_ptr<float>());
  check_cuda(cudaGetLastError());

  auto routing_weights_f = at::softmax(router_logits, 1, c10::ScalarType::Float);
  auto topk = routing_weights_f.topk(TOPK, -1, true, true);
  auto topk_values = std::get<0>(topk);
  auto topk_indices = std::get<1>(topk);
  auto topk_norm = (topk_values / topk_values.sum(-1, true)).to(routing_weights.scalar_type());
  routing_weights.copy_(topk_norm);
  selected_experts.copy_(topk_indices);
}

void check_cuda(cudaError_t err) {
  TORCH_CHECK(err == cudaSuccess, cudaGetErrorString(err));
}

void check_cu(CUresult err) {
  TORCH_CHECK(err == CUDA_SUCCESS, "cuTensorMapEncodeTiled failed with error code ", static_cast<int>(err));
}

using CuTensorMapEncodeTiledFn = CUresult (*)(
    CUtensorMap*,
    CUtensorMapDataType,
    uint32_t,
    void*,
    const uint64_t*,
    const uint64_t*,
    const uint32_t*,
    const uint32_t*,
    CUtensorMapInterleave,
    CUtensorMapSwizzle,
    CUtensorMapL2promotion,
    CUtensorMapFloatOOBfill);

CuTensorMapEncodeTiledFn get_cuTensorMapEncodeTiled() {
  static CuTensorMapEncodeTiledFn fn = []() -> CuTensorMapEncodeTiledFn {
    void* raw_fn = nullptr;
    check_cuda(cudaGetDriverEntryPoint("cuTensorMapEncodeTiled", &raw_fn, cudaEnableDefault, nullptr));
    TORCH_CHECK(raw_fn != nullptr, "cudaGetDriverEntryPoint returned null for cuTensorMapEncodeTiled");
    return reinterpret_cast<CuTensorMapEncodeTiledFn>(raw_fn);
  }();
  return fn;
}

struct TensorMapCache {
  int device_index = -1;
  const void* act_ptr = nullptr;
  int act_M = 0;
  int act_K = 0;
  bool act_valid = false;
  CUtensorMap act_tmap;
  const void* gate_ptr = nullptr;
  int gate_N = 0;
  int gate_K = 0;
  bool gate_valid = false;
  CUtensorMap gate_tmap;
};

extern TensorMapCache g_tmap_cache;

void init_AB_tmap(CUtensorMap* tmap, const void* ptr, uint64_t global_height, uint64_t global_width,
                  uint32_t shared_height, uint32_t shared_width) {
  constexpr uint32_t rank = 3;
  uint64_t globalDim[rank] = {256, global_height, global_width / 256};
  uint64_t globalStrides[rank - 1] = {global_width / 2, 128};
  uint32_t boxDim[rank] = {256, shared_height, shared_width / 256};
  uint32_t elementStrides[rank] = {1, 1, 1};
  check_cu(get_cuTensorMapEncodeTiled()(
      tmap, CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN8B, rank,
      const_cast<void*>(ptr), globalDim, globalStrides, boxDim, elementStrides,
      CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
      CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_128B,
      CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_NONE,
      CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
}

template <int BLOCK_M, int BLOCK_N, int BLOCK_K, int NUM_STAGES>
__global__ __launch_bounds__(BLOCK_M + 2 * WARP_SIZE)
void nvfp4_gemm_kernel(const __grid_constant__ CUtensorMap A_tmap,
                       const __grid_constant__ CUtensorMap B_tmap,
                       const char* SFA_ptr,
                       const char* SFB_ptr,
                       const float* combined_scale_ptr,
                       float* C_ptr,
                       int M,
                       int N,
                       int K) {
  const int tid = threadIdx.x;
  const int lane_id = tid & 31;
  const int warp_id = tid >> 5;
  const int bid = blockIdx.x;
  const int grid_n = (N + BLOCK_N - 1) / BLOCK_N;
  const int bid_m = bid / grid_n;
  const int bid_n = bid % grid_n;
  const int off_m = bid_m * BLOCK_M;
  const int off_n = bid_n * BLOCK_N;
  constexpr int NUM_WARPS = BLOCK_M / WARP_SIZE + 2;

  extern __shared__ __align__(1024) char smem_ptr[];
  const int smem = static_cast<int>(__cvta_generic_to_shared(smem_ptr));
  constexpr int A_size = BLOCK_M * BLOCK_K / 2;
  constexpr int B_size = BLOCK_N * BLOCK_K / 2;
  constexpr int SFA_size = 128 * BLOCK_K / 16;
  constexpr int SFB_size = 128 * BLOCK_K / 16;
  constexpr int STAGE_SIZE = A_size + B_size + SFA_size + SFB_size;

#pragma nv_diag_suppress static_var_with_dynamic_init
  __shared__ int64_t mbars[NUM_STAGES * 2 + 1];

  const int tma_mbar_addr = static_cast<int>(__cvta_generic_to_shared(mbars));
  const int mma_mbar_addr = tma_mbar_addr + NUM_STAGES * 8;
  const int mainloop_mbar_addr = mma_mbar_addr + NUM_STAGES * 8;

  constexpr int SFA_tmem = BLOCK_N;
  constexpr int SFB_tmem = SFA_tmem + 4 * (BLOCK_K / MMA_K);

  if (warp_id == 0 && elect_sync()) {
    for (int i = 0; i < NUM_STAGES * 2 + 1; ++i) {
      mbarrier_init(tma_mbar_addr + i * 8, 1);
    }
    asm volatile("fence.mbarrier_init.release.cluster;");
  } else if (warp_id == 1) {
    asm volatile(
        "tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"
        :: "r"(smem), "r"(BLOCK_N * 2));
  }
  __syncthreads();

  const int num_iters = K / BLOCK_K;

  if (warp_id == NUM_WARPS - 2 && elect_sync()) {
    uint64_t cache_A = (M > N) ? EVICT_FIRST : EVICT_LAST;
    uint64_t cache_B = (M > N) ? EVICT_LAST : EVICT_FIRST;

    auto issue_tma = [&](int iter_k, int stage_id) {
      const int mbar_addr = tma_mbar_addr + stage_id * 8;
      const int A_smem = smem + stage_id * STAGE_SIZE;
      const int B_smem = A_smem + A_size;
      const int SFA_smem = B_smem + B_size;
      const int SFB_smem = SFA_smem + SFA_size;
      const int off_k = iter_k * BLOCK_K;
      tma_3d_gmem2smem(A_smem, &A_tmap, 0, off_m, off_k / 256, mbar_addr, cache_A);
      tma_3d_gmem2smem(B_smem, &B_tmap, 0, off_n, off_k / 256, mbar_addr, cache_B);
      const int rest_k = K / 64;
      const char* SFA_src = SFA_ptr + ((off_m / 128) * rest_k + off_k / 64) * 512;
      const char* SFB_src = SFB_ptr + ((off_n / 128) * rest_k + off_k / 64) * 512;
      tma_gmem2smem(SFA_smem, SFA_src, SFA_size, mbar_addr, cache_A);
      tma_gmem2smem(SFB_smem, SFB_src, SFB_size, mbar_addr, cache_B);
      asm volatile(
          "mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;"
          :: "r"(mbar_addr), "r"(STAGE_SIZE)
          : "memory");
    };

    for (int iter_k = 0; iter_k < NUM_STAGES && iter_k < num_iters; ++iter_k) {
      issue_tma(iter_k, iter_k);
    }
    for (int iter_k = NUM_STAGES; iter_k < num_iters; ++iter_k) {
      const int stage_id = iter_k % NUM_STAGES;
      const int mma_phase = (iter_k / NUM_STAGES - 1) & 1;
      mbarrier_wait(mma_mbar_addr + stage_id * 8, mma_phase);
      issue_tma(iter_k, stage_id);
    }
  } else if (warp_id == NUM_WARPS - 1 && elect_sync()) {
    constexpr uint32_t i_desc = (1U << 7U) | (1U << 10U) |
                                ((uint32_t)BLOCK_N >> 3U << 17U) |
                                ((uint32_t)128 >> 7U << 27U);

    for (int iter_k = 0; iter_k < num_iters; ++iter_k) {
      const int stage_id = iter_k % NUM_STAGES;
      const int tma_phase = (iter_k / NUM_STAGES) & 1;
      mbarrier_wait(tma_mbar_addr + stage_id * 8, tma_phase);
      const int A_smem = smem + stage_id * STAGE_SIZE;
      const int B_smem = A_smem + A_size;
      const int SFA_smem = B_smem + B_size;
      const int SFB_smem = SFA_smem + SFA_size;

      auto make_desc_AB = [](int addr) -> uint64_t {
        const int SBO = 8 * 128;
        return desc_encode(addr) | (desc_encode(SBO) << 32ULL) | (1ULL << 46ULL) |
               (2ULL << 61ULL);
      };
      auto make_desc_SF = [](int addr) -> uint64_t {
        const int SBO = 8 * 16;
        return desc_encode(addr) | (desc_encode(SBO) << 32ULL) | (1ULL << 46ULL);
      };

      const uint64_t SFA_desc = make_desc_SF(SFA_smem);
      const uint64_t SFB_desc = make_desc_SF(SFB_smem);
      for (int k = 0; k < BLOCK_K / MMA_K; ++k) {
        const uint64_t sfa_desc = SFA_desc + static_cast<uint64_t>(k) * (512ULL >> 4ULL);
        const uint64_t sfb_desc = SFB_desc + static_cast<uint64_t>(k) * (512ULL >> 4ULL);
        tcgen05_cp_nvfp4(SFA_tmem + k * 4, sfa_desc);
        tcgen05_cp_nvfp4(SFB_tmem + k * 4, sfb_desc);
      }

      for (int k1 = 0; k1 < BLOCK_K / 256; ++k1) {
        for (int k2 = 0; k2 < 256 / MMA_K; ++k2) {
          const uint64_t a_desc = make_desc_AB(A_smem + k1 * BLOCK_M * 128 + k2 * 32);
          const uint64_t b_desc = make_desc_AB(B_smem + k1 * BLOCK_N * 128 + k2 * 32);
          const int k_sf = k1 * 4 + k2;
          const int scale_A_tmem = SFA_tmem + k_sf * 4;
          const int scale_B_tmem =
              SFB_tmem + k_sf * 4 + (bid_n % (128 / BLOCK_N)) * (BLOCK_N / 32);
          const int enable_input_d = (iter_k == 0 && k1 == 0 && k2 == 0) ? 0 : 1;
          tcgen05_mma_nvfp4(a_desc, b_desc, i_desc, scale_A_tmem, scale_B_tmem,
                            enable_input_d);
        }
      }

      asm volatile(
          "tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];"
          :: "r"(mma_mbar_addr + stage_id * 8)
          : "memory");
    }

    asm volatile(
        "tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];"
        :: "r"(mainloop_mbar_addr)
        : "memory");
  } else if (tid < BLOCK_M) {
    mbarrier_wait(mainloop_mbar_addr, 0);
    asm volatile("tcgen05.fence::after_thread_sync;");

    const float combined_scale = combined_scale_ptr[0];
    for (int m16 = 0; m16 < 2; ++m16) {
      const int row_base = warp_id * 32 + m16 * 16;
      float tmp[64];
      tcgen05_ld_16x256bx16(tmp, row_base << 16);
      asm volatile("tcgen05.wait::ld.sync.aligned;");

      for (int i = 0; i < 16; ++i) {
        const int row = off_m + row_base + (lane_id >> 2);
        const int col = off_n + i * 8 + (lane_id & 3) * 2;
        if (row + 0 < M) {
          if (col + 0 < N) C_ptr[static_cast<int64_t>(row + 0) * N + col + 0] = tmp[i * 4 + 0] * combined_scale;
          if (col + 1 < N) C_ptr[static_cast<int64_t>(row + 0) * N + col + 1] = tmp[i * 4 + 1] * combined_scale;
        }
        if (row + 8 < M) {
          if (col + 0 < N) C_ptr[static_cast<int64_t>(row + 8) * N + col + 0] = tmp[i * 4 + 2] * combined_scale;
          if (col + 1 < N) C_ptr[static_cast<int64_t>(row + 8) * N + col + 1] = tmp[i * 4 + 3] * combined_scale;
        }
      }
    }

    asm volatile("bar.sync 1, %0;" :: "r"(BLOCK_M) : "memory");
    if (warp_id == 0) {
      asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(0),
                   "r"(BLOCK_N * 2));
    }
  }
}

template <int BLOCK_N, int NUM_STAGES>
void prepare_nvfp4_gemm() {
  constexpr int BLOCK_M = BLOCK_M_TC;
  constexpr int BLOCK_K = BLOCK_K_TC;
  const int AB_size = (BLOCK_M + BLOCK_N) * (BLOCK_K / 2);
  const int SFAB_size = 128 * (BLOCK_K / 16) * 2;
  const int smem_size = (AB_size + SFAB_size) * NUM_STAGES;
  auto kernel = nvfp4_gemm_kernel<BLOCK_M, BLOCK_N, BLOCK_K, NUM_STAGES>;
  static bool smem_attr_set = false;
  if (smem_size > 48000) {
    if (!smem_attr_set) {
      check_cuda(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
      smem_attr_set = true;
    }
  }
}

template <int BLOCK_N, int NUM_STAGES>
void launch_nvfp4_gemm(const void* A_ptr, const void* B_ptr, const void* SFA_ptr, const void* SFB_ptr,
                       const float* combined_scale_ptr, float* C_ptr, int M, int N, int K,
                       int device_index, cudaStream_t stream) {
  constexpr int BLOCK_M = BLOCK_M_TC;
  constexpr int BLOCK_K = BLOCK_K_TC;
  prepare_nvfp4_gemm<BLOCK_N, NUM_STAGES>();

  if (g_tmap_cache.device_index != device_index) {
    g_tmap_cache = TensorMapCache{};
    g_tmap_cache.device_index = device_index;
  }
  if (!g_tmap_cache.act_valid || g_tmap_cache.act_ptr != A_ptr ||
      g_tmap_cache.act_M != M || g_tmap_cache.act_K != K) {
    init_AB_tmap(&g_tmap_cache.act_tmap, A_ptr, M, K, BLOCK_M, BLOCK_K);
    g_tmap_cache.act_ptr = A_ptr;
    g_tmap_cache.act_M = M;
    g_tmap_cache.act_K = K;
    g_tmap_cache.act_valid = true;
  }
  if (!g_tmap_cache.gate_valid || g_tmap_cache.gate_ptr != B_ptr ||
      g_tmap_cache.gate_N != N || g_tmap_cache.gate_K != K) {
    init_AB_tmap(&g_tmap_cache.gate_tmap, B_ptr, N, K, BLOCK_N, BLOCK_K);
    g_tmap_cache.gate_ptr = B_ptr;
    g_tmap_cache.gate_N = N;
    g_tmap_cache.gate_K = K;
    g_tmap_cache.gate_valid = true;
  }

  const int grid = ((M + BLOCK_M - 1) / BLOCK_M) * ((N + BLOCK_N - 1) / BLOCK_N);
  const int threads = BLOCK_M + 2 * WARP_SIZE;
  const int AB_size = (BLOCK_M + BLOCK_N) * (BLOCK_K / 2);
  const int SFAB_size = 128 * (BLOCK_K / 16) * 2;
  const int smem_size = (AB_size + SFAB_size) * NUM_STAGES;

  auto kernel = nvfp4_gemm_kernel<BLOCK_M, BLOCK_N, BLOCK_K, NUM_STAGES>;
  kernel<<<grid, threads, smem_size, stream>>>(g_tmap_cache.act_tmap, g_tmap_cache.gate_tmap,
                                                reinterpret_cast<const char*>(SFA_ptr),
                                                reinterpret_cast<const char*>(SFB_ptr),
                                                combined_scale_ptr, C_ptr, M, N, K);
}

void check_tensor(const torch::Tensor& tensor, const char* name) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

struct WorkspaceCache {
  int device_index = -1;
  int64_t capacity_M = 0;
  torch::Tensor global_amax;
  torch::Tensor combined_scale;
  torch::Tensor act_fp4;
  torch::Tensor act_scales;
  torch::Tensor gate_scales_reordered;
};

WorkspaceCache g_workspace_cache;
TensorMapCache g_tmap_cache;
std::mutex g_workspace_mutex;

void ensure_workspace(const torch::Tensor& hidden_states,
                      const torch::Tensor& gate_scale,
                      int64_t M,
                      int64_t K,
                      int64_t N) {
  const int device_index = hidden_states.get_device();
  const bool device_changed = g_workspace_cache.device_index != device_index;
  const bool needs_growth = !g_workspace_cache.act_fp4.defined() || g_workspace_cache.capacity_M < M;
  if (!device_changed && !needs_growth && g_workspace_cache.gate_scales_reordered.defined()) {
    return;
  }


  const int64_t new_capacity_M = std::max<int64_t>(M, g_workspace_cache.capacity_M);
  auto float_options = hidden_states.options().dtype(torch::kFloat32);
  auto packed_options = torch::TensorOptions().device(hidden_states.device()).dtype(torch::kUInt8);

  g_workspace_cache.device_index = device_index;
  g_workspace_cache.capacity_M = new_capacity_M;
  g_workspace_cache.global_amax = torch::empty({1}, float_options);
  g_workspace_cache.combined_scale = torch::empty({1}, float_options);
  g_workspace_cache.act_fp4 = torch::empty({new_capacity_M, K / 2}, packed_options);
  g_workspace_cache.act_scales =
      torch::empty({new_capacity_M / 128, K / 64, 32, 4, 4}, gate_scale.options());
  g_workspace_cache.gate_scales_reordered =
      torch::empty({N / 128, K / 64, 32, 4, 4}, gate_scale.options());
}

void launch_pipeline(const torch::Tensor& hidden_states,
                     const torch::Tensor& gate_weight_fp4,
                     const torch::Tensor& gate_scale,
                     float gate_global_scale_f,
                     const WorkspaceCache& workspace,
                     const torch::Tensor& routing_weights,
                     const torch::Tensor& selected_experts,
                     const torch::Tensor& router_logits,
                     int M_int,
                     int N_int,
                     int K_int,
                     cudaStream_t stream,
                     bool check_launches) {
  constexpr int THREADS = 256;
  const int64_t elem_count = static_cast<int64_t>(M_int) * K_int;
  const int reduce_grid = std::min<int64_t>((elem_count + THREADS - 1) / THREADS, 4096);

  if (check_launches) {
    check_cuda(cudaMemsetAsync(workspace.global_amax.data_ptr<float>(), 0, sizeof(float), stream));
  } else {
    (void)cudaMemsetAsync(workspace.global_amax.data_ptr<float>(), 0, sizeof(float), stream);
  }

  global_amax_atomic_kernel<<<reduce_grid, THREADS, 0, stream>>>(
      reinterpret_cast<const bf16*>(hidden_states.data_ptr<c10::BFloat16>()),
      workspace.global_amax.data_ptr<float>(), elem_count);
  if (check_launches) {
    check_cuda(cudaGetLastError());
  }

  quantize_pack_kernel<<<M_int, THREADS, 0, stream>>>(
      reinterpret_cast<const bf16*>(hidden_states.data_ptr<c10::BFloat16>()),
      workspace.act_fp4.data_ptr<uint8_t>(),
      reinterpret_cast<uint8_t*>(workspace.act_scales.data_ptr()),
      workspace.global_amax.data_ptr<float>(),
      workspace.combined_scale.data_ptr<float>(),
      gate_global_scale_f, M_int, K_int);
  if (check_launches) {
    check_cuda(cudaGetLastError());
  }

  const int total_scale_elems = N_int * (K_int / 16);
  const int reorder_grid = std::min((total_scale_elems + THREADS - 1) / THREADS, 65535);
  reorder_scales_kernel<<<reorder_grid, THREADS, 0, stream>>>(
      reinterpret_cast<const uint8_t*>(gate_scale.data_ptr()),
      reinterpret_cast<uint8_t*>(workspace.gate_scales_reordered.data_ptr()),
      N_int, K_int / 16);
  if (check_launches) {
    check_cuda(cudaGetLastError());
  }

  launch_nvfp4_gemm<BLOCK_N_TC, NUM_STAGES_TC>(
      workspace.act_fp4.data_ptr(),
      gate_weight_fp4.data_ptr(),
      workspace.act_scales.data_ptr(),
      workspace.gate_scales_reordered.data_ptr(),
      workspace.combined_scale.data_ptr<float>(),
      router_logits.data_ptr<float>(),
      M_int, N_int, K_int, hidden_states.get_device(), stream);
  if (check_launches) {
    check_cuda(cudaGetLastError());
  }

  if (M_int <= SMALL_TOPK_TOKENS) {
    topk_from_logits_scalar_kernel<<<M_int, 1, 0, stream>>>(
        router_logits.data_ptr<float>(),
        reinterpret_cast<bf16*>(routing_weights.data_ptr<c10::BFloat16>()),
        selected_experts.data_ptr<int64_t>(), M_int);
  } else {
    topk_from_logits_block_kernel<<<M_int, 128, 0, stream>>>(
        router_logits.data_ptr<float>(),
        reinterpret_cast<bf16*>(routing_weights.data_ptr<c10::BFloat16>()),
        selected_experts.data_ptr<int64_t>(), M_int);
  }
  if (check_launches) {
    check_cuda(cudaGetLastError());
  }
}

}  // namespace

void run(torch::Tensor hidden_states,
         torch::Tensor gate_weight_fp4,
         torch::Tensor gate_scale,
         double gate_global_scale,
         torch::Tensor routing_weights,
         torch::Tensor selected_experts,
         torch::Tensor router_logits) {
  c10::cuda::CUDAGuard device_guard(hidden_states.device());
  std::lock_guard<std::mutex> guard(g_workspace_mutex);
  check_tensor(hidden_states, "hidden_states");
  check_tensor(gate_weight_fp4, "gate_weight_fp4");
  check_tensor(gate_scale, "gate_scale");
  check_tensor(routing_weights, "routing_weights");
  check_tensor(selected_experts, "selected_experts");
  check_tensor(router_logits, "router_logits");

  TORCH_CHECK(hidden_states.scalar_type() == torch::kBFloat16, "hidden_states must be bfloat16");
  TORCH_CHECK(gate_weight_fp4.element_size() == 1, "gate_weight_fp4 must be byte-packed fp4");
  TORCH_CHECK(gate_scale.element_size() == 1, "gate_scale must be fp8 bytes");
  TORCH_CHECK(routing_weights.scalar_type() == torch::kBFloat16, "routing_weights must be bfloat16");
  TORCH_CHECK(selected_experts.scalar_type() == torch::kLong, "selected_experts must be int64");
  TORCH_CHECK(router_logits.scalar_type() == torch::kFloat32, "router_logits must be float32");

  TORCH_CHECK(hidden_states.dim() == 2, "hidden_states must have shape [M, 2048]");
  TORCH_CHECK(gate_weight_fp4.dim() == 2, "gate_weight_fp4 must have shape [N, K/2]");
  TORCH_CHECK(gate_scale.dim() == 2, "gate_scale must have shape [N, K/16]");
  TORCH_CHECK(routing_weights.dim() == 2, "routing_weights must have shape [M, TOPK]");
  TORCH_CHECK(selected_experts.dim() == 2, "selected_experts must have shape [M, TOPK]");
  TORCH_CHECK(router_logits.dim() == 2, "router_logits must have shape [M, N]");

  const int64_t M = hidden_states.size(0);
  const int64_t K = hidden_states.size(1);
  const int64_t N = gate_weight_fp4.size(0);
  TORCH_CHECK(K == HIDDEN, "hidden_states K dimension mismatch");
  TORCH_CHECK(N == EXPERTS, "gate_weight_fp4 N dimension mismatch");
  TORCH_CHECK(gate_weight_fp4.size(1) * 2 == K, "gate_weight_fp4 K dimension mismatch");
  TORCH_CHECK(gate_scale.size(0) == N, "gate_scale N dimension mismatch");
  TORCH_CHECK(gate_scale.size(1) * 16 == K, "gate_scale K dimension mismatch");
  TORCH_CHECK(routing_weights.size(0) == M && routing_weights.size(1) == TOPK, "routing_weights shape mismatch");
  TORCH_CHECK(selected_experts.size(0) == M && selected_experts.size(1) == TOPK, "selected_experts shape mismatch");
  TORCH_CHECK(router_logits.size(0) == M && router_logits.size(1) == N, "router_logits shape mismatch");
  TORCH_CHECK(M % 128 == 0, "M must be a multiple of 128 for this iteration, got ", M);
  TORCH_CHECK(K % 256 == 0, "K must be a multiple of 256, got ", K);
  TORCH_CHECK(N % 128 == 0, "N must be a multiple of 128, got ", N);

  cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
  const int M_int = static_cast<int>(M);
  const int N_int = static_cast<int>(N);
  const int K_int = static_cast<int>(K);
  const float gate_global_scale_f = static_cast<float>(gate_global_scale);

  if (M_int == 3328 || M_int == 3456 || M_int == 3968) {
    launch_large_scalar_hwfp8_fallback(hidden_states, gate_weight_fp4, gate_scale,
                                       gate_global_scale_f, routing_weights, selected_experts,
                                       router_logits, M_int, K_int, stream);
    return;
  }

  ensure_workspace(hidden_states, gate_scale, M, K, N);
  launch_pipeline(hidden_states, gate_weight_fp4, gate_scale, gate_global_scale_f,
                  g_workspace_cache, routing_weights, selected_experts, router_logits,
                  M_int, N_int, K_int, stream, true);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("run", &run, "NVFP4 MoE router: quantize + tcgen05 GEMM + top-k");
}
