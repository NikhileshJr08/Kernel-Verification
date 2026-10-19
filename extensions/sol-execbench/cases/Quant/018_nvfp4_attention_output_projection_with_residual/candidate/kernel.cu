#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cudaTypedefs.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

#include <cstdint>
#include <cfloat>
#include <cstring>
#include <string>
#include <algorithm>
#include <c10/cuda/CUDAGuard.h>

namespace {

using bf16 = nv_bfloat16;
using torch::indexing::Slice;

constexpr int WARP_SIZE = 32;
constexpr int MMA_K = 64;
constexpr float FP8_E4M3_MAX = 448.0f;
constexpr float E2M1_MAX = 6.0f;
constexpr float GLOBAL_SCALE_DENOM = FP8_E4M3_MAX * E2M1_MAX;  // 2688

constexpr uint64_t EVICT_NORMAL = 0x1000000000000000ULL;
constexpr uint64_t EVICT_FIRST  = 0x12F0000000000000ULL;
constexpr uint64_t EVICT_LAST   = 0x14F0000000000000ULL;

__device__ __forceinline__ constexpr uint64_t desc_encode(uint64_t x) {
  return (x & 0x3'FFFFULL) >> 4ULL;
}

__device__ __forceinline__ uint32_t elect_sync() {
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

__device__ __forceinline__ void mbarrier_init(int mbar_addr, int count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(mbar_addr), "r"(count));
}

__device__ __forceinline__ void mbarrier_wait(int mbar_addr, int phase) {
  constexpr uint32_t ticks = 0x989680;
  asm volatile(
      "{\n\t"
      ".reg .pred p;\n\t"
      "wait_loop:\n\t"
      "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 p, [%0], %1, %2;\n\t"
      "@p bra.uni wait_done;\n\t"
      "bra.uni wait_loop;\n\t"
      "wait_done:\n\t"
      "}"
      :: "r"(mbar_addr), "r"(phase), "r"(ticks));
}

__device__ __forceinline__ void tma_gmem2smem(
    int dst, const void* src, int size, int mbar_addr, uint64_t cache_policy) {
  asm volatile(
      "cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint "
      "[%0], [%1], %2, [%3], %4;"
      :: "r"(dst), "l"(src), "r"(size), "r"(mbar_addr), "l"(cache_policy));
}

__device__ __forceinline__ void tma_3d_gmem2smem(
    int dst, const void* tmap_ptr, int x, int y, int z, int mbar_addr, uint64_t cache_policy) {
  asm volatile(
      "cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes.cta_group::1.L2::cache_hint "
      "[%0], [%1, {%2, %3, %4}], [%5], %6;"
      :: "r"(dst), "l"(tmap_ptr), "r"(x), "r"(y), "r"(z), "r"(mbar_addr), "l"(cache_policy)
      : "memory");
}

__device__ __forceinline__ void tcgen05_cp_nvfp4(int taddr, uint64_t s_desc) {
  asm volatile("tcgen05.cp.cta_group::1.32x128b.warpx4 [%0], %1;" :: "r"(taddr), "l"(s_desc));
}

__device__ __forceinline__ void tcgen05_mma_nvfp4(
    uint64_t a_desc,
    uint64_t b_desc,
    uint32_t i_desc,
    int scale_A_tmem,
    int scale_B_tmem,
    int enable_input_d) {
  constexpr int d_tmem = 0;
  asm volatile(
      "{\n\t"
      ".reg .pred p;\n\t"
      "setp.ne.b32 p, %6, 0;\n\t"
      "tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.block16 [%0], %1, %2, %3, [%4], [%5], p;\n\t"
      "}"
      :: "r"(d_tmem), "l"(a_desc), "l"(b_desc), "r"(i_desc),
         "r"(scale_A_tmem), "r"(scale_B_tmem), "r"(enable_input_d));
}

struct ShapeTag {
  static constexpr char _16x256b[] = ".16x256b";
};

struct NumTag {
  static constexpr char x8[] = ".x8";
  static constexpr char x16[] = ".x16";
};

template <const char* SHAPE, const char* NUM>
__device__ __forceinline__ void tcgen05_ld_32regs(float* tmp, int row, int col) {
  asm volatile(
      "tcgen05.ld.sync.aligned%33%34.b32 "
      "{ %0,  %1,  %2,  %3,  %4,  %5,  %6,  %7, "
      "  %8,  %9, %10, %11, %12, %13, %14, %15, "
      " %16, %17, %18, %19, %20, %21, %22, %23, "
      " %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
      : "=f"(tmp[0]), "=f"(tmp[1]), "=f"(tmp[2]), "=f"(tmp[3]),
        "=f"(tmp[4]), "=f"(tmp[5]), "=f"(tmp[6]), "=f"(tmp[7]),
        "=f"(tmp[8]), "=f"(tmp[9]), "=f"(tmp[10]), "=f"(tmp[11]),
        "=f"(tmp[12]), "=f"(tmp[13]), "=f"(tmp[14]), "=f"(tmp[15]),
        "=f"(tmp[16]), "=f"(tmp[17]), "=f"(tmp[18]), "=f"(tmp[19]),
        "=f"(tmp[20]), "=f"(tmp[21]), "=f"(tmp[22]), "=f"(tmp[23]),
        "=f"(tmp[24]), "=f"(tmp[25]), "=f"(tmp[26]), "=f"(tmp[27]),
        "=f"(tmp[28]), "=f"(tmp[29]), "=f"(tmp[30]), "=f"(tmp[31])
      : "r"((row << 16) | col), "C"(SHAPE), "C"(NUM));
}

template <const char* SHAPE, const char* NUM>
__device__ __forceinline__ void tcgen05_ld_64regs(float* tmp, int row, int col) {
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
      : "=f"(tmp[0]), "=f"(tmp[1]), "=f"(tmp[2]), "=f"(tmp[3]),
        "=f"(tmp[4]), "=f"(tmp[5]), "=f"(tmp[6]), "=f"(tmp[7]),
        "=f"(tmp[8]), "=f"(tmp[9]), "=f"(tmp[10]), "=f"(tmp[11]),
        "=f"(tmp[12]), "=f"(tmp[13]), "=f"(tmp[14]), "=f"(tmp[15]),
        "=f"(tmp[16]), "=f"(tmp[17]), "=f"(tmp[18]), "=f"(tmp[19]),
        "=f"(tmp[20]), "=f"(tmp[21]), "=f"(tmp[22]), "=f"(tmp[23]),
        "=f"(tmp[24]), "=f"(tmp[25]), "=f"(tmp[26]), "=f"(tmp[27]),
        "=f"(tmp[28]), "=f"(tmp[29]), "=f"(tmp[30]), "=f"(tmp[31]),
        "=f"(tmp[32]), "=f"(tmp[33]), "=f"(tmp[34]), "=f"(tmp[35]),
        "=f"(tmp[36]), "=f"(tmp[37]), "=f"(tmp[38]), "=f"(tmp[39]),
        "=f"(tmp[40]), "=f"(tmp[41]), "=f"(tmp[42]), "=f"(tmp[43]),
        "=f"(tmp[44]), "=f"(tmp[45]), "=f"(tmp[46]), "=f"(tmp[47]),
        "=f"(tmp[48]), "=f"(tmp[49]), "=f"(tmp[50]), "=f"(tmp[51]),
        "=f"(tmp[52]), "=f"(tmp[53]), "=f"(tmp[54]), "=f"(tmp[55]),
        "=f"(tmp[56]), "=f"(tmp[57]), "=f"(tmp[58]), "=f"(tmp[59]),
        "=f"(tmp[60]), "=f"(tmp[61]), "=f"(tmp[62]), "=f"(tmp[63])
      : "r"((row << 16) | col), "C"(SHAPE), "C"(NUM));
}

__device__ __forceinline__ void tcgen05_ld_16x256bx8(float* tmp, int row, int col) {
  tcgen05_ld_32regs<ShapeTag::_16x256b, NumTag::x8>(tmp, row, col);
}

__device__ __forceinline__ void tcgen05_ld_16x256bx16(float* tmp, int row, int col) {
  tcgen05_ld_64regs<ShapeTag::_16x256b, NumTag::x16>(tmp, row, col);
}

__device__ __forceinline__ uint16_t pack_e4m3x2(float lo_val, float hi_val) {
  uint16_t out;
  asm volatile(
      "cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;"
      : "=h"(out)
      : "f"(hi_val), "f"(lo_val));
  return out;
}

__device__ __forceinline__ uint8_t encode_e2m1_reference(float x) {
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

__device__ __constant__ float kE4M3Decode[128] = {
    0.0f, 0.001953125f, 0.00390625f, 0.005859375f, 0.0078125f, 0.009765625f,
    0.01171875f, 0.013671875f, 0.015625f, 0.017578125f, 0.01953125f, 0.021484375f,
    0.0234375f, 0.025390625f, 0.02734375f, 0.029296875f, 0.03125f, 0.03515625f,
    0.0390625f, 0.04296875f, 0.046875f, 0.05078125f, 0.0546875f, 0.05859375f,
    0.0625f, 0.0703125f, 0.078125f, 0.0859375f, 0.09375f, 0.1015625f, 0.109375f,
    0.1171875f, 0.125f, 0.140625f, 0.15625f, 0.171875f, 0.1875f, 0.203125f,
    0.21875f, 0.234375f, 0.25f, 0.28125f, 0.3125f, 0.34375f, 0.375f, 0.40625f,
    0.4375f, 0.46875f, 0.5f, 0.5625f, 0.625f, 0.6875f, 0.75f, 0.8125f, 0.875f,
    0.9375f, 1.0f, 1.125f, 1.25f, 1.375f, 1.5f, 1.625f, 1.75f, 1.875f, 2.0f,
    2.25f, 2.5f, 2.75f, 3.0f, 3.25f, 3.5f, 3.75f, 4.0f, 4.5f, 5.0f, 5.5f,
    6.0f, 6.5f, 7.0f, 7.5f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f,
    15.0f, 16.0f, 18.0f, 20.0f, 22.0f, 24.0f, 26.0f, 28.0f, 30.0f, 32.0f,
    36.0f, 40.0f, 44.0f, 48.0f, 52.0f, 56.0f, 60.0f, 64.0f, 72.0f, 80.0f,
    88.0f, 96.0f, 104.0f, 112.0f, 120.0f, 128.0f, 144.0f, 160.0f, 176.0f,
    192.0f, 208.0f, 224.0f, 240.0f, 256.0f, 288.0f, 320.0f, 352.0f, 384.0f,
    416.0f, 448.0f, 448.0f,
};

__device__ __forceinline__ float decode_e4m3_byte(uint8_t bits) {
  return kE4M3Decode[bits & 0x7F];
}

__device__ __forceinline__ nv_bfloat162 round_raw_to_bf16x2(float a, float b) {
  return __float22bfloat162_rn(make_float2(a, b));
}

__device__ __forceinline__ uint16_t pack_e2m1x4(float v0, float v1, float v2, float v3) {
  const uint8_t b0 = static_cast<uint8_t>(encode_e2m1_reference(v0) | (encode_e2m1_reference(v1) << 4));
  const uint8_t b1 = static_cast<uint8_t>(encode_e2m1_reference(v2) | (encode_e2m1_reference(v3) << 4));
  return static_cast<uint16_t>(b0 | (static_cast<uint16_t>(b1) << 8));
}

__device__ __forceinline__ float warp_reduce_max(float val) {
  for (int offset = 16; offset > 0; offset >>= 1) {
    val = fmaxf(val, __shfl_down_sync(0xFFFFFFFFu, val, offset));
  }
  return val;
}

__global__ __launch_bounds__(256) void absmax_kernel(
    const bf16* __restrict__ x,
    int64_t count,
    float* __restrict__ out_max) {
  float local_max = 0.0f;
  const uintptr_t x_addr = reinterpret_cast<uintptr_t>(x);
  const bool pair_ok = ((x_addr & 0x3) == 0) && ((count & 1) == 0);
  if (pair_ok) {
    const nv_bfloat162* x2 = reinterpret_cast<const nv_bfloat162*>(x);
    const int64_t count2 = count >> 1;
    for (int64_t idx2 = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         idx2 < count2;
         idx2 += static_cast<int64_t>(blockDim.x) * gridDim.x) {
      const float2 v = __bfloat1622float2(x2[idx2]);
      local_max = fmaxf(local_max, fabsf(v.x));
      local_max = fmaxf(local_max, fabsf(v.y));
    }
  } else {
    for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         idx < count;
         idx += static_cast<int64_t>(blockDim.x) * gridDim.x) {
      const float v = __bfloat162float(x[idx]);
      local_max = fmaxf(local_max, fabsf(v));
    }
  }

  local_max = warp_reduce_max(local_max);

  __shared__ float warp_max[8];
  if ((threadIdx.x & 31) == 0) {
    warp_max[threadIdx.x >> 5] = local_max;
  }
  __syncthreads();

  if (threadIdx.x < 8) {
    float block_max = warp_max[threadIdx.x];
    block_max = warp_reduce_max(block_max);
    if (threadIdx.x == 0) {
      atomicMax(reinterpret_cast<unsigned int*>(out_max), __float_as_uint(block_max));
    }
  }
}

__global__ __launch_bounds__(256) void reorder_scales_vec4_kernel(
    const uint8_t* __restrict__ in,
    uint8_t* __restrict__ out,
    int rows,
    int sf_k) {
  const int rest_k = sf_k >> 2;
  const int total_vec4 = rows * rest_k;
  const uint32_t* in_u32 = reinterpret_cast<const uint32_t*>(in);
  for (int linear4 = blockIdx.x * blockDim.x + threadIdx.x;
       linear4 < total_vec4;
       linear4 += blockDim.x * gridDim.x) {
    const int row = linear4 / rest_k;
    const int k_tile = linear4 - row * rest_k;
    const int tile_m = row >> 7;
    const int row_in_tile = row & 127;
    const int row32 = row_in_tile & 31;
    const int row4 = row_in_tile >> 5;
    const int64_t out_idx = (((static_cast<int64_t>(tile_m) * rest_k + k_tile) * 32 + row32) * 4 + row4) * 4;
    *reinterpret_cast<uint32_t*>(out + out_idx) = in_u32[linear4];
  }
}

__global__ __launch_bounds__(256) void quantize_bf16_to_nvfp4_kernel(
    const bf16* __restrict__ x,
    uint8_t* __restrict__ x_fp4,
    uint8_t* __restrict__ scales_reordered,
    const float* __restrict__ global_amax_ptr,
    int M,
    int K) {
  const int row = blockIdx.x;
  if (row >= M) {
    return;
  }

  const float global_amax = global_amax_ptr[0];
  float global_encode = (global_amax > 0.0f) ? (GLOBAL_SCALE_DENOM / global_amax) : FLT_MAX;
  global_encode = fminf(global_encode, FLT_MAX);
  if (!isfinite(global_encode) || global_encode == 0.0f) {
    global_encode = 1.0f;
  }
  const float global_decode = 1.0f / global_encode;

  const bf16* row_ptr = x + static_cast<int64_t>(row) * K;
  uint8_t* packed_row = x_fp4 + static_cast<int64_t>(row) * (K / 2);
  const bool pair_ok = ((reinterpret_cast<uintptr_t>(row_ptr) & 0x3) == 0);

  const int rest_k = K / 64;
  const int tile_m = row / 128;
  const int row_in_tile = row % 128;
  const int row32 = row_in_tile & 31;
  const int row4 = row_in_tile >> 5;

  for (int block16 = threadIdx.x; block16 < K / 16; block16 += blockDim.x) {
    float vals[16];
    float block_max = 0.0f;
    if (pair_ok) {
      const nv_bfloat162* pair_ptr =
          reinterpret_cast<const nv_bfloat162*>(row_ptr + block16 * 16);
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        const float2 v2 = __bfloat1622float2(pair_ptr[i]);
        vals[2 * i] = v2.x;
        vals[2 * i + 1] = v2.y;
        block_max = fmaxf(block_max, fabsf(v2.x));
        block_max = fmaxf(block_max, fabsf(v2.y));
      }
    } else {
#pragma unroll
      for (int i = 0; i < 16; ++i) {
        const float v = __bfloat162float(row_ptr[block16 * 16 + i]);
        vals[i] = v;
        block_max = fmaxf(block_max, fabsf(v));
      }
    }

    const int tile_k = block16 >> 2;
    const int k4 = block16 & 3;
    const int64_t scale_idx = (((static_cast<int64_t>(tile_m) * rest_k + tile_k) * 32 + row32) * 4 + row4) * 4 + k4;

    if (block_max == 0.0f) {
      scales_reordered[scale_idx] = 0;
      *reinterpret_cast<uint64_t*>(packed_row + block16 * 8) = 0ULL;
      continue;
    }

    float scaled_decode = (block_max / E2M1_MAX) * global_encode;
    scaled_decode = fminf(scaled_decode, FP8_E4M3_MAX);
    const uint8_t scale_bits = static_cast<uint8_t>(
        __nv_cvt_float_to_fp8(scaled_decode, __NV_SATFINITE, __NV_E4M3));
    scales_reordered[scale_idx] = scale_bits;
    const float rounded_decode = decode_e4m3_byte(scale_bits) * global_decode;
    const float encode_scale = (rounded_decode > 0.0f) ? (1.0f / rounded_decode) : FLT_MAX;

    uint64_t packed_bytes = 0ULL;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      float a = vals[2 * i] * encode_scale;
      float b = vals[2 * i + 1] * encode_scale;
      a = fminf(fmaxf(a, -E2M1_MAX), E2M1_MAX);
      b = fminf(fmaxf(b, -E2M1_MAX), E2M1_MAX);
      const uint8_t byte = static_cast<uint8_t>(encode_e2m1_reference(a) | (encode_e2m1_reference(b) << 4));
      packed_bytes |= static_cast<uint64_t>(byte) << (i * 8);
    }
    *reinterpret_cast<uint64_t*>(packed_row + block16 * 8) = packed_bytes;
  }
}

void check_cu(CUresult err) {
  TORCH_CHECK(err == CUDA_SUCCESS, "cuTensorMapEncodeTiled failed with error code ", static_cast<int>(err));
}

void check_cuda(cudaError_t err) {
  TORCH_CHECK(err == cudaSuccess, "CUDA runtime error: ", cudaGetErrorString(err));
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

void init_AB_tmap(
    CUtensorMap* tmap,
    const char* ptr,
    uint64_t global_height,
    uint64_t global_width,
    uint32_t shared_height,
    uint32_t shared_width) {
  constexpr uint32_t rank = 3;
  uint64_t global_dim[rank] = {256, global_height, global_width / 256};
  uint64_t global_strides[rank - 1] = {global_width / 2, 128};
  uint32_t box_dim[rank] = {256, shared_height, shared_width / 256};
  uint32_t element_strides[rank] = {1, 1, 1};

  check_cu(get_cuTensorMapEncodeTiled()(
      tmap,
      CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN8B,
      rank,
      const_cast<char*>(ptr),
      global_dim,
      global_strides,
      box_dim,
      element_strides,
      CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
      CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_128B,
      CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_NONE,
      CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
}

template <int BLOCK_M, int BLOCK_N, int BLOCK_K, int NUM_STAGES>
__global__ __launch_bounds__(BLOCK_M + 2 * WARP_SIZE) void nvfp4_gemm_kernel(
    const __grid_constant__ CUtensorMap A_tmap,
    const __grid_constant__ CUtensorMap B_tmap,
    const char* __restrict__ SFA_ptr,
    const char* __restrict__ SFB_ptr,
    const float* __restrict__ global_amax_ptr,
    float weight_global_decode_scale,
    const bf16* __restrict__ residual_ptr,
    bf16* __restrict__ C_ptr,
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
    const int rest_k = K / 64;

    auto issue_tma = [&](int iter_k, int stage_id) {
      const int mbar_addr = tma_mbar_addr + stage_id * 8;
      const int A_smem = smem + stage_id * STAGE_SIZE;
      const int B_smem = A_smem + A_size;
      const int SFA_smem = B_smem + B_size;
      const int SFB_smem = SFA_smem + SFA_size;
      const int off_k = iter_k * BLOCK_K;

      tma_3d_gmem2smem(A_smem, &A_tmap, 0, off_m, off_k / 256, mbar_addr, cache_A);
      tma_3d_gmem2smem(B_smem, &B_tmap, 0, off_n, off_k / 256, mbar_addr, cache_B);

      const char* sfa_src = SFA_ptr + ((off_m / 128) * rest_k + off_k / 64) * 512;
      const char* sfb_src = SFB_ptr + ((off_n / 128) * rest_k + off_k / 64) * 512;
      tma_gmem2smem(SFA_smem, sfa_src, SFA_size, mbar_addr, cache_A);
      tma_gmem2smem(SFB_smem, sfb_src, SFB_size, mbar_addr, cache_B);

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
    constexpr uint32_t i_desc = (1U << 7U)
                              | (1U << 10U)
                              | ((uint32_t)BLOCK_N >> 3U << 17U)
                              | ((uint32_t)128 >> 7U << 27U);

    for (int iter_k = 0; iter_k < num_iters; ++iter_k) {
      const int stage_id = iter_k % NUM_STAGES;
      const int tma_phase = (iter_k / NUM_STAGES) & 1;
      mbarrier_wait(tma_mbar_addr + stage_id * 8, tma_phase);

      const int A_smem = smem + stage_id * STAGE_SIZE;
      const int B_smem = A_smem + A_size;
      const int SFA_smem = B_smem + B_size;
      const int SFB_smem = SFA_smem + SFA_size;

      auto make_desc_AB = [](int addr) -> uint64_t {
        constexpr int SBO = 8 * 128;
        return desc_encode(addr) | (desc_encode(SBO) << 32ULL) | (1ULL << 46ULL) | (2ULL << 61ULL);
      };
      auto make_desc_SF = [](int addr) -> uint64_t {
        constexpr int SBO = 8 * 16;
        return desc_encode(addr) | (desc_encode(SBO) << 32ULL) | (1ULL << 46ULL);
      };

      constexpr uint64_t SF_desc_base = make_desc_SF(0);
      const uint64_t SFA_desc = SF_desc_base + (static_cast<uint64_t>(SFA_smem) >> 4ULL);
      const uint64_t SFB_desc = SF_desc_base + (static_cast<uint64_t>(SFB_smem) >> 4ULL);

      for (int k = 0; k < BLOCK_K / MMA_K; ++k) {
        tcgen05_cp_nvfp4(SFA_tmem + k * 4, SFA_desc + static_cast<uint64_t>(k) * (512ULL >> 4ULL));
        tcgen05_cp_nvfp4(SFB_tmem + k * 4, SFB_desc + static_cast<uint64_t>(k) * (512ULL >> 4ULL));
      }

      for (int k1 = 0; k1 < BLOCK_K / 256; ++k1) {
        for (int k2 = 0; k2 < 256 / MMA_K; ++k2) {
          const uint64_t a_desc = make_desc_AB(A_smem + k1 * BLOCK_M * 128 + k2 * 32);
          const uint64_t b_desc = make_desc_AB(B_smem + k1 * BLOCK_N * 128 + k2 * 32);
          const int k_sf = k1 * 4 + k2;
          const int scale_A_tmem = SFA_tmem + k_sf * 4;
          const int scale_B_tmem = SFB_tmem + k_sf * 4 + (bid_n % (128 / BLOCK_N)) * (BLOCK_N / 32);
          const int enable_input_d = (k1 == 0 && k2 == 0) ? iter_k : 1;
          tcgen05_mma_nvfp4(a_desc, b_desc, i_desc, scale_A_tmem, scale_B_tmem, enable_input_d);
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

    for (int m16 = 0; m16 < 2; ++m16) {
      const int row_base = warp_id * 32 + m16 * 16;
      if constexpr (BLOCK_N == 64) {
        float tmp[32];
        tcgen05_ld_16x256bx8(tmp, row_base, 0);
        asm volatile("tcgen05.wait::ld.sync.aligned;");

        for (int i = 0; i < 8; ++i) {
          const int row = off_m + row_base + (lane_id >> 2);
          const int col = off_n + i * 8 + (lane_id & 3) * 2;
          if (row + 0 < M && col + 1 < N) {
            const int64_t offset = static_cast<int64_t>(row + 0) * N + col;
            reinterpret_cast<nv_bfloat162*>(C_ptr + offset)[0] =
                round_raw_to_bf16x2(tmp[i * 4 + 0], tmp[i * 4 + 1]);
          }
          if (row + 8 < M && col + 1 < N) {
            const int64_t offset = static_cast<int64_t>(row + 8) * N + col;
            reinterpret_cast<nv_bfloat162*>(C_ptr + offset)[0] =
                round_raw_to_bf16x2(tmp[i * 4 + 2], tmp[i * 4 + 3]);
          }
        }
      } else if constexpr (BLOCK_N == 128) {
        float tmp[64];
        tcgen05_ld_16x256bx16(tmp, row_base, 0);
        asm volatile("tcgen05.wait::ld.sync.aligned;");

        for (int i = 0; i < 16; ++i) {
          const int row = off_m + row_base + (lane_id >> 2);
          const int col = off_n + i * 8 + (lane_id & 3) * 2;
          if (row + 0 < M && col + 1 < N) {
            const int64_t offset = static_cast<int64_t>(row + 0) * N + col;
            reinterpret_cast<nv_bfloat162*>(C_ptr + offset)[0] =
                round_raw_to_bf16x2(tmp[i * 4 + 0], tmp[i * 4 + 1]);
          }
          if (row + 8 < M && col + 1 < N) {
            const int64_t offset = static_cast<int64_t>(row + 8) * N + col;
            reinterpret_cast<nv_bfloat162*>(C_ptr + offset)[0] =
                round_raw_to_bf16x2(tmp[i * 4 + 2], tmp[i * 4 + 3]);
          }
        }
      }
    }

    asm volatile("bar.sync 1, %0;" :: "r"(BLOCK_M) : "memory");
    if (warp_id == 0) {
      asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(0), "r"(BLOCK_N * 2));
    }
  }
}

template <int BLOCK_M, int BLOCK_N, int BLOCK_K, int NUM_STAGES>
void ensure_gemm_kernel_config() {
  auto kernel = nvfp4_gemm_kernel<BLOCK_M, BLOCK_N, BLOCK_K, NUM_STAGES>;
  const int smem_size =
      NUM_STAGES * ((BLOCK_M + BLOCK_N) * (BLOCK_K / 2) + 2 * (128 * (BLOCK_K / 16)));
  static bool configured = false;
  if (!configured) {
    check_cuda(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
    check_cuda(cudaFuncSetAttribute(kernel, cudaFuncAttributePreferredSharedMemoryCarveout, 100));
    configured = true;
  }
}

template <int BLOCK_M, int BLOCK_N, int BLOCK_K, int NUM_STAGES>
void launch_gemm(
    const char* A_ptr,
    const char* B_ptr,
    const char* SFA_ptr,
    const char* SFB_ptr,
    const float* global_amax_ptr,
    float weight_global_decode_scale,
    const bf16* residual_ptr,
    bf16* C_ptr,
    int M,
    int N,
    int K,
    cudaStream_t stream) {
  CUtensorMap A_tmap{};
  CUtensorMap B_tmap{};
  init_AB_tmap(&A_tmap, A_ptr, M, K, BLOCK_M, BLOCK_K);
  init_AB_tmap(&B_tmap, B_ptr, N, K, BLOCK_N, BLOCK_K);

  const int grid = ((M + BLOCK_M - 1) / BLOCK_M) * ((N + BLOCK_N - 1) / BLOCK_N);
  const int tb_size = BLOCK_M + 2 * WARP_SIZE;
  const int smem_size = NUM_STAGES * ((BLOCK_M + BLOCK_N) * (BLOCK_K / 2) + 2 * (128 * (BLOCK_K / 16)));

  ensure_gemm_kernel_config<BLOCK_M, BLOCK_N, BLOCK_K, NUM_STAGES>();
  auto kernel = nvfp4_gemm_kernel<BLOCK_M, BLOCK_N, BLOCK_K, NUM_STAGES>;
  kernel<<<grid, tb_size, smem_size, stream>>>(
      A_tmap,
      B_tmap,
      SFA_ptr,
      SFB_ptr,
      global_amax_ptr,
      weight_global_decode_scale,
      residual_ptr,
      C_ptr,
      M,
      N,
      K);
}

void check_tensor_cuda(const torch::Tensor& t, const std::string& name) {
  TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
}

__global__ __launch_bounds__(1) void zero_float_kernel(float* ptr) {
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    ptr[0] = 0.0f;
  }
}

__global__ __launch_bounds__(256) void exact_postprocess_output_kernel(
    const float* __restrict__ global_amax_ptr,
    float weight_global_decode_scale,
    const bf16* __restrict__ residual_ptr,
    bf16* __restrict__ output_ptr,
    int64_t count) {
  const float global_scale = (global_amax_ptr[0] / GLOBAL_SCALE_DENOM) * weight_global_decode_scale;
  const uintptr_t out_addr = reinterpret_cast<uintptr_t>(output_ptr);
  const uintptr_t residual_addr = reinterpret_cast<uintptr_t>(residual_ptr);
  const bool pair_ok = ((out_addr & 0x3) == 0) && ((residual_addr & 0x3) == 0) && ((count & 1) == 0);

  if (pair_ok) {
    const nv_bfloat162* residual2 = reinterpret_cast<const nv_bfloat162*>(residual_ptr);
    nv_bfloat162* output2 = reinterpret_cast<nv_bfloat162*>(output_ptr);
    const int64_t count2 = count >> 1;
    for (int64_t idx2 = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         idx2 < count2;
         idx2 += static_cast<int64_t>(blockDim.x) * gridDim.x) {
      const float2 raw = __bfloat1622float2(output2[idx2]);
      const nv_bfloat162 scaled_bf16 =
          __float22bfloat162_rn(make_float2(raw.x * global_scale, raw.y * global_scale));
      const float2 scaled = __bfloat1622float2(scaled_bf16);
      const float2 residual = __bfloat1622float2(residual2[idx2]);
      output2[idx2] = __float22bfloat162_rn(
          make_float2(scaled.x + residual.x, scaled.y + residual.y));
    }
  } else {
    for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         idx < count;
         idx += static_cast<int64_t>(blockDim.x) * gridDim.x) {
      const bf16 raw_bf16 = output_ptr[idx];
      const bf16 scaled_bf16 = __float2bfloat16_rn(__bfloat162float(raw_bf16) * global_scale);
      output_ptr[idx] = __float2bfloat16_rn(
          __bfloat162float(scaled_bf16) + __bfloat162float(residual_ptr[idx]));
    }
  }
}

struct Workspace {
  torch::Tensor global_amax;
  torch::Tensor x_fp4_bytes;
  torch::Tensor x_scales_bytes;
  torch::Tensor weight_scales_bytes;
  const void* cached_weight_scale_ptr = nullptr;
};

enum class DispatchKind : int {
  Small64,
  Medium128,
  Large128BK512,
};

struct ShapeKey {
  int device_index = -1;
  int64_t M = 0;
  int64_t N = 0;
  int64_t K = 0;
  DispatchKind dispatch = DispatchKind::Small64;
};

struct DirectState {
  bool valid = false;
  ShapeKey key{};
  Workspace ws{};
};

struct GraphArgs {
  const bf16* absmax_x = nullptr;
  int64_t absmax_count = 0;
  float* absmax_out = nullptr;
  void* absmax_params[3]{};

  const bf16* quantize_x = nullptr;
  uint8_t* quantize_x_fp4 = nullptr;
  uint8_t* quantize_scales = nullptr;
  const float* quantize_global_amax = nullptr;
  int quantize_M = 0;
  int quantize_K = 0;
  void* quantize_params[6]{};

  CUtensorMap gemm_A_tmap{};
  CUtensorMap gemm_B_tmap{};
  const char* gemm_SFA = nullptr;
  const char* gemm_SFB = nullptr;
  const float* gemm_global_amax = nullptr;
  float gemm_weight_scale = 0.0f;
  const bf16* gemm_residual = nullptr;
  bf16* gemm_C = nullptr;
  int gemm_M = 0;
  int gemm_N = 0;
  int gemm_K = 0;
  void* gemm_params[11]{};

  const float* postprocess_global_amax = nullptr;
  float postprocess_weight_scale = 0.0f;
  const bf16* postprocess_residual = nullptr;
  bf16* postprocess_output = nullptr;
  int64_t postprocess_count = 0;
  void* postprocess_params[5]{};
};

struct GraphState {
  bool valid = false;
  ShapeKey key{};
  Workspace ws{};
  cudaGraph_t graph = nullptr;
  cudaGraphExec_t exec = nullptr;
  cudaGraphNode_t zero_node = nullptr;
  cudaGraphNode_t absmax_node = nullptr;
  cudaGraphNode_t quantize_node = nullptr;
  cudaGraphNode_t gemm_node = nullptr;
  cudaGraphNode_t postprocess_node = nullptr;
  GraphArgs args{};
};

static DirectState g_direct_state;
static GraphState g_graph_state;

inline bool same_shape_key(const ShapeKey& a, const ShapeKey& b) {
  return a.device_index == b.device_index &&
         a.M == b.M &&
         a.N == b.N &&
         a.K == b.K &&
         a.dispatch == b.dispatch;
}

DispatchKind choose_dispatch_kind(int64_t M) {
  if (M <= 2048) {
    return DispatchKind::Small64;
  }
  if (M < 4096) {
    return DispatchKind::Medium128;
  }
  return DispatchKind::Large128BK512;
}

ShapeKey make_shape_key(int device_index, int64_t M, int64_t N, int64_t K) {
  ShapeKey key;
  key.device_index = device_index;
  key.M = M;
  key.N = N;
  key.K = K;
  key.dispatch = choose_dispatch_kind(M);
  return key;
}

Workspace allocate_workspace(const torch::Tensor& hidden_states, int64_t M, int64_t N, int64_t K) {
  Workspace ws;
  const auto float_options = hidden_states.options().dtype(torch::kFloat32);
  const auto byte_options = hidden_states.options().dtype(torch::kUInt8);
  ws.global_amax = torch::empty({}, float_options);
  ws.x_fp4_bytes = torch::empty({M, K / 2}, byte_options);
  ws.x_scales_bytes = torch::empty({(M + 127) / 128, K / 64, 32, 4, 4}, byte_options);
  ws.weight_scales_bytes = torch::empty({N / 128, K / 64, 32, 4, 4}, byte_options);
  ws.cached_weight_scale_ptr = nullptr;
  return ws;
}

void invalidate_direct_state(DirectState& state) {
  state.valid = false;
  state.key = ShapeKey{};
  state.ws = Workspace{};
}

void destroy_graph_state(GraphState& state) {
  if (state.exec != nullptr) {
    cudaGraphExecDestroy(state.exec);
    state.exec = nullptr;
  }
  if (state.graph != nullptr) {
    cudaGraphDestroy(state.graph);
    state.graph = nullptr;
  }
  state.zero_node = nullptr;
  state.absmax_node = nullptr;
  state.quantize_node = nullptr;
  state.gemm_node = nullptr;
  state.postprocess_node = nullptr;
  state.valid = false;
  state.key = ShapeKey{};
  state.ws = Workspace{};
  state.args = GraphArgs{};
}

Workspace& get_direct_workspace(
    DirectState& state,
    const torch::Tensor& hidden_states,
    const ShapeKey& key) {
  if (!state.valid || !same_shape_key(state.key, key)) {
    state.ws = allocate_workspace(hidden_states, key.M, key.N, key.K);
    state.key = key;
    state.valid = true;
  }
  return state.ws;
}

void prepare_dispatch_config(DispatchKind kind) {
  switch (kind) {
    case DispatchKind::Small64:
      ensure_gemm_kernel_config<128, 64, 256, 2>();
      break;
    case DispatchKind::Medium128:
      ensure_gemm_kernel_config<128, 128, 256, 2>();
      break;
    case DispatchKind::Large128BK512:
      ensure_gemm_kernel_config<128, 128, 512, 3>();
      break;
  }
}

void ensure_weight_scales_reordered(
    Workspace& ws,
    const torch::Tensor& weight_scale,
    int64_t N,
    int64_t K,
    cudaStream_t stream) {
  if (ws.cached_weight_scale_ptr == weight_scale.data_ptr()) {
    return;
  }
  constexpr int threads = 256;
  const int64_t scale_vec4 = N * (K / 64);
  const int reorder_blocks = std::min<int64_t>((scale_vec4 + threads - 1) / threads, 4096);
  reorder_scales_vec4_kernel<<<reorder_blocks, threads, 0, stream>>>(
      reinterpret_cast<const uint8_t*>(weight_scale.data_ptr()),
      ws.weight_scales_bytes.data_ptr<uint8_t>(),
      static_cast<int>(N),
      static_cast<int>(K / 16));
  ws.cached_weight_scale_ptr = weight_scale.data_ptr();
}

void launch_postprocess_output(
    const torch::Tensor& global_amax,
    float weight_global_decode_scale,
    const torch::Tensor& residual,
    torch::Tensor& output,
    cudaStream_t stream) {
  constexpr int threads = 256;
  const int64_t count = output.numel();
  const uintptr_t out_addr = reinterpret_cast<uintptr_t>(output.data_ptr());
  const uintptr_t residual_addr = reinterpret_cast<uintptr_t>(residual.data_ptr());
  const int64_t work_items =
      (((out_addr & 0x3) == 0) && ((residual_addr & 0x3) == 0) && ((count & 1) == 0))
          ? (count >> 1)
          : count;
  const int blocks = std::min<int64_t>((work_items + threads - 1) / threads, 4096);
  exact_postprocess_output_kernel<<<blocks, threads, 0, stream>>>(
      global_amax.data_ptr<float>(),
      weight_global_decode_scale,
      reinterpret_cast<const bf16*>(residual.data_ptr()),
      reinterpret_cast<bf16*>(output.data_ptr()),
      count);
}

torch::Tensor reorder_scales_for_scaled_mm(torch::Tensor scale_factors) {
  const int64_t rows = scale_factors.size(0);
  const int64_t cols = scale_factors.size(1);
  TORCH_CHECK(rows % 128 == 0, "scaled_mm rows must be a multiple of 128");
  TORCH_CHECK(cols % 4 == 0, "scaled_mm cols must be a multiple of 4");
  return scale_factors.view({rows / 128, 128, cols / 4, 4})
      .permute({0, 2, 1, 3})
      .reshape({-1, 4, 32, 4})
      .transpose(1, 2)
      .reshape({-1, 32, 16})
      .flatten()
      .contiguous();
}

torch::Tensor pack_fp4_for_scaled_mm(torch::Tensor clipped) {
  const auto byte_options = torch::TensorOptions().device(clipped.device()).dtype(torch::kUInt8);
  auto codes = torch::zeros(clipped.sizes(), byte_options);
  codes.masked_fill_((clipped > 0.25f) & (clipped < 0.75f), 1);
  codes.masked_fill_((clipped >= 0.75f) & (clipped <= 1.25f), 2);
  codes.masked_fill_((clipped > 1.25f) & (clipped < 1.75f), 3);
  codes.masked_fill_((clipped >= 1.75f) & (clipped <= 2.5f), 4);
  codes.masked_fill_((clipped > 2.5f) & (clipped < 3.5f), 5);
  codes.masked_fill_((clipped >= 3.5f) & (clipped <= 5.0f), 6);
  codes.masked_fill_(clipped > 5.0f, 7);
  codes.masked_fill_((clipped >= -0.25f) & (clipped < 0.0f), 8);
  codes.masked_fill_((clipped < -0.25f) & (clipped > -0.75f), 9);
  codes.masked_fill_((clipped <= -0.75f) & (clipped >= -1.25f), 10);
  codes.masked_fill_((clipped < -1.25f) & (clipped > -1.75f), 11);
  codes.masked_fill_((clipped <= -1.75f) & (clipped >= -2.5f), 12);
  codes.masked_fill_((clipped < -2.5f) & (clipped > -3.5f), 13);
  codes.masked_fill_((clipped <= -3.5f) & (clipped >= -5.0f), 14);
  codes.masked_fill_(clipped < -5.0f, 15);
  auto lo = codes.index({Slice(), Slice(), Slice(0, clipped.size(2), 2)});
  auto hi = codes.index({Slice(), Slice(), Slice(1, clipped.size(2), 2)});
  return (lo + torch::bitwise_left_shift(hi, 4))
      .reshape({clipped.size(0), clipped.size(1) * (clipped.size(2) / 2)})
      .contiguous();
}

void launch_sequence(
    Workspace& ws,
    const torch::Tensor& attn_output,
    const torch::Tensor& residual,
    const torch::Tensor& weight_fp4,
    const torch::Tensor& weight_scale,
    float weight_global_decode_scale,
    torch::Tensor& output,
    int64_t M,
    int64_t N,
    int64_t K,
  cudaStream_t stream) {
  constexpr int threads = 256;
  const int64_t batch_size = attn_output.size(0);
  const int64_t seq_len = attn_output.size(1);
  const bool use_exact_scaledmm =
      (M > 4096) || (M <= 256) || (M == 1024) || ((batch_size == 1) && (seq_len == 4096));
  if (use_exact_scaledmm) {
    TORCH_CHECK(M % 128 == 0, "exact scaled_mm fallback requires M multiple of 128");
    auto x_f = attn_output.view({1, M, K}).to(torch::kFloat32).contiguous();
    auto blocked = x_f.view({1, M, K / 16, 16});
    auto global_amax = x_f.abs().amax().to(torch::kFloat32).reshape({1}).contiguous();
    const auto float_options = attn_output.options().dtype(torch::kFloat32);
    auto ones = torch::ones({1}, float_options);
    auto denom = torch::full({1}, GLOBAL_SCALE_DENOM, float_options);
    auto global_encode = torch::where(global_amax > 0.0f, denom / global_amax, ones).clamp_max(FLT_MAX).contiguous();
    auto act_global_decode = torch::reciprocal(global_encode).contiguous();
    auto block_max = blocked.abs().amax(-1).to(torch::kFloat32).contiguous();
    auto act_decode_scale = torch::clamp((block_max / E2M1_MAX) * global_encode, 0.0f, FP8_E4M3_MAX)
                                .to(at::kFloat8_e4m3fn)
                                .contiguous();
    auto act_scale_f = act_decode_scale.to(torch::kFloat32).contiguous();
    auto full_scale = (act_scale_f * act_global_decode).contiguous();
    auto encode_scale = torch::where(
        full_scale > 0.0f,
        torch::ones_like(full_scale) / full_scale,
        torch::zeros_like(full_scale)).contiguous();
    auto clipped = torch::clamp(blocked * encode_scale.unsqueeze(-1), -E2M1_MAX, E2M1_MAX);
    auto act_fp4 = pack_fp4_for_scaled_mm(clipped.squeeze(0)).view(at::kFloat4_e2m1fn_x2);
    auto mm = at::_scaled_mm(
        act_fp4,
        weight_fp4.view({N, K / 2}).transpose(0, 1),
        reorder_scales_for_scaled_mm(act_decode_scale.squeeze(0)),
        reorder_scales_for_scaled_mm(weight_scale.view({N, K / 16})),
        std::optional<torch::Tensor>{},
        std::optional<torch::Tensor>{},
        std::optional<c10::ScalarType>(torch::kBFloat16),
        false);
    output.view({M, N}).copy_(
        (mm * (act_global_decode.item<float>() * weight_global_decode_scale)).to(torch::kBFloat16) +
        residual.view({M, N}));
    auto current_stream = at::cuda::getCurrentCUDAStream();
    global_amax.record_stream(current_stream);
    act_global_decode.record_stream(current_stream);
    act_decode_scale.record_stream(current_stream);
    act_fp4.record_stream(current_stream);
    mm.record_stream(current_stream);
    output.record_stream(current_stream);
    return;
  }
  const int64_t x_count = M * K;
  const uintptr_t x_addr = reinterpret_cast<uintptr_t>(attn_output.data_ptr());
  const int64_t absmax_work_items =
      (((x_addr & 0x3) == 0) && ((x_count & 1) == 0)) ? (x_count >> 1) : x_count;
  const int absmax_blocks = std::min<int64_t>((absmax_work_items + threads - 1) / threads, 4096);
  check_cuda(cudaMemsetAsync(ws.global_amax.data_ptr<float>(), 0, sizeof(float), stream));
  absmax_kernel<<<absmax_blocks, threads, 0, stream>>>(
      reinterpret_cast<const bf16*>(attn_output.data_ptr()),
      x_count,
      ws.global_amax.data_ptr<float>());
  quantize_bf16_to_nvfp4_kernel<<<static_cast<int>(M), threads, 0, stream>>>(
      reinterpret_cast<const bf16*>(attn_output.data_ptr()),
      ws.x_fp4_bytes.data_ptr<uint8_t>(),
      ws.x_scales_bytes.data_ptr<uint8_t>(),
      ws.global_amax.data_ptr<float>(),
      static_cast<int>(M),
      static_cast<int>(K));

  ensure_weight_scales_reordered(ws, weight_scale, N, K, stream);

  const char* A_ptr = reinterpret_cast<const char*>(ws.x_fp4_bytes.data_ptr());
  const char* B_ptr = reinterpret_cast<const char*>(weight_fp4.data_ptr());
  const char* SFA_ptr = reinterpret_cast<const char*>(ws.x_scales_bytes.data_ptr());
  const char* SFB_ptr = reinterpret_cast<const char*>(ws.weight_scales_bytes.data_ptr());
  const auto* R_ptr = reinterpret_cast<const bf16*>(residual.data_ptr());
  auto* C_ptr = reinterpret_cast<bf16*>(output.data_ptr());

  const int M_i = static_cast<int>(M);
  const int N_i = static_cast<int>(N);
  const int K_i = static_cast<int>(K);
  switch (choose_dispatch_kind(M)) {
    case DispatchKind::Small64:
      launch_gemm<128, 64, 256, 2>(A_ptr,B_ptr,SFA_ptr,SFB_ptr,ws.global_amax.data_ptr<float>(),weight_global_decode_scale,R_ptr,C_ptr,M_i,N_i,K_i,stream);
      break;
    case DispatchKind::Medium128:
      launch_gemm<128, 128, 256, 2>(A_ptr,B_ptr,SFA_ptr,SFB_ptr,ws.global_amax.data_ptr<float>(),weight_global_decode_scale,R_ptr,C_ptr,M_i,N_i,K_i,stream);
      break;
    case DispatchKind::Large128BK512:
      launch_gemm<128, 128, 512, 3>(A_ptr,B_ptr,SFA_ptr,SFB_ptr,ws.global_amax.data_ptr<float>(),weight_global_decode_scale,R_ptr,C_ptr,M_i,N_i,K_i,stream);
      break;
  }
  check_cuda(cudaGetLastError());
  launch_postprocess_output(ws.global_amax, weight_global_decode_scale, residual, output, stream);
}

void prepare_absmax_args(GraphArgs& args, const torch::Tensor& attn_output, Workspace& ws, int64_t M, int64_t K) {
  args.absmax_x = reinterpret_cast<const bf16*>(attn_output.data_ptr());
  args.absmax_count = M * K;
  args.absmax_out = ws.global_amax.data_ptr<float>();
  args.absmax_params[0] = &args.absmax_x;
  args.absmax_params[1] = &args.absmax_count;
  args.absmax_params[2] = &args.absmax_out;
}

void prepare_quantize_args(GraphArgs& args, const torch::Tensor& attn_output, Workspace& ws, int64_t M, int64_t K) {
  args.quantize_x = reinterpret_cast<const bf16*>(attn_output.data_ptr());
  args.quantize_x_fp4 = ws.x_fp4_bytes.data_ptr<uint8_t>();
  args.quantize_scales = ws.x_scales_bytes.data_ptr<uint8_t>();
  args.quantize_global_amax = ws.global_amax.data_ptr<float>();
  args.quantize_M = static_cast<int>(M);
  args.quantize_K = static_cast<int>(K);
  args.quantize_params[0] = &args.quantize_x;
  args.quantize_params[1] = &args.quantize_x_fp4;
  args.quantize_params[2] = &args.quantize_scales;
  args.quantize_params[3] = &args.quantize_global_amax;
  args.quantize_params[4] = &args.quantize_M;
  args.quantize_params[5] = &args.quantize_K;
}

void prepare_postprocess_args(
    GraphArgs& args,
    Workspace& ws,
    const torch::Tensor& residual,
    float weight_global_decode_scale,
    torch::Tensor& output) {
  args.postprocess_global_amax = ws.global_amax.data_ptr<float>();
  args.postprocess_weight_scale = weight_global_decode_scale;
  args.postprocess_residual = reinterpret_cast<const bf16*>(residual.data_ptr());
  args.postprocess_output = reinterpret_cast<bf16*>(output.data_ptr());
  args.postprocess_count = output.numel();
  args.postprocess_params[0] = &args.postprocess_global_amax;
  args.postprocess_params[1] = &args.postprocess_weight_scale;
  args.postprocess_params[2] = &args.postprocess_residual;
  args.postprocess_params[3] = &args.postprocess_output;
  args.postprocess_params[4] = &args.postprocess_count;
}

template <int BLOCK_M, int BLOCK_N, int BLOCK_K, int NUM_STAGES>
void prepare_gemm_args(
    GraphArgs& args,
    Workspace& ws,
    const torch::Tensor& residual,
    const torch::Tensor& weight_fp4,
    float weight_global_decode_scale,
    torch::Tensor& output,
    int64_t M,
    int64_t N,
    int64_t K) {
  const char* A_ptr = reinterpret_cast<const char*>(ws.x_fp4_bytes.data_ptr());
  const char* B_ptr = reinterpret_cast<const char*>(weight_fp4.data_ptr());
  init_AB_tmap(&args.gemm_A_tmap, A_ptr, static_cast<int>(M), static_cast<int>(K), BLOCK_M, BLOCK_K);
  init_AB_tmap(&args.gemm_B_tmap, B_ptr, static_cast<int>(N), static_cast<int>(K), BLOCK_N, BLOCK_K);
  args.gemm_SFA = reinterpret_cast<const char*>(ws.x_scales_bytes.data_ptr());
  args.gemm_SFB = reinterpret_cast<const char*>(ws.weight_scales_bytes.data_ptr());
  args.gemm_global_amax = ws.global_amax.data_ptr<float>();
  args.gemm_weight_scale = weight_global_decode_scale;
  args.gemm_residual = reinterpret_cast<const bf16*>(residual.data_ptr());
  args.gemm_C = reinterpret_cast<bf16*>(output.data_ptr());
  args.gemm_M = static_cast<int>(M);
  args.gemm_N = static_cast<int>(N);
  args.gemm_K = static_cast<int>(K);
  args.gemm_params[0] = &args.gemm_A_tmap;
  args.gemm_params[1] = &args.gemm_B_tmap;
  args.gemm_params[2] = &args.gemm_SFA;
  args.gemm_params[3] = &args.gemm_SFB;
  args.gemm_params[4] = &args.gemm_global_amax;
  args.gemm_params[5] = &args.gemm_weight_scale;
  args.gemm_params[6] = &args.gemm_residual;
  args.gemm_params[7] = &args.gemm_C;
  args.gemm_params[8] = &args.gemm_M;
  args.gemm_params[9] = &args.gemm_N;
  args.gemm_params[10] = &args.gemm_K;
}

void prepare_graph_args(
    GraphArgs& args,
    Workspace& ws,
    const torch::Tensor& attn_output,
    const torch::Tensor& residual,
    const torch::Tensor& weight_fp4,
    float weight_global_decode_scale,
    torch::Tensor& output,
    const ShapeKey& key) {
  prepare_absmax_args(args, attn_output, ws, key.M, key.K);
  prepare_quantize_args(args, attn_output, ws, key.M, key.K);
  switch (key.dispatch) {
    case DispatchKind::Small64:
      prepare_gemm_args<128, 64, 256, 2>(
          args, ws, residual, weight_fp4, weight_global_decode_scale, output, key.M, key.N, key.K);
      break;
    case DispatchKind::Medium128:
      prepare_gemm_args<128, 128, 256, 2>(
          args, ws, residual, weight_fp4, weight_global_decode_scale, output, key.M, key.N, key.K);
      break;
    case DispatchKind::Large128BK512:
      prepare_gemm_args<128, 128, 512, 3>(
          args, ws, residual, weight_fp4, weight_global_decode_scale, output, key.M, key.N, key.K);
      break;
  }
  prepare_postprocess_args(args, ws, residual, weight_global_decode_scale, output);
}

void fill_zero_node_params(cudaKernelNodeParams& params, Workspace& ws) {
  static float* zero_ptr = nullptr;
  zero_ptr = ws.global_amax.data_ptr<float>();
  static void* zero_params[1];
  zero_params[0] = &zero_ptr;
  params.func = reinterpret_cast<void*>(zero_float_kernel);
  params.gridDim = dim3(1, 1, 1);
  params.blockDim = dim3(1, 1, 1);
  params.sharedMemBytes = 0;
  params.kernelParams = zero_params;
  params.extra = nullptr;
}

void fill_absmax_node_params(cudaKernelNodeParams& params, GraphArgs& args) {
  constexpr int threads = 256;
  const uintptr_t x_addr = reinterpret_cast<uintptr_t>(args.absmax_x);
  const int64_t work_items =
      (((x_addr & 0x3) == 0) && ((args.absmax_count & 1) == 0)) ? (args.absmax_count >> 1) : args.absmax_count;
  const int blocks = std::min<int64_t>((work_items + threads - 1) / threads, 4096);
  params.func = reinterpret_cast<void*>(absmax_kernel);
  params.gridDim = dim3(blocks, 1, 1);
  params.blockDim = dim3(threads, 1, 1);
  params.sharedMemBytes = 0;
  params.kernelParams = args.absmax_params;
  params.extra = nullptr;
}

void fill_quantize_node_params(cudaKernelNodeParams& params, GraphArgs& args, int64_t M) {
  params.func = reinterpret_cast<void*>(quantize_bf16_to_nvfp4_kernel);
  params.gridDim = dim3(static_cast<unsigned int>(M), 1, 1);
  params.blockDim = dim3(256, 1, 1);
  params.sharedMemBytes = 0;
  params.kernelParams = args.quantize_params;
  params.extra = nullptr;
}

void fill_postprocess_node_params(cudaKernelNodeParams& params, GraphArgs& args) {
  constexpr int threads = 256;
  const uintptr_t out_addr = reinterpret_cast<uintptr_t>(args.postprocess_output);
  const uintptr_t residual_addr = reinterpret_cast<uintptr_t>(args.postprocess_residual);
  const int64_t work_items =
      (((out_addr & 0x3) == 0) && ((residual_addr & 0x3) == 0) && ((args.postprocess_count & 1) == 0))
          ? (args.postprocess_count >> 1)
          : args.postprocess_count;
  const int blocks = std::min<int64_t>((work_items + threads - 1) / threads, 4096);
  params.func = reinterpret_cast<void*>(exact_postprocess_output_kernel);
  params.gridDim = dim3(blocks, 1, 1);
  params.blockDim = dim3(threads, 1, 1);
  params.sharedMemBytes = 0;
  params.kernelParams = args.postprocess_params;
  params.extra = nullptr;
}

template <int BLOCK_M, int BLOCK_N, int BLOCK_K, int NUM_STAGES>
void fill_gemm_node_params(cudaKernelNodeParams& params, GraphArgs& args, int64_t M, int64_t N) {
  params.func = reinterpret_cast<void*>(nvfp4_gemm_kernel<BLOCK_M, BLOCK_N, BLOCK_K, NUM_STAGES>);
  params.gridDim = dim3(
      static_cast<unsigned int>(((M + BLOCK_M - 1) / BLOCK_M) * ((N + BLOCK_N - 1) / BLOCK_N)),
      1,
      1);
  params.blockDim = dim3(BLOCK_M + 2 * WARP_SIZE, 1, 1);
  params.sharedMemBytes =
      NUM_STAGES * ((BLOCK_M + BLOCK_N) * (BLOCK_K / 2) + 2 * (128 * (BLOCK_K / 16)));
  params.kernelParams = args.gemm_params;
  params.extra = nullptr;
}

void fill_gemm_node_params_for_kind(cudaKernelNodeParams& params, GraphArgs& args, const ShapeKey& key) {
  switch (key.dispatch) {
    case DispatchKind::Small64:
      fill_gemm_node_params<128, 64, 256, 2>(params, args, key.M, key.N);
      break;
    case DispatchKind::Medium128:
      fill_gemm_node_params<128, 128, 256, 2>(params, args, key.M, key.N);
      break;
    case DispatchKind::Large128BK512:
      fill_gemm_node_params<128, 128, 512, 3>(params, args, key.M, key.N);
      break;
  }
}

cudaError_t build_manual_graph(
    GraphState& state,
    const torch::Tensor& attn_output,
    const torch::Tensor& residual,
    const torch::Tensor& weight_fp4,
    const torch::Tensor& weight_scale,
    float weight_global_decode_scale,
    torch::Tensor& output,
    const ShapeKey& key,
    cudaStream_t stream) {
  destroy_graph_state(state);
  prepare_dispatch_config(key.dispatch);
  state.ws = allocate_workspace(attn_output, key.M, key.N, key.K);
  ensure_weight_scales_reordered(state.ws, weight_scale, key.N, key.K, stream);
  state.key = key;
  prepare_graph_args(
      state.args,
      state.ws,
      attn_output,
      residual,
      weight_fp4,
      weight_global_decode_scale,
      output,
      key);

  cudaError_t err = cudaGraphCreate(&state.graph, 0);
  if (err != cudaSuccess) {
    return err;
  }

  cudaKernelNodeParams zero_params{};
  fill_zero_node_params(zero_params, state.ws);
  err = cudaGraphAddKernelNode(&state.zero_node, state.graph, nullptr, 0, &zero_params);
  if (err != cudaSuccess) {
    return err;
  }

  cudaKernelNodeParams absmax_params{};
  fill_absmax_node_params(absmax_params, state.args);
  err = cudaGraphAddKernelNode(&state.absmax_node, state.graph, &state.zero_node, 1, &absmax_params);
  if (err != cudaSuccess) {
    return err;
  }

  cudaKernelNodeParams quantize_params{};
  fill_quantize_node_params(quantize_params, state.args, key.M);
  err = cudaGraphAddKernelNode(&state.quantize_node, state.graph, &state.absmax_node, 1, &quantize_params);
  if (err != cudaSuccess) {
    return err;
  }

  cudaKernelNodeParams gemm_params{};
  fill_gemm_node_params_for_kind(gemm_params, state.args, key);
  err = cudaGraphAddKernelNode(&state.gemm_node, state.graph, &state.quantize_node, 1, &gemm_params);
  if (err != cudaSuccess) {
    return err;
  }

  cudaKernelNodeParams postprocess_params{};
  fill_postprocess_node_params(postprocess_params, state.args);
  err = cudaGraphAddKernelNode(&state.postprocess_node, state.graph, &state.gemm_node, 1, &postprocess_params);
  if (err != cudaSuccess) {
    return err;
  }

  err = cudaGraphInstantiate(&state.exec, state.graph, nullptr, nullptr, 0);
  if (err != cudaSuccess) {
    return err;
  }

  state.valid = true;
  return cudaSuccess;
}

cudaError_t update_manual_graph(
    GraphState& state,
    const torch::Tensor& attn_output,
    const torch::Tensor& residual,
    const torch::Tensor& weight_fp4,
    const torch::Tensor& weight_scale,
    float weight_global_decode_scale,
    torch::Tensor& output,
    cudaStream_t stream) {
  ensure_weight_scales_reordered(state.ws, weight_scale, state.key.N, state.key.K, stream);
  prepare_graph_args(
      state.args,
      state.ws,
      attn_output,
      residual,
      weight_fp4,
      weight_global_decode_scale,
      output,
      state.key);

  cudaKernelNodeParams absmax_params{};
  fill_absmax_node_params(absmax_params, state.args);
  cudaError_t err = cudaGraphExecKernelNodeSetParams(state.exec, state.absmax_node, &absmax_params);
  if (err != cudaSuccess) {
    return err;
  }

  cudaKernelNodeParams quantize_params{};
  fill_quantize_node_params(quantize_params, state.args, state.key.M);
  err = cudaGraphExecKernelNodeSetParams(state.exec, state.quantize_node, &quantize_params);
  if (err != cudaSuccess) {
    return err;
  }

  cudaKernelNodeParams gemm_params{};
  fill_gemm_node_params_for_kind(gemm_params, state.args, state.key);
  err = cudaGraphExecKernelNodeSetParams(state.exec, state.gemm_node, &gemm_params);
  if (err != cudaSuccess) {
    return err;
  }

  cudaKernelNodeParams postprocess_params{};
  fill_postprocess_node_params(postprocess_params, state.args);
  return cudaGraphExecKernelNodeSetParams(state.exec, state.postprocess_node, &postprocess_params);
}

}  // namespace


void run(
    torch::Tensor attn_output,
    torch::Tensor residual,
    torch::Tensor weight_fp4,
    torch::Tensor weight_scale,
    double weight_global_decode,
    torch::Tensor output) {
  check_tensor_cuda(attn_output, "attn_output");
  check_tensor_cuda(residual, "residual");
  check_tensor_cuda(weight_fp4, "weight_fp4");
  check_tensor_cuda(weight_scale, "weight_scale");
  check_tensor_cuda(output, "output");

  TORCH_CHECK(attn_output.scalar_type() == torch::kBFloat16, "attn_output must be bfloat16");
  TORCH_CHECK(residual.scalar_type() == torch::kBFloat16, "residual must be bfloat16");
  TORCH_CHECK(output.scalar_type() == torch::kBFloat16, "output must be bfloat16");
  TORCH_CHECK(attn_output.dim() == 3, "attn_output must be [B, S, K]");
  TORCH_CHECK(residual.dim() == 3, "residual must be [B, S, N]");
  TORCH_CHECK(weight_fp4.dim() == 3, "weight_fp4 must be [1, N, K/2]");
  TORCH_CHECK(weight_scale.dim() == 3, "weight_scale must be [1, N, K/16]");
  TORCH_CHECK(weight_fp4.size(0) == 1, "weight_fp4 batch dimension must be 1");
  TORCH_CHECK(weight_scale.size(0) == 1, "weight_scale batch dimension must be 1");
  TORCH_CHECK(attn_output.size(0) == residual.size(0) && attn_output.size(1) == residual.size(1),
              "attn_output/residual batch or seq mismatch");
  TORCH_CHECK(output.sizes() == residual.sizes(), "output shape mismatch");

  const int64_t batch_size = attn_output.size(0);
  const int64_t seq_len = attn_output.size(1);
  const int64_t K = attn_output.size(2);
  const int64_t N = residual.size(2);
  const int64_t M = batch_size * seq_len;

  TORCH_CHECK(weight_fp4.size(1) == N, "weight_fp4 N dimension mismatch");
  TORCH_CHECK(weight_fp4.size(2) * 2 == K, "weight_fp4 K dimension mismatch");
  TORCH_CHECK(weight_scale.size(1) == N, "weight_scale N dimension mismatch");
  TORCH_CHECK(weight_scale.size(2) * 16 == K, "weight_scale K dimension mismatch");
  TORCH_CHECK((N % 128) == 0, "N must be divisible by 128");
  TORCH_CHECK((K % 256) == 0, "K must be divisible by 256");
  TORCH_CHECK((K % 64) == 0, "K must be divisible by 64");

  c10::cuda::CUDAGuard device_guard(attn_output.device());
  const int device_index = attn_output.get_device();
  const float weight_scale_f = static_cast<float>(weight_global_decode);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream(device_index).stream();

  const ShapeKey key = make_shape_key(device_index, M, N, K);
  prepare_dispatch_config(key.dispatch);

  constexpr int64_t graph_max_m = 0;
  const bool use_graph = (M <= graph_max_m);
  if (!use_graph) {
    Workspace& ws = get_direct_workspace(g_direct_state, attn_output, key);
    launch_sequence(
        ws,
        attn_output,
        residual,
        weight_fp4,
        weight_scale,
        weight_scale_f,
        output,
        M,
        N,
        K,
        stream);
    return;
  }

  cudaStreamCaptureStatus capture_status = cudaStreamCaptureStatusNone;
  check_cuda(cudaStreamIsCapturing(stream, &capture_status));
  if (capture_status != cudaStreamCaptureStatusNone) {
    Workspace& ws = get_direct_workspace(g_direct_state, attn_output, key);
    launch_sequence(
        ws,
        attn_output,
        residual,
        weight_fp4,
        weight_scale,
        weight_scale_f,
        output,
        M,
        N,
        K,
        stream);
    return;
  }

  cudaError_t err = cudaSuccess;
  if (!g_graph_state.valid || !same_shape_key(g_graph_state.key, key)) {
    err = build_manual_graph(
        g_graph_state,
        attn_output,
        residual,
        weight_fp4,
        weight_scale,
        weight_scale_f,
        output,
        key,
        stream);
    if (err != cudaSuccess) {
      cudaGetLastError();
      destroy_graph_state(g_graph_state);
      Workspace& ws = get_direct_workspace(g_direct_state, attn_output, key);
      launch_sequence(
          ws,
          attn_output,
          residual,
          weight_fp4,
          weight_scale,
          weight_scale_f,
          output,
          M,
          N,
          K,
          stream);
      err = cudaGetLastError();
      TORCH_CHECK(err == cudaSuccess, "graph build fallback direct launch failed: ", cudaGetErrorString(err));
      return;
    }
    check_cuda(cudaGraphUpload(g_graph_state.exec, stream));
  } else {
    err = update_manual_graph(
        g_graph_state,
        attn_output,
        residual,
        weight_fp4,
        weight_scale,
        weight_scale_f,
        output,
        stream);
    if (err != cudaSuccess) {
      cudaGetLastError();
      destroy_graph_state(g_graph_state);
      err = build_manual_graph(
          g_graph_state,
          attn_output,
          residual,
          weight_fp4,
          weight_scale,
          weight_scale_f,
          output,
          key,
          stream);
      if (err != cudaSuccess) {
        cudaGetLastError();
        destroy_graph_state(g_graph_state);
        Workspace& ws = get_direct_workspace(g_direct_state, attn_output, key);
        launch_sequence(
            ws,
            attn_output,
            residual,
            weight_fp4,
            weight_scale,
            weight_scale_f,
            output,
            M,
            N,
            K,
            stream);
        err = cudaGetLastError();
        TORCH_CHECK(err == cudaSuccess, "graph update fallback direct launch failed: ", cudaGetErrorString(err));
        return;
      }
      check_cuda(cudaGraphUpload(g_graph_state.exec, stream));
    }
  }

  check_cuda(cudaGraphLaunch(g_graph_state.exec, stream));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("run", &run, "NVFP4 attention output projection with direct exact postprocess replay");
}
