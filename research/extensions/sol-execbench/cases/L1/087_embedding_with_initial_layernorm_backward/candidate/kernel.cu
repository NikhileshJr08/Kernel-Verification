#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

#include <cstdint>
#include <mutex>
#include <unordered_map>

namespace {

constexpr int HIDDEN_SIZE = 4096;
constexpr int VOCAB_SIZE = 65536;
constexpr int BLOCK_THREADS = 256;
constexpr int WARP_SIZE = 32;
constexpr int GRAD_NORM_SHARDS = 2048;
constexpr int MIN_GRAD_NORM_SHARDS = 16;
constexpr int TOKENS_PER_GRAD_NORM_SHARD = 16;
static_assert((GRAD_NORM_SHARDS & (GRAD_NORM_SHARDS - 1)) == 0, "GRAD_NORM_SHARDS must be power of two");
static_assert((MIN_GRAD_NORM_SHARDS & (MIN_GRAD_NORM_SHARDS - 1)) == 0, "MIN_GRAD_NORM_SHARDS must be power of two");

std::mutex workspace_mutex;
std::unordered_map<int, at::Tensor> grad_norm_workspace_by_device;
std::unordered_map<int, at::Tensor> row_state_workspace_by_device;

inline void check_cuda(const torch::Tensor& t, const char* name) {
    TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
}

inline at::Tensor get_grad_norm_workspace(const torch::Tensor& ref) {
    const int device = ref.get_device();
    std::lock_guard<std::mutex> guard(workspace_mutex);
    auto it = grad_norm_workspace_by_device.find(device);
    if (it == grad_norm_workspace_by_device.end() || !it->second.defined()) {
        auto options = ref.options().dtype(torch::kFloat32);
        auto tensor = torch::empty({GRAD_NORM_SHARDS, HIDDEN_SIZE}, options);
        it = grad_norm_workspace_by_device.emplace(device, tensor).first;
    }
    return it->second;
}

inline at::Tensor get_row_state_workspace(const torch::Tensor& ref) {
    const int device = ref.get_device();
    std::lock_guard<std::mutex> guard(workspace_mutex);
    auto it = row_state_workspace_by_device.find(device);
    if (it == row_state_workspace_by_device.end() || !it->second.defined()) {
        auto options = ref.options().dtype(torch::kInt32);
        auto tensor = torch::empty({VOCAB_SIZE}, options);
        it = row_state_workspace_by_device.emplace(device, tensor).first;
    }
    return it->second;
}

inline int compute_grad_norm_shards(int num_tokens) {
    int target = (num_tokens + TOKENS_PER_GRAD_NORM_SHARD - 1) / TOKENS_PER_GRAD_NORM_SHARD;
    if (target < MIN_GRAD_NORM_SHARDS) {
        target = MIN_GRAD_NORM_SHARDS;
    }
    int shards = 1;
    while (shards < target && shards < GRAD_NORM_SHARDS) {
        shards <<= 1;
    }
    if (shards > GRAD_NORM_SHARDS) {
        shards = GRAD_NORM_SHARDS;
    }
    return shards;
}

__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffffu, val, offset);
    }
    return val;
}

__device__ __forceinline__ void red_global_v4_f32(float* addr, float v0, float v1, float v2, float v3) {
#if __CUDA_ARCH__ >= 900
    asm volatile(
        "red.global.v4.f32.add [%0], {%1, %2, %3, %4};\n"
        :
        : "l"(addr), "f"(v0), "f"(v1), "f"(v2), "f"(v3)
        : "memory");
#else
    atomicAdd(addr + 0, v0);
    atomicAdd(addr + 1, v1);
    atomicAdd(addr + 2, v2);
    atomicAdd(addr + 3, v3);
#endif
}

union BFloat162Bits {
    __nv_bfloat162 value;
    uint32_t bits;
};

__device__ __forceinline__ uint32_t bf162_to_bits(__nv_bfloat162 value) {
    BFloat162Bits packed;
    packed.value = value;
    return packed.bits;
}

__device__ __forceinline__ __nv_bfloat162 bits_to_bf162(uint32_t bits) {
    BFloat162Bits packed;
    packed.bits = bits;
    return packed.value;
}

__device__ __forceinline__ void red_global_v4_bf16x2(uint32_t* addr, uint32_t v0, uint32_t v1, uint32_t v2, uint32_t v3) {
#if __CUDA_ARCH__ >= 900
    asm volatile(
        "red.global.v4.bf16x2.add.noftz [%0], {%1, %2, %3, %4};\n"
        :
        : "l"(addr), "r"(v0), "r"(v1), "r"(v2), "r"(v3)
        : "memory");
#else
    auto* addr_pairs = reinterpret_cast<__nv_bfloat162*>(addr);
    atomicAdd(addr_pairs + 0, bits_to_bf162(v0));
    atomicAdd(addr_pairs + 1, bits_to_bf162(v1));
    atomicAdd(addr_pairs + 2, bits_to_bf162(v2));
    atomicAdd(addr_pairs + 3, bits_to_bf162(v3));
#endif
}

__device__ __forceinline__ void st_global_v4_b32(uint32_t* addr, uint32_t v0, uint32_t v1, uint32_t v2, uint32_t v3) {
#if __CUDA_ARCH__ >= 900
    asm volatile(
        "st.global.v4.b32 [%0], {%1, %2, %3, %4};\n"
        :
        : "l"(addr), "r"(v0), "r"(v1), "r"(v2), "r"(v3)
        : "memory");
#else
    addr[0] = v0;
    addr[1] = v1;
    addr[2] = v2;
    addr[3] = v3;
#endif
}

__global__ __launch_bounds__(BLOCK_THREADS, 2)
void fused_scatter_and_grad_norm_kernel(
    const __nv_bfloat16* __restrict__ grad_output,
    const int64_t* __restrict__ input_ids,
    const float* __restrict__ hidden_states_fp32,
    const float* __restrict__ rstd,
    const __nv_bfloat16* __restrict__ norm_weight,
    float* __restrict__ grad_norm_workspace,
    int* __restrict__ row_state_workspace,
    __nv_bfloat16* __restrict__ grad_embed_weight,
    int grad_norm_mask,
    int num_tokens) {

    const int token = blockIdx.x;
    if (token >= num_tokens) {
        return;
    }

    __shared__ float hidden_norm_cache[HIDDEN_SIZE];
    __shared__ __nv_bfloat162 grad_pair_cache[HIDDEN_SIZE / 2];
    __shared__ float warp_sums[BLOCK_THREADS / WARP_SIZE];
    __shared__ float shared_mean;
    __shared__ int shared_direct_store;

    const auto* grad_row = grad_output + static_cast<int64_t>(token) * HIDDEN_SIZE;
    const auto* hidden_row = hidden_states_fp32 + static_cast<int64_t>(token) * HIDDEN_SIZE;
    const float r = rstd[token];
    const int token_id = static_cast<int>(input_ids[token]);

    const auto* grad_row_pairs = reinterpret_cast<const __nv_bfloat162*>(grad_row);
    const auto* weight_pairs = reinterpret_cast<const __nv_bfloat162*>(norm_weight);
    auto* out_row_pairs = reinterpret_cast<__nv_bfloat162*>(grad_embed_weight + static_cast<int64_t>(token_id) * HIDDEN_SIZE);
    float* shard_row = grad_norm_workspace + static_cast<int64_t>(token & grad_norm_mask) * HIDDEN_SIZE;

    float local_dot = 0.0f;
    for (int pair_base = threadIdx.x * 4; pair_base < HIDDEN_SIZE / 2; pair_base += BLOCK_THREADS * 4) {
        float dgamma_vals[8];

        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            const int pair_idx = pair_base + k;
            const int elem_idx = pair_idx * 2;

            const __nv_bfloat162 g_pair = grad_row_pairs[pair_idx];
            grad_pair_cache[pair_idx] = g_pair;

            const float2 g = __bfloat1622float2(g_pair);
            const float h0 = hidden_row[elem_idx + 0] * r;
            const float h1 = hidden_row[elem_idx + 1] * r;
            hidden_norm_cache[elem_idx + 0] = h0;
            hidden_norm_cache[elem_idx + 1] = h1;

            const float2 w = __bfloat1622float2(weight_pairs[pair_idx]);
            local_dot += g.x * w.x * h0 + g.y * w.y * h1;

            dgamma_vals[2 * k + 0] = g.x * h0;
            dgamma_vals[2 * k + 1] = g.y * h1;
        }

        red_global_v4_f32(shard_row + pair_base * 2 + 0, dgamma_vals[0], dgamma_vals[1], dgamma_vals[2], dgamma_vals[3]);
        red_global_v4_f32(shard_row + pair_base * 2 + 4, dgamma_vals[4], dgamma_vals[5], dgamma_vals[6], dgamma_vals[7]);
    }

    float warp_sum = warp_reduce_sum(local_dot);
    const int lane = threadIdx.x & (WARP_SIZE - 1);
    const int warp = threadIdx.x / WARP_SIZE;
    if (lane == 0) {
        warp_sums[warp] = warp_sum;
    }
    __syncthreads();

    float block_sum = 0.0f;
    if (warp == 0) {
        block_sum = (threadIdx.x < BLOCK_THREADS / WARP_SIZE) ? warp_sums[lane] : 0.0f;
        block_sum = warp_reduce_sum(block_sum);
        if (lane == 0) {
            shared_mean = block_sum * (1.0f / static_cast<float>(HIDDEN_SIZE));
        }
    }
    __syncthreads();

    const float mean = shared_mean;
    if (threadIdx.x == 0) {
        int prior = atomicCAS(row_state_workspace + token_id, 0, 1);
        if (prior == 0) {
            shared_direct_store = 1;
        } else {
            while (prior == 1) {
                __nanosleep(64);
                prior = atomicAdd(row_state_workspace + token_id, 0);
            }
            shared_direct_store = 0;
        }
    }
    __syncthreads();
    const bool direct_store = shared_direct_store != 0;

    for (int pair_base = threadIdx.x * 4; pair_base < HIDDEN_SIZE / 2; pair_base += BLOCK_THREADS * 4) {
        uint32_t out_bits[4];

        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            const int pair_idx = pair_base + k;
            const int elem_idx = pair_idx * 2;

            const __nv_bfloat162 g_pair = grad_pair_cache[pair_idx];
            const float2 g = __bfloat1622float2(g_pair);
            const float2 w = __bfloat1622float2(weight_pairs[pair_idx]);

            const float hn0 = hidden_norm_cache[elem_idx + 0];
            const float hn1 = hidden_norm_cache[elem_idx + 1];

            const float out0 = r * (g.x * w.x - mean * hn0);
            const float out1 = r * (g.y * w.y - mean * hn1);
            const __nv_bfloat162 out_pair = __floats2bfloat162_rn(out0, out1);
            out_bits[k] = bf162_to_bits(out_pair);
        }

        uint32_t* out_ptr = reinterpret_cast<uint32_t*>(out_row_pairs + pair_base);
        if (direct_store) {
            st_global_v4_b32(out_ptr, out_bits[0], out_bits[1], out_bits[2], out_bits[3]);
        } else {
            red_global_v4_bf16x2(out_ptr, out_bits[0], out_bits[1], out_bits[2], out_bits[3]);
        }
    }

    if (direct_store) {
        __syncthreads();
        if (threadIdx.x == 0) {
            __threadfence();
            atomicExch(row_state_workspace + token_id, 2);
        }
    }
}

__global__ __launch_bounds__(BLOCK_THREADS, 2)
void reduce_grad_norm_workspace_kernel(
    const float* __restrict__ grad_norm_workspace,
    __nv_bfloat16* __restrict__ grad_norm_weight,
    int grad_norm_shards) {
    const int col = blockIdx.x * BLOCK_THREADS + threadIdx.x;
    if (col >= HIDDEN_SIZE) {
        return;
    }

    float acc = 0.0f;
    for (int shard = 0; shard < grad_norm_shards; ++shard) {
        acc += grad_norm_workspace[static_cast<int64_t>(shard) * HIDDEN_SIZE + col];
    }
    grad_norm_weight[col] = __float2bfloat16_rn(acc);
}

}  // namespace

void run(
    torch::Tensor grad_output,
    torch::Tensor input_ids,
    torch::Tensor hidden_states_fp32,
    torch::Tensor rstd,
    torch::Tensor norm_weight,
    torch::Tensor grad_embed_weight,
    torch::Tensor grad_norm_weight) {

    TORCH_CHECK(grad_output.scalar_type() == torch::kBFloat16, "grad_output must be bfloat16");
    TORCH_CHECK(input_ids.scalar_type() == torch::kInt64, "input_ids must be int64");
    TORCH_CHECK(hidden_states_fp32.scalar_type() == torch::kFloat32, "hidden_states_fp32 must be float32");
    TORCH_CHECK(rstd.scalar_type() == torch::kFloat32, "rstd must be float32");
    TORCH_CHECK(norm_weight.scalar_type() == torch::kBFloat16, "norm_weight must be bfloat16");
    TORCH_CHECK(grad_embed_weight.scalar_type() == torch::kBFloat16, "grad_embed_weight must be bfloat16");
    TORCH_CHECK(grad_norm_weight.scalar_type() == torch::kBFloat16, "grad_norm_weight must be bfloat16");

    check_cuda(grad_output, "grad_output");
    check_cuda(input_ids, "input_ids");
    check_cuda(hidden_states_fp32, "hidden_states_fp32");
    check_cuda(rstd, "rstd");
    check_cuda(norm_weight, "norm_weight");
    check_cuda(grad_embed_weight, "grad_embed_weight");
    check_cuda(grad_norm_weight, "grad_norm_weight");

    TORCH_CHECK(grad_output.dim() == 3, "grad_output must be [batch, seq, hidden]");
    TORCH_CHECK(hidden_states_fp32.dim() == 3, "hidden_states_fp32 must be [batch, seq, hidden]");
    TORCH_CHECK(rstd.dim() == 3, "rstd must be [batch, seq, 1]");
    TORCH_CHECK(input_ids.dim() == 2, "input_ids must be [batch, seq]");
    TORCH_CHECK(norm_weight.dim() == 1, "norm_weight must be [hidden]");
    TORCH_CHECK(grad_embed_weight.dim() == 2, "grad_embed_weight must be [vocab, hidden]");
    TORCH_CHECK(grad_norm_weight.dim() == 1, "grad_norm_weight must be [hidden]");

    TORCH_CHECK(grad_output.size(0) == input_ids.size(0) && grad_output.size(1) == input_ids.size(1),
                "grad_output and input_ids shape mismatch");
    TORCH_CHECK(hidden_states_fp32.sizes() == grad_output.sizes(), "hidden_states_fp32 shape mismatch");
    TORCH_CHECK(rstd.size(0) == grad_output.size(0) && rstd.size(1) == grad_output.size(1) && rstd.size(2) == 1,
                "rstd must match [batch, seq, 1]");
    TORCH_CHECK(grad_output.size(2) == HIDDEN_SIZE, "hidden size must be 4096");
    TORCH_CHECK(norm_weight.size(0) == HIDDEN_SIZE, "norm_weight size must be 4096");
    TORCH_CHECK(grad_embed_weight.size(0) == VOCAB_SIZE && grad_embed_weight.size(1) == HIDDEN_SIZE,
                "grad_embed_weight must be [65536, 4096]");
    TORCH_CHECK(grad_norm_weight.size(0) == HIDDEN_SIZE, "grad_norm_weight size must be 4096");

    const int num_tokens = static_cast<int>(grad_output.size(0) * grad_output.size(1));
    if (num_tokens == 0) {
        return;
    }

    auto grad_norm_workspace = get_grad_norm_workspace(grad_output);
    auto row_state_workspace = get_row_state_workspace(grad_output);
    auto stream = at::cuda::getDefaultCUDAStream();
    const int grad_norm_shards = compute_grad_norm_shards(num_tokens);
    const int grad_norm_mask = grad_norm_shards - 1;

    C10_CUDA_CHECK(cudaMemsetAsync(
        grad_norm_workspace.data_ptr(),
        0,
        static_cast<size_t>(grad_norm_shards) * HIDDEN_SIZE * sizeof(float),
        stream.stream()));
    C10_CUDA_CHECK(cudaMemsetAsync(
        row_state_workspace.data_ptr(),
        0,
        static_cast<size_t>(VOCAB_SIZE) * sizeof(int),
        stream.stream()));

    const int scatter_blocks = num_tokens;
    fused_scatter_and_grad_norm_kernel<<<scatter_blocks, BLOCK_THREADS, 0, stream.stream()>>>(
        reinterpret_cast<const __nv_bfloat16*>(grad_output.data_ptr<at::BFloat16>()),
        input_ids.data_ptr<int64_t>(),
        hidden_states_fp32.data_ptr<float>(),
        rstd.data_ptr<float>(),
        reinterpret_cast<const __nv_bfloat16*>(norm_weight.data_ptr<at::BFloat16>()),
        grad_norm_workspace.data_ptr<float>(),
        row_state_workspace.data_ptr<int>(),
        reinterpret_cast<__nv_bfloat16*>(grad_embed_weight.data_ptr<at::BFloat16>()),
        grad_norm_mask,
        num_tokens);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    const int reduce_blocks = HIDDEN_SIZE / BLOCK_THREADS;
    reduce_grad_norm_workspace_kernel<<<reduce_blocks, BLOCK_THREADS, 0, stream.stream()>>>(
        grad_norm_workspace.data_ptr<float>(),
        reinterpret_cast<__nv_bfloat16*>(grad_norm_weight.data_ptr<at::BFloat16>()),
        grad_norm_shards);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run", &run, "Embedding + initial layernorm backward (fused scatter / sharded grad-norm)");
}
