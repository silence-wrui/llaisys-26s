#include "self_attention_nvidia.cuh"

#include "../../../utils.hpp"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cfloat>
#include <stdexcept>
#include <string>

namespace {

constexpr unsigned int BLOCK_SIZE = 256;

template <typename T>
__device__ float toFloat(T value);

template <>
__device__ float toFloat<float>(float value) {
    return value;
}

template <>
__device__ float toFloat<__half>(__half value) {
    return __half2float(value);
}

template <>
__device__ float toFloat<__nv_bfloat16>(
    __nv_bfloat16 value) {

    return __bfloat162float(value);
}

template <typename T>
__device__ T fromFloat(float value);

template <>
__device__ float fromFloat<float>(float value) {
    return value;
}

template <>
__device__ __half fromFloat<__half>(float value) {
    return __float2half_rn(value);
}

template <>
__device__ __nv_bfloat16
fromFloat<__nv_bfloat16>(float value) {
    return __float2bfloat16_rn(value);
}

template <typename T>
__global__ void selfAttentionKernel(T *attn_val,
    const T *q,
    const T *k,
    const T *v,
    size_t q_len,
    size_t kv_len,
    size_t num_heads,
    size_t num_kv_heads,
    size_t qk_dim,
    size_t value_dim,
    float scale) {

    extern __shared__ float shared_memory[];

    float *reduction = shared_memory;

    float *output_accumulator = shared_memory + BLOCK_SIZE;

    __shared__ float running_max;
    __shared__ float running_sum;
    __shared__ float old_scale;
    __shared__ float new_scale;

    const unsigned int thread_id = threadIdx.x;

    const size_t query_head_index = static_cast<size_t>(blockIdx.x);

    const size_t query_pos = query_head_index / num_heads;

    const size_t query_head = query_head_index % num_heads;

    if (query_pos >= q_len) {
        return;
    }

    const size_t heads_per_kv_head = num_heads / num_kv_heads;

    const size_t kv_head = query_head / heads_per_kv_head;

    const size_t past_len = kv_len - q_len;

    // 因果遮罩允许访问的最后一个key。
    const size_t max_key_pos = past_len + query_pos;

    for (size_t value_col = thread_id; value_col < value_dim; value_col += blockDim.x) {

        output_accumulator[value_col] = 0.0f;
    }

    if (thread_id == 0) {
        running_max = -FLT_MAX;
        running_sum = 0.0f;
        old_scale = 0.0f;
        new_scale = 0.0f;
    }

    __syncthreads();

    for (size_t key_pos = 0; key_pos <= max_key_pos; ++key_pos) {

        float partial_dot = 0.0f;

        for (size_t dim = thread_id; dim < qk_dim; dim += blockDim.x) {

            const size_t q_index = (query_pos * num_heads + query_head) * qk_dim + dim;

            const size_t k_index = (key_pos * num_kv_heads + kv_head) * qk_dim + dim;

            partial_dot += toFloat<T>(q[q_index]) * toFloat<T>(k[k_index]);
        }

        reduction[thread_id] = partial_dot;
        __syncthreads();

        // 归约Q·K。
        for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {

            if (thread_id < stride) {
                reduction[thread_id] += reduction[thread_id + stride];
            }

            __syncthreads();
        }

        if (thread_id == 0) {
            const float score = reduction[0] * scale;

            const float updated_max = fmaxf(running_max, score);

            // 旧Softmax累加结果需要按新的最大值缩放。
            old_scale = expf(running_max - updated_max);

            // 当前key的Softmax分子。
            new_scale = expf(score - updated_max);

            running_sum = running_sum * old_scale + new_scale;

            running_max = updated_max;
        }

        __syncthreads();

        // 在线累加Softmax(QK) × V的分子。
        for (size_t value_col = thread_id; value_col < value_dim; value_col += blockDim.x) {

            const size_t v_index = (key_pos * num_kv_heads + kv_head) * value_dim + value_col;

            output_accumulator[value_col] = output_accumulator[value_col] * old_scale
                + new_scale * toFloat<T>(v[v_index]);
        }

        __syncthreads();
    }

    // 除以Softmax分母并写回。
    for (size_t value_col = thread_id; value_col < value_dim; value_col += blockDim.x) {

        const size_t output_index = (query_pos * num_heads + query_head) * value_dim + value_col;

        attn_val[output_index] = fromFloat<T>(output_accumulator[value_col] / running_sum);
    }
}

template <typename T>
void launchSelfAttention(std::byte *attn_val,
    const std::byte *q,
    const std::byte *k,
    const std::byte *v,
    size_t q_len,
    size_t kv_len,
    size_t num_heads,
    size_t num_kv_heads,
    size_t qk_dim,
    size_t value_dim,
    float scale,
    llaisysStream_t stream) {

    const size_t block_count = q_len * num_heads;

    if (block_count == 0) {
        return;
    }

    const size_t shared_memory_size = (BLOCK_SIZE + value_dim) * sizeof(float);

    selfAttentionKernel<T><<<static_cast<unsigned int>(block_count),
        BLOCK_SIZE,
        shared_memory_size,
        reinterpret_cast<cudaStream_t>(stream)>>>(
        reinterpret_cast<T *>(attn_val),
        reinterpret_cast<const T *>(q),
        reinterpret_cast<const T *>(k),
        reinterpret_cast<const T *>(v),
        q_len,
        kv_len,
        num_heads,
        num_kv_heads,
        qk_dim,
        value_dim,
        scale);

    const cudaError_t status = cudaGetLastError();

    if (status != cudaSuccess) {
        throw std::runtime_error(
            std::string("CUDA SelfAttention kernel failed: ")
            + cudaGetErrorString(status));
    }
}

} // namespace

namespace llaisys::ops::nvidia {

void self_attention(std::byte *attn_val,
    const std::byte *q,
    const std::byte *k,
    const std::byte *v,
    llaisysDataType_t dtype,
    size_t q_len,
    size_t kv_len,
    size_t num_heads,
    size_t num_kv_heads,
    size_t qk_dim,
    size_t value_dim,
    float scale,
    llaisysStream_t stream) {

    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        return launchSelfAttention<float>(attn_val,
            q,
            k,
            v,
            q_len,
            kv_len,
            num_heads,
            num_kv_heads,
            qk_dim,
            value_dim,
            scale,
            stream);
    case LLAISYS_DTYPE_F16:
        return launchSelfAttention<__half>(attn_val,
            q,
            k,
            v,
            q_len,
            kv_len,
            num_heads,
            num_kv_heads,
            qk_dim,
            value_dim,
            scale,
            stream);
    case LLAISYS_DTYPE_BF16:
        return launchSelfAttention<__nv_bfloat16>(attn_val,
            q,
            k,
            v,
            q_len,
            kv_len,
            num_heads,
            num_kv_heads,
            qk_dim,
            value_dim,
            scale,
            stream);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}
} // namespace llaisys::ops::nvidia