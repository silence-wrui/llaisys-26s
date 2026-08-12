#include "argmax_musa.muh"

#include "../../../utils.hpp"

#include <musa_bf16.h>
#include <musa_fp16.h>
#include <musa_runtime.h>

#include <cstdint>
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
__device__ float toFloat<__mt_bfloat16>(
    __mt_bfloat16 value) {

    return __bfloat162float(value);
}

__device__ bool isBetter(float candidate_value,
    int64_t candidate_index,
    float current_value,
    int64_t current_index) {

    if (candidate_index < 0) {
        return false;
    }

    if (current_index < 0) {
        return true;
    }

    // 遇到 NaN 时按较小索引选择，
    // 与从前向后扫描时的行为保持一致。
    if (isnan(candidate_value) || isnan(current_value)) {
        return candidate_index < current_index;
    }

    return candidate_value > current_value || (candidate_value == current_value && candidate_index < current_index);
}

template <typename T>
__global__ void argmaxKernel(int64_t *max_idx, T *max_val, const T *vals, size_t numel) {

    __shared__ float shared_values[BLOCK_SIZE];
    __shared__ int64_t shared_indices[BLOCK_SIZE];

    const unsigned int thread_id = threadIdx.x;

    float local_value = 0.0f;
    int64_t local_index = -1;

    // 每个线程扫描多个输入元素。
    for (size_t index = thread_id; index < numel; index += blockDim.x) {

        const float value = toFloat<T>(vals[index]);

        const int64_t index_i64 = static_cast<int64_t>(index);

        if (isBetter(value, index_i64, local_value, local_index)) {

            local_value = value;
            local_index = index_i64;
        }
    }

    shared_values[thread_id] = local_value;

    shared_indices[thread_id] = local_index;

    __syncthreads();

    // 块内归约，同时归约最大值和对应索引。
    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {

        if (thread_id < stride) {
            const float candidate_value = shared_values[thread_id + stride];

            const int64_t candidate_index = shared_indices[thread_id + stride];

            if (isBetter(candidate_value, candidate_index, shared_values[thread_id], shared_indices[thread_id])) {

                shared_values[thread_id] = candidate_value;

                shared_indices[thread_id] = candidate_index;
            }
        }

        __syncthreads();
    }

    if (thread_id == 0) {
        const int64_t best_index = shared_indices[0];

        *max_idx = best_index;

        // 直接复制原始值，避免 F16/BF16
        // 经过额外转换后改变二进制结果。
        *max_val = vals[best_index];
    }
}

template <typename T>
void launchArgmax(std::byte *max_idx,
    std::byte *max_val,
    const std::byte *vals,
    size_t numel,
    llaisysStream_t stream) {

    argmaxKernel<T><<<1, BLOCK_SIZE, 0, reinterpret_cast<musaStream_t>(stream)>>>(
        reinterpret_cast<int64_t *>(max_idx),
        reinterpret_cast<T *>(max_val),
        reinterpret_cast<const T *>(vals),
        numel);

    const musaError_t status = musaGetLastError();

    if (status != musaSuccess) {
        throw std::runtime_error(
            std::string("MUSA Argmax kernel launch failed: ")
            + musaGetErrorString(status));
    }
}

} // namespace

namespace llaisys::ops::musa {

void argmax(std::byte *max_idx,
    std::byte *max_val,
    const std::byte *vals,
    llaisysDataType_t dtype,
    size_t numel,
    llaisysStream_t stream) {

    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        return launchArgmax<float>(max_idx,
            max_val,
            vals,
            numel,
            stream);
    case LLAISYS_DTYPE_F16:
        return launchArgmax<__half>(max_idx,
            max_val,
            vals,
            numel,
            stream);
    case LLAISYS_DTYPE_BF16:
        return launchArgmax<__mt_bfloat16>(max_idx,
            max_val,
            vals,
            numel,
            stream);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}
} // namespace llaisys::ops::musa