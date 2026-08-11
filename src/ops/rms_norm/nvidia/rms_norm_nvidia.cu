#include "rms_norm_nvidia.cuh"

#include "../../../utils.hpp"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

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
__global__ void rmsNormKernel(
    T *out,
    const T *in,
    const T *weight,
    size_t hidden_size,
    float eps) {

    extern __shared__ float shared[];

    const size_t row = static_cast<size_t>(blockIdx.x);

    const unsigned int thread_id = threadIdx.x;

    float square_sum = 0.0f;

    // 每个线程处理本行中的若干个元素。
    for (size_t col = thread_id; col < hidden_size; col += blockDim.x) {

        const size_t index = row * hidden_size + col;

        const float value = toFloat<T>(in[index]);

        square_sum += value * value;
    }

    shared[thread_id] = square_sum;
    __syncthreads();

    // 块内归约，求出整行的平方和。
    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {

        if (thread_id < stride) {
            shared[thread_id] += shared[thread_id + stride];
        }

        __syncthreads();
    }

    // 只计算一次 inverse RMS。
    if (thread_id == 0) {
        const float mean_square = shared[0] / static_cast<float>(hidden_size);

        shared[0] = rsqrtf(mean_square + eps);
    }

    __syncthreads();

    const float inverse_rms = shared[0];

    // 同一个 block 中的线程共同写回该行。
    for (size_t col = thread_id; col < hidden_size; col += blockDim.x) {

        const size_t index = row * hidden_size + col;

        const float input_value = toFloat<T>(in[index]);

        const float weight_value = toFloat<T>(weight[col]);

        out[index] = fromFloat<T>(input_value * inverse_rms * weight_value);
    }
}

template <typename T>
void launchRmsNorm(std::byte *out,
    const std::byte *in,
    const std::byte *weight,
    size_t num_rows,
    size_t hidden_size,
    float eps,
    llaisysStream_t stream) {

    if (num_rows == 0) {
        return;
    }

    rmsNormKernel<T><<<
        static_cast<unsigned int>(num_rows),
        BLOCK_SIZE,
        BLOCK_SIZE * sizeof(float),
        reinterpret_cast<cudaStream_t>(stream)>>>(
        reinterpret_cast<T *>(out),
        reinterpret_cast<const T *>(in),
        reinterpret_cast<const T *>(weight),
        hidden_size,
        eps);

    const cudaError_t status = cudaGetLastError();

    if (status != cudaSuccess) {
        throw std::runtime_error(
            std::string("CUDA RMSNorm kernel launch failed: ")
            + cudaGetErrorString(status));
    }
}

} // namespace

namespace llaisys::ops::nvidia {

void rms_norm(std::byte *out,
    const std::byte *in,
    const std::byte *weight,
    llaisysDataType_t dtype,
    size_t num_rows,
    size_t hidden_size,
    float eps,
    llaisysStream_t stream) {

    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        return launchRmsNorm<float>(
            out,
            in,
            weight,
            num_rows,
            hidden_size,
            eps,
            stream);
    case LLAISYS_DTYPE_F16:
        return launchRmsNorm<__half>(
            out,
            in,
            weight,
            num_rows,
            hidden_size,
            eps,
            stream);
    case LLAISYS_DTYPE_BF16:
        return launchRmsNorm<__nv_bfloat16>(
            out,
            in,
            weight,
            num_rows,
            hidden_size,
            eps,
            stream);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}
} // namespace llaisys::ops::nvidia