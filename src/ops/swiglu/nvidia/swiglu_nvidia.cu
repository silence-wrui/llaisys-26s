#include "swiglu_nvidia.cuh"

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
__global__ void swigluKernel(T *out, const T *gate, const T *up, size_t numel) {

    const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    if (index >= numel) {
        return;
    }

    const float gate_value = toFloat<T>(gate[index]);

    const float up_value = toFloat<T>(up[index]);

    float silu_value;

    // 避免 gate 很小时计算 exp(-gate) 溢出。
    if (gate_value >= 0.0f) {
        silu_value = gate_value / (1.0f + expf(-gate_value));
    } else {
        const float exp_gate = expf(gate_value);

        silu_value = gate_value * exp_gate / (1.0f + exp_gate);
    }

    out[index] = fromFloat<T>(up_value * silu_value);
}

template <typename T>
void launchSwiglu(std::byte *out,
    const std::byte *gate,
    const std::byte *up,
    size_t numel,
    llaisysStream_t stream) {

    if (numel == 0) {
        return;
    }

    const unsigned int grid_size = static_cast<unsigned int>((numel + BLOCK_SIZE - 1) / BLOCK_SIZE);

    swigluKernel<T><<<grid_size, BLOCK_SIZE, 0, reinterpret_cast<cudaStream_t>(stream)>>>(
        reinterpret_cast<T *>(out),
        reinterpret_cast<const T *>(gate),
        reinterpret_cast<const T *>(up),
        numel);

    const cudaError_t status = cudaGetLastError();

    if (status != cudaSuccess) {
        throw std::runtime_error(
            std::string("CUDA SwiGLU kernel launch failed: ")
            + cudaGetErrorString(status));
    }
}

} // namespace

namespace llaisys::ops::nvidia {

void swiglu(std::byte *out,
    const std::byte *gate,
    const std::byte *up,
    llaisysDataType_t dtype,
    size_t numel,
    llaisysStream_t stream) {

    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        return launchSwiglu<float>(out, gate, up, numel, stream);
    case LLAISYS_DTYPE_F16:
        return launchSwiglu<__half>(out, gate, up, numel, stream);
    case LLAISYS_DTYPE_BF16:
        return launchSwiglu<__nv_bfloat16>(out, gate, up, numel, stream);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}
} // namespace llaisys::ops::nvidia