#include "linear_nvidia.cuh"

#include "../../../device/nvidia/nvidia_resource.cuh"
#include "../../../utils.hpp"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cublas_v2.h>

#include <stdexcept>
#include <string>

namespace {

constexpr unsigned int BLOCK_SIZE = 256;

void checkCublas(
    cublasStatus_t status,
    const char *operation) {

    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error(std::string(operation)
            + " failed: "
            + cublasGetStatusString(status));
    }
}

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
__global__ void addBiasKernel(T *out, const T *bias, size_t numel, size_t out_features) {

    const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    if (index >= numel) {
        return;
    }

    const size_t column = index % out_features;

    out[index] = fromFloat<T>( toFloat<T>(out[index]) + toFloat<T>(bias[column]));
}

template <typename T>
void launchLinear(std::byte *out,
    const std::byte *in,
    const std::byte *weight,
    const std::byte *bias,
    cudaDataType_t data_type,
    size_t num_rows,
    size_t in_features,
    size_t out_features,
    int device_id,
    llaisysStream_t stream) {

    if (num_rows == 0 || out_features == 0) {
        return;
    }

    auto &resource = llaisys::device::nvidia::getResource( device_id);

    const cublasHandle_t handle = resource.cublasHandle();

    checkCublas(
        cublasSetStream(handle, reinterpret_cast<cudaStream_t>(stream)),
        "cublasSetStream");

    const int m = static_cast<int>(num_rows);

    const int k = static_cast<int>(in_features);

    const int n = static_cast<int>(out_features);

    const float alpha = 1.0f;
    const float beta = 0.0f;

    const cublasGemmAlgo_t algorithm = data_type == CUDA_R_32F ? CUBLAS_GEMM_DEFAULT : CUBLAS_GEMM_DEFAULT_TENSOR_OP;

    /*
     * 行主序目标：
     *
     * out[M, N] = in[M, K] * weight[N, K]^T
     *
     * cuBLAS按列主序解释内存，因此计算转置后的等式：
     *
     * out^T[N, M] = weight[N, K] * in^T[K, M]
     */
    checkCublas(
        cublasGemmEx(handle,
            CUBLAS_OP_T,
            CUBLAS_OP_N,
            n,
            m,
            k,
            &alpha,
            weight,
            data_type,
            k,
            in,
            data_type,
            k,
            &beta,
            out,
            data_type,
            n,
            CUBLAS_COMPUTE_32F,
            algorithm),
        "cublasGemmEx");

    if (bias != nullptr) {
        const size_t output_numel = num_rows * out_features;

        const unsigned int grid_size = static_cast<unsigned int>((output_numel + BLOCK_SIZE - 1) / BLOCK_SIZE);

        addBiasKernel<T><<<grid_size, BLOCK_SIZE, 0, reinterpret_cast<cudaStream_t>(stream)>>>(
            reinterpret_cast<T *>(out),
            reinterpret_cast<const T *>(bias),
            output_numel,
            out_features);

        const cudaError_t status = cudaGetLastError();

        if (status != cudaSuccess) {
            throw std::runtime_error(
                std::string("CUDA Linear bias kernel failed: ")
                + cudaGetErrorString(status));
        }
    }
}

} // namespace

namespace llaisys::ops::nvidia {

void linear(std::byte *out,
    const std::byte *in,
    const std::byte *weight,
    const std::byte *bias,
    llaisysDataType_t dtype,
    size_t num_rows,
    size_t in_features,
    size_t out_features,
    int device_id,
    llaisysStream_t stream) {

    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        return launchLinear<float>(out,
            in,
            weight,
            bias,
            CUDA_R_32F,
            num_rows,
            in_features,
            out_features,
            device_id,
            stream);
    case LLAISYS_DTYPE_F16:
        return launchLinear<__half>(out,
            in,
            weight,
            bias,
            CUDA_R_16F,
            num_rows,
            in_features,
            out_features,
            device_id,
            stream);
    case LLAISYS_DTYPE_BF16:
        return launchLinear<__nv_bfloat16>(out,
            in,
            weight,
            bias,
            CUDA_R_16BF,
            num_rows,
            in_features,
            out_features,
            device_id,
            stream);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}
} // namespace llaisys::ops::nvidia