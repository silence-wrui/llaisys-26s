#include "add_nvidia.cuh"

#include "../../../utils.hpp"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

namespace {

constexpr unsigned int BLOCK_SIZE = 256;

template <typename T>
__global__ void addKernel(T *c, const T *a, const T *b, size_t numel) {

    const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    if (index < numel) {
        c[index] = a[index] + b[index];
    }
}

template <typename T>
void launchAdd(
    std::byte *c,
    const std::byte *a,
    const std::byte *b,
    size_t numel,
    llaisysStream_t stream) {

    if (numel == 0) {
        return;
    }

    const unsigned int grid_size =
        static_cast<unsigned int>((numel + BLOCK_SIZE - 1) / BLOCK_SIZE);

    addKernel<T><<<grid_size, BLOCK_SIZE, 0, reinterpret_cast<cudaStream_t>(stream)>>>(
        reinterpret_cast<T *>(c),
        reinterpret_cast<const T *>(a),
        reinterpret_cast<const T *>(b),
        numel);

    const cudaError_t status = cudaGetLastError();

    if (status != cudaSuccess) {
        throw std::runtime_error(
            std::string("CUDA Add kernel launch failed: ")
            + cudaGetErrorString(status));
    }
}

} // namespace

namespace llaisys::ops::nvidia {

void add(std::byte *c,
    const std::byte *a,
    const std::byte *b,
    llaisysDataType_t type,
    size_t numel,
    llaisysStream_t stream) {

    switch (type) {
    case LLAISYS_DTYPE_F32:
        return launchAdd<float>(c, a, b, numel, stream);
    case LLAISYS_DTYPE_F16:
        return launchAdd<__half>(c, a, b, numel, stream);
    case LLAISYS_DTYPE_BF16:
        return launchAdd<__nv_bfloat16>(c, a, b, numel, stream);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::nvidia