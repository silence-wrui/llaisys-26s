#include "add_musa.muh"

#include "../../../utils.hpp"

#include <musa_bf16.h>
#include <musa_fp16.h>
#include <musa_runtime.h>

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

    addKernel<T><<<grid_size, BLOCK_SIZE, 0, reinterpret_cast<musaStream_t>(stream)>>>(
        reinterpret_cast<T *>(c),
        reinterpret_cast<const T *>(a),
        reinterpret_cast<const T *>(b),
        numel);

    const musaError_t status = musaGetLastError();

    if (status != musaSuccess) {
        throw std::runtime_error(
            std::string("MUSA Add kernel launch failed: ")
            + musaGetErrorString(status));
    }
}

} // namespace

namespace llaisys::ops::musa {

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
        return launchAdd<__mt_bfloat16>(c, a, b, numel, stream);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::musa