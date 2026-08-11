#include "embedding_nvidia.cuh"

#include "../../../utils.hpp"

#include <cuda_runtime.h>

#include <cstdint>
#include <stdexcept>
#include <string>

namespace {

constexpr unsigned int BLOCK_SIZE = 256;

template <typename T>
__global__ void embeddingKernel(T *out,
    const int64_t *index,
    const T *weight,
    size_t num_indices,
    size_t num_embeddings,
    size_t embedding_dim) {

    const size_t output_index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    const size_t output_numel = num_indices * embedding_dim;

    if (output_index >= output_numel) {
        return;
    }

    const size_t index_position = output_index / embedding_dim;

    const size_t column = output_index % embedding_dim;

    const int64_t row = index[index_position];

    // 防止非法索引造成显存越界访问。
    if (row < 0 || static_cast<size_t>(row) >= num_embeddings) {
        return;
    }

    const size_t weight_offset = static_cast<size_t>(row) * embedding_dim + column;

    out[output_index] = weight[weight_offset];
}

template <typename T>
void launchEmbedding(std::byte *out,
    const int64_t *index,
    const std::byte *weight,
    size_t num_indices,
    size_t num_embeddings,
    size_t embedding_dim,
    llaisysStream_t stream) {

    const size_t output_numel = num_indices * embedding_dim;

    if (output_numel == 0) {
        return;
    }

    const unsigned int grid_size = static_cast<unsigned int>((output_numel + BLOCK_SIZE - 1) / BLOCK_SIZE);

    embeddingKernel<T><<<grid_size, BLOCK_SIZE, 0, reinterpret_cast<cudaStream_t>(stream)>>>(
        reinterpret_cast<T *>(out),
        index,
        reinterpret_cast<const T *>(weight),
        num_indices,
        num_embeddings,
        embedding_dim);

    const cudaError_t status = cudaGetLastError();

    if (status != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA Embedding kernel launch failed: ")
            + cudaGetErrorString(status));
    }
}

} // namespace

namespace llaisys::ops::nvidia {

void embedding(std::byte *out,
    const int64_t *index,
    const std::byte *weight,
    llaisysDataType_t dtype,
    size_t num_indices,
    size_t num_embeddings,
    size_t embedding_dim,
    llaisysStream_t stream) {

    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        return launchEmbedding<float>(out,
            index,
            weight,
            num_indices,
            num_embeddings,
            embedding_dim,
            stream);
    case LLAISYS_DTYPE_F16:
    case LLAISYS_DTYPE_BF16:
        return launchEmbedding<uint16_t>(out,
            index,
            weight,
            num_indices,
            num_embeddings,
            embedding_dim,
            stream);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}
} // namespace llaisys::ops::nvidia