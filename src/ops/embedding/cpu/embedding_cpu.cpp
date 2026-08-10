#include "embedding_cpu.hpp"

#include "../../../utils.hpp"

template <typename T>
void embedding_(T *out,
                const int64_t *index,
                const T *weight,
                size_t num_indices,
                size_t num_embeddings,
                size_t embedding_dim) {

    // Embedding 只是复制数据，所以不需要像 add、argmax 那样把 FP16/BF16 转成 FP32

    for (size_t i = 0; i < num_indices; ++i) {
        const int64_t row = index[i];

        // 防止负数索引
        CHECK_ARGUMENT(row >= 0 && static_cast<size_t>(row) < num_embeddings,
                       "Embedding index is out of range");

        const size_t source_offset = static_cast<size_t>(row) * embedding_dim;

        const size_t output_offset = i * embedding_dim;

        for (size_t j = 0; j < embedding_dim; ++j) {
            out[output_offset + j] = weight[source_offset + j];
        }
    }
}

namespace llaisys::ops::cpu {
void embedding(std::byte *out,
               const int64_t *index,
               const std::byte *weight,
               llaisysDataType_t dtype,
               size_t num_indices,
               size_t num_embeddings,
               size_t embedding_dim) {

    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        return embedding_(reinterpret_cast<float *>(out),
                          index,
                          reinterpret_cast<const float *>(weight),
                          num_indices,
                          num_embeddings,
                          embedding_dim);

    case LLAISYS_DTYPE_F16:
        return embedding_(reinterpret_cast<llaisys::fp16_t *>(out),
                          index,
                          reinterpret_cast<const llaisys::fp16_t *>(weight),
                          num_indices,
                          num_embeddings,
                          embedding_dim);

    case LLAISYS_DTYPE_BF16:
        return embedding_(reinterpret_cast<llaisys::bf16_t *>(out),
                          index,
                          reinterpret_cast<const llaisys::bf16_t *>(weight),
                          num_indices,
                          num_embeddings,
                          embedding_dim);

    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}
} // namespace llaisys::ops::cpu