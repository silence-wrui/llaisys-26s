#pragma once
#include "llaisys.h"

#include <cstddef>
#include <cstdint>

// dtype：weight 和 out 的元素类型
namespace llaisys::ops::cpu {
void embedding(std::byte *out,
               const int64_t *index,
               const std::byte *weight,
               llaisysDataType_t dtype,
               size_t num_indices,    // 索引数量，即输出行数
               size_t num_embeddings, // 权重矩阵行数
               size_t embedding_dim   // 每行元素数
);

} // namespace llaisys::ops::cpu