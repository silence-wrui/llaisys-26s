#pragma once
#include "llaisys.h"

#include <cstddef>
#include <cstdint>

namespace llaisys::ops::cpu {
void rope(std::byte *out,
          const std::byte *in,
          const int64_t *pos_ids,
          llaisysDataType_t dtype,
          size_t seq_len,   //第一维 输入in、输出out、位置pos_ids向量中的行数 
          size_t num_heads, //第二维
          size_t head_dim,  //第三位 向量长度d
          float theta);
} // namespace llaisys::ops::cpu