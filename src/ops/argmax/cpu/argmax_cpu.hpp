#pragma once
#include "llaisys.h"

#include <cstddef>

namespace llaisys::ops::cpu {
// std::byte* 表示一段尚未解释类型的原始内存
// vals 使用 const，因为它是输入
void argmax(std::byte *max_idx,
            std::byte *max_val,
            const std::byte *vals,
            llaisysDataType_t dtype,
            size_t numel);
} // namespace llaisys::ops::cpu