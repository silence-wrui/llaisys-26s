#pragma once
#include "llaisys.h"

#include <cstddef>

// mythink:bias 可能为空指针，表示不使用偏置
namespace llaisys::ops::cpu {
void linear(std::byte *out,
            const std::byte *in,
            const std::byte *weight,
            const std::byte *bias,
            llaisysDataType_t dtype,
            size_t num_rows,
            size_t in_features,
            size_t out_features);
} // namespace llaisys::ops::cpu