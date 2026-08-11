#pragma once
#include "llaisys.h"

#include <cstddef>
namespace llaisys::ops::cpu {
void rms_norm(
    std::byte *out,
    const std::byte *in,
    const std::byte *weight,
    llaisysDataType_t dtype,
    size_t num_rows,  //行数
    size_t hidden_size,  //列数，隐藏层数
    float eps
);
} // namespace llaisys::ops::cpu