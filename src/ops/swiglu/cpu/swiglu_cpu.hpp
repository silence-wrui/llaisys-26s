#pragma once
#include "llaisys.h"

#include <cstddef>

//SwiGLU 是逐元素操作，所以 CPU 内核只需要知道总元素数 numel，不需要分别传入行数和列数
namespace llaisys::ops::cpu {
void swiglu(std::byte *out,
            const std::byte *gate,
            const std::byte *up,
            llaisysDataType_t dtype,
            size_t numel);
} // namespace llaisys::ops::cpu