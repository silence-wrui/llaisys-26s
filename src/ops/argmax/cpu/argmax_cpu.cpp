#include "argmax_cpu.hpp"

#include "../../../utils.hpp"

// 以add为例学习

// 实际获取张量vals最大值及索引的函数
// max_idx 是 int64_t*，作业明确要求下标为 I64
template <typename T>
void argmax_(int64_t *max_idx, T *max_val, const T *vals, size_t numel) {

    // 先将第一个元素设为最大值
    size_t best_idx = 0;
    // 不能令初始最大值为 0，万一有负数
    float best_value = llaisys::utils::cast<float>(vals[0]);

    // 第 0 个元素已经被用作初始最大值，所以只需要从第 1 个元素开始比较
    for (size_t i = 1; i < numel; ++i) {
        // 取出当前元素
        float current = llaisys::utils::cast<float>(vals[i]);

        // 使用 > 保证最大值相同时选择第一个下标
        if (current > best_value) {
            best_value = current;
            best_idx = i;
        }
    }

    *max_idx = static_cast<int64_t>(best_idx);
    // 直接复制原值，避免再次转换带来的精度变化
    *max_val = vals[best_idx];
}

// F16/BF16 转成 float 后比较
// argmax先区分数据类型，然后调用argmax_
namespace llaisys::ops::cpu {
void argmax(std::byte *max_idx,
            std::byte *max_val,
            const std::byte *vals,
            llaisysDataType_t dtype,
            size_t numel) {

    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        return argmax_(reinterpret_cast<int64_t *>(max_idx),
                       reinterpret_cast<float *>(max_val),
                       reinterpret_cast<const float *>(vals),
                       numel);
    case LLAISYS_DTYPE_F16:
        return argmax_(reinterpret_cast<int64_t *>(max_idx),
                       reinterpret_cast<llaisys::fp16_t *>(max_val),
                       reinterpret_cast<const llaisys::fp16_t *>(vals),
                       numel);
    case LLAISYS_DTYPE_BF16:
        return argmax_(reinterpret_cast<int64_t *>(max_idx),
                       reinterpret_cast<llaisys::bf16_t *>(max_val),
                       reinterpret_cast<const llaisys::bf16_t *>(vals),
                       numel);

    // 不支持上述的三种
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}
} // namespace llaisys::ops::cpu