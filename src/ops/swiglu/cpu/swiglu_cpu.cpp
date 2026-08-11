#include "swiglu_cpu.hpp"

#include "../../../utils.hpp"

#include <cmath>

template <typename T>
void swiglu_(T *out, const T *gate, const T *up, size_t numel) {
    //遍历所有元素
    for (size_t i = 0; i < numel; ++i) {
        //取出gate、up中的数据
        const float gate_value = llaisys::utils::cast<float>(gate[i]);
        const float up_value = llaisys::utils::cast<float>(up[i]);

        float silu_value;

        // 使用数值稳定的方式计算 sigmoid
        if (gate_value >= 0.0f) {   //gate为正
            silu_value = gate_value / (1.0f + std::exp(-gate_value));
        } else {   //gate为负 避免溢出
            const float exp_gate = std::exp(gate_value);
            silu_value = gate_value * exp_gate / (1.0f + exp_gate);
        }

        //计算out存入
        const float result = up_value * silu_value;
        out[i] = llaisys::utils::cast<T>(result);
    }
}

namespace llaisys::ops::cpu {
void swiglu(std::byte *out,
            const std::byte *gate,
            const std::byte *up,
            llaisysDataType_t dtype,
            size_t numel) {
    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        return swiglu_(reinterpret_cast<float *>(out),
                       reinterpret_cast<const float *>(gate),
                       reinterpret_cast<const float *>(up),
                       numel);
    case LLAISYS_DTYPE_F16:
        return swiglu_(reinterpret_cast<llaisys::fp16_t *>(out),
                       reinterpret_cast<const llaisys::fp16_t *>(gate),
                       reinterpret_cast<const llaisys::fp16_t *>(up),
                       numel);
    case LLAISYS_DTYPE_BF16:
        return swiglu_(reinterpret_cast<llaisys::bf16_t *>(out),
                       reinterpret_cast<const llaisys::bf16_t *>(gate),
                       reinterpret_cast<const llaisys::bf16_t *>(up),
                       numel);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}
} // namespace llaisys::ops::cpu