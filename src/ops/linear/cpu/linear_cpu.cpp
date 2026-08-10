#include "linear_cpu.hpp"

#include "../../../utils.hpp"

template <typename T>
void linear_(T *out,
             const T *in,
             const T *weight,
             const T *bias,
             size_t num_rows,
             size_t in_features,    // in的列数，一行的元素个数
             size_t out_features) { // out的列数，一行的元素个数

    for (size_t row = 0; row < num_rows; ++row) {
        for (size_t out_col = 0; out_col < out_features; ++out_col) {

            float result = 0.0f;

            // 偏置b
            if (bias != nullptr) {
                result = llaisys::utils::cast<float>(bias[out_col]);
            }

            for (size_t in_col = 0; in_col < in_features; ++in_col) {
                const float input_value = llaisys::utils::cast<float>(in[row * in_features + in_col]);
                // weight[out_col * in_features + in_col]正好转置了
                // weight 在内存中的实际形状是 [out_features, in_features]，没有真的转置
                const float weight_value = llaisys::utils::cast<float>(weight[out_col * in_features + in_col]);

                result += input_value * weight_value;
            }

            out[row * out_features + out_col] = llaisys::utils::cast<T>(result);
        }
    }
}

namespace llaisys::ops::cpu {
void linear(std::byte *out,
            const std::byte *in,
            const std::byte *weight,
            const std::byte *bias,
            llaisysDataType_t dtype,
            size_t num_rows,
            size_t in_features,
            size_t out_features) {
    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        return linear_(reinterpret_cast<float *>(out),
                       reinterpret_cast<const float *>(in),
                       reinterpret_cast<const float *>(weight),
                       reinterpret_cast<const float *>(bias),
                       num_rows,
                       in_features,
                       out_features);

    case LLAISYS_DTYPE_F16:
        return linear_(reinterpret_cast<llaisys::fp16_t *>(out),
                       reinterpret_cast<const llaisys::fp16_t *>(in),
                       reinterpret_cast<const llaisys::fp16_t *>(weight),
                       reinterpret_cast<const llaisys::fp16_t *>(bias),
                       num_rows,
                       in_features,
                       out_features);

    case LLAISYS_DTYPE_BF16:
        return linear_(reinterpret_cast<llaisys::bf16_t *>(out),
                       reinterpret_cast<const llaisys::bf16_t *>(in),
                       reinterpret_cast<const llaisys::bf16_t *>(weight),
                       reinterpret_cast<const llaisys::bf16_t *>(bias),
                       num_rows,
                       in_features,
                       out_features);

    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}
} // namespace llaisys::ops::cpu