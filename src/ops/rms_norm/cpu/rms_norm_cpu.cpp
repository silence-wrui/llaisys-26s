#include "rms_norm_cpu.hpp"

#include "../../../utils.hpp"

#include <cmath>

template <typename T>
void rms_norm_(T *out,
               const T *in,
               const T *weight,
               size_t num_rows,
               size_t hidden_size,
               float eps) {

    for (size_t row = 0; row < num_rows; ++row) {
        float square_sum = 0.0f;

        //标准化沿输入张量的最后一个维度（即每一行，长度为 d 即 hidden_size）执行
        // 对一行中x平方求和 hidden_size个x
        for (size_t col = 0; col < hidden_size; ++col) {
            // 算出在一维数组下标索引
            const size_t index = row * hidden_size + col;
            // 取值转float FP16/BF16能表示的有效精度较低 误差
            const float value = llaisys::utils::cast<float>(in[index]); 
            square_sum += value * value;                                // 平方和
        }

        // 求平均数
        const float mean_square = square_sum / static_cast<float>(hidden_size);
        // 分母根号下那一串的倒数 之后计算除法变乘法 
        // 没有对每个元素重复计算 input_value / sqrt(...) 避免同一行重复执行昂贵的平方根和除法
        const float inverse_rms = 1.0f / std::sqrt(mean_square + eps);

        for (size_t col = 0; col < hidden_size; ++col) {
            // 算出在一维数组下标索引
            const size_t index = row * hidden_size + col;
            //取出weight、x相应的值
            const float input_value = llaisys::utils::cast<float>(in[index]);
            const float weight_value = llaisys::utils::cast<float>(weight[col]);
            //算出相对应的y
            const float result = input_value * inverse_rms * weight_value;
            //存入y
            out[index] = llaisys::utils::cast<T>(result);
        }
    }
}

namespace llaisys::ops::cpu {
void rms_norm(std::byte *out,
              const std::byte *in,
              const std::byte *weight,
              llaisysDataType_t dtype,
              size_t num_rows,
              size_t hidden_size,
              float eps) {
    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        return rms_norm_(reinterpret_cast<float *>(out),
                         reinterpret_cast<const float *>(in),
                         reinterpret_cast<const float *>(weight),
                         num_rows,
                         hidden_size,
                         eps);
    case LLAISYS_DTYPE_F16:
        return rms_norm_(reinterpret_cast<llaisys::fp16_t *>(out),
                         reinterpret_cast<const llaisys::fp16_t *>(in),
                         reinterpret_cast<const llaisys::fp16_t *>(weight),
                         num_rows,
                         hidden_size,
                         eps);
    case LLAISYS_DTYPE_BF16:
        return rms_norm_(reinterpret_cast<llaisys::bf16_t *>(out),
                         reinterpret_cast<const llaisys::bf16_t *>(in),
                         reinterpret_cast<const llaisys::bf16_t *>(weight),
                         num_rows,
                         hidden_size,
                         eps);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}
} // namespace llaisys::ops::cpu