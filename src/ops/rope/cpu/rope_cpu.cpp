#include "rope_cpu.hpp"

#include "../../../utils.hpp"

#include <cmath>
#include <vector>

template <typename T>
void rope_(T *out,
           const T *in,
           const int64_t *pos_ids,
           size_t seq_len,
           size_t num_heads,
           size_t head_dim,
           float theta) {

    const size_t half_dim = head_dim / 2; //向量按照最后一维拆成两半了，每一半长度d
    // theta^(2j/d) 只与j有关，与token、head、position无关，所以提前计算，避免在内层循环反复调用 pow
    // std::pow 比普通乘法昂贵
    std::vector<float> denominators(half_dim);
    for (size_t j = 0; j < half_dim; ++j) {
        //计算2j/d
        const float exponent = 2.0f * static_cast<float>(j) / static_cast<float>(head_dim);
        denominators[j] = std::pow(theta, exponent);  //计算theta^(2j/d)
    }

    for (size_t token = 0; token < seq_len; ++token) {
        //取出序列中每个token的位置id
        const int64_t position = pos_ids[token];
        //位置id不为负数
        CHECK_ARGUMENT(position >= 0, "RoPE position id must not be negative");

        for (size_t j = 0; j < half_dim; ++j) {
            //计算 RoPE 的角度
            const float angle = static_cast<float>(position) / denominators[j];
            //计算sin、cos值
            const float cos_value = std::cos(angle);
            const float sin_value = std::sin(angle);

            // 同一个token的所有head使用相同位置和频率，因此sin/cos可以在head循环外计算
            for (size_t head = 0; head < num_heads; ++head) {
                //每个向量的起点 索引
                const size_t base = (token * num_heads + head) * head_dim;

                const size_t first_index = base + j;             //前半部分第 j 个元素：
                const size_t second_index = base + half_dim + j; //后半部分对应元素：
                const float a = llaisys::utils::cast<float>(in[first_index]);
                const float b = llaisys::utils::cast<float>(in[second_index]);

                //输出向量a、b的计算
                const float rotated_a = a * cos_value - b * sin_value;
                const float rotated_b = b * cos_value + a * sin_value;
                
                //存入out
                out[first_index] = llaisys::utils::cast<T>(rotated_a);
                out[second_index] = llaisys::utils::cast<T>(rotated_b);
            }
        }
    }
}

namespace llaisys::ops::cpu {
void rope(std::byte *out,
          const std::byte *in,
          const int64_t *pos_ids,
          llaisysDataType_t dtype,
          size_t seq_len,
          size_t num_heads,
          size_t head_dim,
          float theta) {
    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        return rope_(reinterpret_cast<float *>(out),
                     reinterpret_cast<const float *>(in),
                     pos_ids,
                     seq_len,
                     num_heads,
                     head_dim,
                     theta);
    case LLAISYS_DTYPE_F16:
        return rope_(reinterpret_cast<llaisys::fp16_t *>(out),
                     reinterpret_cast<const llaisys::fp16_t *>(in),
                     pos_ids,
                     seq_len,
                     num_heads,
                     head_dim,
                     theta);
    case LLAISYS_DTYPE_BF16:
        return rope_(reinterpret_cast<llaisys::bf16_t *>(out),
                     reinterpret_cast<const llaisys::bf16_t *>(in),
                     pos_ids,
                     seq_len,
                     num_heads,
                     head_dim,
                     theta);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}
} // namespace llaisys::ops::cpu