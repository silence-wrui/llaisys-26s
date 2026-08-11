#include "self_attention_cpu.hpp"

#include "../../../utils.hpp"

#include <cmath>
#include <limits>
#include <vector>

template <typename T>
void self_attention_(T *attn_val,
                     const T *q,
                     const T *k,
                     const T *v,
                     size_t q_len,
                     size_t kv_len,
                     size_t num_heads,
                     size_t num_kv_heads,
                     size_t qk_dim,
                     size_t value_dim,
                     float scale) {
    //几个num_head公用一个kv_head
    const size_t heads_per_kv_head = num_heads / num_kv_heads;
    //历史token
    const size_t past_len = kv_len - q_len;
    // 保存当前一个 query head 对所有有效 key 的分数。
    // 在不同 query/head 之间重复使用这块内存。
    std::vector<float> scores(kv_len);
    //遍历q
    for (size_t query_pos = 0; query_pos < q_len; ++query_pos) {
        // 当前 query 在整个上下文中的位置。
        const size_t max_key_pos = past_len + query_pos;
        for (size_t query_head = 0; query_head < num_heads; ++query_head) {
            // GQA：多个 query head 共享一个 KV head。
            const size_t kv_head = query_head / heads_per_kv_head;
            float max_score = -std::numeric_limits<float>::infinity();

            // 第一步：计算 QK^T，并寻找最大分数。
            for (size_t key_pos = 0; key_pos <= max_key_pos; ++key_pos) {
                float dot_product = 0.0f;
                for (size_t dim = 0; dim < qk_dim; ++dim) {
                    const size_t q_index = (query_pos * num_heads + query_head) * qk_dim + dim;
                    const size_t k_index = (key_pos * num_kv_heads + kv_head) * qk_dim + dim;
                    const float q_value = llaisys::utils::cast<float>(q[q_index]);
                    const float k_value = llaisys::utils::cast<float>(k[k_index]);
                    dot_product += q_value * k_value;
                }
                const float score = dot_product * scale;
                scores[key_pos] = score;
                if (score > max_score) {
                    max_score = score;
                }
            }

            // 第二步：稳定 Softmax。
            float exp_sum = 0.0f;
            for (size_t key_pos = 0; key_pos <= max_key_pos; ++key_pos) {
                const float exp_score = std::exp(scores[key_pos] - max_score);
                scores[key_pos] = exp_score;
                exp_sum += exp_score;
            }
            const float inverse_exp_sum = 1.0f / exp_sum;
            for (size_t key_pos = 0; key_pos <= max_key_pos; ++key_pos) {
                scores[key_pos] *= inverse_exp_sum;
            }

            // 第三步：Softmax(A) × V。
            for (size_t value_col = 0; value_col < value_dim; ++value_col) {
                float result = 0.0f;
                for (size_t key_pos = 0; key_pos <= max_key_pos; ++key_pos) {
                    const size_t v_index = (key_pos * num_kv_heads + kv_head)
                                                    * value_dim + value_col;
                    const float v_value = llaisys::utils::cast<float>(v[v_index]);
                    result += scores[key_pos] * v_value;
                }

                const size_t out_index = (query_pos * num_heads + query_head)
                                                    * value_dim + value_col;
                attn_val[out_index] = llaisys::utils::cast<T>(result);
            }
        }
    }
}

namespace llaisys::ops::cpu {
void self_attention(std::byte *attn_val,
                    const std::byte *q,
                    const std::byte *k,
                    const std::byte *v,
                    llaisysDataType_t dtype,
                    size_t q_len,
                    size_t kv_len,
                    size_t num_heads,
                    size_t num_kv_heads,
                    size_t qk_dim,
                    size_t value_dim,
                    float scale) {
    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        return self_attention_(reinterpret_cast<float *>(attn_val),
                               reinterpret_cast<const float *>(q),
                               reinterpret_cast<const float *>(k),
                               reinterpret_cast<const float *>(v),
                               q_len,
                               kv_len,
                               num_heads,
                               num_kv_heads,
                               qk_dim,
                               value_dim,
                               scale);
    case LLAISYS_DTYPE_F16:
        return self_attention_(reinterpret_cast<llaisys::fp16_t *>(attn_val),
                               reinterpret_cast<const llaisys::fp16_t *>(q),
                               reinterpret_cast<const llaisys::fp16_t *>(k),
                               reinterpret_cast<const llaisys::fp16_t *>(v),
                               q_len,
                               kv_len,
                               num_heads,
                               num_kv_heads,
                               qk_dim,
                               value_dim,
                               scale);
    case LLAISYS_DTYPE_BF16:
        return self_attention_(reinterpret_cast<llaisys::bf16_t *>(attn_val),
                               reinterpret_cast<const llaisys::bf16_t *>(q),
                               reinterpret_cast<const llaisys::bf16_t *>(k),
                               reinterpret_cast<const llaisys::bf16_t *>(v),
                               q_len,
                               kv_len,
                               num_heads,
                               num_kv_heads,
                               qk_dim,
                               value_dim,
                               scale);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}
} // namespace llaisys::ops::cpu