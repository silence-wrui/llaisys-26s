#pragma once   //防止头文件重复包含

#include "llaisys/models/qwen2.h"   //引入公共的LlaisysQwen2Meta 内部模型直接保存一份 Meta
#include "../../tensor/tensor.hpp"

#include <vector>

namespace llaisys::models {
struct Qwen2Weights {
    tensor_t in_embed;
    tensor_t out_embed;
    tensor_t out_norm_w;

    std::vector<tensor_t> attn_norm_w;

    std::vector<tensor_t> attn_q_w;
    std::vector<tensor_t> attn_q_b;

    std::vector<tensor_t> attn_k_w;
    std::vector<tensor_t> attn_k_b;

    std::vector<tensor_t> attn_v_w;
    std::vector<tensor_t> attn_v_b;

    std::vector<tensor_t> attn_o_w;

    std::vector<tensor_t> mlp_norm_w;
    std::vector<tensor_t> mlp_gate_w;
    std::vector<tensor_t> mlp_up_w;
    std::vector<tensor_t> mlp_down_w;
};

class Qwen2Model {
private:
    LlaisysQwen2Meta _meta;   //保存模型配置副本
    //保存模型所在设备
    llaisysDeviceType_t _device;
    int _device_id;

    //真正拥有模型全部权重 创建模型时，这些字段最初为空
    //allocateWeights() 会为它们创建 Tensor，Python 再通过 bridge 把 safetensors 数据加载进去
    Qwen2Weights _weights;

    // 阶段 2 再使用。
    size_t _past_len = 0;   //KV Cache 中已经保存多少个历史 token 初始值为 0
    size_t _cache_capacity = 0; //当前 KV Cache 最多能装多少个 token

    //每层都有一份 K Cache 和 V Cache
    std::vector<tensor_t> _k_cache;
    std::vector<tensor_t> _v_cache;

    void allocateWeights();

public:
    Qwen2Model(
        const LlaisysQwen2Meta &meta,
        llaisysDeviceType_t device,
        int device_id);

    Qwen2Model(const Qwen2Model &) = delete;
    Qwen2Model &operator=(const Qwen2Model &) = delete;

    const LlaisysQwen2Meta &meta() const;
    Qwen2Weights &weights();

    llaisysDeviceType_t deviceType() const;
    int deviceId() const;

    // 阶段 2 实现。
    int64_t infer(
        const int64_t *token_ids,
        size_t ntoken);
};
} // namespace llaisys::models