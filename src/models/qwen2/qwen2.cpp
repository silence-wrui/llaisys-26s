#include "qwen2.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "../../ops/add/op.hpp"
#include "../../ops/argmax/op.hpp"
#include "../../ops/embedding/op.hpp"
#include "../../ops/linear/op.hpp"
#include "../../ops/rms_norm/op.hpp"
#include "../../ops/rope/op.hpp"
#include "../../ops/self_attention/op.hpp"
#include "../../ops/swiglu/op.hpp"

#include <algorithm>
#include <cmath>
#include <utility>
#include <vector>

namespace llaisys::models {
Qwen2Model::Qwen2Model(
    const LlaisysQwen2Meta &meta,
    llaisysDeviceType_t device,
    int device_id)
    : _meta(meta),
      _device(device),
      _device_id(device_id) {

    CHECK_ARGUMENT(_meta.nlayer > 0,
                   "Qwen2 must contain at least one layer");

    CHECK_ARGUMENT(_meta.hs > 0,
                   "Qwen2 hidden size must be positive");

    CHECK_ARGUMENT(_meta.nh > 0 && _meta.nkvh > 0,
                   "Qwen2 head counts must be positive");

    CHECK_ARGUMENT(_meta.nh % _meta.nkvh == 0,
                   "Query heads must be divisible by KV heads");

    CHECK_ARGUMENT(_meta.dh > 0,
                   "Qwen2 head dimension must be positive");

    CHECK_ARGUMENT(_meta.hs == _meta.nh * _meta.dh,
                   "Hidden size must equal num_heads * head_dim");

    CHECK_ARGUMENT(_meta.di > 0,
                   "Qwen2 intermediate size must be positive");

    CHECK_ARGUMENT(_meta.voc > 0,
                   "Qwen2 vocabulary size must be positive");

    CHECK_ARGUMENT(_meta.maxseq > 0,
                   "Qwen2 maximum sequence length must be positive");

    core::context().setDevice(_device, _device_id);

    allocateWeights();
}

void Qwen2Model::ensureCacheCapacity(
    size_t required) {

    CHECK_ARGUMENT(
        required <= _meta.maxseq,
        "Qwen2 sequence length exceeds maxseq");

    if (required <= _cache_capacity) {
        return;
    }

    core::context().setDevice(
        _device,
        _device_id);

    // 第一次至少分配 16 个 token，之后每次容量翻倍。
    size_t new_capacity =
        _cache_capacity == 0
            ? std::min<size_t>(16, _meta.maxseq)
            : _cache_capacity;

    while (new_capacity < required) {
        if (new_capacity > _meta.maxseq / 2) {
            new_capacity = _meta.maxseq;
        } else {
            new_capacity *= 2;
        }
    }

    std::vector<tensor_t> new_k_cache(
        _meta.nlayer);

    std::vector<tensor_t> new_v_cache(
        _meta.nlayer);

    for (size_t layer = 0;
         layer < _meta.nlayer;
         ++layer) {

        new_k_cache[layer] =
            Tensor::create(
                {
                    new_capacity,
                    _meta.nkvh,
                    _meta.dh,
                },
                _meta.dtype,
                _device,
                _device_id);

        new_v_cache[layer] =
            Tensor::create(
                {
                    new_capacity,
                    _meta.nkvh,
                    _meta.dh,
                },
                _meta.dtype,
                _device,
                _device_id);

        // 如果原缓存已经有历史 token，需要复制到新缓存。
        if (_past_len > 0) {
            const size_t bytes =
                _past_len
                * _meta.nkvh
                * _meta.dh
                * new_k_cache[layer]->elementSize();

            core::context()
                .runtime()
                .api()
                ->memcpy_sync(
                    new_k_cache[layer]->data(),
                    _k_cache[layer]->data(),
                    bytes,
                    LLAISYS_MEMCPY_D2D);

            core::context()
                .runtime()
                .api()
                ->memcpy_sync(
                    new_v_cache[layer]->data(),
                    _v_cache[layer]->data(),
                    bytes,
                    LLAISYS_MEMCPY_D2D);
        }
    }

    _k_cache = std::move(new_k_cache);
    _v_cache = std::move(new_v_cache);
    _cache_capacity = new_capacity;
}

void Qwen2Model::allocateWeights() {
    auto create = [&](const std::vector<size_t> &shape) {
        return Tensor::create(
            shape,
            _meta.dtype,
            _device,
            _device_id);
    };

    const size_t q_size = _meta.nh * _meta.dh;
    const size_t kv_size = _meta.nkvh * _meta.dh;

    _weights.in_embed =
        create({_meta.voc, _meta.hs});

    _weights.out_embed =
        create({_meta.voc, _meta.hs});

    _weights.out_norm_w =
        create({_meta.hs});

    _weights.attn_norm_w.resize(_meta.nlayer);

    _weights.attn_q_w.resize(_meta.nlayer);
    _weights.attn_q_b.resize(_meta.nlayer);

    _weights.attn_k_w.resize(_meta.nlayer);
    _weights.attn_k_b.resize(_meta.nlayer);

    _weights.attn_v_w.resize(_meta.nlayer);
    _weights.attn_v_b.resize(_meta.nlayer);

    _weights.attn_o_w.resize(_meta.nlayer);

    _weights.mlp_norm_w.resize(_meta.nlayer);
    _weights.mlp_gate_w.resize(_meta.nlayer);
    _weights.mlp_up_w.resize(_meta.nlayer);
    _weights.mlp_down_w.resize(_meta.nlayer);

    for (size_t layer = 0;
         layer < _meta.nlayer;
         ++layer) {

        _weights.attn_norm_w[layer] =
            create({_meta.hs});

        _weights.attn_q_w[layer] =
            create({q_size, _meta.hs});

        _weights.attn_q_b[layer] =
            create({q_size});

        _weights.attn_k_w[layer] =
            create({kv_size, _meta.hs});

        _weights.attn_k_b[layer] =
            create({kv_size});

        _weights.attn_v_w[layer] =
            create({kv_size, _meta.hs});

        _weights.attn_v_b[layer] =
            create({kv_size});

        _weights.attn_o_w[layer] =
            create({_meta.hs, q_size});

        _weights.mlp_norm_w[layer] =
            create({_meta.hs});

        _weights.mlp_gate_w[layer] =
            create({_meta.di, _meta.hs});

        _weights.mlp_up_w[layer] =
            create({_meta.di, _meta.hs});

        _weights.mlp_down_w[layer] =
            create({_meta.hs, _meta.di});
    }
}

const LlaisysQwen2Meta &Qwen2Model::meta() const {
    return _meta;
}

Qwen2Weights &Qwen2Model::weights() {
    return _weights;
}

llaisysDeviceType_t Qwen2Model::deviceType() const {
    return _device;
}

int Qwen2Model::deviceId() const {
    return _device_id;
}

int64_t Qwen2Model::infer(
    const int64_t *token_ids,
    size_t ntoken) {

    CHECK_ARGUMENT(
        token_ids != nullptr,
        "Qwen2 token ids must not be null");

    CHECK_ARGUMENT(
        ntoken > 0,
        "Qwen2 token count must be positive");

    CHECK_ARGUMENT(
        ntoken <= _meta.maxseq - _past_len,
        "Qwen2 sequence length exceeds maxseq");

    core::context().setDevice(
        _device,
        _device_id);

    const size_t total_len =
        _past_len + ntoken;

    ensureCacheCapacity(total_len);

    auto create = [&](
                      const std::vector<size_t> &shape,
                      llaisysDataType_t dtype) {

        return Tensor::create(
            shape,
            dtype,
            _device,
            _device_id);
    };

    auto createModelTensor =
        [&](const std::vector<size_t> &shape) {

            return create(
                shape,
                _meta.dtype);
        };

    /*
     * 第一部分：准备 token id 和位置 id
     */

    auto input_ids =
        create(
            {ntoken},
            LLAISYS_DTYPE_I64);

    input_ids->load(token_ids);

    std::vector<int64_t> host_positions(
        ntoken);

    for (size_t i = 0; i < ntoken; ++i) {
        host_positions[i] =
            static_cast<int64_t>(
                _past_len + i);
    }

    auto position_ids =
        create(
            {ntoken},
            LLAISYS_DTYPE_I64);

    position_ids->load(
        host_positions.data());

    /*
     * 第二部分：Embedding
     */

    auto hidden =
        createModelTensor(
            {ntoken, _meta.hs});

    ops::embedding(
        hidden,
        input_ids,
        _weights.in_embed);

    /*
     * 第三部分：创建各层可以重复使用的临时 Tensor
     */

    const size_t q_size =
        _meta.nh * _meta.dh;

    const size_t kv_size =
        _meta.nkvh * _meta.dh;

    auto norm =
        createModelTensor(
            {ntoken, _meta.hs});

    auto q_projection =
        createModelTensor(
            {ntoken, q_size});

    auto k_projection =
        createModelTensor(
            {ntoken, kv_size});

    auto v_projection =
        createModelTensor(
            {ntoken, kv_size});

    auto q =
        q_projection->view(
            {ntoken, _meta.nh, _meta.dh});

    auto k =
        k_projection->view(
            {ntoken, _meta.nkvh, _meta.dh});

    auto v =
        v_projection->view(
            {ntoken, _meta.nkvh, _meta.dh});

    auto rotated_q =
        createModelTensor(
            {ntoken, _meta.nh, _meta.dh});

    auto rotated_k =
        createModelTensor(
            {ntoken, _meta.nkvh, _meta.dh});

    auto attention_value =
        createModelTensor(
            {ntoken, _meta.nh, _meta.dh});

    auto attention_flat =
        attention_value->view(
            {ntoken, q_size});

    auto attention_projection =
        createModelTensor(
            {ntoken, _meta.hs});

    auto attention_residual =
        createModelTensor(
            {ntoken, _meta.hs});

    auto gate =
        createModelTensor(
            {ntoken, _meta.di});

    auto up =
        createModelTensor(
            {ntoken, _meta.di});

    auto activated =
        createModelTensor(
            {ntoken, _meta.di});

    auto down =
        createModelTensor(
            {ntoken, _meta.hs});

    auto mlp_residual =
        createModelTensor(
            {ntoken, _meta.hs});

    const float attention_scale =
        1.0f / std::sqrt(
                   static_cast<float>(_meta.dh));

    /*
     * 第四部分：依次执行所有 Transformer 层
     */

    for (size_t layer = 0;
         layer < _meta.nlayer;
         ++layer) {

        /*
         * 4.1 Attention 前的 RMSNorm
         */

        ops::rms_norm(
            norm,
            hidden,
            _weights.attn_norm_w[layer],
            _meta.epsilon);

        /*
         * 4.2 计算 Q、K、V
         */

        ops::linear(
            q_projection,
            norm,
            _weights.attn_q_w[layer],
            _weights.attn_q_b[layer]);

        ops::linear(
            k_projection,
            norm,
            _weights.attn_k_w[layer],
            _weights.attn_k_b[layer]);

        ops::linear(
            v_projection,
            norm,
            _weights.attn_v_w[layer],
            _weights.attn_v_b[layer]);

        /*
         * 4.3 给 Q、K 加入旋转位置编码
         */

        ops::rope(
            rotated_q,
            q,
            position_ids,
            _meta.theta);

        ops::rope(
            rotated_k,
            k,
            position_ids,
            _meta.theta);

        /*
         * 4.4 把本轮 K、V 写入当前层的 KV Cache
         */

        auto k_destination =
            _k_cache[layer]->slice(
                0,
                _past_len,
                total_len);

        auto v_destination =
            _v_cache[layer]->slice(
                0,
                _past_len,
                total_len);

        core::context()
            .runtime()
            .api()
            ->memcpy_sync(
                k_destination->data(),
                rotated_k->data(),
                rotated_k->numel()
                    * rotated_k->elementSize(),
                LLAISYS_MEMCPY_D2D);

        core::context()
            .runtime()
            .api()
            ->memcpy_sync(
                v_destination->data(),
                v->data(),
                v->numel()
                    * v->elementSize(),
                LLAISYS_MEMCPY_D2D);

        /*
         * 4.5 Attention 需要读取：
         *
         * [历史 token + 本轮 token]
         */

        auto all_k =
            _k_cache[layer]->slice(
                0,
                0,
                total_len);

        auto all_v =
            _v_cache[layer]->slice(
                0,
                0,
                total_len);

        ops::self_attention(
            attention_value,
            rotated_q,
            all_k,
            all_v,
            attention_scale);

        /*
         * 4.6 拼回所有 Query Head，并执行 O Projection
         */

        ops::linear(
            attention_projection,
            attention_flat,
            _weights.attn_o_w[layer],
            nullptr);

        /*
         * 4.7 第一次残差连接
         *
         * attention_residual =
         *     hidden + attention_projection
         */

        ops::add(
            attention_residual,
            hidden,
            attention_projection);

        /*
         * 4.8 MLP 前的 RMSNorm
         */

        ops::rms_norm(
            norm,
            attention_residual,
            _weights.mlp_norm_w[layer],
            _meta.epsilon);

        /*
         * 4.9 Gate Projection 和 Up Projection
         */

        ops::linear(
            gate,
            norm,
            _weights.mlp_gate_w[layer],
            nullptr);

        ops::linear(
            up,
            norm,
            _weights.mlp_up_w[layer],
            nullptr);

        /*
         * 4.10 SwiGLU
         *
         * activated = SiLU(gate) * up
         */

        ops::swiglu(
            activated,
            gate,
            up);

        /*
         * 4.11 Down Projection
         */

        ops::linear(
            down,
            activated,
            _weights.mlp_down_w[layer],
            nullptr);

        /*
         * 4.12 第二次残差连接
         *
         * mlp_residual =
         *     attention_residual + down
         */

        ops::add(
            mlp_residual,
            attention_residual,
            down);

        hidden = mlp_residual;
    }

    /*
     * 第五部分：最终 RMSNorm
     */

    auto final_norm =
        createModelTensor(
            {ntoken, _meta.hs});

    ops::rms_norm(
        final_norm,
        hidden,
        _weights.out_norm_w,
        _meta.epsilon);

    /*
     * 只需要最后一个 token 的输出预测下一个 token。
     */

    auto last_hidden =
        final_norm->slice(
            0,
            ntoken - 1,
            ntoken);

    auto logits =
        createModelTensor(
            {1, _meta.voc});

    ops::linear(
        logits,
        last_hidden,
        _weights.out_embed,
        nullptr);

    /*
     * Argmax 算子要求输入是一维 Tensor。
     */

    auto logits_1d =
        logits->view(
            {_meta.voc});

    auto max_index =
        create(
            {1},
            LLAISYS_DTYPE_I64);

    auto max_value =
        createModelTensor(
            {1});

    ops::argmax(
        max_index,
        max_value,
        logits_1d);

    /*
     * 所有层都使用完旧的 _past_len 之后，
     * 才能更新缓存长度。
     */

    _past_len = total_len;

    /*
     * 把结果 token 从模型设备复制回 CPU。
     */

    int64_t next_token = -1;

    const llaisysMemcpyKind_t copy_kind =
        _device == LLAISYS_DEVICE_CPU
            ? LLAISYS_MEMCPY_H2H
            : LLAISYS_MEMCPY_D2H;

    core::context()
        .runtime()
        .api()
        ->memcpy_sync(
            &next_token,
            max_index->data(),
            sizeof(next_token),
            copy_kind);

    return next_token;
}

} // namespace llaisys::models