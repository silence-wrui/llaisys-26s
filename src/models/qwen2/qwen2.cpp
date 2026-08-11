#include "qwen2.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

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
    const int64_t *,
    size_t) {

    // 阶段 2 再实现。
    TO_BE_IMPLEMENTED();
    return -1;
}

} // namespace llaisys::models