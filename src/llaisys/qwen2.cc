#include "llaisys/models/qwen2.h"

#include "llaisys_tensor.hpp"

#include "../models/qwen2/qwen2.hpp"
#include "../utils.hpp"

#include <vector>

struct LlaisysQwen2Model {
    llaisys::models::Qwen2Model *impl = nullptr;

    LlaisysQwen2Weights exposed_weights{};

    std::vector<llaisysTensor_t> owned_handles;
};

namespace {

llaisysTensor_t wrapTensor(
    LlaisysQwen2Model *model,
    const llaisys::tensor_t &tensor) {

    auto handle =
        new LlaisysTensor{tensor};

    model->owned_handles.push_back(handle);

    return handle;
}

void wrapTensorArray(
    LlaisysQwen2Model *model,
    llaisysTensor_t *&destination,
    const std::vector<llaisys::tensor_t> &source) {

    destination =
        new llaisysTensor_t[source.size()];

    for (size_t i = 0; i < source.size(); ++i) {
        destination[i] =
            wrapTensor(model, source[i]);
    }
}

void destroyWeightArrays(
    LlaisysQwen2Weights &weights) {

    delete[] weights.attn_norm_w;

    delete[] weights.attn_q_w;
    delete[] weights.attn_q_b;

    delete[] weights.attn_k_w;
    delete[] weights.attn_k_b;

    delete[] weights.attn_v_w;
    delete[] weights.attn_v_b;

    delete[] weights.attn_o_w;

    delete[] weights.mlp_norm_w;
    delete[] weights.mlp_gate_w;
    delete[] weights.mlp_up_w;
    delete[] weights.mlp_down_w;
}

} // namespace

__C {

LlaisysQwen2Model *llaisysQwen2ModelCreate(
    const LlaisysQwen2Meta *meta,
    llaisysDeviceType_t device,
    int *device_ids,
    int ndevice) {

    CHECK_ARGUMENT(meta != nullptr,
                   "Qwen2 meta must not be null");

    CHECK_ARGUMENT(device_ids != nullptr,
                   "Qwen2 device ids must not be null");

    // 当前 Assignment 3 先只支持单设备。
    CHECK_ARGUMENT(ndevice == 1,
                   "Qwen2 currently supports exactly one device");

    auto model = new LlaisysQwen2Model;

    model->impl =
        new llaisys::models::Qwen2Model(
            *meta,
            device,
            device_ids[0]);

    auto &internal =
        model->impl->weights();

    auto &exposed =
        model->exposed_weights;

    exposed.in_embed =
        wrapTensor(model, internal.in_embed);

    exposed.out_embed =
        wrapTensor(model, internal.out_embed);

    exposed.out_norm_w =
        wrapTensor(model, internal.out_norm_w);

    wrapTensorArray(
        model,
        exposed.attn_norm_w,
        internal.attn_norm_w);

    wrapTensorArray(
        model,
        exposed.attn_q_w,
        internal.attn_q_w);

    wrapTensorArray(
        model,
        exposed.attn_q_b,
        internal.attn_q_b);

    wrapTensorArray(
        model,
        exposed.attn_k_w,
        internal.attn_k_w);

    wrapTensorArray(
        model,
        exposed.attn_k_b,
        internal.attn_k_b);

    wrapTensorArray(
        model,
        exposed.attn_v_w,
        internal.attn_v_w);

    wrapTensorArray(
        model,
        exposed.attn_v_b,
        internal.attn_v_b);

    wrapTensorArray(
        model,
        exposed.attn_o_w,
        internal.attn_o_w);

    wrapTensorArray(
        model,
        exposed.mlp_norm_w,
        internal.mlp_norm_w);

    wrapTensorArray(
        model,
        exposed.mlp_gate_w,
        internal.mlp_gate_w);

    wrapTensorArray(
        model,
        exposed.mlp_up_w,
        internal.mlp_up_w);

    wrapTensorArray(
        model,
        exposed.mlp_down_w,
        internal.mlp_down_w);

    return model;
}

void llaisysQwen2ModelDestroy(
    LlaisysQwen2Model *model) {

    if (model == nullptr) {
        return;
    }

    // wrapper 只持有 shared_ptr，不会提前破坏内部模型。
    for (llaisysTensor_t handle :
         model->owned_handles) {
        delete handle;
    }

    destroyWeightArrays(
        model->exposed_weights);

    delete model->impl;
    model->impl = nullptr;

    delete model;
}

LlaisysQwen2Weights *llaisysQwen2ModelWeights(
    LlaisysQwen2Model *model) {

    CHECK_ARGUMENT(model != nullptr,
                   "Qwen2 model must not be null");

    return &model->exposed_weights;
}

int64_t llaisysQwen2ModelInfer(
    LlaisysQwen2Model *model,
    int64_t *token_ids,
    size_t ntoken) {

    CHECK_ARGUMENT(model != nullptr,
                   "Qwen2 model must not be null");

    CHECK_ARGUMENT(token_ids != nullptr,
                   "Qwen2 token ids must not be null");

    CHECK_ARGUMENT(ntoken > 0,
                   "Qwen2 token count must be positive");

    return model->impl->infer(
        token_ids,
        ntoken);
}

} // extern C