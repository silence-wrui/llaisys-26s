#include "llaisys/ops.h"

#include "api_guard.hpp"
#include "llaisys_tensor.hpp"

#include "../ops/add/op.hpp"
#include "../ops/argmax/op.hpp"
#include "../ops/embedding/op.hpp"
#include "../ops/linear/op.hpp"
#include "../ops/rearrange/op.hpp"
#include "../ops/rms_norm/op.hpp"
#include "../ops/rope/op.hpp"
#include "../ops/self_attention/op.hpp"
#include "../ops/swiglu/op.hpp"
#include "../utils.hpp"

__C {

void llaisysAdd(llaisysTensor_t c, llaisysTensor_t a, llaisysTensor_t b) {
    llaisys::api::guardVoid([&] {
        CHECK_ARGUMENT(c != nullptr && a != nullptr && b != nullptr, "Add tensor handles must not be null");
        llaisys::ops::add(c->tensor, a->tensor, b->tensor);
    });
}

void llaisysArgmax(llaisysTensor_t max_idx, llaisysTensor_t max_val, llaisysTensor_t vals) {
    llaisys::api::guardVoid([&] {
        CHECK_ARGUMENT(
            max_idx != nullptr && max_val != nullptr && vals != nullptr,
            "Argmax tensor handles must not be null");
        llaisys::ops::argmax(max_idx->tensor, max_val->tensor, vals->tensor);
    });
}

void llaisysEmbedding(llaisysTensor_t out, llaisysTensor_t index, llaisysTensor_t weight) {
    llaisys::api::guardVoid([&] {
        CHECK_ARGUMENT(
            out != nullptr && index != nullptr && weight != nullptr,
            "Embedding tensor handles must not be null");
        llaisys::ops::embedding(out->tensor, index->tensor, weight->tensor);
    });
}

void llaisysLinear(
    llaisysTensor_t out, llaisysTensor_t in, llaisysTensor_t weight, llaisysTensor_t bias) {
    llaisys::api::guardVoid([&] {
        CHECK_ARGUMENT(
            out != nullptr && in != nullptr && weight != nullptr,
            "Linear tensor handles must not be null");
        llaisys::tensor_t bias_tensor = bias == nullptr ? nullptr : bias->tensor;
        llaisys::ops::linear(out->tensor, in->tensor, weight->tensor, bias_tensor);
    });
}

void llaisysRearrange(llaisysTensor_t out, llaisysTensor_t in) {
    llaisys::api::guardVoid([&] {
        CHECK_ARGUMENT(out != nullptr && in != nullptr, "Rearrange tensor handles must not be null");
        llaisys::ops::rearrange(out->tensor, in->tensor);
    });
}

void llaisysRmsNorm(llaisysTensor_t out, llaisysTensor_t in, llaisysTensor_t weight, float eps) {
    llaisys::api::guardVoid([&] {
        CHECK_ARGUMENT(
            out != nullptr && in != nullptr && weight != nullptr,
            "RMSNorm tensor handles must not be null");
        llaisys::ops::rms_norm(out->tensor, in->tensor, weight->tensor, eps);
    });
}

void llaisysROPE(llaisysTensor_t out, llaisysTensor_t in, llaisysTensor_t pos_ids, float theta) {
    llaisys::api::guardVoid([&] {
        CHECK_ARGUMENT(
            out != nullptr && in != nullptr && pos_ids != nullptr,
            "RoPE tensor handles must not be null");
        llaisys::ops::rope(out->tensor, in->tensor, pos_ids->tensor, theta);
    });
}

void llaisysSelfAttention(
    llaisysTensor_t attn_val, llaisysTensor_t q, llaisysTensor_t k,
    llaisysTensor_t v, float scale) {
    llaisys::api::guardVoid([&] {
        CHECK_ARGUMENT(
            attn_val != nullptr && q != nullptr && k != nullptr && v != nullptr,
            "SelfAttention tensor handles must not be null");
        llaisys::ops::self_attention(attn_val->tensor, q->tensor, k->tensor, v->tensor, scale);
    });
}

void llaisysSwiGLU(llaisysTensor_t out, llaisysTensor_t gate, llaisysTensor_t up) {
    llaisys::api::guardVoid([&] {
        CHECK_ARGUMENT(
            out != nullptr && gate != nullptr && up != nullptr,
            "SwiGLU tensor handles must not be null");
        llaisys::ops::swiglu(out->tensor, gate->tensor, up->tensor);
    });
}

} // extern C