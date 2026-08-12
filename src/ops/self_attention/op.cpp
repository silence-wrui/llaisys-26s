#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/self_attention_cpu.hpp"

#ifdef ENABLE_NVIDIA_API
#include "nvidia/self_attention_nvidia.cuh"
#endif

#ifdef ENABLE_MUSA_API
#include "musa/self_attention_musa.muh"
#endif

namespace llaisys::ops {
void self_attention(tensor_t attn_val, tensor_t q, tensor_t k, tensor_t v, float scale) {
    CHECK_SAME_DEVICE(attn_val, q, k, v);

    CHECK_ARGUMENT(q->ndim() == 3, "SelfAttention q must be a 3D tensor");
    CHECK_ARGUMENT(k->ndim() == 3, "SelfAttention k must be a 3D tensor");
    CHECK_ARGUMENT(v->ndim() == 3, "SelfAttention v must be a 3D tensor");
    CHECK_ARGUMENT(attn_val->ndim() == 3, "SelfAttention output must be a 3D tensor");

    CHECK_ARGUMENT(k->shape()[0] == v->shape()[0],
                   "SelfAttention k and v lengths must match");
    CHECK_ARGUMENT(k->shape()[1] == v->shape()[1],
                   "SelfAttention k and v head counts must match");
    CHECK_ARGUMENT(q->shape()[2] == k->shape()[2],
                   "SelfAttention q and k dimensions must match");

    CHECK_ARGUMENT(attn_val->shape()[0] == q->shape()[0], "SelfAttention output length must match q length");
    CHECK_ARGUMENT(attn_val->shape()[1] == q->shape()[1], "SelfAttention output heads must match q heads");
    CHECK_ARGUMENT(attn_val->shape()[2] == v->shape()[2], "SelfAttention output dimension must match v dimension");

    CHECK_ARGUMENT(k->shape()[0] >= q->shape()[0], "SelfAttention KV length must not be shorter than q length");
    CHECK_ARGUMENT(q->shape()[0] > 0, "SelfAttention q length must be positive");
    CHECK_ARGUMENT(q->shape()[1] > 0 && k->shape()[1] > 0, "SelfAttention head counts must be positive");
    CHECK_ARGUMENT(q->shape()[1] % k->shape()[1] == 0, "SelfAttention query heads must be divisible by KV heads");
    CHECK_ARGUMENT(q->shape()[2] > 0 && v->shape()[2] > 0, "SelfAttention head dimensions must be positive");

    CHECK_SAME_DTYPE(attn_val->dtype(), q->dtype(), k->dtype(), v->dtype());

    ASSERT(attn_val->isContiguous()
               && q->isContiguous()
               && k->isContiguous()
               && v->isContiguous(),
           "SelfAttention tensors must be contiguous");

    if (attn_val->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::self_attention(
            attn_val->data(),
            q->data(),
            k->data(),
            v->data(),
            attn_val->dtype(),
            q->shape()[0],
            k->shape()[0],
            q->shape()[1],
            k->shape()[1],
            q->shape()[2],
            v->shape()[2],
            scale);
    }

    llaisys::core::context().setDevice(attn_val->deviceType(), attn_val->deviceId());

    switch (attn_val->deviceType()) {
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        return nvidia::self_attention(attn_val->data(),
            q->data(),
            k->data(),
            v->data(),
            attn_val->dtype(),
            q->shape()[0],
            k->shape()[0],
            q->shape()[1],
            k->shape()[1],
            q->shape()[2],
            v->shape()[2],
            scale,
            llaisys::core::context().runtime().stream());
#endif
#ifdef ENABLE_MUSA_API
    case LLAISYS_DEVICE_MUSA:
        return musa::self_attention(
            attn_val->data(),
            q->data(),
            k->data(),
            v->data(),
            attn_val->dtype(),
            q->shape()[0],
            k->shape()[0],
            q->shape()[1],
            k->shape()[1],
            q->shape()[2],
            v->shape()[2],
            scale,
            llaisys::core::context().runtime().stream());
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace llaisys::ops