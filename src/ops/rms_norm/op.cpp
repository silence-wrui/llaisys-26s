#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/rms_norm_cpu.hpp"

#ifdef ENABLE_NVIDIA_API
#include "nvidia/rms_norm_nvidia.cuh"
#endif

namespace llaisys::ops {
void rms_norm(tensor_t out, tensor_t in, tensor_t weight, float eps) {
    CHECK_SAME_DEVICE(out, in, weight);

    //检查维度
    CHECK_ARGUMENT(in->ndim() == 2, "RMSNorm input must be a 2D tensor");
    CHECK_ARGUMENT(out->ndim() == 2, "RMSNorm output must be a 2D tensor");
    CHECK_ARGUMENT(weight->ndim() == 1, "RMSNorm weight must be a 1D tensor");
    //检查形状一致 RMSNorm 不改变形状，所以输入输出必须完全一致
    CHECK_SAME_SHAPE(out->shape(), in->shape());

    CHECK_ARGUMENT(in->shape()[1] > 0, "RMSNorm hidden size must be greater than zero");
    // weight 对应最后一个维度，它会被重复用于每一行 访问weight[col]
    CHECK_ARGUMENT(weight->shape()[0] == in->shape()[1],
                   "RMSNorm weight size must match input hidden size");
    CHECK_ARGUMENT(eps >= 0.0f, "RMSNorm eps must not be negative");

    CHECK_SAME_DTYPE(out->dtype(), in->dtype(), weight->dtype());

    ASSERT(out->isContiguous() && in->isContiguous() && weight->isContiguous(),
           "RMSNorm tensors must be contiguous");

    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::rms_norm(out->data(),
                             in->data(),
                             weight->data(),
                             out->dtype(),
                             in->shape()[0],
                             in->shape()[1],
                             eps);
    }

    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());

    switch (out->deviceType()) {
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        return nvidia::rms_norm(
            out->data(),
            in->data(),
            weight->data(),
            out->dtype(),
            in->shape()[0],
            in->shape()[1],
            eps,
            llaisys::core::context()
                .runtime()
                .stream());
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace llaisys::ops