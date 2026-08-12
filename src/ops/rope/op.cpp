#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/rope_cpu.hpp"

#ifdef ENABLE_NVIDIA_API
#include "nvidia/rope_nvidia.cuh"
#endif

#ifdef ENABLE_MUSA_API
#include "musa/rope_musa.muh"
#endif

namespace llaisys::ops {
void rope(tensor_t out, tensor_t in, tensor_t pos_ids, float theta) {
    CHECK_SAME_DEVICE(out, in, pos_ids);

    CHECK_ARGUMENT(in->ndim() == 3, "RoPE input must be a 3D tensor");
    CHECK_ARGUMENT(out->ndim() == 3, "RoPE output must be a 3D tensor");
    CHECK_ARGUMENT(pos_ids->ndim() == 1, "RoPE position ids must be a 1D tensor");

    CHECK_SAME_SHAPE(out->shape(), in->shape());

    CHECK_ARGUMENT(pos_ids->shape()[0] == in->shape()[0], "RoPE position count must match sequence length");

    CHECK_ARGUMENT(pos_ids->dtype() == LLAISYS_DTYPE_I64,
                   "RoPE position ids must have int64 dtype");

    CHECK_SAME_DTYPE(out->dtype(), in->dtype());

    CHECK_ARGUMENT(in->shape()[2] > 0 && in->shape()[2] % 2 == 0,
                   "RoPE head dimension must be positive and even");

    CHECK_ARGUMENT(theta > 0.0f, "RoPE theta must be positive");

    ASSERT(out->isContiguous() && in->isContiguous() && pos_ids->isContiguous(),
           "RoPE tensors must be contiguous");

    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::rope(out->data(),
                         in->data(),
                         reinterpret_cast<const int64_t *>(pos_ids->data()),
                         out->dtype(),
                         in->shape()[0],
                         in->shape()[1],
                         in->shape()[2],
                         theta);
    }

    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());

    switch (out->deviceType()) {
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        return nvidia::rope(out->data(),
            in->data(),
            reinterpret_cast<const int64_t *>(pos_ids->data()),
            out->dtype(),
            in->shape()[0],
            in->shape()[1],
            in->shape()[2],
            theta,
            llaisys::core::context().runtime().stream());
#endif
#ifdef ENABLE_MUSA_API
    case LLAISYS_DEVICE_MUSA:
        return musa::rope(
            out->data(),
            in->data(),
            reinterpret_cast<const int64_t *>(pos_ids->data()),
            out->dtype(),
            in->shape()[0],
            in->shape()[1],
            in->shape()[2],
            theta,
            llaisys::core::context().runtime().stream());
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace llaisys::ops