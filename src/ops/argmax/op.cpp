#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/argmax_cpu.hpp"

#ifdef ENABLE_NVIDIA_API
#include "nvidia/argmax_nvidia.cuh"
#endif

namespace llaisys::ops {
void argmax(tensor_t max_idx, tensor_t max_val, tensor_t vals) {
    // 要求三个 Tensor 位于同一个设备，否则可能出现 CPU 函数读取 GPU 地址等严重错误
    CHECK_SAME_DEVICE(max_idx, max_val, vals);
    CHECK_ARGUMENT(vals->ndim() == 1, "Argmax input must be a 1D tensor");
    CHECK_ARGUMENT(vals->numel() > 0, "Argmax input must not be empty");
    // 最大值只有一个
    CHECK_ARGUMENT(max_idx->ndim() == 1 && max_idx->numel() == 1,
                   "Argmax index output must contain one element");
    CHECK_ARGUMENT(max_val->ndim() == 1 && max_val->numel() == 1,
                   "Argmax value output must contain one element");
    CHECK_ARGUMENT(max_idx->dtype() == LLAISYS_DTYPE_I64, "Argmax index output must have int64 dtype");

    CHECK_SAME_DTYPE(max_val->dtype(), vals->dtype());

    ASSERT(vals->isContiguous() && max_idx->isContiguous() && max_val->isContiguous(),
           "Argmax tensors must be contiguous");

    if (vals->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::argmax(max_idx->data(),
                           max_val->data(),
                           vals->data(),
                           vals->dtype(),
                           vals->numel());
    }

    llaisys::core::context().setDevice(vals->deviceType(), vals->deviceId());

    switch (vals->deviceType()) {
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        return nvidia::argmax(max_idx->data(),
            max_val->data(),
            vals->data(),
            vals->dtype(),
            vals->numel(),
            llaisys::core::context().runtime().stream());
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace llaisys::ops
