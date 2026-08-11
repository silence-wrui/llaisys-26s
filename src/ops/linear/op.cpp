#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/linear_cpu.hpp"

#ifdef ENABLE_NVIDIA_API
#include "nvidia/linear_nvidia.cuh"
#endif

namespace llaisys::ops {
void linear(tensor_t out, tensor_t in, tensor_t weight, tensor_t bias) {
    CHECK_SAME_DEVICE(out, in, weight);

    // 提供偏执的情况
    if (bias != nullptr) {
        CHECK_SAME_DEVICE(out, bias);
    }

    CHECK_ARGUMENT(in->ndim() == 2, "Linear input must be a 2D tensor");
    CHECK_ARGUMENT(weight->ndim() == 2, "Linear weight must be a 2D tensor");
    CHECK_ARGUMENT(out->ndim() == 2, "Linear output must be a 2D tensor");

    // 矩阵参数MNK的对应 相应维度对应
    CHECK_ARGUMENT(in->shape()[1] == weight->shape()[1],
                   "Linear input features must match weight input features");
    CHECK_ARGUMENT(out->shape()[0] == in->shape()[0],
                   "Linear output rows must match input rows");
    CHECK_ARGUMENT(out->shape()[1] == weight->shape()[0],
                   "Linear output features must match weight rows");

    // 数据类型相同
    CHECK_SAME_DTYPE(out->dtype(), in->dtype(), weight->dtype());

    // 提供bias时的检查
    if (bias != nullptr) {
        CHECK_ARGUMENT(bias->ndim() == 1, "Linear bias must be a 1D tensor");

        CHECK_ARGUMENT(bias->shape()[0] == weight->shape()[0], "Linear bias size must match output features");

        CHECK_SAME_DTYPE(out->dtype(), bias->dtype());
    }

    // 检查在内存上连续
    ASSERT(out->isContiguous() && in->isContiguous() && weight->isContiguous() && (bias == nullptr || bias->isContiguous()),
           "Linear tensors must be contiguous");

    // 设置bias，原来的代码直接访问 bias->tensor 当 bias == nullptr 时会崩溃
    // 把 C 空指针转换成空的 C++ shared_ptr<Tensor>
    const std::byte *bias_data = bias == nullptr ? nullptr : bias->data();

    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::linear(out->data(),
                           in->data(),
                           weight->data(),
                           bias_data,
                           out->dtype(),
                           in->shape()[0],
                           in->shape()[1],
                           weight->shape()[0]);
    }

    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());

    switch (out->deviceType()) {
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        return nvidia::linear(out->data(),
            in->data(),
            weight->data(),
            bias_data,
            out->dtype(),
            in->shape()[0],
            in->shape()[1],
            weight->shape()[0],
            out->deviceId(),
            llaisys::core::context().runtime().stream());
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace llaisys::ops