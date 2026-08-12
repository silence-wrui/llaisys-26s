#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/embedding_cpu.hpp"

#ifdef ENABLE_NVIDIA_API
#include "nvidia/embedding_nvidia.cuh"
#endif

#ifdef ENABLE_MUSA_API
#include "musa/embedding_musa.muh"
#endif

namespace llaisys::ops {
void embedding(tensor_t out,
               tensor_t index,
               tensor_t weight) {

    CHECK_SAME_DEVICE(out, index, weight);

    CHECK_ARGUMENT(index->ndim() == 1, "Embedding index must be a 1D tensor");
    CHECK_ARGUMENT(index->dtype() == LLAISYS_DTYPE_I64, "Embedding index must have int64 dtype");
    CHECK_ARGUMENT(weight->ndim() == 2, "Embedding weight must be a 2D tensor");
    CHECK_ARGUMENT(out->ndim() == 2, "Embedding output must be a 2D tensor");
    // 输出行数等于索引数量
    CHECK_ARGUMENT(out->shape()[0] == index->shape()[0],
                   "Embedding output rows must match index length");
    // 输出每行宽度等于 Embedding 每行宽度
    CHECK_ARGUMENT(
        out->shape()[1] == weight->shape()[1],
        "Embedding output width must match weight width");

    CHECK_SAME_DTYPE(out->dtype(), weight->dtype());

    ASSERT(out->isContiguous() && index->isContiguous() && weight->isContiguous(),
           "Embedding tensors must be contiguous");

    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::embedding(out->data(),
                              reinterpret_cast<const int64_t *>(index->data()),
                              weight->data(),
                              weight->dtype(),
                              index->numel(),
                              weight->shape()[0],
                              weight->shape()[1]);
    }

    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());

    switch (out->deviceType()) {
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        return nvidia::embedding(
            out->data(),
            reinterpret_cast<const int64_t *>(
                index->data()),
            weight->data(),
            weight->dtype(),
            index->numel(),
            weight->shape()[0],
            weight->shape()[1],
            llaisys::core::context()
                .runtime()
                .stream());
#endif
#ifdef ENABLE_MUSA_API
    case LLAISYS_DEVICE_MUSA:
        return musa::embedding(
            out->data(),
            reinterpret_cast<const int64_t *>(index->data()),
            weight->data(),
            weight->dtype(),
            index->numel(),
            weight->shape()[0],
            weight->shape()[1],
            llaisys::core::context().runtime().stream());
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace llaisys::ops