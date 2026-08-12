#include "rope_musa.muh"

#include "../../../utils.hpp"

#include <musa_bf16.h>
#include <musa_fp16.h>
#include <musa_runtime.h>

#include <stdexcept>
#include <string>

namespace {

constexpr unsigned int BLOCK_SIZE = 256;

template <typename T>
__device__ float toFloat(T value);

template <>
__device__ float toFloat<float>(float value) {
    return value;
}

template <>
__device__ float toFloat<__half>(__half value) {
    return __half2float(value);
}

template <>
__device__ float toFloat<__mt_bfloat16>(
    __mt_bfloat16 value) {

    return __bfloat162float(value);
}

template <typename T>
__device__ T fromFloat(float value);

template <>
__device__ float fromFloat<float>(float value) {
    return value;
}

template <>
__device__ __half fromFloat<__half>(float value) {
    return __float2half_rn(value);
}

template <>
__device__ __mt_bfloat16
fromFloat<__mt_bfloat16>(float value) {
    return __float2bfloat16_rn(value);
}

template <typename T>
__global__ void ropeKernel(T *out,
    const T *in,
    const int64_t *pos_ids,
    size_t seq_len,
    size_t num_heads,
    size_t head_dim,
    float theta) {

    const size_t half_dim = head_dim / 2;

    const size_t frequency_index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    const size_t frequency_count = seq_len * half_dim;

    if (frequency_index >= frequency_count) {
        return;
    }

    const size_t token = frequency_index / half_dim;

    const size_t j = frequency_index % half_dim;

    const int64_t position = pos_ids[token];

    if (position < 0) {
        return;
    }

    const float exponent = 2.0f * static_cast<float>(j) / static_cast<float>(head_dim);

    const float denominator = powf(theta, exponent);

    const float angle = static_cast<float>(position) / denominator;

    float sin_value;
    float cos_value;

    sincosf(angle, &sin_value, &cos_value);

    // 同一 token 和频率 j 的所有 head
    // 使用相同的 sin/cos。
    for (size_t head = 0; head < num_heads; ++head) {

        const size_t base = (token * num_heads + head) * head_dim;

        const size_t first_index = base + j;

        const size_t second_index = base + half_dim + j;

        const float a = toFloat<T>(in[first_index]);

        const float b = toFloat<T>(in[second_index]);

        const float rotated_a = a * cos_value - b * sin_value;

        const float rotated_b = b * cos_value + a * sin_value;

        out[first_index] = fromFloat<T>(rotated_a);

        out[second_index] = fromFloat<T>(rotated_b);
    }
}

template <typename T>
void launchRope(std::byte *out,
    const std::byte *in,
    const int64_t *pos_ids,
    size_t seq_len,
    size_t num_heads,
    size_t head_dim,
    float theta,
    llaisysStream_t stream) {

    const size_t half_dim = head_dim / 2;

    const size_t frequency_count = seq_len * half_dim;

    if (frequency_count == 0) {
        return;
    }

    const unsigned int grid_size = static_cast<unsigned int>((frequency_count + BLOCK_SIZE - 1) / BLOCK_SIZE);

    ropeKernel<T><<<grid_size, BLOCK_SIZE, 0, reinterpret_cast<musaStream_t>(stream)>>>(
        reinterpret_cast<T *>(out),
        reinterpret_cast<const T *>(in),
        pos_ids,
        seq_len,
        num_heads,
        head_dim,
        theta);

    const musaError_t status = musaGetLastError();

    if (status != musaSuccess) {
        throw std::runtime_error(
            std::string("MUSA RoPE kernel launch failed: ")
             + musaGetErrorString(status));
    }
}

} // namespace

namespace llaisys::ops::musa {
void rope(std::byte *out,
    const std::byte *in,
    const int64_t *pos_ids,
    llaisysDataType_t dtype,
    size_t seq_len,
    size_t num_heads,
    size_t head_dim,
    float theta,
    llaisysStream_t stream) {

    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        return launchRope<float>(out,
            in,
            pos_ids,
            seq_len,
            num_heads,
            head_dim,
            theta,
            stream);
    case LLAISYS_DTYPE_F16:
        return launchRope<__half>(
            out,
            in,
            pos_ids,
            seq_len,
            num_heads,
            head_dim,
            theta,
            stream);
    case LLAISYS_DTYPE_BF16:
        return launchRope<__mt_bfloat16>(
            out,
            in,
            pos_ids,
            seq_len,
            num_heads,
            head_dim,
            theta,
            stream);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}
} // namespace llaisys::ops::musa