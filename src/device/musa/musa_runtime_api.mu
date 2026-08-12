#include "../runtime_api.hpp"

#include <musa_runtime.h>

#include <sstream>
#include <stdexcept>

#define MUSA_CHECK(expression) \
    checkMusa(                  \
        (expression),           \
        #expression,            \
        __FILE__,               \
        __LINE__)

namespace llaisys::device::musa {

namespace {

void checkMusa(
    musaError_t status,
    const char *expression,
    const char *file,
    int line) {

    if (status == musaSuccess) {
        return;
    }

    std::ostringstream message;
    message
        << "MUSA call failed: "
        << expression
        << " at "
        << file
        << ":"
        << line
        << ", error="
        << musaGetErrorName(status)
        << " ("
        << musaGetErrorString(status)
        << ")";

    throw std::runtime_error(message.str());
}

musaMemcpyKind musaMemcpyKindFrom(
    llaisysMemcpyKind_t kind) {

    switch (kind) {
    case LLAISYS_MEMCPY_H2H:
        return musaMemcpyHostToHost;

    case LLAISYS_MEMCPY_H2D:
        return musaMemcpyHostToDevice;

    case LLAISYS_MEMCPY_D2H:
        return musaMemcpyDeviceToHost;

    case LLAISYS_MEMCPY_D2D:
        return musaMemcpyDeviceToDevice;

    default:
        throw std::invalid_argument(
            "Unsupported LLAISYS memcpy kind");
    }
}

} // namespace

namespace runtime_api {
int getDeviceCount() {
    int count = 0;
    MUSA_CHECK(musaGetDeviceCount(&count));
    return count;
}

void setDevice(int device) {
    MUSA_CHECK(musaSetDevice(device));
}

void deviceSynchronize() {
    MUSA_CHECK(musaDeviceSynchronize());
}

llaisysStream_t createStream() {
    musaStream_t stream = nullptr;

    MUSA_CHECK(musaStreamCreate(&stream));

    return reinterpret_cast<llaisysStream_t>(stream);
}

void destroyStream(llaisysStream_t stream) {
    if (stream == nullptr) {
        return;
    }

    MUSA_CHECK(
        musaStreamDestroy(
            reinterpret_cast<musaStream_t>(
                stream)));
}

void streamSynchronize(llaisysStream_t stream) {
    MUSA_CHECK(
        musaStreamSynchronize(
            reinterpret_cast<musaStream_t>(
                stream)));
}

void *mallocDevice(size_t size) {
    void *ptr = nullptr;

    MUSA_CHECK(musaMalloc(&ptr, size));

    return ptr;
}

void freeDevice(void *ptr) {
    if (ptr == nullptr) {
        return;
    }

    MUSA_CHECK(musaFree(ptr));
}

void *mallocHost(size_t size) {
    void *ptr = nullptr;

    MUSA_CHECK(musaMallocHost(&ptr, size));

    return ptr;
}

void freeHost(void *ptr) {
    if (ptr == nullptr) {
        return;
    }

    MUSA_CHECK(musaFreeHost(ptr));
}

void memcpySync(void *dst, const void *src, size_t size, llaisysMemcpyKind_t kind) {

    if (size == 0) {
        return;
    }

    MUSA_CHECK(musaMemcpy(dst, src, size, musaMemcpyKindFrom(kind)));
}

void memcpyAsync(
    void *dst,
    const void *src,
    size_t size,
    llaisysMemcpyKind_t kind,
    llaisysStream_t stream) {

    if (size == 0) {
        return;
    }

    MUSA_CHECK(
        musaMemcpyAsync(
            dst,
            src,
            size,
            musaMemcpyKindFrom(kind),
            reinterpret_cast<musaStream_t>(
                stream)));
}

static const LlaisysRuntimeAPI RUNTIME_API = {
    &getDeviceCount,
    &setDevice,
    &deviceSynchronize,
    &createStream,
    &destroyStream,
    &streamSynchronize,
    &mallocDevice,
    &freeDevice,
    &mallocHost,
    &freeHost,
    &memcpySync,
    &memcpyAsync};

} // namespace runtime_api

const LlaisysRuntimeAPI *getRuntimeAPI() {
    return &runtime_api::RUNTIME_API;
}
} // namespace llaisys::device::musa
