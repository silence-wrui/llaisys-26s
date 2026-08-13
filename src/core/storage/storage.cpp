#include "storage.hpp"

#include "../runtime/runtime.hpp"

#include <cstdio>
#include <exception>

namespace llaisys::core {
Storage::Storage(std::byte *memory, size_t size, Runtime &runtime, bool is_host)
    : _memory(memory), _size(size), _runtime(runtime), _is_host(is_host) {}

Storage::~Storage() noexcept {
    try {
        _runtime.freeStorage(this);
    } catch (const std::exception &error) {
        std::fprintf(
            stderr,
            "[ERROR] Failed to release storage: %s\n",
            error.what());
    } catch (...) {
        std::fprintf(
            stderr,
            "[ERROR] Failed to release storage: unknown error\n");
    }
}

std::byte *Storage::memory() const {
    return _memory;
}

size_t Storage::size() const {
    return _size;
}

llaisysDeviceType_t Storage::deviceType() const {
    if (isHost()) {
        return LLAISYS_DEVICE_CPU;
    } else {
        return _runtime.deviceType();
    }
}

int Storage::deviceId() const {
    if (isHost()) {
        return 0;
    } else {
        return _runtime.deviceId();
    }
}

bool Storage::isHost() const {
    return _is_host;
}
} // namespace llaisys::core