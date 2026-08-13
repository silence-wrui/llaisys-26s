#include "error.hpp"

#include "llaisys.h"

#include <cstddef>
#include <cstdio>

namespace {

constexpr std::size_t ERROR_BUFFER_SIZE = 2048;
thread_local char error_buffer[ERROR_BUFFER_SIZE] = {};

} // namespace

namespace llaisys::api {

void clearLastError() noexcept {
    error_buffer[0] = '\0';
}

void setLastError(const char *message) noexcept {
    if (message == nullptr) {
        message = "Unknown error";
    }

    std::snprintf(error_buffer, ERROR_BUFFER_SIZE, "%s", message);
}

} // namespace llaisys::api

__C const char *llaisysGetLastError(void) {
    return error_buffer[0] == '\0' ? nullptr : error_buffer;
}

__C void llaisysClearLastError(void) {
    llaisys::api::clearLastError();
}