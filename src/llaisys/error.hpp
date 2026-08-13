#pragma once

namespace llaisys::api {
void clearLastError() noexcept;
void setLastError(const char *message) noexcept;
} // namespace llaisys::api