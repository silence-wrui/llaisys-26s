#pragma once

#include "error.hpp"

#include <exception>
#include <utility>

namespace llaisys::api {

template <typename Function>
void guardVoid(Function &&function) noexcept {
    clearLastError();

    try {
        std::forward<Function>(function)();
    } catch (const std::exception &error) {
        setLastError(error.what());
    } catch (...) {
        setLastError("Unknown C++ exception");
    }
}

} // namespace llaisys::api