#include "musa_resource.muh"

#include <musa_runtime.h>

#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace {
void checkMusa(
    musaError_t status,
    const char *operation) {
    if (status != musaSuccess) {
        throw std::runtime_error(std::string(operation)
            + " failed: "
            + musaGetErrorString(status));
    }
}

void checkMublas(mublasStatus_t status, const char *operation) {

    if (status != MUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error(std::string(operation)
            + " failed: "
            + std::to_string(static_cast<int>(status)));
    }
}

} // namespace

namespace llaisys::device::musa {
Resource::Resource(int device_id) : llaisys::device::DeviceResource(
          LLAISYS_DEVICE_MUSA,
          device_id),
      _mublas_handle(nullptr) {

    checkMusa(musaSetDevice(device_id), "musaSetDevice");

    checkMublas(mublasCreate(&_mublas_handle), "mublasCreate");

    checkMublas(
        mublasSetPointerMode(
            _mublas_handle,
            MUBLAS_POINTER_MODE_HOST),
        "mublasSetPointerMode");
}

Resource::~Resource() {
    if (_mublas_handle != nullptr) {
        // 析构函数不能抛异常。
        musaSetDevice(getDeviceId());
        mublasDestroy(_mublas_handle);
        _mublas_handle = nullptr;
    }
}

mublasHandle_t Resource::mublasHandle() const {
    return _mublas_handle;
}

Resource &getResource(int device_id) {
    thread_local std::unordered_map<int, std::unique_ptr<Resource>> resources;

    auto iterator = resources.find(device_id);

    if (iterator == resources.end()) {
        iterator = resources.emplace(device_id, std::make_unique<Resource>(device_id)).first;
    }

    return *iterator->second;
}
} // namespace llaisys::device::musa