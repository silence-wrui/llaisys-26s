#include "nvidia_resource.cuh"

#include <cuda_runtime.h>

#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace {
void checkCuda(
    cudaError_t status,
    const char *operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(operation)
            + " failed: "
            + cudaGetErrorString(status));
    }
}

void checkCublas(cublasStatus_t status, const char *operation) {

    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error(std::string(operation)
            + " failed: "
            + cublasGetStatusString(status));
    }
}

} // namespace

namespace llaisys::device::nvidia {
Resource::Resource(int device_id) : llaisys::device::DeviceResource(
          LLAISYS_DEVICE_NVIDIA,
          device_id),
      _cublas_handle(nullptr) {

    checkCuda(cudaSetDevice(device_id), "cudaSetDevice");

    checkCublas(cublasCreate(&_cublas_handle), "cublasCreate");

    checkCublas(
        cublasSetPointerMode(
            _cublas_handle,
            CUBLAS_POINTER_MODE_HOST),
        "cublasSetPointerMode");
}

Resource::~Resource() {
    if (_cublas_handle != nullptr) {
        // 析构函数不能抛异常。
        cudaSetDevice(getDeviceId());
        cublasDestroy(_cublas_handle);
        _cublas_handle = nullptr;
    }
}

cublasHandle_t Resource::cublasHandle() const {
    return _cublas_handle;
}

Resource &getResource(int device_id) {
    thread_local std::unordered_map<int, std::unique_ptr<Resource>> resources;

    auto iterator = resources.find(device_id);

    if (iterator == resources.end()) {
        iterator = resources.emplace(device_id, std::make_unique<Resource>(device_id)).first;
    }

    return *iterator->second;
}
} // namespace llaisys::device::nvidia