#pragma once

#include "../device_resource.hpp"

#include <cublas_v2.h>

namespace llaisys::device::nvidia {
class Resource : public llaisys::device::DeviceResource {
private:
    cublasHandle_t _cublas_handle;
public:
    explicit Resource(int device_id);
    ~Resource();
    cublasHandle_t cublasHandle() const;
};
Resource &getResource(int device_id);
} // namespace llaisys::device::nvidia