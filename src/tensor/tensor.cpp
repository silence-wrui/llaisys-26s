#include "tensor.hpp"

#include "../utils.hpp"

#include <cstring>
#include <numeric>
#include <sstream>

namespace llaisys {

Tensor::Tensor(TensorMeta meta, core::storage_t storage, size_t offset)
    : _meta(std::move(meta)), _storage(std::move(storage)), _offset(offset) {}

tensor_t Tensor::create(const std::vector<size_t> &shape,
                        llaisysDataType_t dtype,
                        llaisysDeviceType_t device_type,
                        int device) {

    //计算strides 每一维+1内存跨过的元素
    size_t ndim_ = shape.size();
    std::vector<ptrdiff_t> strides(ndim_);
    size_t stride = 1;
    for (size_t i = 1; i <= ndim_; i++) {
        strides[ndim_ - i] = stride;
        stride *= shape[ndim_ - i];
    }

    TensorMeta meta{dtype, shape, strides};
    size_t total_elems = stride;
    size_t dtype_size = utils::dsize(dtype);

    if (device_type == LLAISYS_DEVICE_CPU && core::context().runtime().deviceType() != LLAISYS_DEVICE_CPU) {
        auto storage = core::context().runtime().allocateHostStorage(total_elems * dtype_size);
        return std::shared_ptr<Tensor>(new Tensor(meta, storage));
    } else {
        core::context().setDevice(device_type, device);
        auto storage = core::context().runtime().allocateDeviceStorage(total_elems * dtype_size);
        return std::shared_ptr<Tensor>(new Tensor(meta, storage));
    }
}

std::byte *Tensor::data() {
    return _storage->memory() + _offset;
}

const std::byte *Tensor::data() const {
    return _storage->memory() + _offset;
}

//shape维度
size_t Tensor::ndim() const {
    return _meta.shape.size();
}

const std::vector<size_t> &Tensor::shape() const {
    return _meta.shape;
}

const std::vector<ptrdiff_t> &Tensor::strides() const {
    return _meta.strides;
}

llaisysDataType_t Tensor::dtype() const {
    return _meta.dtype;
}

//获取设备类型
llaisysDeviceType_t Tensor::deviceType() const {
    return _storage->deviceType();
}

int Tensor::deviceId() const {
    return _storage->deviceId();
}

//计算元素数量
size_t Tensor::numel() const {
    return std::accumulate(_meta.shape.begin(), _meta.shape.end(), size_t(1), std::multiplies<size_t>());
}

//每个元素占多少字节
size_t Tensor::elementSize() const {
    return utils::dsize(_meta.dtype);
}

//这里 numel() * elementSize() 算tensor占用字节数

std::string Tensor::info() const {
    std::stringstream ss;

    ss << "Tensor: "
       << "shape[ ";
    for (auto s : this->shape()) {
        ss << s << " ";
    }
    ss << "] strides[ ";
    for (auto s : this->strides()) {
        ss << s << " ";
    }
    ss << "] dtype=" << this->dtype();

    return ss.str();
}

template <typename T>
void print_data(const T *data, const std::vector<size_t> &shape, const std::vector<ptrdiff_t> &strides, size_t dim) {
    if (dim == shape.size() - 1) {
        for (size_t i = 0; i < shape[dim]; i++) {
            if constexpr (std::is_same_v<T, bf16_t> || std::is_same_v<T, fp16_t>) {
                std::cout << utils::cast<float>(data[i * strides[dim]]) << " ";
            } else {
                std::cout << data[i * strides[dim]] << " ";
            }
        }
        std::cout << std::endl;
    } else if (dim < shape.size() - 1) {
        for (size_t i = 0; i < shape[dim]; i++) {
            print_data(data + i * strides[dim], shape, strides, dim + 1);
        }
    }
}

void debug_print(const std::byte *data, const std::vector<size_t> &shape, const std::vector<ptrdiff_t> &strides, llaisysDataType_t dtype) {
    switch (dtype) {
    case LLAISYS_DTYPE_BYTE:
        return print_data(reinterpret_cast<const char *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_BOOL:
        return print_data(reinterpret_cast<const bool *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_I8:
        return print_data(reinterpret_cast<const int8_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_I16:
        return print_data(reinterpret_cast<const int16_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_I32:
        return print_data(reinterpret_cast<const int32_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_I64:
        return print_data(reinterpret_cast<const int64_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_U8:
        return print_data(reinterpret_cast<const uint8_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_U16:
        return print_data(reinterpret_cast<const uint16_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_U32:
        return print_data(reinterpret_cast<const uint32_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_U64:
        return print_data(reinterpret_cast<const uint64_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_F16:
        return print_data(reinterpret_cast<const fp16_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_F32:
        return print_data(reinterpret_cast<const float *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_F64:
        return print_data(reinterpret_cast<const double *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_BF16:
        return print_data(reinterpret_cast<const bf16_t *>(data), shape, strides, 0);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}

void Tensor::debug() const {
    core::context().setDevice(this->deviceType(), this->deviceId());
    core::context().runtime().api()->device_synchronize();
    std::cout << this->info() << std::endl;
    if (this->deviceType() == LLAISYS_DEVICE_CPU) {
        debug_print(this->data(), this->shape(), this->strides(), this->dtype());
    } else {
        auto tmp_tensor = create({this->_storage->size()}, this->dtype());
        core::context().runtime().api()->memcpy_sync(
            tmp_tensor->data(),
            this->data(),
            this->numel() * this->elementSize(),
            LLAISYS_MEMCPY_D2H);
        debug_print(tmp_tensor->data(), this->shape(), this->strides(), this->dtype());
    }
}

//作业1.2 Contiguous连续性
//_offset不影响连续性
bool Tensor::isContiguous() const {
    //空tensor直接返回true
    if (numel() == 0) {
        return true;
    }

    //最后一维步长味为1
    ptrdiff_t expected_stride = 1;

    //size_t无符号整数 i从0-1会得到一个极大的正数 死循环
    for (size_t i = ndim(); i-- > 0;) {
        //维度为1 不影响元素是否紧密排列
        if (_meta.shape[i] == 1) {
            continue;
        }

        if (_meta.strides[i] != expected_stride) {
            return false;
        }

        expected_stride *= static_cast<ptrdiff_t>(_meta.shape[i]);
    }

    return true;
}

tensor_t Tensor::permute(const std::vector<size_t> &order) const {
    TO_BE_IMPLEMENTED();
    return std::shared_ptr<Tensor>(new Tensor(_meta, _storage));
}

//作业1.3 仅支持连续tensor
tensor_t Tensor::view(const std::vector<size_t> &shape) const {
    //新shape元素的个数
    size_t new_numel = std::accumulate(shape.begin(), shape.end(), size_t(1), std::multiplies<size_t>());

    //新元素shape个数=原shape元素个数
    CHECK_ARGUMENT(new_numel == numel(), "View shape must have the same number of elements!");

    CHECK_ARGUMENT(isContiguous(), "This simple version only supports contiguous tensors");

    std::vector<ptrdiff_t> new_strides(shape.size());

    ptrdiff_t stride = 1;
    for(size_t i = shape.size(); i-- > 0;) {
        new_strides[i] = stride;
        stride *= static_cast<ptrdiff_t>(shape[i]);
    }

    TensorMeta new_meta{_meta.dtype, shape, new_strides};
    
    return std::shared_ptr<Tensor>(new Tensor(new_meta, _storage, _offset));
}

tensor_t Tensor::slice(size_t dim, size_t start, size_t end) const {
    TO_BE_IMPLEMENTED();
    return std::shared_ptr<Tensor>(new Tensor(_meta, _storage));
}

//作业1.1
//src_是CPU内存地址，目标tensor可能在CPU/GPU
void Tensor::load(const void *src_) {
    //空tensor不复制：如果元素数量是0，不访问src_
    if (numel() == 0) {
        return;
    }

    //检查空指针：如果源地址空，则复制造成非法访问
    CHECK_ARGUMENT(src_ != nullptr, "Source pointer should not be null!!!!");

    //切换到tensor所在设备
    core::context().setDevice(deviceType(), deviceId());

    //选择复制方向 to cpu/gpu
    auto kind = (deviceType() == LLAISYS_DEVICE_CPU
                ? LLAISYS_MEMCPY_H2H : LLAISYS_MEMCPY_H2D);
                
    //调用同步复制
    core::context().runtime().api()->memcpy_sync(data(), src_, numel()*elementSize(), kind);
}

tensor_t Tensor::contiguous() const {
    TO_BE_IMPLEMENTED();
    return std::shared_ptr<Tensor>(new Tensor(_meta, _storage));
}

tensor_t Tensor::reshape(const std::vector<size_t> &shape) const {
    TO_BE_IMPLEMENTED();
    return std::shared_ptr<Tensor>(new Tensor(_meta, _storage));
}

tensor_t Tensor::to(llaisysDeviceType_t device_type, int device) const {
    TO_BE_IMPLEMENTED();
    return std::shared_ptr<Tensor>(new Tensor(_meta, _storage));
}

} // namespace llaisys
