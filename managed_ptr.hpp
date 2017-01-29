# pragma once

#include <memory>
#include <new>
#include <cstdlib>

#include <cuda.h>
#include <cuda_runtime.h>


// TODO
//  - check cuda errors from cudaMalloc etc.
//  - support for arbitrary cuda streams
//  - assertions or execeptions for erroneous actions like dereferencing nullptr

// used to indicate that the type pointed to by the managed_ptr is to be
// constructed in the managed_ptr constructor
struct construct_in_place_tag {};

//
// manged_ptr
//
// Like  std::unique_ptr, but for CUDA managed memory.
template <typename T>
class managed_ptr {
    public:

    using element_type = T;
    using pointer = element_type*;
    using reference = element_type&;

    managed_ptr() = default;

    managed_ptr(const managed_ptr& other) = delete;

    template <typename... Args>
    managed_ptr(construct_in_place_tag, Args&&... args) {
        auto success = cudaMallocManaged(&data_, sizeof(element_type));
        synchronize();
        data_ = new (data_) element_type(std::forward<Args>(args)...);
    }

    managed_ptr(managed_ptr&& other) {
        std::swap(other.data_, data_);
    }

    __host__ __device__
    pointer get() const {
        return data_;
    }

    __host__ __device__
    reference operator *() const {
        return *data_;
    }

    __host__ __device__
    pointer operator->() const {
        return get();
    }

    managed_ptr& operator=(managed_ptr&& other) {
        swap(std::move(other));
        return *this;
    }

    ~managed_ptr() {
        if (is_allocated()) {
            synchronize(); // required to ensure that memory is not in use on GPU
            data_->~element_type();
            cudaFree(data_);
        }
    }

    void swap(managed_ptr&& other) {
        std::swap(other.data_, data_);
    }

    __host__ __device__
    operator bool() const {
        return is_allocated();
    }

    void synchronize() const {
        cudaDeviceSynchronize();
    }

    private:

    __host__ __device__
    bool is_allocated() const {
        return data_!=nullptr;
    }

    pointer data_ = nullptr;
};

// The correct way to construct a type in managed memory.
// Equivalent to std::make_unique_ptr for std::unique_ptr
template <typename T, typename... Args>
managed_ptr<T> make_managed_ptr(Args&&... args) {
    return managed_ptr<T>(construct_in_place_tag(), std::forward<Args>(args)...);
}

