# A Smart Pointer for CUDA Managed Memory

Simplify the use of managed memory, with well defined lifetime management equivalent to `std::unique_ptr`.

## Rationale

An obvious question is "why implement the `managed_ptr` when you could just use `std::unique_ptr`?".
A solution based on `std::unique_ptr` would look something like

```C++
// tested with cuda 8 on gtx 1070
//      nvcc -arch=sm_60 -std=c++11

#include <cassert>
#include <memory>

template <typename T>
struct managed_deletor {
    void operator()(T* p) const {
        p->~T();
        cudaFree(p);
    }
};

template <typename T>
using managed_ptr = std::unique_ptr<T, managed_deletor<T>>;

template <typename T, typename... Args>
managed_ptr<T> make_managed_ptr(Args&&... args) {
    T* ptr;
    cudaMallocManaged(&ptr, sizeof(T));
    cudaDeviceSynchronize();
    ptr = new (ptr) T(std::forward<Args>(args)...);
    return managed_ptr<T>(ptr);
}

__global__ void kernel(int& x) { x = 4; }

int main(void) {
    // allocate and set initial value for an int in managed memory on the host
    auto ptr = make_managed_ptr<int>(3);
    assert(*ptr == 3);       // access the value from the host
    kernel<<<1, 1>>>(*ptr);  // update the value on the device
    cudaDeviceSynchronize(); // wait for kernel to finish execution
    assert(*ptr == 4);       // access the value from the host
}
```

This implementation defines a `managed_ptr` as a `std::unique_ptr` with a custom deleter for managed memory.
A convenience function `make_managed_ptr` simplifies the fiddly bussiness of allocating managed memory and constructing a type in place in the newly allocated memory.

The implementation is adequate in many situations, particularly when working with trivially constructable types.
The main motivation behind a full implementation of `managed_ptr` is to simplify "deep copy", which has been a challenge with CUDA programming in the past.

`managed_ptr` provides both `__host__` and `__device__` interfaces that make `managed_ptr` useful for creating and using nested hierarchies of objects on both host and device code.

## Usage

### Creating managed pointers

The `managed_ptr` is designed to be constructed using the `make_managed_ptr` helper template funtion:

```C++
// int initialized to 3
managed_ptr<int> p1 = make_managed_ptr<int>(3);

// the default constructor is called with an empty argument list
// in this case, *p2==0 after construction, because int defaults to zero
managed_ptr<int> p2 = make_managed_ptr<int>();
```

The operations and semantics that apply to `std::unique_ptr` also apply to `managed_ptr`.
For example, `managed_ptr` practices exclusive ownership of memory, so that copy construction and copy are not permitted.

### Using managed memory with CUDA kernels

```C++
managed_ptr<int> p = make_managed_ptr<int>();

// Dereference of manged pointer passes reference to underlying storage,
// which is visible in device code.
// kernel has prototype : __global__ kernel(int&);
kernel<<<1, 1>>>(*p);
```

### Host-Device synchronization

Working with CUDA managed memory requires additional care to ensure that memory accesses between host and device code are synchronized.
Specifically, CUDA dictates that host code should not access managed memory that is in use on the GPU.

The `managed_ptr::synchronize()` will return once all currently running kernels and memory transfers on the device have finished.
This is a blocking operation, so it is important to realize that a call to `synchronize()` could block execution in the calling host thread for a significant amount of time.
Currently it is a simple wrapper around `cudaDeviceSynchronize`, which is quite a heavy handed.
A possible improvement would be to optionally associate the `managed_ptr` with a CUDA stream, so that synchronization would be on a singled stream, and not across the entire device.

```C++
managed_ptr<int> p = make_managed_ptr<int>();

// kernel has prototype : __global__ kernel(int&);
kernel<<<1, 1>>>(*p);

// wait for kernel to finish before using the (potentially updated) value in p
p.synchronize();
int i = *p;
```

