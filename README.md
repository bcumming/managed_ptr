# A Smart Pointer for CUDA Managed Memory

Simplify the use of managed memory, with well defined lifetime management equivalent to `std::unique_ptr`.

## Rationale

An obvious question is "why implement the `managed_ptr` when you could just use `std::unique_ptr`?".
A solution based on `std::unique_ptr` would look something like

```C++
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

// ...

__global__ kernel(int& x) { x = 4; }

// allocate and set initial value for an int in managed memory on the host
auto ptr = make_managed_ptr<int>)(3);
assert(*ptr == 3);       // access the value from the host
kernel<<<1, 1>>>(*ptr);  // update the value on the device
cudaDeviceSynchronize(); // wait for kernel to finish execution
assert(*ptr == 4);       // access the value from the host

```

This implementation defines a `managed_ptr` to be a `std::unique_ptr` with a custom type for freeing managed memory, i.e. memory allocated using `cudaMallocManaged()`.
A convenience function `make_managed_ptr` simplifies the fiddly bussiness of allocating managed memory and constructing a type in place in the newly allocated memory.


