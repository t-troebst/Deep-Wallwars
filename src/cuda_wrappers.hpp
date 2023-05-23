#pragma once

#include <cuda_runtime.h>

#include <optional>
#include <span>

class CudaStream {
public:
    CudaStream();

    CudaStream(CudaStream const& other) = delete;
    CudaStream(CudaStream&& other) noexcept;

    CudaStream& operator=(CudaStream const& other) = delete;
    CudaStream& operator=(CudaStream&& other) noexcept;

    ~CudaStream();

    cudaStream_t get();
    void synchronize();

private:
    std::optional<cudaStream_t> m_stream;
};

template <typename T>
class CudaBuffer {
public:
    CudaBuffer();
    CudaBuffer(std::size_t size);

    CudaBuffer(CudaBuffer const& other);
    CudaBuffer(CudaBuffer&& other) noexcept;

    CudaBuffer& operator=(CudaBuffer const& other);
    CudaBuffer& operator=(CudaBuffer&& other) noexcept;

    ~CudaBuffer();

    T* device_ptr() const;
    std::size_t size() const;

    void to_device(std::span<T> in, CudaStream& stream);
    void to_host(std::span<T> out, CudaStream& stream);
    
private:
    T* m_device_ptr;
    std::size_t m_size;
};

#include "cuda_wrappers_inl.hpp"
