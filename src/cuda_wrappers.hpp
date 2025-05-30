#pragma once

#include <cuda_runtime.h>

#include <optional>
#include <span>
#include <stdexcept>

class CudaException : public std::runtime_error {
    using std::runtime_error::runtime_error;
};

void cuda_check(cudaError_t err);

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
class PinnedBuffer {
public:
    PinnedBuffer(std::size_t size) : m_size{size} {
        cuda_check(cudaMallocHost(&m_data, size * sizeof(T)));
        std::fill(m_data, m_data + size, T{});
    }

    ~PinnedBuffer() {
        cuda_check(cudaFreeHost(m_data));
    }

    PinnedBuffer(PinnedBuffer const& other) : m_size{other.m_size} {
        cuda_check(cudaMallocHost(&m_data, m_size * sizeof(T)));
        std::copy(other.m_data, other.m_data + m_size, m_data);
    }

    PinnedBuffer(PinnedBuffer&& other) noexcept : PinnedBuffer() {
        swap(*this, other);
    }

    friend void swap(PinnedBuffer& lhs, PinnedBuffer& rhs) noexcept {
        std::swap(lhs.m_data, rhs.m_data);
        std::swap(lhs.m_size, rhs.m_size);
    }

    PinnedBuffer& operator=(PinnedBuffer other) noexcept {
        swap(*this, other);
        return *this;
    }

    T& operator[](std::size_t idx) {
        return m_data[idx];
    }

    T const& operator[](std::size_t idx) const {
        return m_data[idx];
    }

    T* data() const {
        return m_data;
    }

    std::size_t size() const {
        return m_size;
    }

    operator std::span<T>() const {
        return std::span<T>(m_data, m_size);
    }

private:
    T* m_data;
    std::size_t m_size;
};

template <typename T>
class CudaBuffer {
public:
    CudaBuffer() : m_device_ptr{nullptr}, m_size{0} {}

    CudaBuffer(std::size_t size) : m_size{size} {
        cuda_check(cudaMalloc(&m_device_ptr, size * sizeof(T)));
        cuda_check(cudaMemset(m_device_ptr, 0, size * sizeof(T)));
    }

    CudaBuffer(CudaBuffer const& other) : m_size{other.m_size} {
        cuda_check(cudaMalloc(&m_device_ptr, m_size * sizeof(T)));
        cuda_check(cudaMemcpy(&m_device_ptr, other.m_device_ptr, m_size * sizeof(T),
                              cudaMemcpyDeviceToDevice));
    }

    CudaBuffer(CudaBuffer&& other) noexcept : CudaBuffer() {
        swap(*this, other);
    }

    friend void swap(CudaBuffer& lhs, CudaBuffer& rhs) noexcept {
        std::swap(lhs.m_device_ptr, rhs.m_device_ptr);
        std::swap(lhs.m_size, rhs.m_size);
    }

    CudaBuffer& operator=(CudaBuffer other) noexcept {
        swap(*this, other);
        return *this;
    }

    ~CudaBuffer() {
        cuda_check(cudaFree(m_device_ptr));
    }

    T* device_ptr() const {
        return m_device_ptr;
    }

    std::size_t size() const {
        return m_size;
    }

    void to_device(std::span<T> in, CudaStream& stream) {
        if (in.size() > m_size) {
            throw std::runtime_error("Cannot upload buffer to device - too large!");
        }

        cuda_check(cudaMemcpyAsync(m_device_ptr, in.data(), in.size() * sizeof(T),
                                   cudaMemcpyHostToDevice, stream.get()));
    }

    void to_host(std::span<T> out, CudaStream& stream) {
        if (m_size > out.size()) {
            throw std::runtime_error("Cannot download buffer to host - too large!");
        }

        cuda_check(cudaMemcpyAsync(out.data(), m_device_ptr, m_size * sizeof(T),
                                   cudaMemcpyDeviceToHost, stream.get()));
    }

private:
    T* m_device_ptr;
    std::size_t m_size;
};
