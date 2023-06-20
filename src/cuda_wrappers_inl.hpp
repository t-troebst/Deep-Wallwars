#pragma once

#include <cuda_runtime.h>

#include <algorithm>
#include <exception>
#include <ranges>

#include "cuda_wrappers.hpp"

template <typename T>
CudaBuffer<T>::CudaBuffer() : m_device_ptr{nullptr}, m_size{0} {}

template <typename T>
CudaBuffer<T>::CudaBuffer(std::size_t size) : m_size{size} {
    cuda_check(cudaMalloc(&m_device_ptr, size));
}

template <typename T>
CudaBuffer<T>::CudaBuffer(CudaBuffer&& other) noexcept
    : m_device_ptr{other.m_device_address}, m_size{other.m_size} {
    other.m_device_ptr = nullptr;
    other.m_size = 0;
}

template <typename T>
CudaBuffer<T>::CudaBuffer(CudaBuffer const& other) : m_size{other.m_size} {
    cuda_check(cudaMalloc(&m_device_ptr, m_size));
    cuda_check(cudaMemcpy(&m_device_ptr, other.m_device_ptr, m_size, cudaMemcpyDeviceToDevice));
}

template <typename T>
CudaBuffer<T>& CudaBuffer<T>::operator=(CudaBuffer&& other) noexcept {
    cuda_check(cudaFree(m_device_ptr));

    m_device_ptr = other.m_device_ptr;
    m_size = other.m_size;

    other.m_device_ptr = nullptr;
    other.m_size = 0;

    return *this;
}

template <typename T>
CudaBuffer<T>::~CudaBuffer() {
    cuda_check(cudaFree(m_device_ptr));
}

template <typename T>
CudaBuffer<T>& CudaBuffer<T>::operator=(CudaBuffer const& other) {
    cuda_check(cudaFree(m_device_ptr));

    cuda_check(cudaMalloc(m_device_ptr, other.m_size));
    m_size = other.m_size;

    cuda_check(cudaMemcpy(&m_device_ptr, other.m_device_ptr, m_size, cudaMemcpyDeviceToDevice));

    return *this;
}

template <typename T>
T* CudaBuffer<T>::device_ptr() const {
    return m_device_ptr;
}

template <typename T>
std::size_t CudaBuffer<T>::size() const {
    return m_size;
}

template <typename T>
void CudaBuffer<T>::to_device(std::span<T> in, CudaStream& stream) {
    if (in.size() > m_size) {
        throw std::runtime_error("Cannot upload buffer to device - too large!");
    }

    cuda_check(
        cudaMemcpyAsync(m_device_ptr, in.data(), in.size(), cudaMemcpyHostToDevice, stream.get()));
}

template <typename T>
void CudaBuffer<T>::to_host(std::span<T> out, CudaStream& stream) {
    if (m_size > out.size()) {
        throw std::runtime_error("Cannot download buffer to host - too large!");
    }

    cuda_check(
        cudaMemcpyAsync(out.data(), m_device_ptr, m_size, cudaMemcpyDeviceToHost, stream.get()));
}
