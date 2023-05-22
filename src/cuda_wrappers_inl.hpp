#pragma once

#include <cuda_runtime.h>

#include <algorithm>
#include <ranges>

#include "cuda_wrappers.hpp"

template <typename T>
CudaBuffer<T>::CudaBuffer(std::size_t size) : m_size{size}, m_host_ptr{new T[size]} {
    cudaMalloc(&m_device_ptr, size);
}

template <typename T>
CudaBuffer<T>::CudaBuffer(CudaBuffer&& other) noexcept
    : m_device_ptr{other.m_device_address}, m_host_ptr{other.m_host_address}, m_size{other.m_size} {
    other.m_device_ptr = nullptr;
    other.m_host_ptr = nullptr;
    other.m_size = 0;
}

template <typename T>
CudaBuffer<T>::CudaBuffer(CudaBuffer const& other)
    : m_host_ptr{new T[other.m_size]}, m_size{other.m_size} {
    cudaMalloc(&m_device_ptr, m_size);
    cudaMemcpy(&m_device_ptr, other.m_device_ptr, m_size, cudaMemcpyDeviceToDevice);
    std::ranges::copy_n(other.m_host_ptr, m_size, m_host_ptr);
}

template <typename T>
CudaBuffer<T>& CudaBuffer<T>::operator=(CudaBuffer&& other) noexcept {
    cudaFree(m_device_ptr);
    delete[] m_host_ptr;

    m_device_ptr = other.m_device_ptr;
    m_host_ptr = other.m_host_ptr;
    m_size = other.m_size;

    other.m_device_ptr = nullptr;
    other.m_host_ptr = nullptr;
    other.m_size = 0;

    return *this;
}

template <typename T>
CudaBuffer<T>::~CudaBuffer() {
    cudaFree(m_device_ptr);
    delete[] m_host_ptr;
}

template <typename T>
CudaBuffer<T>& CudaBuffer<T>::operator=(CudaBuffer const& other) {
    cudaFree(m_device_ptr);
    delete[] m_host_ptr;

    cudaMalloc(m_device_ptr, other.m_size);
    m_host_ptr = new T[other.m_size];
    m_size = other.m_size;

    cudaMemcpy(&m_device_ptr, other.m_device_ptr, m_size, cudaMemcpyDeviceToDevice);
    std::ranges::copy_n(other.m_host_ptr, m_size, m_host_ptr);

    return *this;
}

template <typename T>
T* CudaBuffer<T>::device_ptr() const {
    return m_device_ptr;
}

template <typename T>
T* CudaBuffer<T>::host_ptr() const {
    return m_host_ptr;
}

template <typename T>
std::size_t CudaBuffer<T>::size() const {
    return m_size;
}

template <typename T>
CudaBuffer<T>::operator std::span<T>() {
    return {m_host_ptr, m_size};
}

template <typename T>
T* CudaBuffer<T>::begin() {
    return m_host_ptr;
}

template <typename T>
T* CudaBuffer<T>::end() {
    return m_host_ptr + m_size;
}

template <typename T>
T const* CudaBuffer<T>::begin() const {
    return m_host_ptr;
}

template <typename T>
T const* CudaBuffer<T>::end() const {
    return m_host_ptr + m_size;
}

template <typename T>
T& CudaBuffer<T>::operator[](std::size_t i) {
    return m_host_ptr[i];
}

template <typename T>
T const& CudaBuffer<T>::operator[](std::size_t i) const {
    return m_host_ptr[i];
}

template <typename T>
void CudaBuffer<T>::to_device(CudaStream& stream) {
    cudaMemcpyAsync(m_device_ptr, m_host_ptr, m_size, cudaMemcpyHostToDevice, stream.get());
}

template <typename T>
void CudaBuffer<T>::to_host(CudaStream& stream) {
    cudaMemcpyAsync(m_host_ptr, m_device_ptr, m_size, cudaMemcpyDeviceToHost, stream.get());
}
