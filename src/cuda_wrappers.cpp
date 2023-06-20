#include "cuda_wrappers.hpp"

void cuda_check(cudaError_t err) {
    if (err != cudaSuccess) {
        throw CudaException(cudaGetErrorString(err));
    }
}

CudaStream::CudaStream() {
    cudaStream_t stream;
    cuda_check(cudaStreamCreate(&stream));
    m_stream = stream;
}

CudaStream::CudaStream(CudaStream&& other) noexcept : m_stream{other.m_stream} {
    other.m_stream.reset();
}

CudaStream& CudaStream::operator=(CudaStream&& other) noexcept {
    if (m_stream) {
        synchronize();
        cuda_check(cudaStreamDestroy(*m_stream));
    }

    m_stream = other.m_stream;
    other.m_stream.reset();

    return *this;
}

CudaStream::~CudaStream() {
    synchronize();
    cuda_check(cudaStreamDestroy(*m_stream));
}

cudaStream_t CudaStream::get() {
    return *m_stream;
}

void CudaStream::synchronize() {
    cuda_check(cudaStreamSynchronize(*m_stream));
}
