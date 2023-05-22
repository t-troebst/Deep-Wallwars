#include "cuda_wrappers.hpp"

CudaStream::CudaStream() {
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    m_stream = stream;
}

CudaStream::CudaStream(CudaStream&& other) noexcept : m_stream{other.m_stream} {
    other.m_stream.reset();
}

CudaStream& CudaStream::operator=(CudaStream&& other) noexcept {
    if (m_stream) {
        synchronize();
        cudaStreamDestroy(*m_stream);
    }

    m_stream = other.m_stream;
    other.m_stream.reset();

    return *this;
}

CudaStream::~CudaStream() {
    synchronize();
    cudaStreamDestroy(*m_stream);
}

cudaStream_t CudaStream::get() {
    return *m_stream;
}

void CudaStream::synchronize() {
    cudaStreamSynchronize(*m_stream);
}
