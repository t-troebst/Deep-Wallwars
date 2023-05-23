#pragma once

#include <memory>

#include "cuda_wrappers.hpp"
#include "model.hpp"

namespace nvinfer1 {
class ICudaEngine;
class IExecutionContext;
};  // namespace nvinfer1

class TensorRTModel : public Model {
public:
    TensorRTModel(nvinfer1::ICudaEngine& engine, int batch_size);

    void inference(std::span<float> states, Output const& out) override;

private:
    std::unique_ptr<nvinfer1::IExecutionContext> m_context;

    CudaStream m_stream;
    CudaBuffer<float> m_states;
    CudaBuffer<float> m_wall_priors;
    CudaBuffer<float> m_step_priors;
    CudaBuffer<float> m_values;
};
