#pragma once

#include <iosfwd>
#include <memory>

#include "cuda_wrappers.hpp"
#include "model.hpp"

namespace nvinfer1 {
class IRuntime;
class ICudaEngine;
class IExecutionContext;
};  // namespace nvinfer1

class TensorRTModel : public Model {
public:
    TensorRTModel(std::shared_ptr<nvinfer1::ICudaEngine> engine);

    void inference(std::span<float> states, Output const& out) override;

private:
    std::shared_ptr<nvinfer1::ICudaEngine> m_engine;  // Keep engine alive
    std::unique_ptr<nvinfer1::IExecutionContext> m_context;

    CudaStream m_stream;
    CudaBuffer<float> m_states;
    CudaBuffer<float> m_priors;
    CudaBuffer<float> m_values;
};

std::shared_ptr<nvinfer1::ICudaEngine> load_serialized_engine(nvinfer1::IRuntime& runtime,
                                                              std::istream& binary_in);
