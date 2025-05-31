#include "tensorrt_model.hpp"

#include <NvInfer.h>
#include <NvInferRuntime.h>
#include <folly/logging/xlog.h>

#include <iostream>
#include <vector>

namespace nv = nvinfer1;

TensorRTModel::TensorRTModel(std::shared_ptr<nv::ICudaEngine> engine)
    : m_engine(std::move(engine)), m_context{m_engine->createExecutionContext()} {
    auto const states_dims = m_engine->getTensorShape("States");

    if (states_dims.nbDims != 4) {
        throw std::runtime_error("Invalid input shape for \"States\" tensor!");
    }

    int const columns = states_dims.d[2];
    int const rows = states_dims.d[3];
    m_batch_size = states_dims.d[0];
    m_state_size = states_dims.d[1] * columns * rows;
    m_wall_prior_size = 2 * columns * rows;

    auto const wall_priors_dims = m_engine->getTensorShape("Priors");

    if (wall_priors_dims.nbDims != 2 || wall_priors_dims.d[0] != m_batch_size ||
        wall_priors_dims.d[1] != m_wall_prior_size + 4) {
        throw std::runtime_error("Invalid input shape for \"Priors\" tensor!");
    }

    auto values_dims = m_engine->getTensorShape("Values");

    if (values_dims.nbDims != 2 || values_dims.d[0] != m_batch_size || values_dims.d[1] != 1) {
        throw std::runtime_error("Invalid input shape for \"Values\" tensor!");
    }

    m_states = CudaBuffer<float>(m_state_size * m_batch_size);
    m_priors = CudaBuffer<float>((m_wall_prior_size + 4) * m_batch_size);
    m_values = CudaBuffer<float>(m_batch_size);

    m_context->setTensorAddress("States", m_states.device_ptr());
    m_context->setTensorAddress("Priors", m_priors.device_ptr());
    m_context->setTensorAddress("Values", m_values.device_ptr());
}

void TensorRTModel::inference(std::span<float> states, Output const& out) {
    m_states.to_device(states, m_stream);
    m_context->enqueueV3(m_stream.get());
    m_priors.to_host(out.priors, m_stream);
    m_values.to_host(out.values, m_stream);
    m_stream.synchronize();
}

std::shared_ptr<nv::ICudaEngine> load_serialized_engine(nv::IRuntime& runtime,
                                                        std::istream& binary_in) {
    // Fetch file size first to initialize the vector with the correct size
    binary_in.seekg(0, std::ios::end);
    auto size = binary_in.tellg();
    binary_in.seekg(0, std::ios::beg);

    std::vector<char> serialized_engine(size);
    binary_in.read(serialized_engine.data(), size);

    XLOGF(INFO, "Loaded engine size: {} MiB", size / 1024.0 / 1024.0);

    auto* raw_engine = runtime.deserializeCudaEngine(serialized_engine.data(), size);
    if (!raw_engine) {
        throw std::runtime_error("Failed to deserialize CUDA engine - this usually indicates CUDA is out of memory or the model file is corrupted");
    }

    return std::shared_ptr<nv::ICudaEngine>(raw_engine);
}
