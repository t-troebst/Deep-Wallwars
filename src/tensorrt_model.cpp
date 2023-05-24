#include "tensorrt_model.hpp"

#include <NvInfer.h>
#include <NvInferRuntime.h>

#include <iostream>

namespace nv = nvinfer1;

TensorRTModel::TensorRTModel(nv::ICudaEngine& engine, int batch_size)
    : m_context{engine.createExecutionContext()} {
    auto const states_dims = engine.getTensorShape("States");

    if (states_dims.nbDims != 4 || states_dims.d[0] != batch_size) {
        throw std::runtime_error("Invalid input shape for \"States\" tensor!");
    }

    int const columns = states_dims.d[2];
    int const rows = states_dims.d[3];
    m_batch_size = batch_size;
    m_state_size = states_dims.d[1] * columns * rows;
    m_wall_prior_size = 2 * columns * rows;

    auto const wall_priors_dims = engine.getTensorShape("WallPriors");

    if (wall_priors_dims.nbDims != 2 || wall_priors_dims.d[0] != batch_size ||
        wall_priors_dims.d[1] != m_wall_prior_size) {
        throw std::runtime_error("Invalid input shape for \"WallPriors\" tensor!");
    }

    auto const step_priors_dims = engine.getTensorShape("StepPriors");

    if (step_priors_dims.nbDims != 2 || step_priors_dims.d[0] != batch_size ||
        step_priors_dims.d[1] != 4) {
        throw std::runtime_error("Invalid input shape for \"StepPriors\" tensor!");
    }

    auto values_dims = engine.getTensorShape("Values");

    if (values_dims.nbDims != 2 || values_dims.d[0] != batch_size || values_dims.d[1] != 1) {
        throw std::runtime_error("Invalid input shape for \"Values\" tensor!");
    }

    m_states = CudaBuffer<float>(m_state_size * m_batch_size);
    m_wall_priors = CudaBuffer<float>(m_wall_prior_size * m_batch_size);
    m_step_priors = CudaBuffer<float>(4 * m_batch_size);
    m_values = CudaBuffer<float>(m_batch_size);

    m_context->setTensorAddress("States", m_states.device_ptr());
    m_context->setTensorAddress("WallPriors", m_wall_priors.device_ptr());
    m_context->setTensorAddress("StepPriors", m_step_priors.device_ptr());
    m_context->setTensorAddress("Values", m_values.device_ptr());
}

void TensorRTModel::inference(std::span<float> states, Output const& out) {
    m_states.to_device(states, m_stream);
    m_context->enqueueV3(m_stream.get());
    m_wall_priors.to_host(out.wall_priors, m_stream);
    m_step_priors.to_host(out.step_priors, m_stream);
    m_values.to_host(out.values, m_stream);
    m_stream.synchronize();
}

std::unique_ptr<nv::ICudaEngine> load_serialized_engine(nv::IRuntime& runtime,
                                                        std::istream& binary_in) {
    // Yes I should memory map or whatever but this is fine. :)
    std::vector<char> serialized_engine{std::istreambuf_iterator<char>(binary_in),
                                        std::istreambuf_iterator<char>()};

    return std::unique_ptr<nv::ICudaEngine>(
        runtime.deserializeCudaEngine(serialized_engine.data(), serialized_engine.size()));
}
