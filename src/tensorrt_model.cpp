#include "tensorrt_model.hpp"

#include <NvInfer.h>
#include <NvInferRuntime.h>

namespace nv = nvinfer1;

constexpr int kNumWallTypes = 2;
constexpr int kNumDirections = 4;

TensorRTModel::TensorRTModel(nv::ICudaEngine& engine, int batch_size)
    : m_context{engine.createExecutionContext()} {
    m_batch_size = batch_size;

    auto states_dims = engine.getTensorShape("States");

    if (states_dims.nbDims != 4 || states_dims.d[0] != -1) {
        throw std::runtime_error("Invalid input shape for \"States\" tensor!");
    }

    states_dims.d[0] = batch_size;
    m_context->setInputShape("States", states_dims);
    int const columns = states_dims.d[2];
    int const rows = states_dims.d[3];
    m_state_size = states_dims.d[1] * columns * rows;

    auto wall_priors_dims = engine.getTensorShape("WallPriors");

    if (wall_priors_dims.nbDims != 4 || wall_priors_dims.d[0] != -1 ||
        wall_priors_dims.d[1] != kNumWallTypes || wall_priors_dims.d[2] != columns ||
        wall_priors_dims.d[3] != rows) {
        throw std::runtime_error("Invalid input shape for \"WallPriors\" tensor!");
    }

    wall_priors_dims.d[0] = batch_size;
    m_context->setInputShape("WallPriors", wall_priors_dims);
    m_wall_prior_size = kNumWallTypes * columns * rows;

    auto step_priors_dims = engine.getTensorShape("StepPriors");

    if (step_priors_dims.nbDims != 2 || step_priors_dims.d[0] != -1 ||
        step_priors_dims.d[1] != kNumDirections) {
        throw std::runtime_error("Invalid input shape for \"StepPriors\" tensor!");
    }

    step_priors_dims.d[0] = batch_size;
    m_context->setInputShape("StepPriors", step_priors_dims);

    auto values_dims = engine.getTensorShape("Values");

    if (values_dims.nbDims != 1 || values_dims.d[0] != -1) {
        throw std::runtime_error("Invalid input shape for \"Values\" tensor!");
    }

    values_dims.d[0] = batch_size;
    m_context->setInputShape("Values", values_dims);

    m_states = CudaBuffer<float>(m_state_size * m_batch_size);
    m_wall_priors = CudaBuffer<float>(m_wall_prior_size * m_batch_size);
    m_step_priors = CudaBuffer<float>(kNumDirections * m_batch_size);
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