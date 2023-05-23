#include "batched_model.hpp"

#include <NvInfer.h>
#include <NvInferRuntime.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <ranges>

namespace nv = nvinfer1;

constexpr int kNumWallTypes = 2;
constexpr int kNumDirections = 4;
constexpr int kDefaultBatchesInQueue = 16;

BatchedModel::BatchedModel(nv::ICudaEngine& engine, int batch_size)
    : BatchedModel(engine, batch_size, kDefaultBatchesInQueue * batch_size) {}

BatchedModel::BatchedModel(nv::ICudaEngine& engine, int batch_size, int queue_size)
    : m_batch_size{batch_size}, m_context{engine.createExecutionContext()}, m_tasks(queue_size) {
    // TODO: validate model
    auto const states_dims = engine.getTensorShape("States");

    if (states_dims.nbDims != 4 || states_dims.d[0] != -1) {
        throw std::runtime_error("Invalid input shape for \"States\" tensor!");
    }

    int const columns = states_dims.d[2];
    int const rows = states_dims.d[3];
    m_state_size = states_dims.d[1] * columns * rows;

    auto const wall_priors_dims = engine.getTensorShape("WallPriors");

    if (wall_priors_dims.nbDims != 4 || wall_priors_dims.d[0] != -1 ||
        wall_priors_dims.d[1] != kNumWallTypes || wall_priors_dims.d[2] != columns ||
        wall_priors_dims.d[3] != rows) {
        throw std::runtime_error("Invalid input shape for \"WallPriors\" tensor!");
    }

    m_wall_prior_size = kNumWallTypes * columns * rows;

    auto const step_priors_dims = engine.getTensorShape("StepPriors");

    if (step_priors_dims.nbDims != 2 || step_priors_dims.d[0] != -1 ||
        step_priors_dims.d[1] != kNumDirections) {
        throw std::runtime_error("Invalid input shape for \"StepPriors\" tensor!");
    }

    auto const values_dims = engine.getTensorShape("Values");

    if (values_dims.nbDims != 1 || values_dims.d[0] != -1) {
        throw std::runtime_error("Invalid input shape for \"Values\" tensor!");
    }

    m_states = CudaBuffer<float>(m_state_size * m_batch_size);
    m_wall_priors = CudaBuffer<float>(m_wall_prior_size * m_batch_size);
    m_step_priors = CudaBuffer<float>(kNumDirections * m_batch_size);
    m_values = CudaBuffer<float>(m_batch_size);

    m_context->setTensorAddress("States", m_states.device_ptr());
    m_context->setTensorAddress("WallPriors", m_wall_priors.device_ptr());
    m_context->setTensorAddress("StepPriors", m_step_priors.device_ptr());
    m_context->setTensorAddress("Values", m_values.device_ptr());

    m_dequeued_promises.reserve(m_batch_size);
    m_worker = std::jthread{std::bind_front(&BatchedModel::run_worker, this)};
}

BatchedModel::~BatchedModel() {
    // Sentinel value so the worker stops (eventually)
    m_tasks.blockingWrite(InferenceTask{});
}

folly::SemiFuture<BatchedModel::Output> BatchedModel::inference(Input input) {
    InferenceTask task{std::move(input), {}};
    auto result = task.output.getSemiFuture();

    m_tasks.blockingWrite(std::move(task));

    return result;
}

void BatchedModel::run_worker() {
    while (true) {
        m_dequeued_promises.clear();

        for (int i = 0; i < m_batch_size; ++i) {
            InferenceTask task;

            if (!m_tasks.read(task)) {
                break;
            }

            if (task.input.state.empty()) {
                return;
            }

            std::ranges::copy(task.input.state, m_states.begin() + m_state_size * i);
            m_dequeued_promises.push_back(std::move(task.output));
        }

        m_states.to_device(m_stream);
        m_context->enqueueV3(m_stream.get());
        m_wall_priors.to_host(m_stream);
        m_step_priors.to_host(m_stream);
        m_values.to_host(m_stream);

        // TODO: eliminate this barrier
        m_stream.synchronize();

        for (std::size_t i = 0; i < m_dequeued_promises.size(); ++i) {
            Output output{{m_wall_priors.begin() + m_wall_prior_size * i,
                           m_wall_priors.begin() + m_wall_prior_size * (i + 1)},
                          {m_step_priors.begin() + kNumDirections * i,
                           m_step_priors.begin() + kNumDirections * (i + 1)},
                          m_values[i]};

            m_dequeued_promises[i].setValue(std::move(output));
        }
    }
}
