#include "batched_model.hpp"

#include <NvInfer.h>
#include <NvInferRuntime.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <ranges>

namespace nv = nvinfer1;

constexpr int kLayersPerHistory = 7;

BatchedModel::BatchedModel(std::shared_ptr<nv::IRuntime> runtime, std::span<std::byte> model,
                           Options const& opts)
    : m_board_width{opts.board_width},
      m_board_height{opts.board_height},
      m_history_length{opts.history_length},
      m_batch_size{opts.batch_size},
      m_runtime{runtime},
      m_engine{runtime->deserializeCudaEngine(model.data(), model.size())},
      m_context{m_engine->createExecutionContext()},
      m_states(opts.batch_size * opts.history_length * opts.board_width * opts.board_height *
               (kLayersPerHistory + 1)),
      m_priors(opts.batch_size * (opts.board_width * opts.board_height * 2 + 4)),
      m_values(opts.batch_size),
      m_tasks(opts.queue_size) {
    // TODO: validate model and throw exception if it doesn't match

    m_context->setTensorAddress("Priors", m_priors.device_ptr());
    m_context->setTensorAddress("Values", m_values.device_ptr());

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
    std::vector<folly::Promise<Output>> output_promises;
    output_promises.reserve(m_batch_size);

    int const input_size =
        m_board_width * m_board_height * kLayersPerHistory * m_history_length + 1;
    int const priors_size = m_board_width * m_board_height * 2 + 4;

    while (true) {
        output_promises.clear();

        for (int i = 0; i < m_batch_size; ++i) {
            InferenceTask task;
            m_tasks.blockingRead(task);

            if (task.input.state.empty()) {
                return;
            }

            std::ranges::copy(task.input.state, m_states.begin() + input_size * i);
            output_promises.push_back(std::move(task.output));
        }

        m_states.to_device(m_stream);
        m_context->enqueueV3(m_stream.get());
        m_priors.to_host(m_stream);
        m_values.to_host(m_stream);

        // TODO: eliminate this barrier
        m_stream.synchronize();

        for (int i = 0; i < m_batch_size; ++i) {
            Output output{
                {m_priors.begin() + priors_size * i, m_priors.begin() + priors_size * (i + 1)},
                m_values[i]};

            output_promises[i].setValue(std::move(output));
        }
    }
}
