#pragma once

#include <folly/MPMCQueue.h>
#include <folly/futures/Future.h>
#include <folly/experimental/coro/Task.h>

#include <memory>
#include <thread>

#include "cuda_wrappers.hpp"

namespace nvinfer1 {
class IRuntime;
class ICudaEngine;
class IExecutionContext;
};  // namespace nvinfer1

class BatchedModel {
public:
    struct Input {
        std::vector<float> state;
    };

    struct Output {
        std::vector<float> wall_prior;
        std::vector<float> step_prior;
        float value;
    };

    BatchedModel(nvinfer1::ICudaEngine& engine, int batch_size);
    BatchedModel(nvinfer1::ICudaEngine& engine, int batch_size, int queue_size);

    ~BatchedModel();

    BatchedModel(BatchedModel const& other) = delete;
    BatchedModel(BatchedModel&& other) noexcept = delete;

    BatchedModel& operator=(BatchedModel const& other) = delete;
    BatchedModel& operator=(BatchedModel&& other) noexcept = delete;

    folly::SemiFuture<Output> inference(Input input);

private:
    struct InferenceTask {
        Input input;
        folly::Promise<Output> output;
    };

    int m_state_size;
    int m_wall_prior_size;
    int m_batch_size;

    std::unique_ptr<nvinfer1::IExecutionContext> m_context;

    CudaStream m_stream;
    CudaBuffer<float> m_states;
    CudaBuffer<float> m_wall_priors;
    CudaBuffer<float> m_step_priors;
    CudaBuffer<float> m_values;

    folly::MPMCQueue<InferenceTask> m_tasks;
    std::vector<folly::Promise<Output>> m_dequeued_promises;
    std::jthread m_worker;

    void run_worker();
};
