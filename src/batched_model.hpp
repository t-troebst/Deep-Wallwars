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
        std::vector<double> state;
    };

    struct Output {
        std::vector<double> wall_prior;
        std::vector<double> step_prior;
        double value;
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
    CudaBuffer<double> m_states;
    CudaBuffer<double> m_wall_priors;
    CudaBuffer<double> m_step_priors;
    CudaBuffer<double> m_values;

    folly::MPMCQueue<InferenceTask> m_tasks;
    std::vector<folly::Promise<Output>> m_dequeued_promises;
    std::jthread m_worker;

    void run_worker();
};
