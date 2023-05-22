#pragma once

#include <folly/MPMCQueue.h>
#include <folly/futures/Future.h>

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
        std::vector<double> priors;
        double value;
    };

    struct Options {
        int board_width;
        int board_height;
        int history_length = 1;
        int batch_size = 64;
        int queue_size = batch_size * 16;
    };

    BatchedModel(std::shared_ptr<nvinfer1::IRuntime> runtime, std::span<std::byte> model,
                 Options const& opts);

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

    int m_board_width;
    int m_board_height;
    int m_history_length;
    int m_batch_size;

    std::shared_ptr<nvinfer1::IRuntime> m_runtime;
    std::unique_ptr<nvinfer1::ICudaEngine> m_engine;
    std::unique_ptr<nvinfer1::IExecutionContext> m_context;

    CudaStream m_stream;
    CudaBuffer<double> m_states;
    CudaBuffer<double> m_priors;
    CudaBuffer<double> m_values;

    folly::MPMCQueue<InferenceTask> m_tasks;
    std::jthread m_worker;

    void run_worker();
};
