#include <NvInfer.h>
#include <NvInferRuntime.h>
#include <folly/executors/CPUThreadPoolExecutor.h>
#include <folly/experimental/coro/BlockingWait.h>
#include <folly/logging/xlog.h>
#include <gflags/gflags.h>

#include <chrono>
#include <fstream>
#include <iostream>
#include <memory>

#include "batched_model.hpp"
#include "batched_model_policy.hpp"
#include "cached_policy.hpp"
#include "mcts.hpp"
#include "play.hpp"
#include "simple_policy.hpp"
#include "state_conversions.hpp"
#include "tensorrt_model.hpp"

namespace nv = nvinfer1;

DEFINE_string(model1, "model.trt", "Serialized TensorRT Model 1 ");
DEFINE_string(model2, "", "Serialized TensorRT Model 2");
DEFINE_string(output, "data", "Folder to print training data to");
DEFINE_uint32(seed, 42, "Random seed");
DEFINE_uint64(cache_size, 100'000, "Size of the internal evaluation cache");

DEFINE_int32(columns, 5, "Number of columns");
DEFINE_int32(rows, 5, "Number of rows");

DEFINE_int32(games, 100, "Number of games to play");
DEFINE_int32(samples, 500, "Number of MCST samples per action");
DEFINE_int32(j, 8, "Number of threads");

DEFINE_double(move_prior, 0.3, "Move prior of simple agent");
DEFINE_double(good_move, 1.5, "Good move bias of simple agent");
DEFINE_double(bad_move, 0.75, "Bad move bias of simple agent");

DEFINE_bool(interactive, false, "Enable interactive play against the AI");

struct Logger : nv::ILogger {
    void log(Severity severity, char const* msg) noexcept {
        switch (severity) {
            case Severity::kINTERNAL_ERROR:
            case Severity::kERROR:
                XLOG(ERR, msg);
                break;
            case Severity::kWARNING:
                XLOG(WARN, msg);
                break;
            case Severity::kINFO:
                XLOG(INFO, msg);
                break;
            default:
                break;
        }
    }
};

void train(nv::IRuntime& runtime, std::string const& model) {
    std::ifstream model_file(model, std::ios::binary);
    auto engine = load_serialized_engine(runtime, model_file);

    std::vector<std::unique_ptr<Model>> tensor_rt_models;
    tensor_rt_models.push_back(std::make_unique<TensorRTModel>(*engine));
    tensor_rt_models.push_back(std::make_unique<TensorRTModel>(*engine));
    tensor_rt_models.push_back(std::make_unique<TensorRTModel>(*engine));

    auto batched_model = std::make_shared<BatchedModel>(std::move(tensor_rt_models), 4096);
    TrainingDataPrinter training_data_printer(FLAGS_output, 0.5);
    BatchedModelPolicy batched_model_policy(batched_model);
    CachedPolicy cached_policy(batched_model_policy, FLAGS_cache_size);
    Board board{FLAGS_columns, FLAGS_rows};

    folly::CPUThreadPoolExecutor thread_pool(FLAGS_j);
    folly::coro::blockingWait(computer_play(board, cached_policy, cached_policy, FLAGS_games,
                                            {
                                                .samples = FLAGS_samples,
                                                .seed = FLAGS_seed,
                                                .on_complete = training_data_printer,
                                            })
                                  .scheduleOn(&thread_pool));

    XLOGF(INFO, "{} cache hits, {} cache misses during play.", cached_policy.cache_hits(),
          cached_policy.cache_misses());
    auto inferences = batched_model->total_inferences();
    auto batches = batched_model->total_batches();
    XLOGF(INFO, "{} inferences were sent in {} batches ({} per batch)", inferences, batches,
          double(inferences) / batches);
}

void train_simple() {
    SimplePolicy simple_policy(FLAGS_move_prior, FLAGS_good_move, FLAGS_bad_move);

    Board board{FLAGS_columns, FLAGS_rows};
    TrainingDataPrinter training_data_printer(FLAGS_output, 0.5);

    folly::CPUThreadPoolExecutor thread_pool(FLAGS_j);
    folly::coro::blockingWait(computer_play(board, simple_policy, simple_policy, FLAGS_games,
                                            {
                                                .samples = FLAGS_samples,
                                                .seed = FLAGS_seed,
                                                .on_complete = training_data_printer,
                                            })
                                  .scheduleOn(&thread_pool));
}

void evaluate(nv::IRuntime& runtime, std::string const& model1, std::string const& model2) {
    std::ifstream model1_file(model1, std::ios::binary);
    auto engine1 = load_serialized_engine(runtime, model1_file);
    auto batched_model1 =
        std::make_shared<BatchedModel>(std::make_unique<TensorRTModel>(*engine1), 4096);
    BatchedModelPolicy batched_model_policy1(batched_model1);
    CachedPolicy cached_policy1(batched_model_policy1, FLAGS_cache_size);

    std::ifstream model2_file(model2, std::ios::binary);
    auto engine2 = load_serialized_engine(runtime, model2_file);
    auto batched_model2 =
        std::make_shared<BatchedModel>(std::make_unique<TensorRTModel>(*engine2), 4096);
    BatchedModelPolicy batched_model_policy2(batched_model2);
    CachedPolicy cached_policy2(batched_model_policy1, FLAGS_cache_size);

    Board board{FLAGS_columns, FLAGS_rows};

    folly::CPUThreadPoolExecutor thread_pool(FLAGS_j);
    folly::coro::blockingWait(computer_play(board, cached_policy1, cached_policy2, FLAGS_games,
                                            {
                                                .samples = FLAGS_samples,
                                                .temperature = 0.0,
                                                .seed = FLAGS_seed,
                                            })
                                  .scheduleOn(&thread_pool));
}

void evaluate_simple(nv::IRuntime& runtime, std::string const& model1, bool model_goes_first) {
    std::ifstream model1_file(model1, std::ios::binary);
    auto engine1 = load_serialized_engine(runtime, model1_file);
    auto batched_model =
        std::make_shared<BatchedModel>(std::make_unique<TensorRTModel>(*engine1), 4096);
    BatchedModelPolicy batched_model_policy(batched_model);
    CachedPolicy cached_policy(batched_model_policy, FLAGS_cache_size);
    SimplePolicy simple_policy(FLAGS_move_prior, FLAGS_good_move, FLAGS_bad_move);

    Board board{FLAGS_columns, FLAGS_rows};

    folly::CPUThreadPoolExecutor thread_pool(FLAGS_j);

    if (model_goes_first) {
        folly::coro::blockingWait(computer_play(board, cached_policy, simple_policy, FLAGS_games,
                                                {
                                                    .samples = FLAGS_samples,
                                                    .temperature = 0.0,
                                                    .seed = FLAGS_seed,
                                                })
                                      .scheduleOn(&thread_pool));
    } else {
        folly::coro::blockingWait(computer_play(board, simple_policy, cached_policy, FLAGS_games,
                                                {
                                                    .samples = FLAGS_samples,
                                                    .temperature = 0.0,
                                                    .seed = FLAGS_seed,
                                                })
                                      .scheduleOn(&thread_pool));
    }

    XLOGF(INFO, "{} cache hits, {} cache misses during play.", cached_policy.cache_hits(),
          cached_policy.cache_misses());
    auto inferences = batched_model->total_inferences();
    auto batches = batched_model->total_batches();
    XLOGF(INFO, "{} inferences were sent in {} batches ({} per batch)", inferences, batches,
          double(inferences) / batches);
}

void interactive_simple() {
    SimplePolicy simple_policy(FLAGS_move_prior, FLAGS_good_move, FLAGS_bad_move);

    Board board{FLAGS_columns, FLAGS_rows};
    folly::CPUThreadPoolExecutor thread_pool(FLAGS_j);
    folly::coro::blockingWait(human_play(board, simple_policy,
                                         {
                                             .samples = FLAGS_samples,
                                             .seed = FLAGS_seed,
                                         })
                                  .scheduleOn(&thread_pool));
}

void interactive(nv::IRuntime& runtime, std::string const& model) {
    std::ifstream model1_file(model, std::ios::binary);
    auto engine1 = load_serialized_engine(runtime, model1_file);
    auto batched_model1 =
        std::make_shared<BatchedModel>(std::make_unique<TensorRTModel>(*engine1), 4096);
    BatchedModelPolicy batched_model_policy1(batched_model1);
    CachedPolicy cached_policy1(batched_model_policy1, FLAGS_cache_size);

    Board board{FLAGS_columns, FLAGS_rows};
    folly::CPUThreadPoolExecutor thread_pool(FLAGS_j);
    folly::coro::blockingWait(human_play(board, cached_policy1,
                                         {
                                             .samples = FLAGS_samples,
                                             .seed = FLAGS_seed,
                                         })
                                  .scheduleOn(&thread_pool));
}

int main(int argc, char** argv) {
    gflags::ParseCommandLineFlags(&argc, &argv, true);

    Logger logger;
    std::unique_ptr<nv::IRuntime> runtime{nv::createInferRuntime(logger)};

    auto start = std::chrono::high_resolution_clock::now();

    if (FLAGS_interactive) {
        if (FLAGS_model1 == "simple") {
            interactive_simple();
        } else {
            interactive(*runtime, FLAGS_model1);
        }
    } else if (FLAGS_model1 == "simple" && FLAGS_model2 == "simple") {
        train_simple();
    } else if (FLAGS_model2 == "") {
        train(*runtime, FLAGS_model1);
    } else if (FLAGS_model2 == "simple") {
        evaluate_simple(*runtime, FLAGS_model1, true);
    } else if (FLAGS_model1 == "simple") {
        evaluate_simple(*runtime, FLAGS_model2, false);
    } else {
        evaluate(*runtime, FLAGS_model1, FLAGS_model2);
    }

    auto stop = std::chrono::high_resolution_clock::now();

    XLOGF(INFO, "Completed in {} seconds.",
          std::chrono::duration_cast<std::chrono::seconds>(stop - start).count());
}
