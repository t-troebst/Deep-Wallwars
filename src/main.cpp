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
#include "tensorrt_model.hpp"

namespace nv = nvinfer1;

DEFINE_string(model, "model.trt", "Serialized TensorRT Model");
DEFINE_string(snapshots, "snapshots.csv", "Output for training");
DEFINE_uint32(seed, 42, "Random seed");
DEFINE_uint64(cache_size, 100'000, "Size of the internal evaluation cache");

DEFINE_int32(columns, 6, "Number of columns");
DEFINE_int32(rows, 6, "Number of rows");

DEFINE_int32(games, 100, "Number of games to play");
DEFINE_int32(samples, 500, "Number of MCST samples per action");
DEFINE_int32(j, 8, "Number of threads");

DEFINE_double(move_prior, 0.3, "Move prior of simple agent");
DEFINE_double(good_move, 1.5, "Good move bias of simple agent");
DEFINE_double(bad_move, 0.75, "Bad move bias of simple agent");

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

std::vector<std::unique_ptr<Model>> get_models(nv::ICudaEngine& engine, int n) {
    std::vector<std::unique_ptr<Model>> result;

    for (int i = 0; i < n; ++i) {
        result.push_back(std::make_unique<TensorRTModel>(engine));
    }

    return result;
}

int main(int argc, char** argv) {
    gflags::ParseCommandLineFlags(&argc, &argv, true);

    Logger logger;
    std::unique_ptr<nv::IRuntime> runtime{nv::createInferRuntime(logger)};
    std::ifstream model_file(FLAGS_model, std::ios::binary);
    auto engine = load_serialized_engine(*runtime, model_file);
    auto trt_models = get_models(*engine, 3);

    // For some reason, the second model created by TensorRT is misaligned in GPU memory. This is an
    // extremely ridiculous work-around. Nvidia what the hell?
    std::vector<std::unique_ptr<Model>> safe_models;
    safe_models.push_back(std::move(trt_models[0]));
    safe_models.push_back(std::move(trt_models[2]));

    auto batched_model = std::make_shared<BatchedModel>(std::move(safe_models), 1024);

    auto snapshots_file = std::make_shared<std::ofstream>(FLAGS_snapshots);
    auto sp1 = std::make_shared<BatchedModelPolicy>(batched_model, snapshots_file);
    auto sp1_cached = std::make_shared<CachedPolicy>(sp1, FLAGS_cache_size);
    auto sp2 = std::make_shared<SimplePolicy>(FLAGS_move_prior, FLAGS_good_move, FLAGS_bad_move);

    Board board{FLAGS_columns, FLAGS_rows};

    folly::CPUThreadPoolExecutor thread_pool(FLAGS_j);
    auto start = std::chrono::high_resolution_clock::now();
    folly::coro::blockingWait(computer_play(board, sp1_cached, sp1_cached, FLAGS_games,
                                            {.samples = FLAGS_samples, .seed = FLAGS_seed})
                                  .scheduleOn(&thread_pool));
    auto stop = std::chrono::high_resolution_clock::now();

    XLOGF(INFO, "{} cache hits, {} cache misses during play.", sp1_cached->cache_hits(),
          sp1_cached->cache_misses());
    XLOGF(INFO, "Completed in {} seconds.",
          std::chrono::duration_cast<std::chrono::seconds>(stop - start).count());
}
