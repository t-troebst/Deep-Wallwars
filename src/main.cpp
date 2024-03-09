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

DEFINE_string(ranking, "", "Folder of *.trt models to rank against each other");
DEFINE_int32(rank_last, 5, "Number of models that each model plays against during ranking");
DEFINE_int32(models_to_rank, 0, "Number of models that play games for ranking (0 for all)");

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
    folly::coro::blockingWait(training_play(board, FLAGS_games,
                                            {
                                                .model1 = cached_policy,
                                                .model2 = cached_policy,
                                                .samples = FLAGS_samples,
                                                .on_complete = training_data_printer,
                                                .seed = FLAGS_seed,
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
    folly::coro::blockingWait(training_play(board, FLAGS_games,
                                            {
                                                .model1 = simple_policy,
                                                .model2 = simple_policy,
                                                .samples = FLAGS_samples,
                                                .on_complete = training_data_printer,
                                                .seed = FLAGS_seed,
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
    auto recorders =
        folly::coro::blockingWait(evaluation_play(board, FLAGS_games,
                                                  {
                                                      .model1 = {cached_policy1, "Model1"},
                                                      .model2 = {cached_policy2, "Model2"},
                                                      .samples = FLAGS_samples,
                                                      .seed = FLAGS_seed,
                                                  })
                                      .scheduleOn(&thread_pool));

    for (auto const& [player, results] : tally_results(recorders)) {
        XLOGF(INFO, "{} has a W/L/D of {}/{}/{}.", player, results.wins, results.losses,
              results.draws);
    }
}

void evaluate_simple(nv::IRuntime& runtime, std::string const& model1) {
    std::ifstream model1_file(model1, std::ios::binary);
    auto engine1 = load_serialized_engine(runtime, model1_file);
    auto batched_model =
        std::make_shared<BatchedModel>(std::make_unique<TensorRTModel>(*engine1), 4096);
    BatchedModelPolicy batched_model_policy(batched_model);
    CachedPolicy cached_policy(batched_model_policy, FLAGS_cache_size);
    SimplePolicy simple_policy(FLAGS_move_prior, FLAGS_good_move, FLAGS_bad_move);

    Board board{FLAGS_columns, FLAGS_rows};

    folly::CPUThreadPoolExecutor thread_pool(FLAGS_j);

    folly::coro::blockingWait(evaluation_play(board, FLAGS_games,
                                              {
                                                  .model1 = {cached_policy, "Model"},
                                                  .model2 = {simple_policy, "Simple"},
                                                  .samples = FLAGS_samples,
                                                  .seed = FLAGS_seed,
                                              })
                                  .scheduleOn(&thread_pool));

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
    folly::coro::blockingWait(interactive_play(board,
                                               {
                                                   .model = simple_policy,
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
    folly::coro::blockingWait(interactive_play(board,
                                               {
                                                   .model = cached_policy1,
                                                   .samples = FLAGS_samples,
                                                   .seed = FLAGS_seed,
                                               })
                                  .scheduleOn(&thread_pool));
}

void ranking(nv::IRuntime& runtime) {
    std::filesystem::path ranking_folder(FLAGS_ranking);
    std::map<std::filesystem::file_time_type, std::filesystem::path> model_paths;
    for (auto const& dir_entry : std::filesystem::directory_iterator{ranking_folder}) {
        if (dir_entry.path().extension() == ".trt") {
            model_paths.insert({dir_entry.last_write_time(), dir_entry.path()});
        }
    }

    std::vector<std::unique_ptr<nv::ICudaEngine>> engines;
    std::vector<NamedModel> models;

    for (auto const& [_, model_path] : model_paths) {
        std::ifstream model_file{model_path, std::ios_base::binary};
        engines.push_back(load_serialized_engine(runtime, model_file));
        auto batched_model =
            std::make_shared<BatchedModel>(std::make_unique<TensorRTModel>(*engines.back()), 4096);
        BatchedModelPolicy batched_model_policy(std::move(batched_model));
        CachedPolicy cached_policy1(std::move(batched_model_policy), FLAGS_cache_size);
        models.push_back({std::move(cached_policy1), model_path.filename()});
    }

    Board board{FLAGS_columns, FLAGS_rows};
    folly::CPUThreadPoolExecutor thread_pool(FLAGS_j);
    XLOGF(INFO, "Collected {} models. Starting ranking now.", models.size());

    auto recorders =
        folly::coro::blockingWait(ranking_play(board, FLAGS_games,
                                               {.models = std::move(models),
                                                .max_matchup_distance = FLAGS_rank_last,
                                                .models_to_rank = FLAGS_models_to_rank,
                                                .samples = FLAGS_samples,
                                                .seed = FLAGS_seed}));

    std::string json = all_to_json(recorders);
    std::ofstream json_file{ranking_folder / "games.json", std::ios_base::app};
    json_file << json;

    std::string pgn = all_to_pgn(recorders);
    std::ofstream pgn_file{ranking_folder / "games.pgn", std::ios_base::app};
    pgn_file << pgn;

    XLOGF(INFO, "Output written to {}.pgn/json.", (ranking_folder / "games").string());
}

int main(int argc, char** argv) {
    gflags::ParseCommandLineFlags(&argc, &argv, true);

    Logger logger;
    std::unique_ptr<nv::IRuntime> runtime{nv::createInferRuntime(logger)};

    auto start = std::chrono::high_resolution_clock::now();

    if (FLAGS_ranking != "") {
        ranking(*runtime);
    } else if (FLAGS_interactive) {
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
        evaluate_simple(*runtime, FLAGS_model1);
    } else if (FLAGS_model1 == "simple") {
        evaluate_simple(*runtime, FLAGS_model2);
    } else {
        evaluate(*runtime, FLAGS_model1, FLAGS_model2);
    }

    auto stop = std::chrono::high_resolution_clock::now();

    XLOGF(INFO, "Completed in {} seconds.",
          std::chrono::duration_cast<std::chrono::seconds>(stop - start).count());
}
