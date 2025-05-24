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
#include <sstream>

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
DEFINE_int32(samples, 500, "Number of MCTS samples per action");
DEFINE_int32(j, 8, "Number of threads");

DEFINE_double(move_prior, 0.3, "Move prior of simple agent");
DEFINE_double(good_move, 1.5, "Good move bias of simple agent");
DEFINE_double(bad_move, 0.75, "Bad move bias of simple agent");

DEFINE_bool(interactive, false, "Enable interactive play against the AI");

DEFINE_string(ranking, "", "Folder of *.trt models to rank against each other");
DEFINE_int32(rank_last, 5, "Number of models that each model plays against during ranking");
DEFINE_int32(models_to_rank, 0, "Number of models that play games for ranking (0 for all)");

const int kBatchedModelQueueSize = 4096;

// Factory functions for policy creation
SimplePolicy create_simple_policy() {
    return SimplePolicy(FLAGS_move_prior, FLAGS_good_move, FLAGS_bad_move);
}

CachedPolicy create_model_policy(nv::IRuntime& runtime, std::string const& model_path, int num_models) {
    std::ifstream model_file(model_path, std::ios::binary);
    if (!model_file) {
        throw std::runtime_error("Failed to open model file: " + model_path);
    }
    auto engine = load_serialized_engine(runtime, model_file);
    if (!engine) {
        throw std::runtime_error("Failed to load TensorRT engine from: " + model_path);
    }
    
    std::vector<std::unique_ptr<Model>> tensor_rt_models;
    // Probably two is enough, more models = more parallelizable work for the
    // scheduler on the GPU so you can get slightly higher GPU utilization
    for (int i = 0; i < num_models; i++) {
        tensor_rt_models.push_back(std::make_unique<TensorRTModel>(engine));
    }
    auto batched_model = std::make_shared<BatchedModel>(
        std::move(tensor_rt_models),
        kBatchedModelQueueSize
    );
    BatchedModelPolicy batched_model_policy(std::move(batched_model));
    return CachedPolicy(std::move(batched_model_policy), FLAGS_cache_size);
}

std::string get_usage_message() {
    std::ostringstream oss;
    oss << "Deep Wallwars Usage:\n\n"
        << "RANKING: Rank all models in a folder against each other\n"
        << "  ./deep_ww --ranking <model_folder>\n"
        << "  Options:\n"
        << "    --rank_last N      # Number of models each model plays against (default 5)\n"
        << "    --models_to_rank N # Number of models to rank (0 for all)\n"
        << "INTERACTIVE: Play against the AI\n"
        << "  ./deep_ww --interactive --model1 <model.trt>\n"
        << "  ./deep_ww --interactive --model1 simple\n"
        << "TRAINING: Generate training data via self-play\n"
        << "  ./deep_ww --model1 <model.trt>\n"
        << "  ./deep_ww --model1 simple\n"
        << "  Options:\n"
        << "    --output DIR # Output folder (default 'data')\n"
        << "EVALUATION: Evaluate models against each other\n"
        << "  ./deep_ww --model1 <model1.trt> --model2 <model2.trt>\n"
        << "  ./deep_ww --model1 <model.trt> --model2 simple\n"
        << "  ./deep_ww --model1 simple --model2 <model.trt>\n"
        << "COMMON OPTIONS:\n"
        << "  --games N             # Number of games to play (default 100)\n"
        << "  --samples N           # MCTS samples per action (default 500)\n"
        << "  --columns N --rows N  # Board size (default 5x5)\n"
        << "  --j N                 # Thread count (default 8)\n"
        << "  --seed N              # Random seed (default 42)\n"
        << "  --cache_size N        # MCTS cache size (default 100k)\n"
        << "SIMPLE POLICY OPTIONS: policy that primarily tries to move towards the goal\n"
        << "  --move_prior N  # How likely it is to choose a pawn move (default 0.3)\n"
        << "  --good_move N   # Bias for pawn moves that get closer to the goal (default 1.5)\n"
        << "  --bad_move N    # Bias for pawn moves that get further from the goal (default 0.75)\n"
        << "See --help for all options\n";
    return oss.str();
}

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
    auto cached_policy = create_model_policy(runtime, model, 3);
    TrainingDataPrinter training_data_printer(FLAGS_output, 0.5);
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
    
    // Get batched model stats
    if (auto batched_model = cached_policy.get_batched_model()) {
        auto inferences = batched_model->total_inferences();
        auto batches = batched_model->total_batches();
        XLOGF(INFO, "{} inferences were sent in {} batches ({} per batch)", inferences, batches,
              double(inferences) / batches);
    }
}

void train_simple() {
    auto simple_policy = create_simple_policy();
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
    auto cached_policy1 = create_model_policy(runtime, model1, 1);
    auto cached_policy2 = create_model_policy(runtime, model2, 1);
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
    auto cached_policy = create_model_policy(runtime, model1, 1);
    auto simple_policy = create_simple_policy();

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
    
    // Get batched model stats
    if (auto batched_model = cached_policy.get_batched_model()) {
        auto inferences = batched_model->total_inferences();
        auto batches = batched_model->total_batches();
        XLOGF(INFO, "{} inferences were sent in {} batches ({} per batch)", inferences, batches,
              double(inferences) / batches);
    }
}

void interactive_simple() {
    auto simple_policy = create_simple_policy();
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
    auto cached_policy = create_model_policy(runtime, model, 1);
    Board board{FLAGS_columns, FLAGS_rows};
    
    folly::CPUThreadPoolExecutor thread_pool(FLAGS_j);
    folly::coro::blockingWait(interactive_play(board,
                                               {
                                                   .model = cached_policy,
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
        auto cached_policy = create_model_policy(runtime, model_path.string(), 1);
        models.push_back({std::move(cached_policy), model_path.filename()});
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
    // If no arguments are provided (only program name), print usage and exit.
    if (argc == 1) {
        XLOG(ERR, "No arguments provided.");
        std::cout << get_usage_message() << std::endl;
        return 1;
    }

    gflags::SetUsageMessage(get_usage_message());
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
    } else if (FLAGS_model2 == "") {
        // If only one model is provided, generate training data with self-play.
        // This is called from the training script, it is not intended to be used
        // directly.
        if (FLAGS_model1 == "simple") {
            train_simple();
        } else {
            train(*runtime, FLAGS_model1);
        }
    } else {
        // If two models are provided, evaluate them against each other.
        // The key output is the win/loss/draw record.
        if (FLAGS_model1 == "simple" && FLAGS_model2 == "simple") {
            XLOG(ERR, "Simple vs Simple evaluation is not supported.");
            return 1;
        } else if (FLAGS_model2 == "simple") {
            evaluate_simple(*runtime, FLAGS_model1);
        } else if (FLAGS_model1 == "simple") {
            evaluate_simple(*runtime, FLAGS_model2);
        } else {
            evaluate(*runtime, FLAGS_model1, FLAGS_model2);
        }
    }

    auto stop = std::chrono::high_resolution_clock::now();
    XLOGF(INFO, "Completed in {} seconds.",
          std::chrono::duration_cast<std::chrono::seconds>(stop - start).count());
}
