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
#ifdef GUI_ENABLED
#include "gui/game_gui.hpp"
#endif

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
DEFINE_bool(gui, false, "Use GUI instead of console for interactive mode");

DEFINE_string(ranking, "", "Folder of *.trt models to rank against each other");
DEFINE_int32(rank_last, 5, "Number of models that each model plays against during ranking");
DEFINE_int32(models_to_rank, 0, "Number of models that play games for ranking (0 for all)");

int const kBatchedModelQueueSize = 4096;

enum class Mode {
    Train,
    Evaluate,
    Interactive,
    Ranking
};

// Creates and validates a model, returning it as an EvaluationFunction
EvaluationFunction create_and_validate_model(nv::IRuntime& runtime, std::string const& model_flag,
                                             Mode mode) {
    if (model_flag == "simple") {
        return SimplePolicy(FLAGS_move_prior, FLAGS_good_move, FLAGS_bad_move);
    }

    // Load and validate TensorRT model
    std::ifstream model_file(model_flag, std::ios::binary);
    if (!model_file) {
        throw std::runtime_error("Failed to open model file: " + model_flag);
    }  
    XLOGF(INFO, "Loading TensorRT engine from: {}", model_flag);  
    std::shared_ptr<nv::ICudaEngine> engine;
    try {
        engine = load_serialized_engine(runtime, model_file);
    } catch (const std::exception& e) {
        throw std::runtime_error("Failed to load TensorRT engine from " + model_flag + ": " + e.what());
    }
    if (!engine) {
        throw std::runtime_error("Failed to load TensorRT engine from: " + model_flag);
    }

    std::vector<std::unique_ptr<Model>> tensor_rt_models;
    // Use two models to improve GPU utilization.
    int num_models = mode == Mode::Train ? 2 : 1;
    for (int i = 0; i < num_models; i++) {
        tensor_rt_models.push_back(std::make_unique<TensorRTModel>(engine));
    }
    auto batched_model =
        std::make_shared<BatchedModel>(std::move(tensor_rt_models), kBatchedModelQueueSize);
    BatchedModelPolicy batched_model_policy(std::move(batched_model));
    return CachedPolicy(std::move(batched_model_policy), FLAGS_cache_size);
}

std::string get_usage_message() {
    std::ostringstream oss;
    oss << "Deep Wallwars Usage:\n\n"
        << "RANKING: Rank all models in a folder against each other\n"
        << "    ./deep_ww --ranking <model_folder>\n"
        << "  Options:\n"
        << "    --rank_last N      # Number of models each model plays against (default 5)\n"
        << "    --models_to_rank N # Number of models to rank (0 for all)\n"
        << "INTERACTIVE: Play against the AI\n"
        << "    ./deep_ww --interactive --model1 <model.trt | simple>\n"
        << "    ./deep_ww --interactive --model1 <model.trt | simple> --gui  # Use GUI instead of console\n"
        << "TRAINING: Generate training data via self-play\n"
        << "    ./deep_ww --model1 <model.trt | simple>\n"
        << "  Options:\n"
        << "    --output DIR # Output folder (default 'data')\n"
        << "EVALUATION: Evaluate models against each other\n"
        << "    ./deep_ww --model1 <model1.trt | simple> --model2 <model2.trt | simple>\n"
        << "COMMON OPTIONS:\n"
        << "    --games N             # Number of games to play (default 100)\n"
        << "    --samples N           # MCTS samples per action (default 500)\n"
        << "    --columns N --rows N  # Board size (default 5x5)\n"
        << "    --j N                 # Thread count (default 8)\n"
        << "    --seed N              # Random seed (default 42)\n"
        << "    --cache_size N        # MCTS cache size (default 100k)\n"
        << "SIMPLE POLICY OPTIONS: policy that primarily tries to move towards the goal\n"
        << "    --move_prior N  # How likely it is to choose a pawn move (default 0.3)\n"
        << "    --good_move N   # Bias for pawn moves that get closer to the goal (default 1.5)\n"
        << "    --bad_move N    # Bias for pawn moves that get further from the goal (default "
           "0.75)\n"
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

void train(EvaluationFunction const& eval_fn) {
    Board board{FLAGS_columns, FLAGS_rows};
    TrainingDataPrinter training_data_printer(FLAGS_output, 0.5);

    folly::CPUThreadPoolExecutor thread_pool(FLAGS_j);
    
    XLOGF(INFO, "Created thread pool with {} threads (FLAGS_j = {})", 
          thread_pool.numThreads(), FLAGS_j);
    
    folly::coro::blockingWait(training_play(board, FLAGS_games,
                                            {
                                                .model1 = eval_fn,
                                                .model2 = eval_fn,
                                                .samples = FLAGS_samples,
                                                .on_complete = training_data_printer,
                                                .seed = FLAGS_seed,
                                            })
                                  .scheduleOn(&thread_pool));

    // Get cache stats if available
    if (auto* cached_policy = eval_fn.target<CachedPolicy>()) {
        XLOGF(INFO, "{} cache hits, {} cache misses during play.", cached_policy->cache_hits(),
              cached_policy->cache_misses());

        // Get batched model stats
        if (auto* policy = cached_policy->underlying_policy().target<BatchedModelPolicy>()) {
            auto inferences = policy->total_inferences();
            auto batches = policy->total_batches();
            XLOGF(INFO, "{} inferences were sent in {} batches ({} per batch)", inferences, batches,
                  double(inferences) / batches);
        }
    }
}

void evaluate(EvaluationFunction const& eval_fn1, EvaluationFunction const& eval_fn2) {
    Board board{FLAGS_columns, FLAGS_rows};
    folly::CPUThreadPoolExecutor thread_pool(FLAGS_j);

    auto recorders = folly::coro::blockingWait(evaluation_play(board, FLAGS_games,
                                                               {
                                                                   .model1 = {eval_fn1, "Model1"},
                                                                   .model2 = {eval_fn2, "Model2"},
                                                                   .samples = FLAGS_samples,
                                                                   .seed = FLAGS_seed,
                                                               })
                                                   .scheduleOn(&thread_pool));

    for (auto const& [player, results] : tally_results(recorders)) {
        XLOGF(INFO, "{} has a W/L/D of {}/{}/{}.", player, results.wins, results.losses,
              results.draws);
    }

    // Get cache stats for first model if available
    if (auto* cached_policy = eval_fn1.target<CachedPolicy>()) {
        XLOGF(INFO, "Model1: {} cache hits, {} cache misses during play.",
              cached_policy->cache_hits(), cached_policy->cache_misses());

        // Get batched model stats for first model
        if (auto* policy = cached_policy->underlying_policy().target<BatchedModelPolicy>()) {
            auto inferences = policy->total_inferences();
            auto batches = policy->total_batches();
            XLOGF(INFO, "Model1: {} inferences were sent in {} batches ({} per batch)", inferences,
                  batches, double(inferences) / batches);
        }
    }
}

void interactive(EvaluationFunction const& eval_fn) {
    Board board{FLAGS_columns, FLAGS_rows};
    folly::CPUThreadPoolExecutor thread_pool(FLAGS_j);
    InteractivePlayOptions opts = {
        .model = eval_fn,
        .samples = FLAGS_samples,
        .seed = FLAGS_seed,
    };
    
#ifdef GUI_ENABLED
    if (FLAGS_gui) {
        GUI::interactive_play_gui(board, opts, thread_pool);
        return;
    }
#endif
    folly::coro::blockingWait(interactive_play(board, opts)
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
        auto cached_policy = create_and_validate_model(runtime, model_path.string(), Mode::Ranking);
        models.push_back(NamedModel{std::move(cached_policy), model_path.filename().string()});
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
                                                .seed = FLAGS_seed})
                                      .scheduleOn(&thread_pool));

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
    
    if (!runtime) {
        XLOG(ERR, "Failed to create TensorRT runtime. CUDA may be not available or out of memory.");
        return 1;
    }

    Mode mode;
    if (FLAGS_ranking != "") {
        mode = Mode::Ranking;
    } else if (FLAGS_interactive) {
        mode = Mode::Interactive;
    } else if (FLAGS_model2.empty()) {
        // If only one model is provided, generate training data with self-play.
        // This is called from the training script, it is not intended to be used
        // directly.
        mode = Mode::Train;
    } else {
        // If two models are provided, evaluate them against each other.
        // The key output is the win/loss/draw record.
        mode = Mode::Evaluate;
    }

    // Validate arguments for the selected mode
    if (mode == Mode::Ranking) {
        if (!FLAGS_model1.empty() || !FLAGS_model2.empty()) {
            XLOG(ERR, "Ranking mode does not support --model1 or --model2.");
            return 1;
        }
        if (FLAGS_interactive) {
            XLOG(ERR, "Specified --interactive and --ranking.");
            return 1;
        }
    } else if (mode == Mode::Interactive) {
        if (FLAGS_model1.empty()) {
            XLOG(ERR, "Interactive mode requires --model1.");
            return 1;
        }
        if (!FLAGS_model2.empty()) {
            XLOG(ERR, "Interactive mode does not support --model2.");
            return 1;
        }
#ifndef GUI_ENABLED
        if (FLAGS_gui) {
            XLOG(ERR, "GUI support not available. This build was compiled without SFML. Install libsfml-dev and rebuild to enable GUI support.");
            return 1;
        }
#endif
    }

    EvaluationFunction eval_fn1, eval_fn2;
    if (!FLAGS_model1.empty()) {
        eval_fn1 = create_and_validate_model(*runtime, FLAGS_model1, mode);
    }
    if (!FLAGS_model2.empty()) {
        eval_fn2 = create_and_validate_model(*runtime, FLAGS_model2, mode);
    }

    auto start = std::chrono::high_resolution_clock::now();

    if (mode == Mode::Ranking) {
        ranking(*runtime);
    } else if (mode == Mode::Interactive) {
        interactive(eval_fn1);
    } else if (mode == Mode::Evaluate) {
        evaluate(eval_fn1, eval_fn2);
    } else if (mode == Mode::Train) {
        train(eval_fn1);
    }

    auto stop = std::chrono::high_resolution_clock::now();
    XLOGF(INFO, "Completed in {} seconds.",
          std::chrono::duration_cast<std::chrono::seconds>(stop - start).count());
    return 0;
}
