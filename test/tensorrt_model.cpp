#include "tensorrt_model.hpp"

#include <NvInfer.h>
#include <folly/logging/xlog.h>

#include <catch2/catch_test_macros.hpp>
#include <fstream>
#include <memory>
#include <vector>

#include "gamestate.hpp"
#include "state_conversions.hpp"

namespace nv = nvinfer1;

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

TEST_CASE("TensorRT 5x5 model", "[tensorrt]") {
    Logger logger;
    std::unique_ptr<nv::IRuntime> runtime{nv::createInferRuntime(logger)};
    REQUIRE(runtime != nullptr);

    std::ifstream model_file("5x5_60000.trt", std::ios::binary);
    REQUIRE(model_file.good());

    auto engine = load_serialized_engine(*runtime, model_file);
    REQUIRE(engine != nullptr);

    auto model = std::make_unique<TensorRTModel>(engine);
    REQUIRE(model != nullptr);
    REQUIRE(model->batch_size() == 256);
    REQUIRE(model->wall_prior_size() == 50);
    REQUIRE(model->state_size() == 200);

    SECTION("Run inference on empty board") {
        Board board(5, 5);
        auto single_input = convert_to_model_input(board, Turn{Player::Red, Turn::First});
        REQUIRE(static_cast<int>(single_input.size()) == model->state_size());

        std::vector<float> batched_input(model->batch_size() * model->state_size(), 0.0f);
        std::copy(single_input.begin(), single_input.end(), batched_input.begin());

        std::vector<float> priors(model->batch_size() * (model->wall_prior_size() + 4));
        std::vector<float> values(model->batch_size());
        Model::Output out{priors, values};

        model->inference(batched_input, out);

        REQUIRE(values[0] >= -1.0f);
        REQUIRE(values[0] <= 1.0f);

        float sum_priors = 0.0f;
        for (int i = 0; i < model->wall_prior_size(); ++i) {
            float p = priors[i];
            REQUIRE(p >= 0.0f);
            REQUIRE(p <= 1.0f);
            sum_priors += p;
        }
        REQUIRE(sum_priors > 0.0f);
    }

    SECTION("Run inference on board with wall") {
        Board board(5, 5);
        board.place_wall(Player::Red, Wall{Cell{0, 0}, Wall::Right});

        auto single_input = convert_to_model_input(board, Turn{Player::Red, Turn::First});
        REQUIRE(static_cast<int>(single_input.size()) == model->state_size());

        std::vector<float> batched_input(model->batch_size() * model->state_size(), 0.0f);
        // Test last batch position this time, had a bug before where not all of the data was copied
        std::copy(single_input.begin(), single_input.end(),
                  batched_input.begin() + (model->batch_size() - 1) * model->state_size());

        std::vector<float> priors(model->batch_size() * (model->wall_prior_size() + 4));
        std::vector<float> values(model->batch_size());
        Model::Output out{priors, values};

        model->inference(batched_input, out);

        int last_batch_idx = model->batch_size() - 1;
        REQUIRE(values[last_batch_idx] >= -1.0f);
        REQUIRE(values[last_batch_idx] <= 1.0f);

        auto last_batch_priors =
            std::span<float>(priors.begin() + last_batch_idx * (model->wall_prior_size() + 4),
                             model->wall_prior_size());

        float wall_prior = last_batch_priors[0];
        REQUIRE(wall_prior < 0.01f);

        // Can't move through the blocked wall, should be near zero
        float right_move_prior = last_batch_priors[model->wall_prior_size()];
        REQUIRE(right_move_prior < 0.01f);

        // Moving down is reasonable, should not be zero
        float down_move_prior = last_batch_priors[model->wall_prior_size() + 1];
        REQUIRE(down_move_prior > 0.05f);
    }
}