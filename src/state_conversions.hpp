#pragma once

#include <folly/Synchronized.h>
#include <iosfwd>
#include <vector>

#include "gamestate.hpp"
#include "mcts.hpp"

struct ModelOutput {
    std::vector<float> wall_prior;
    std::array<float, 4> step_prior;
    float value;
};

using ModelInput = std::vector<float>;

// Converts current position in the MCTS tree into the output that we would have expected from the
// ML model. This is used for training.
ModelOutput convert_to_model_output(NodeInfo const& node_info, std::optional<Player> winner);

// Converts current board state into a vector of [0, 1] floats so it can be used for ML models.
ModelInput convert_to_model_input(Board const& board, Turn turn);

// Print a single training data point (input, expected output) to `out_stream`. These will be read
// in from Python for training.
void print_training_data_point(std::ostream& out_stream, ModelInput const& input,
                               ModelOutput const& model_output);

class TrainingDataPrinter {
public:
    TrainingDataPrinter(std::ostream& output);

    void operator()(MCTS const& mcts) const;

private:
    std::shared_ptr<folly::Synchronized<std::ostream*>> m_output;
};
