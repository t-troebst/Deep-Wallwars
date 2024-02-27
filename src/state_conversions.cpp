#include "state_conversions.hpp"

#include <folly/Overload.h>

#include <filesystem>
#include <fstream>

ModelOutput convert_to_model_output(NodeInfo const& node_info, float score_for_red,
                                    float winner_contribution) {
    std::size_t board_size = node_info.board.columns() * node_info.board.rows();
    std::vector<float> priors(2 * board_size + 4);

    // Note: the sum of samples in the children is not equal to the sum of samples in the parent
    // because some samples *end* in the parent. *Typically* only one sample does but due to the
    // depth limit, it can happen that more do.
    int total_samples = 0;
    for (EdgeInfo const& edge_info : node_info.edges) {
        total_samples += edge_info.num_samples;
    }

    for (EdgeInfo const& edge_info : node_info.edges) {
        if (!edge_info.num_samples) {
            continue;
        }

        float prior = float(edge_info.num_samples) / total_samples;

        folly::variant_match(
            edge_info.action, [&](Direction dir) { priors[2 * board_size + int(dir)] = prior; },
            [&](Wall wall) {
                priors[int(wall.type) * board_size + node_info.board.index_from_cell(wall.cell)] =
                    prior;
            });
    }

    float const z_value = node_info.turn.player == Player::Red ? score_for_red : -score_for_red;
    float const expected_value =
        (1 - winner_contribution) * node_info.q_value + winner_contribution * z_value;
    return {std::move(priors), expected_value};
}

std::vector<float> convert_to_model_input(Board const& board, Turn turn) {
    std::size_t board_size = board.columns() * board.rows();
    std::vector<float> state(7 * board_size);

    auto blocked_directions = board.blocked_directions();
    board.fill_relative_distances(board.position(turn.player), {state.begin(), board_size},
                                  blocked_directions);
    board.fill_relative_distances(board.goal(turn.player), {state.begin() + board_size, board_size},
                                  blocked_directions);

    board.fill_relative_distances(board.position(other_player(turn.player)),
                                  {state.begin() + 2 * board_size, board_size}, blocked_directions);
    board.fill_relative_distances(board.goal(other_player(turn.player)),
                                  {state.begin() + 3 * board_size, board_size}, blocked_directions);

    for (int column = 0; column < board.columns(); ++column) {
        for (int row = 0; row < board.rows(); ++row) {
            Cell cell{column, row};
            for (int type = 0; type < 2; ++type) {
                state[(4 + type) * board_size + board.index_from_cell(cell)] =
                    board.is_blocked({cell, Wall::Type(type)});
            }
        }
    }

    if (turn.action == Turn::Second) {
        std::fill(state.begin() + 6 * board_size, state.end(), 1.0);
    }

    return state;
}

void print_training_data_point(std::ostream& out_stream, ModelInput const& model_input,
                               ModelOutput const& model_output) {
    auto it = std::ostream_iterator<float>(out_stream, ", ");

    std::copy(model_input.begin(), model_input.end() - 1, it);
    out_stream << model_input.back() << '\n';
    std::copy(model_output.prior.begin(), model_output.prior.end() - 1, it);
    out_stream << model_output.prior.back() << '\n';
    out_stream << model_output.value << "\n\n";
}

TrainingDataPrinter::TrainingDataPrinter(std::filesystem::path directory, float winner_contribution)
    : m_directory{std::move(directory)}, m_winner_contribution{winner_contribution} {
    std::filesystem::create_directory(m_directory);
}

void TrainingDataPrinter::operator()(MCTS const& out, int index) const {
    float score_for_red = out.current_board().score_for(Player::Red);

    std::ofstream output_file{m_directory / ("game_" + std::to_string(index) + ".csv")};

    for (NodeInfo const& node_info : out.history()) {
        ModelInput model_input = convert_to_model_input(node_info.board, node_info.turn);
        ModelOutput model_output =
            convert_to_model_output(node_info, score_for_red, m_winner_contribution);
        print_training_data_point(output_file, model_input, model_output);
    }

    // The history does not include the actual winning board state.
    NodeInfo const& node_info = out.root_info();
    ModelInput model_input = convert_to_model_input(node_info.board, node_info.turn);
    ModelOutput model_output =
        convert_to_model_output(node_info, score_for_red, m_winner_contribution);
    print_training_data_point(output_file, model_input, model_output);
}
