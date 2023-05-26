#include "batched_model_policy.hpp"

#include <folly/Overload.h>
#include <folly/logging/xlog.h>

#include <algorithm>
#include <thread>

#include "model.hpp"

BatchedModelPolicy::BatchedModelPolicy(std::shared_ptr<BatchedModel> model,
                                       std::shared_ptr<std::ostream> snapshot_stream)
    : m_model{std::move(model)}, m_snapshot_stream{std::move(snapshot_stream)} {}

folly::coro::Task<MCTSPolicy::Evaluation> BatchedModelPolicy::evaluate_position(Board const& board,
                                                                                Turn turn) {
    auto state = convert_to_state(board, turn);
    auto inference_result = co_await m_model->inference(std::move(state));

    Evaluation eval;
    eval.value = inference_result.value;

    Cell const pos = board.position(turn.player);

    for (Direction dir : kDirections) {
        if (!board.is_blocked(Wall{pos, dir})) {
            eval.edges.emplace_back(dir, inference_result.step_prior[int(dir)]);
        }
    }

    auto const legal_walls = board.legal_walls();

    for (Wall wall : legal_walls) {
        int index =
            int(wall.type) * board.columns() * board.rows() + board.index_from_cell(wall.cell);
        eval.edges.emplace_back(wall, inference_result.wall_prior[index]);
    }

    co_return eval;
}

void BatchedModelPolicy::snapshot(NodeInfo const& node_info, std::optional<Player> winner) {
    if (!m_snapshot_stream) {
        return;
    }

    std::vector<float> const state = convert_to_state(node_info.board, node_info.turn);
    BatchedModel::Output const output = convert_to_output(node_info, winner);

    {
        std::lock_guard lock{m_snapshot_mutex};
        std::ostream_iterator<float> it{*m_snapshot_stream, ", "};

        std::copy(state.begin(), state.end() - 1, it);
        *m_snapshot_stream << state.back() << '\n';
        std::copy(output.wall_prior.begin(), output.wall_prior.end() - 1, it);
        *m_snapshot_stream << output.wall_prior.back() << '\n';
        std::copy(output.step_prior.begin(), output.step_prior.end() - 1, it);
        *m_snapshot_stream << output.step_prior.back() << '\n';

        // TODO: these could be combined in a different way
        *m_snapshot_stream << output.value << "\n\n";
    }
}

std::vector<float> BatchedModelPolicy::convert_to_state(Board const& board, Turn turn) const {
    // TODO: currently we hard-code a history length of 1
    std::size_t board_size = board.columns() * board.rows();
    std::vector<float> state(7 * board_size);

    board.fill_relative_distances(board.position(turn.player), {state.begin(), board_size});
    board.fill_relative_distances(board.goal(turn.player),
                                  {state.begin() + board_size, board_size});

    board.fill_relative_distances(board.position(other_player(turn.player)),
                                  {state.begin() + 2 * board_size, board_size});
    board.fill_relative_distances(board.goal(other_player(turn.player)),
                                  {state.begin() + 3 * board_size, board_size});

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

BatchedModel::Output BatchedModelPolicy::convert_to_output(NodeInfo const& node_info,
                                                           std::optional<Player> winner) const {
    std::size_t board_size = node_info.board.columns() * node_info.board.rows();

    std::vector<float> wall_prior(2 * board_size);
    std::array<float, 4> step_prior{};

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
            edge_info.action, [&](Direction dir) { step_prior[int(dir)] = prior; },
            [&](Wall wall) {
                wall_prior[int(wall.type) * board_size +
                           node_info.board.index_from_cell(wall.cell)] = prior;
            });
    }

    float const z_value = winner ? (*winner == node_info.turn.player ? 1 : -1) : 0;
    return {std::move(wall_prior), step_prior, 0.5f * node_info.q_value + 0.5f * z_value};
}
