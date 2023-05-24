#include "batched_model_policy.hpp"

#include <folly/Overload.h>

#include <ranges>
#include <thread>

#include "model.hpp"
#include "util.hpp"

folly::coro::Task<MCTSPolicy::Evaluation> BatchedModelPolicy::evaluate_position(Board const& board,
                                                                                Turn turn,
                                                                                TreeNode const*) {
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

void BatchedModelPolicy::snapshot(TreeNode const& current_root) {
    if (!m_snapshot_stream) {
        return;
    }

    std::vector<float> state = convert_to_state(current_root.board, current_root.turn);
    BatchedModel::Output output = convert_to_output(current_root);

    {
        std::lock_guard lock{m_snapshot_mutex};
        std::ostream_iterator<float> it{*m_snapshot_stream, ", "};

        std::ranges::copy(state, it);
        *m_snapshot_stream << '\n';
        std::ranges::copy(output.wall_prior, it);
        *m_snapshot_stream << '\n';
        std::ranges::copy(output.step_prior, it);
        *m_snapshot_stream << '\n';

        TreeNode::Value val = current_root.value;
        *m_snapshot_stream << val.total_weight / val.total_samples << "\n\n";
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

BatchedModel::Output BatchedModelPolicy::convert_to_output(TreeNode const& node) const {
    std::size_t board_size = node.board.columns() * node.board.rows();

    std::vector<float> wall_prior(2 * board_size);
    std::array<float, 4> step_prior{};

    TreeNode::Value root_val = node.value;

    for (TreeEdge const& te : node.edges) {
        if (!te.child) {
            continue;
        }

        float prior = float(te.child.load()->value.load().total_samples) / root_val.total_samples;

        folly::variant_match(
            te.action, [&](Direction dir) { step_prior[int(dir)] = prior; },
            [&](Wall wall) {
                wall_prior[int(wall.type) * board_size + node.board.index_from_cell(wall.cell)] =
                    prior;
            });
    }

    return {std::move(wall_prior), step_prior, root_val.total_weight / root_val.total_samples};
}
