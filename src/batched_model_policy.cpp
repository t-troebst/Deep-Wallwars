#include "batched_model_policy.hpp"

BatchedModelPolicy::BatchedModelPolicy(std::shared_ptr<BatchedModel> model)
    : m_model{std::move(model)} {}

folly::coro::Task<Evaluation> BatchedModelPolicy::operator()(Board const& board, Turn turn) {
    auto state = convert_to_model_input(board, turn);
    auto inference_result = co_await m_model->inference(std::move(state));

    Evaluation eval;
    eval.value = inference_result.value;

    Cell const pos = board.position(turn.player);
    std::size_t board_size = board.columns() * board.rows();

    for (Direction dir : kDirections) {
        if (!board.is_blocked(Wall{pos, dir})) {
            eval.edges.emplace_back(dir, inference_result.prior[2 * board_size + int(dir)]);
        }
    }

    auto const legal_walls = board.legal_walls();

    for (Wall wall : legal_walls) {
        int index =
            int(wall.type) * board.columns() * board.rows() + board.index_from_cell(wall.cell);
        eval.edges.emplace_back(wall, inference_result.prior[index]);
    }

    co_return eval;
}
