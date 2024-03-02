#include "batched_model_policy.hpp"

BatchedModelPolicy::BatchedModelPolicy(std::shared_ptr<BatchedModel> model)
    : m_model{std::move(model)} {}

folly::coro::Task<Evaluation> BatchedModelPolicy::operator()(
    Board const& board, Turn turn, std::optional<Cell> previous_position) {
    auto state = convert_to_model_input(board, turn);
    auto inference_result = co_await m_model->inference(std::move(state));

    Evaluation eval;
    eval.value = inference_result.value;

    Cell const pos = board.position(turn.player);
    std::size_t board_size = board.columns() * board.rows();
    float total_prior = 0.0;

    for (Direction dir : kDirections) {
        if (!board.is_blocked(Wall{pos, dir})) {
            if (!previous_position || pos.step(dir) != *previous_position) {
                float prior = inference_result.prior[2 * board_size + int(dir)];
                eval.edges.emplace_back(dir, prior);
                total_prior += prior;
            }
        }
    }

    auto const legal_walls = board.legal_walls();

    for (Wall wall : legal_walls) {
        int index =
            int(wall.type) * board.columns() * board.rows() + board.index_from_cell(wall.cell);
        eval.edges.emplace_back(wall, inference_result.prior[index]);
        total_prior += inference_result.prior[index];
    }

    // Renormalize to account for illegal actions.
    for (TreeEdge& edge : eval.edges) {
        edge.prior /= total_prior;
    }

    co_return eval;
}
