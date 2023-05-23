#include "mcts.hpp"

#include <folly/experimental/coro/Collect.h>

#include <algorithm>
#include <random>
#include <ranges>

namespace views = std::ranges::views;

MCTS::MCTS(std::shared_ptr<MCTSPolicy> policy, Board board)
    : MCTS{std::move(policy), std::move(board), {}} {}

MCTS::MCTS(std::shared_ptr<MCTSPolicy> policy, Board board, Options options)
    : m_policy{std::move(policy)},
      m_root{m_policy->evaluate_position(board, options.starting_turn, nullptr).get()},
      m_current_root{m_root},
      m_opts{options},
      m_gamma_dist{options.direchlet_alpha, 1.0},
      m_twister{options.seed} {
    add_root_noise();
}

folly::coro::Task<double> MCTS::sample(int worker_iterations) {
    std::atomic<int> remaining_iters = worker_iterations;

    auto workers = views::iota(0, m_opts.max_parallelism) |
                   views::transform([&](int) { return sample_worker(remaining_iters); });

    co_await folly::coro::collectAllRange(workers);

    TreeNode::Value val = m_root->value;
    co_return val.total_weight / val.total_samples;
}

folly::coro::Task<> MCTS::sample_worker(std::atomic<int>& remaining_iters) {
    while (--remaining_iters > 0) {
        co_await sample_rec(*m_current_root);
    }
}

folly::coro::Task<float> MCTS::sample_rec(TreeNode& root) {
    if (auto winner = root.board.winner(); winner) {
        co_return *winner == root.turn.player ? 1 : -1;
    }

    if (root.depth >= m_opts.max_depth) {
        co_return root.board.score_for(root.turn.player);
    }

    TreeEdge& te = *std::ranges::max_element(root.edges, {}, [&](TreeEdge const& te) -> float {
        TreeNode::Value root_val = root.value;
        TreeNode* child = te.child;

        if (!child) {
            int const active_samples = te.active_samples;

            if (active_samples) {
                // Sampling here would be a total waste so make this expensive
                return -1000 * active_samples;
            }

            return m_opts.puct * te.prior * std::sqrt(root_val.total_samples);
        }

        TreeNode::Value child_val = child->value;
        int const active_samples = te.active_samples;
        child_val.total_weight -= active_samples;
        child_val.total_samples += active_samples;

        return child_val.total_weight / child_val.total_samples +
               m_opts.puct * te.prior * std::sqrt(root_val.total_samples) /
                   (1 + child_val.total_samples);
    });

    ++te.active_samples;
    float value;

    if (te.child == nullptr) {
        Board next_board{root.board};
        next_board.do_action(root.turn.player, te.action);

        TreeNode* new_node =
            co_await m_policy->evaluate_position(std::move(next_board), root.turn.next(), &root);
        TreeNode* expected;

        value = new_node->value.load().total_weight;

        if (!te.child.compare_exchange_strong(expected, new_node)) {
            ++m_wasted_inferences;
            delete new_node;
        }
    } else {
        value = co_await sample_rec(*te.child);
    }

    --te.active_samples;
    root.add_sample(value);

    co_return value;
}

Action MCTS::commit_to_action() {
    TreeEdge const& te =
        *std::ranges::max_element(m_current_root->edges, {}, [&](TreeEdge const& te) {
            return te.child ? te.child.load()->value.load().total_samples : 0;
        });

    m_current_root = te.child;
    return te.action;
}

Action MCTS::commit_to_action(double temperature) {
    auto const weights =
        std::ranges::views::transform(m_current_root->edges, [&](TreeEdge const& te) {
            return te.child
                       ? std::pow(te.child.load()->value.load().total_samples, 1.0 / temperature)
                       : 0;
        });

    std::discrete_distribution<std::size_t> weight_dist(weights.begin(), weights.end());
    TreeEdge const& te = m_current_root->edges[weight_dist(m_twister)];

    m_current_root = te.child;
    add_root_noise();
    return te.action;
}

Move MCTS::commit_to_move() {
    Move result{commit_to_action(), {}};

    if (!m_current_root->board.winner()) {
        result.second = commit_to_action();
    }

    return result;
}

Move MCTS::commit_to_move(double temperature) {
    Move result{commit_to_action(temperature), {}};

    if (!m_current_root->board.winner()) {
        result.second = commit_to_action(temperature);
    }

    return result;
}

void MCTS::force_action(Action const& action) {
    auto const te_it = std::ranges::find_if(
        m_current_root->edges, [&](TreeEdge const& te) { return te.action == action; });

    if (te_it == m_current_root->edges.end()) {
        throw std::runtime_error("Could not find action - not legal?");
    }

    if (!te_it->child) {
        Board board = m_current_root->board;
        board.do_action(m_current_root->turn.player, action);
        te_it->child =
            m_policy
                ->evaluate_position(std::move(board), m_current_root->turn.next(), m_current_root)
                .get();
    }

    m_current_root = te_it->child;
    add_root_noise();
}

void MCTS::force_move(Move const& move) {
    force_action(move.first);

    if (!m_current_root->board.winner()) {
        force_action(move.second);
    }
}

void MCTS::add_root_noise() {
    float total = 0.0;

    std::vector<float> samples(m_current_root->edges.size());

    for (float& s : samples) {
        total += s = m_gamma_dist(m_twister);
    }

    for (std::size_t i = 0; i < samples.size(); ++i) {
        m_current_root->edges[i].prior += samples[i] / total;
    }
}

Board const& MCTS::current_board() const {
    return m_current_root->board;
}

float MCTS::root_value() const {
    TreeNode::Value val = m_current_root->value;
    return val.total_weight / val.total_samples;
}

int MCTS::root_samples() const {
    return m_current_root->value.load().total_samples;
}
