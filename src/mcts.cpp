#include "mcts.hpp"

#include <folly/experimental/coro/Collect.h>

#include <algorithm>
#include <random>
#include <ranges>

namespace views = std::ranges::views;

constexpr float kWastedInferencePenalty = 1000.0;

TreeEdge::TreeEdge(Action action, float prior) : action{action}, prior{prior} {}

TreeEdge::TreeEdge(TreeEdge const& other)
    : action{other.action},
      prior{other.prior},
      active_samples{other.active_samples.load()},
      child{other.child.load()} {}

TreeEdge& TreeEdge::operator=(TreeEdge const& other) {
    action = other.action;
    prior = other.prior;
    active_samples = other.active_samples.load();
    child = other.child.load();

    return *this;
}

void TreeNode::add_sample(float weight) {
    TreeNode::Value old_val = value;
    TreeNode::Value new_val;

    do {
        new_val = {old_val.total_weight + weight, old_val.total_samples + 1};
    } while (!value.compare_exchange_weak(old_val, new_val));
}

TreeNode::~TreeNode() {
    for (auto& te : edges) {
        delete te.child.load();
    }
}

MCTS::MCTS(std::shared_ptr<MCTSPolicy> policy, Board board)
    : MCTS{std::move(policy), std::move(board), {}} {}

MCTS::MCTS(std::shared_ptr<MCTSPolicy> policy, Board board, Options options)
    : m_policy{std::move(policy)},
      m_root{create_tree_node(board, options.starting_turn, nullptr)},
      m_current_root{m_root.get()},
      m_opts{options},
      m_gamma_dist{options.direchlet_alpha, 1.0},
      m_twister{options.seed} {
    add_root_noise();
}

folly::coro::Task<float> MCTS::sample(int worker_iterations) {
    std::atomic<int> remaining_iters = worker_iterations;

    auto workers = views::iota(0, m_opts.max_parallelism) |
                   views::transform([&](int) { return sample_worker(remaining_iters); });

    co_await folly::coro::collectAllRange(workers);

    TreeNode::Value val = m_root->value;
    co_return val.total_weight / val.total_samples;
}

folly::coro::Task<> MCTS::sample_worker(std::atomic<int>& remaining_iters) {
    while (--remaining_iters >= 0) {
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

    if (root.edges.empty()) {
        co_return -1;
    }

    TreeEdge& te = *std::ranges::max_element(root.edges, {}, [&](TreeEdge const& te) {
        TreeNode::Value root_val = root.value;
        TreeNode* child = te.child;

        float const p_root = m_opts.puct * std::sqrt(float(root_val.total_samples));

        if (!child) {
            int const active_samples = te.active_samples;

            if (active_samples) {
                // Sampling here would be a total waste so make this expensive
                return -kWastedInferencePenalty * active_samples;
            }

            return te.prior * p_root;
        }

        TreeNode::Value child_val = child->value;
        int const active_samples = te.active_samples;
        child_val.total_weight -= m_opts.active_sample_penalty * active_samples;
        child_val.total_samples += active_samples;

        return child_val.total_weight / child_val.total_samples +
               te.prior * p_root / (1 + child_val.total_samples);
    });

    ++te.active_samples;
    float value;

    if (te.child == nullptr) {
        Board next_board{root.board};
        next_board.do_action(root.turn.player, te.action);

        TreeNode* new_node =
            co_await create_tree_node_async(std::move(next_board), root.turn.next(), &root);
        TreeNode* expected = nullptr;

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
    if (m_current_root->edges.empty()) {
        throw std::runtime_error("No action available!");
    }

    TreeEdge const& te =
        *std::ranges::max_element(m_current_root->edges, {}, [&](TreeEdge const& te) {
            return te.child ? te.child.load()->value.load().total_samples : 0;
        });

    if (!te.child) {
        throw std::runtime_error("No explored action available!");
    }

    m_current_root = te.child;
    return te.action;
}

Action MCTS::commit_to_action(float temperature) {
    if (m_current_root->edges.empty()) {
        throw std::runtime_error("No action available!");
    }

    auto const weights =
        std::ranges::views::transform(m_current_root->edges, [&](TreeEdge const& te) {
            return te.child
                       ? std::pow(te.child.load()->value.load().total_samples, 1.0 / temperature)
                       : 0;
        });

    std::discrete_distribution<std::size_t> weight_dist(weights.begin(), weights.end());
    TreeEdge const& te = m_current_root->edges[weight_dist(m_twister)];

    if (!te.child) {
        throw std::runtime_error("No explored action available!");
    }

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

Move MCTS::commit_to_move(float temperature) {
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
            create_tree_node(std::move(board), m_current_root->turn.next(), m_current_root);
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

TreeNode* MCTS::create_tree_node(Board board, Turn turn, TreeNode* parent) {
    MCTSPolicy::Evaluation eval = m_policy->evaluate_position(board, turn, parent).get();
    TreeNode* result = new TreeNode{parent,
                                    std::move(board),
                                    turn,
                                    parent ? parent->depth + 1 : 0,
                                    TreeNode::Value{eval.value, 1},
                                    std::move(eval.edges)};

    return result;
}

folly::coro::Task<TreeNode*> MCTS::create_tree_node_async(Board board, Turn turn,
                                                          TreeNode* parent) {
    MCTSPolicy::Evaluation eval = co_await m_policy->evaluate_position(board, turn, parent);
    TreeNode* result = new TreeNode{parent,
                                    std::move(board),
                                    turn,
                                    parent ? parent->depth + 1 : 0,
                                    TreeNode::Value{eval.value, 1},
                                    std::move(eval.edges)};

    co_return result;
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

int MCTS::wasted_inferences() const {
    return m_wasted_inferences;
}
