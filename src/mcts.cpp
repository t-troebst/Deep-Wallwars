#include "mcts.hpp"

#include <folly/experimental/coro/BlockingWait.h>
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

MCTS::MCTS(EvaluationFunction evaluate, Board board)
    : MCTS{std::move(evaluate), std::move(board), {}} {}

MCTS::MCTS(EvaluationFunction evaluate, Board board, Options options)
    : m_evaluate{std::move(evaluate)},
      m_root{folly::coro::blockingWait(create_tree_node(board, options.starting_turn, nullptr))},
      m_opts{options},
      m_gamma_dist{options.direchlet_alpha, 1.0},
      m_twister{options.seed} {
    add_root_noise();
}

folly::coro::Task<float> MCTS::sample(int samples) {
    auto sample_tasks =
        views::iota(0, samples) | views::transform([&](int) { return sample_rec(*m_root); });

    co_await folly::coro::collectAllWindowed(sample_tasks, m_opts.max_parallelism);

    TreeNode::Value val = m_root->value;
    co_return val.total_weight / val.total_samples;
}

TreeEdge& MCTS::get_best_edge(TreeNode& current) const {
    return *std::ranges::max_element(current.edges, {}, [&](TreeEdge const& te) {
        TreeNode::Value root_val = current.value;  // TODO: load this only once maybe?
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

        if (current.turn.action == Turn::Second) {
            child_val.total_weight *= -1;
        }

        int const active_samples = te.active_samples;
        child_val.total_weight -= m_opts.active_sample_penalty * active_samples;
        child_val.total_samples += active_samples;

        return child_val.total_weight / child_val.total_samples +
               te.prior * p_root / (1 + child_val.total_samples);
    });
}

folly::coro::Task<float> MCTS::initialize_child(TreeNode& current, TreeEdge& edge) {
    Board next_board{current.board};
    next_board.do_action(current.turn.player, edge.action);

    TreeNode* new_node =
        co_await create_tree_node(std::move(next_board), current.turn.next(), &current);

    float value = new_node->value.load().total_weight;
    TreeNode* child = nullptr;

    if (!edge.child.compare_exchange_strong(child, new_node)) {
        ++m_wasted_inferences;
        delete new_node;
    }

    co_return value;
}

folly::coro::Task<float> MCTS::sample_rec(TreeNode& current) {
    if (auto winner = current.board.winner(); winner) {
        float value = *winner == current.turn.player ? 1 : -1;
        current.add_sample(value);
        co_return value;
    }

    if (current.depth - m_root->depth >= m_opts.max_depth) {
        float value = current.board.score_for(current.turn.player);
        current.add_sample(value);
        co_return value;
    }

    if (current.edges.empty()) {
        float value = -1;
        current.add_sample(value);
        co_return value;
    }

    TreeEdge& te = get_best_edge(current);
    ++te.active_samples;
    TreeNode* child = te.child;
    float value = co_await (child == nullptr ? initialize_child(current, te) : sample_rec(*child));

    if (current.turn.action == Turn::Second) {
        value *= -1;
    }

    current.add_sample(value);
    --te.active_samples;
    co_return value;
}

void MCTS::move_root(TreeEdge const& edge) {
    m_history.push_back(root_info());

    for (TreeEdge& te2 : m_root->edges) {
        TreeNode* child = te2.child;
        if (child && child != edge.child) {
            delete_subtree(*child);
        }
    }

    TreeNode* old_root = m_root;
    m_root = edge.child;
    delete old_root;
    add_root_noise();
}

Action MCTS::commit_to_action() {
    if (m_root->edges.empty()) {
        throw std::runtime_error("No action available!");
    }

    TreeEdge const& te = *std::ranges::max_element(m_root->edges, {}, [&](TreeEdge const& te) {
        return te.child ? te.child.load()->value.load().total_samples : 0;
    });

    if (!te.child) {
        throw std::runtime_error("No explored action available!");
    }

    Action result = te.action;
    move_root(te);  // invalidates reference to te!
    return result;
}

Action MCTS::commit_to_action(float temperature) {
    if (temperature == 0.0) {
        return commit_to_action();
    }

    if (m_root->edges.empty()) {
        throw std::runtime_error("No action available!");
    }

    auto const weights = std::ranges::views::transform(m_root->edges, [&](TreeEdge const& te) {
        return te.child ? std::pow(te.child.load()->value.load().total_samples, 1.0 / temperature)
                        : 0;
    });

    std::discrete_distribution<std::size_t> weight_dist(weights.begin(), weights.end());
    TreeEdge const& te = m_root->edges[weight_dist(m_twister)];

    if (!te.child) {
        throw std::runtime_error("No explored action available!");
    }

    Action result = te.action;
    move_root(te);  // invalidates reference to te!
    return result;
}

Move MCTS::commit_to_move() {
    Move result{commit_to_action(), {}};

    if (!m_root->board.winner()) {
        result.second = commit_to_action();
    }

    return result;
}

Move MCTS::commit_to_move(float temperature) {
    Move result{commit_to_action(temperature), {}};

    if (!m_root->board.winner()) {
        result.second = commit_to_action(temperature);
    }

    return result;
}

void MCTS::force_action(Action const& action) {
    auto const te_it = std::ranges::find_if(
        m_root->edges, [&](TreeEdge const& te) { return action == te.action; });

    if (te_it == m_root->edges.end()) {
        throw std::runtime_error("Could not find action - not legal?");
    }

    if (!te_it->child) {
        Board board = m_root->board;
        board.do_action(m_root->turn.player, action);
        te_it->child = folly::coro::blockingWait(
            create_tree_node(std::move(board), m_root->turn.next(), m_root));
    }

    move_root(*te_it);
}

void MCTS::force_move(Move const& move) {
    force_action(move.first);

    if (!m_root->board.winner()) {
        force_action(move.second);
    }
}

void MCTS::add_root_noise() {
    float total = 0.0;

    std::vector<float> samples(m_root->edges.size());

    for (float& s : samples) {
        total += (s = m_gamma_dist(m_twister));
    }

    for (std::size_t i = 0; i < samples.size(); ++i) {
        m_root->edges[i].prior = (1 - m_opts.noise_factor) * m_root->edges[i].prior +
                                 m_opts.noise_factor * samples[i] / total;
    }
}

folly::coro::Task<TreeNode*> MCTS::create_tree_node(Board board, Turn turn, TreeNode* parent) {
    Evaluation eval = co_await m_evaluate(board, turn);
    TreeNode* result = new TreeNode{parent,
                                    std::move(board),
                                    turn,
                                    parent ? parent->depth + 1 : 0,
                                    TreeNode::Value{eval.value, 1},
                                    std::move(eval.edges)};

    co_return result;
}

MCTS::~MCTS() {
    delete_subtree(*m_root);
}

void MCTS::delete_subtree(TreeNode& tn) {
    std::vector<TreeNode*> delete_stack{&tn};

    while (!delete_stack.empty()) {
        TreeNode* tn_top = delete_stack.back();
        delete_stack.pop_back();

        for (TreeEdge const& te : tn_top->edges) {
            if (te.child != nullptr) {
                delete_stack.push_back(te.child);
            }
        }

        delete tn_top;
    }
}

Board const& MCTS::current_board() const {
    return m_root->board;
}

float MCTS::root_value() const {
    TreeNode::Value val = m_root->value;
    return val.total_weight / val.total_samples;
}

int MCTS::root_samples() const {
    return m_root->value.load().total_samples;
}

NodeInfo MCTS::root_info() const {
    TreeNode::Value const val = m_root->value;
    NodeInfo result{
        m_root->board, m_root->turn, val.total_weight / val.total_samples, val.total_samples, {}};
    result.edges.reserve(m_root->edges.size());

    for (TreeEdge const& edge : m_root->edges) {
        TreeNode* child = edge.child;
        result.edges.emplace_back(edge.action, child ? child->value.load().total_samples : 0);
    }

    return result;
}

std::vector<NodeInfo> const& MCTS::history() const {
    return m_history;
}

int MCTS::wasted_inferences() const {
    return m_wasted_inferences;
}
