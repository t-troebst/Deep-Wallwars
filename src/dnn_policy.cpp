#include "dnn_policy.hpp"

#include <torch/torch.h>

#include <exception>
#include <execution>
#include <iostream>
#include <random>

#include "simple_policy.hpp"
#include "util.hpp"

DNN::DNN(int width, int height)
    : width{width},
      height{height},
      conv1{register_module("conv1",
                            torch::nn::Conv2d(torch::nn::Conv2dOptions{5, 16, 3}.padding(1)))},
      conv2{register_module("conv2",
                            torch::nn::Conv2d(torch::nn::Conv2dOptions{16, 16, 3}.padding(1)))},
      conv3{register_module("conv3",
                            torch::nn::Conv2d(torch::nn::Conv2dOptions{16, 16, 3}.padding(1)))},
      conv4{register_module("conv4",
                            torch::nn::Conv2d(torch::nn::Conv2dOptions{16, 16, 3}.padding(1)))},

      bn1{register_module("bn1", torch::nn::BatchNorm2d(16))},
      bn2{register_module("bn2", torch::nn::BatchNorm2d(16))},
      bn3{register_module("bn3", torch::nn::BatchNorm2d(16))},
      bn4{register_module("bn4", torch::nn::BatchNorm2d(16))},

      lin1{register_module("lin1", torch::nn::Linear(16 * width * height, 1024))},
      lin2{register_module("lin2", torch::nn::Linear(1024, 512))},
      lin3{register_module("lin3", torch::nn::Linear(512, width * height))},
      lin4{register_module("lin4", torch::nn::Linear(512, width * height))},
      lin5{register_module("lin5", torch::nn::Linear(512, 4))},
      lin6{register_module("lin6", torch::nn::Linear(512, 1))} {}

DNN::Result DNN::forward(torch::Tensor x) {
    x = x.view({-1, 5, width, height});
    x = torch::relu(bn1->forward(conv1->forward(x)));
    x = torch::relu(bn2->forward(conv2->forward(x)));
    x = torch::relu(bn3->forward(conv3->forward(x)));
    x = torch::relu(bn4->forward(conv4->forward(x)));
    x = x.view({-1, 16 * width * height});
    x = torch::dropout(torch::relu(lin1->forward(x)), 0.3, is_training());
    x = torch::dropout(torch::relu(lin2->forward(x)), 0.3, is_training());

    Result result;

    result.up_walls = torch::sigmoid(lin3->forward(x));
    result.right_walls = torch::sigmoid(lin4->forward(x));
    result.steps = torch::sigmoid(lin5->forward(x));
    result.value = torch::tanh(lin6->forward(x));

    return result;
}

torch::Tensor board_to_tensor(Board const& board, Turn turn, torch::Device device) {
    torch::Tensor result =
        torch::zeros({5, board.width(), board.height()}, at::TensorOptions(device));

    Cell player_position = board.position(turn.player);
    Cell opponent_position =
        board.position(turn.player == Player::Red ? Player::Blue : Player::Red);

    result.index_put_({0, player_position.x, player_position.y}, 1);
    result.index_put_({1, opponent_position.x, opponent_position.y}, 1);

    for (int i = 0; i < board.width(); ++i) {
        for (int j = 0; j < board.height(); ++j) {
            if (board.is_blocked({{i, j}, Direction::Right})) {
                result.index_put_({2, i, j}, 1);
            }

            if (board.is_blocked({{i, j}, Direction::Up})) {
                result.index_put_({3, i, j}, 1);
            }
        }
    }

    if (turn.Second) {
        result.index_put_({torch::indexing::Ellipsis, 4}, 1);
    }

    return result;
}

DNNPolicy::DNNPolicy(DNN dnn) : dnn{std::move(dnn)} {}

std::unique_ptr<MCTSPolicy> DNNPolicy::clone() const {
    return std::make_unique<DNNPolicy>(*this);
}

void DNNPolicy::evaluate(TreeNode const& node, Turn turn) {
    if (node.board.width() != dnn.width || node.board.height() != dnn.height) {
        throw std::runtime_error("Incorrect board dimensions for DNN!");
    }

    torch::NoGradGuard guard;
    torch::Tensor board_state =
        board_to_tensor(node.board, turn, dnn.parameters().front().device());
    result = dnn.forward(board_state.view({1, 5, node.board.width(), node.board.height()}));

    result.up_walls = result.up_walls.view({node.board.width() * node.board.height()});
    result.right_walls = result.right_walls.view({node.board.width() * node.board.height()});
    result.steps = result.steps.view({4});
    result.value = result.value.view({1});
}

double DNNPolicy::step_prior(Direction dir) const {
    return result.steps[int(dir)].item<double>();
}

double DNNPolicy::wall_prior(Wall wall) const {
    wall = wall.normalize();

    if (wall.direction == Direction::Up) {
        return result.up_walls[wall.cell.x * dnn.height + wall.cell.y].item<double>();
    }

    return result.right_walls[wall.cell.x * dnn.height + wall.cell.y].item<double>();
}

std::optional<double> DNNPolicy::value() const {
    return result.value.item<double>();
}

torch::data::Example<> node_to_example(TreeNode const& tn, Turn turn) {
    torch::Tensor board_tensor = board_to_tensor(tn.board, turn, torch::kCPU);
    torch::Tensor output = torch::zeros({2 * tn.board.width() * tn.board.height() + 5});

    double total_weight = 0.0;

    for (NodeInfo const& ni : tn.children) {
        total_weight += ni.cumulative_weight;
    }

    for (NodeInfo const& ni : tn.children) {
        std::visit(
            overload{[&](Direction dir) {
                         output.index_put_({2 * tn.board.width() * tn.board.height() + int(dir)},
                                           double(ni.num_samples) / tn.total_samples);
                     },
                     [&](Wall wall) {
                         if (wall.direction == Direction::Up) {
                             output.index_put_({wall.cell.x * tn.board.height() + wall.cell.y},
                                               double(ni.num_samples) / tn.total_samples);
                         } else {
                             output.index_put_({tn.board.width() * tn.board.height() +
                                                wall.cell.x * tn.board.height() + wall.cell.y},
                                               double(ni.num_samples) / tn.total_samples);
                         }
                     }},
            ni.move);
    }

    output.index_put_({2 * tn.board.width() * tn.board.height() + 4},
                      total_weight / tn.total_samples);

    return {board_tensor, output};
}

DNN::Result labels_to_result(int width, int height, torch::Tensor labels) {
    using namespace torch::indexing;
    DNN::Result result;

    result.up_walls = labels.index({Ellipsis, Slice{0, width * height}}).view({-1, width * height});
    result.right_walls = labels.index({Ellipsis, Slice{width * height, 2 * width * height}})
                             .view({-1, width * height});
    result.steps =
        labels.index({Ellipsis, Slice{2 * width * height, 2 * width * height + 4}}).view({-1, 4});
    result.value = labels.index({Ellipsis, Slice{2 * width * height + 4, 2 * width * height + 5}})
                       .view({-1, 1});

    return result;
}

torch::data::Example<> Dataset::get(std::size_t index) {
    return examples[index];
}

torch::optional<std::size_t> Dataset::size() const {
    return examples.size();
}

Dataset::Dataset(std::size_t limit) : limit{limit} {}

std::vector<torch::data::Example<>> generate_examples(Board const& board, DNN const& dnn,
                                                      int max_iters) {
    std::vector<torch::data::Example<>> result;

    MCTS mcts_1{board, {Player::Red, Turn::First}, std::make_unique<DNNPolicy>(dnn)};
    MCTS mcts_2{board, {Player::Red, Turn::First}, std::make_unique<DNNPolicy>(dnn)};

    std::mt19937_64 twister(42);

    for (int k = 0; k < max_iters; ++k) {
        std::cout << "Iteration " << k << '\n';
        for (int i = 0; i < 2; ++i) {
            mcts_1.sample(250);
            result.push_back(node_to_example(mcts_1.current_node(), mcts_1.current_turn()));

            Move const m1 = mcts_1.commit_to_move(twister, 0.2);
            std::cout << Player::Red << " does " << m1 << '\n';

            if (mcts_1.current_board().winner()) {
                return result;
            }

            mcts_2.force_move(m1);
        }

        for (int i = 0; i < 2; ++i) {
            mcts_2.sample(250);
            result.push_back(node_to_example(mcts_2.current_node(), mcts_2.current_turn()));

            Move const m2 = mcts_2.commit_to_move(twister, 0.2);
            std::cout << Player::Blue << " does " << m2 << '\n';

            if (mcts_2.current_board().winner()) {
                return result;
            }

            mcts_1.force_move(m2);
        }
    }

    return result;
}

void append_examples_parallel(Dataset& dataset, Board const& board, const DNN& dnn, int games,
                              int max_iters) {
    std::vector<std::vector<torch::data::Example<>>> examples(games);

    std::for_each(std::execution::par, examples.begin(), examples.end(),
                  [&, dnn = dnn](auto& result) mutable {
                      result = generate_examples(board, dnn, max_iters);
                  });

    for (auto const& vec : examples) {
        dataset.examples.insert(dataset.examples.end(), vec.begin(), vec.end());
    }

    while (dataset.examples.size() >= dataset.limit) {
        dataset.examples.pop_front();
    }
}

void train(DNN& dnn, Dataset& data, int epochs, int batch_size, double learning_rate) {
    auto loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
        data.map(torch::data::transforms::Stack<>()), batch_size);

    torch::optim::SGD optimizer(dnn.parameters(), learning_rate);
    torch::Device device = dnn.parameters().front().device();

    double overall_loss = 0.0;

    for (int i = 0; i < epochs; ++i) {
        for (auto& batch : *loader) {
            optimizer.zero_grad();
            DNN::Result result = dnn.forward(batch.data.to(device));
            DNN::Result labels = labels_to_result(dnn.width, dnn.height, batch.target.to(device));

            torch::Tensor loss_1 = torch::cross_entropy_loss(result.up_walls, labels.up_walls);
            torch::Tensor loss_2 =
                torch::cross_entropy_loss(result.right_walls, labels.right_walls);
            torch::Tensor loss_3 = torch::cross_entropy_loss(result.steps, labels.steps);
            torch::Tensor loss_4 = torch::mse_loss(result.value, labels.value);
            torch::Tensor total_loss = loss_1 + loss_2 + loss_3 + loss_4;

            overall_loss += total_loss.item<double>();
            total_loss.backward();
            optimizer.step();
        }
    }

    std::cout << "Current loss: " << overall_loss << '\n';
}
