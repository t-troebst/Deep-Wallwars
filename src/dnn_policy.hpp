#pragma once

#include <torch/torch.h>

#include <deque>
#include <memory>
#include <mutex>

#include "mcts.hpp"

struct DNN : torch::nn::Module {
    int width, height;

    torch::nn::Conv2d conv1, conv2, conv3, conv4;
    torch::nn::BatchNorm2d bn1, bn2, bn3, bn4;
    torch::nn::Linear lin1, lin2, lin3, lin4, lin5, lin6;

    DNN(int width, int height);

    struct Result {
        torch::Tensor up_walls;
        torch::Tensor right_walls;
        torch::Tensor steps;
        torch::Tensor value;
    };

    Result forward(torch::Tensor x);
};

struct DNNPolicy : MCTSPolicy {
    DNN dnn;
    DNN::Result result;

    DNNPolicy(DNN dnn);

    void evaluate(TreeNode const& node, Turn turn) override;

    double step_prior(Direction dir) const override;
    double wall_prior(Wall wall) const override;

    std::optional<double> value() const override;

    std::unique_ptr<MCTSPolicy> clone() const override;
};

torch::Tensor board_to_tensor(Board const& board, Turn turn, torch::Device device);

struct Dataset : torch::data::Dataset<Dataset> {
    std::size_t limit;
    std::deque<torch::data::Example<>> examples;

    Dataset(std::size_t limit);

    torch::data::Example<> get(std::size_t index) override;
    torch::optional<std::size_t> size() const override;
};

std::vector<torch::data::Example<>> generate_examples(Board const& board, DNN const& dnn, int max_iters);

void append_examples_parallel(Dataset& dataset, Board const& board, DNN const& dnn, int games,
                              int max_iters);

void train(DNN& dnn, Dataset& data, int epochs, int batch_size, double learning_rate);
