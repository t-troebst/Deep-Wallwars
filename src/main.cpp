#include <iostream>

#include "dnn_policy.hpp"
#include "play.hpp"
#include "simple_policy.hpp"

int main() {
    auto dnn = std::make_shared<DNN>(5, 5);
    // torch::load(dnn, "model.pt");
    dnn->to(torch::kCUDA);

    auto dnn_policy = std::make_unique<DNNPolicy>(*dnn);
    Board board{5, 5, {0, 0}, {4, 4}, {4, 0}, {0, 4}};
    Dataset data{10000};

    while (true) {
        append_examples_parallel(data, board, *dnn, 100, 50);

        train(*dnn, data, 10, 64, 0.1);
        torch::save(dnn, "model.pt");
    }
}
