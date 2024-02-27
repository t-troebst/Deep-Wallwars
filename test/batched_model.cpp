#include "batched_model.hpp"

#include <catch2/catch_test_macros.hpp>

#include "model.hpp"

struct MockModel : Model {
    MockModel(int batch_size, int channels, int columns, int rows)
        : Model{batch_size, channels, columns, rows} {}

    void inference(std::span<float> states, Output const& out) override {
        for (int i = 0; i < m_batch_size; ++i) {
            out.priors[m_wall_prior_size * i] = states[m_state_size * i];
            out.values[i] = i;
        }
    }
};

TEST_CASE("Initialization", "[Batched Model]") {
    auto model = std::make_unique<MockModel>(4, 1, 3, 6);
    // If BatchedModel is on the stack, the 128 byte alignment is not respected...
    // Compiler bug?
    auto bm = std::make_unique<BatchedModel>(std::move(model), 12);
}
