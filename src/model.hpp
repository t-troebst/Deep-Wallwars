#pragma once

#include <span>
#include <vector>

class Model {
public:
    struct Output {
        std::span<float> wall_priors;
        std::span<float> step_priors;
        std::span<float> values;
    };

    virtual void inference(std::span<float> states, Output const& out) = 0;

    int batch_size() const;
    int state_size() const;
    int wall_prior_size() const;

    Model(Model const& other) = delete;
    Model(Model&& other) = delete;

    Model& operator=(Model const& other) = delete;
    Model& operator=(Model&& other) = delete;

    virtual ~Model() = default;

protected:
    int m_batch_size;
    int m_wall_prior_size;
    int m_state_size;

    Model() = default;
    Model(int batch_size, int wall_prior_size, int state_size);
};