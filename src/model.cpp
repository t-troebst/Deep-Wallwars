#include "model.hpp"

Model::Model(int batch_size, int wall_prior_size, int state_size)
    : m_batch_size{batch_size}, m_wall_prior_size{wall_prior_size}, m_state_size{state_size} {}

int Model::batch_size() const {
    return m_batch_size;
};

int Model::state_size() const {
    return m_state_size;
}

int Model::wall_prior_size() const {
    return m_wall_prior_size;
}
