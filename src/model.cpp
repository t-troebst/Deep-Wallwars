#include "model.hpp"

Model::Model(int batch_size, int channels, int columns, int rows)
    : m_batch_size{batch_size},
      m_state_size{columns * rows * channels},
      m_wall_prior_size{2 * columns * rows} {}

int Model::batch_size() const {
    return m_batch_size;
};

int Model::state_size() const {
    return m_state_size;
}

int Model::wall_prior_size() const {
    return m_wall_prior_size;
}
