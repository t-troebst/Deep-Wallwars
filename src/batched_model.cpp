#include "batched_model.hpp"

#include <algorithm>
#include <ranges>

#include "model.hpp"

constexpr int kDefaultBatchesInQueue = 16;
constexpr int kNumDirections = 4;

BatchedModel::BatchedModel(std::unique_ptr<Model> model)
    : BatchedModel{std::move(model), kDefaultBatchesInQueue * model->batch_size()} {}

BatchedModel::BatchedModel(std::unique_ptr<Model> model, int queue_size)
    : m_model{std::move(model)},
      m_tasks(queue_size),
      m_worker{std::bind_front(&BatchedModel::run_worker, this)} {}

BatchedModel::~BatchedModel() {
    // Sentinel value so the worker stops (eventually)
    m_tasks.blockingWrite(InferenceTask{});
}

folly::SemiFuture<BatchedModel::Output> BatchedModel::inference(std::vector<float> states) {
    InferenceTask task{std::move(states), {}};
    auto result = task.output.getSemiFuture();

    m_tasks.blockingWrite(std::move(task));

    return result;
}

void BatchedModel::run_worker() {
    std::vector<folly::Promise<Output>> dequeued_promises;
    std::vector<float> states(m_model->batch_size() * m_model->state_size());
    std::vector<float> wall_priors(m_model->batch_size() * m_model->wall_prior_size());
    std::vector<float> step_priors(m_model->batch_size() * kNumDirections);
    std::vector<float> values(m_model->batch_size());

    while (true) {
        for (int i = 0; i < m_model->batch_size(); ++i) {
            InferenceTask task;

            if (!m_tasks.read(task)) {
                break;
            }

            if (task.state.empty()) {
                return;
            }

            std::ranges::copy(task.state, states.begin() + m_model->state_size() * i);
            dequeued_promises.push_back(std::move(task.output));
        }

        m_model->inference(states, {wall_priors, step_priors, values});

        for (std::size_t i = 0; i < dequeued_promises.size(); ++i) {
            Output output{{wall_priors.begin() + m_model->wall_prior_size() * i,
                           wall_priors.begin() + m_model->wall_prior_size() * (i + 1)},
                          {step_priors.begin() + kNumDirections * i,
                           step_priors.begin() + kNumDirections * (i + 1)},
                          values[i]};

            dequeued_promises[i].setValue(std::move(output));
        }

        dequeued_promises.clear();
    }
}
