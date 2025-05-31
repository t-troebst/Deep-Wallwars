#include "batched_model.hpp"

#include <algorithm>

#include "cuda_wrappers.hpp"
#include "model.hpp"

constexpr int kDefaultBatchesInQueue = 16;

BatchedModel::BatchedModel(std::unique_ptr<Model> model)
    : BatchedModel{std::move(model), kDefaultBatchesInQueue * model->batch_size()} {}

BatchedModel::BatchedModel(std::unique_ptr<Model> model, int queue_size) : m_tasks(queue_size) {
    m_models.push_back(std::move(model));
    m_workers.emplace_back([&] { run_worker(0); });
}

BatchedModel::BatchedModel(std::vector<std::unique_ptr<Model>> models, int queue_size)
    : m_tasks(queue_size), m_models{std::move(models)} {
    for (std::size_t i = 0; i < m_models.size(); ++i) {
        m_workers.emplace_back([this, i] { run_worker(i); });
    }
}

BatchedModel::~BatchedModel() {
    // Sentinel values so the workers stop (eventually)
    for (std::size_t i = 0; i < m_workers.size(); ++i) {
        m_tasks.blockingWrite(InferenceTask{});
    }
}

folly::SemiFuture<ModelOutput> BatchedModel::inference(std::vector<float> states) {
    InferenceTask task{std::move(states), {}};
    auto result = task.output.getSemiFuture();

    m_tasks.blockingWrite(std::move(task));

    return result;
}

std::size_t BatchedModel::total_inferences() const {
    return m_inferences;
}

std::size_t BatchedModel::total_batches() const {
    return m_batches;
}

void BatchedModel::run_worker(std::size_t idx) {
    std::vector<folly::Promise<ModelOutput>> dequeued_promises;

    // This substantially improves GPU memory transfer times but feels a bit hacky to use
    // cuda-specific code here...
    PinnedBuffer<float> states(m_models[idx]->batch_size() * m_models[idx]->state_size());
    PinnedBuffer<float> priors(m_models[idx]->batch_size() *
                               (m_models[idx]->wall_prior_size() + 4));
    PinnedBuffer<float> values(m_models[idx]->batch_size());

    while (true) {
        for (int i = 0; i < m_models[idx]->batch_size(); ++i) {
            InferenceTask task;

            if (i == 0) {
                m_tasks.blockingRead(task);
            } else if (!m_tasks.read(task)) {
                break;
            }

            if (task.state.empty()) {
                return;
            }

            std::ranges::copy(task.state, states.data() + m_models[idx]->state_size() * i);
            dequeued_promises.push_back(std::move(task.output));
        }

        m_models[idx]->inference(states, {priors, values});

        for (std::size_t i = 0; i < dequeued_promises.size(); ++i) {
            std::vector<float> prior{
                priors.data() + (m_models[idx]->wall_prior_size() + 4) * i,
                priors.data() + (m_models[idx]->wall_prior_size() + 4) * (i + 1)};

            dequeued_promises[i].setValue(ModelOutput{std::move(prior), values[i]});
        }

        m_batches += 1;
        m_inferences += dequeued_promises.size();

        dequeued_promises.clear();
    }
}
