import torch
import sys
import torch.onnx
import torch.utils
import torch.utils.data
from model import ResNet

device = torch.device("cuda:0")

columns = 6
rows = 6
channels = 64
layers = 10
epochs = 2
training_batch_size = 64
inference_batch_size = 256

data_folder = sys.argv[1]
models_folder = sys.argv[2]
generation = int(sys.argv[3])


class Snapshots(torch.utils.data.Dataset):
    def __init__(self, file_name):
        self.data = [[], [], [], []]
        i = 0
        with open(file_name) as f:
            for line in f.readlines():
                if line.strip() == "":
                    i = 0
                    continue

                t = torch.tensor([float(x) for x in line.split(", ")])

                if i == 0:
                    t = t.view(7, columns, rows)
                self.data[i].append(t)
                i += 1

    def __len__(self):
        return len(self.data[0])

    def __getitem__(self, index):
        return [self.data[x][index] for x in range(4)]


def loss_fn(wp_out, sp_out, vs_out, wp_label, sp_label, vs_label):
    cel = torch.nn.CrossEntropyLoss()
    mse = torch.nn.MSELoss()
    return cel(wp_out, wp_label) + cel(sp_out, sp_label) + mse(vs_out, vs_label)


def save_model(model, folder):
    torch.save(model, f"{folder}/model_{generation}.pt")
    input_names = ["States"]
    output_names = ["WallPriors", "StepPriors", "Values"]
    dummy_input = torch.randn(inference_batch_size, 7, columns, rows).to(device)
    torch.onnx.export(
        model,
        dummy_input,
        f"{folder}/model_{generation}.onnx",
        input_names=input_names,
        output_names=output_names,
    )


if generation == 0:
    model = ResNet(columns, rows, channels, layers).to(device)
    save_model(model, models_folder)
    exit()
else:
    model = torch.load(f"{models_folder}/model_{generation - 1}.pt").to(device)


training_window = range((generation - 1) // 2, generation)
snapshots = torch.utils.data.ConcatDataset(
    [Snapshots(f"{data_folder}/snapshots_{i}.csv") for i in training_window]
)
training_data, eval_data = torch.utils.data.random_split(snapshots, [0.8, 0.2])
training_loader = torch.utils.data.DataLoader(
    training_data,
    batch_size=training_batch_size,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
)
eval_loader = torch.utils.data.DataLoader(
    eval_data,
    batch_size=training_batch_size,
    num_workers=4,
    pin_memory=True,
    shuffle=False,
)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.02)

for epoch in range(epochs):
    for states, wall_priors, step_priors, values in training_loader:
        states = states.to(device)
        wall_priors = wall_priors.to(device)
        step_priors = step_priors.to(device)
        values = values.to(device)

        optimizer.zero_grad()
        wp, sp, vs = model.forward(states)
        loss = loss_fn(wp, sp, vs, wall_priors, step_priors, values)

        loss.backward()
        optimizer.step()
        del loss

    model.train(False)
    eval_loss = 0
    for states, wall_priors, step_priors, values in eval_loader:
        states = states.to(device)
        wall_priors = wall_priors.to(device)
        step_priors = step_priors.to(device)
        values = values.to(device)
        wp, sp, vs = model.forward(states)
        eval_loss += float(loss_fn(wp, sp, vs, wall_priors, step_priors, values))
    print(
        f"Average loss in epoch {epoch} of generation {generation}: {eval_loss / len(eval_loader)}."
    )

save_model(model, models_folder)
