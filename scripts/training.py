import torch
import sys
import torch.onnx
from model import ResNet

device = torch.device("cuda:0")

columns = 6
rows = 6
channels = 64
layers = 10
epochs = 10
batch_size = 64

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


# model = ResNet(columns, rows, channels, layers)
model = torch.load(f"{models_folder}/model_{generation}.pt")
model.to(device)

snapshots = torch.utils.data.ConcatDataset([Snapshots(f"{data_folder}/snapshots_{i}.csv") for i in range(1,
                                           generation + 1)])
training_data, eval_data = torch.utils.data.random_split(snapshots, [0.8, 0.2])
training_loader  = torch.utils.data.DataLoader(training_data, batch_size = 64, shuffle = True, num_workers
                                               = 4, pin_memory = True)

eval_loader  = torch.utils.data.DataLoader(eval_data, batch_size = 64, num_workers
                                               = 4, pin_memory = True, shuffle = False)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

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

    model.train(False)
    eval_loss = 0
    for states, wall_priors, step_priors, values in eval_loader:
        states = states.to(device)
        wall_priors = wall_priors.to(device)
        step_priors = step_priors.to(device)
        values = values.to(device)
        wp, sp, vs = model.forward(states)
        eval_loss += loss_fn(wp, sp, vs, wall_priors, step_priors, values)
    print(f"Loss in epoch {epoch}: {eval_loss}.")

# Save model in pytorch and onnx formats
torch.save(model, f"{models_folder}/model_{generation + 1}.pt")
input_names = ["States"]
output_names = ["WallPriors", "StepPriors", "Values"]
dummy_input = torch.randn(batch_size, 7, columns, rows).to(device)
torch.onnx.export(model, dummy_input, f"{models_folder}/model_{generation + 1}.onnx", input_names = input_names, output_names =
                  output_names)
