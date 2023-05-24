import torch
import torch.nn as nn
import torch.onnx
import torch.nn.functional as fn

class ResNet(nn.Module):
    def __init__(self, columns, rows, hidden_channels, layers):
        super().__init__()

        self.start = nn.Sequential(
            nn.Conv2d(7, hidden_channels, kernel_size = 3, padding = 1, bias = False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU()
        )

        self.layers = nn.ModuleList([ResLayer(hidden_channels) for _ in range(layers)])

        self.priors = nn.Sequential(
            nn.Conv2d(hidden_channels, 32, kernel_size = 3, padding = 1, bias = False),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )

        self.wall_priors = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * columns * rows, 2 * columns * rows),
            nn.Sigmoid(),
            # TODO: not supported...
            # nn.Unflatten(1, [2, columns, rows])
        )

        self.step_priors = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * columns * rows, 4),
            nn.Sigmoid()
        )

        self.value = nn.Sequential(
            nn.Conv2d(hidden_channels, 32, kernel_size = 3, padding = 1, bias = False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * columns * rows, 1),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.start.forward(x)

        for layer in self.layers:
            x = layer.forward(x)

        priors = self.priors.forward(x)
        wall_priors = self.wall_priors.forward(priors)
        step_priors = self.step_priors.forward(priors)
        value = self.value.forward(x)

        return wall_priors, step_priors, value

class ResLayer(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size = 3, padding = 1, bias = False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size = 3, padding = 1, bias = False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        x = fn.relu(self.bn1.forward(self.conv1.forward(x)))
        x = fn.relu(residual + self.bn2.forward(self.conv2.forward(x)))
        return x
        
def main():
    columns = 6
    rows = 6
    channels = 64
    layers = 10
    batch_size = 64

    input_names = ["States"]
    output_names = ["WallPriors", "StepPriors", "Values"]

    model = ResNet(columns, rows, channels, layers).eval()
    dummy_input = torch.randn(batch_size, 7, columns, rows)
    torch.onnx.export(model, dummy_input, "model.onnx", input_names = input_names, output_names =
                      output_names)

if __name__ == "__main__":
    main()
