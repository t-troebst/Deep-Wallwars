import torch.nn as nn
import torch.nn.functional as fn


class ResNet(nn.Module):
    def __init__(self, columns, rows, hidden_channels, layers):
        super().__init__()

        self.start = nn.Sequential(
            nn.Conv2d(7, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(),
        )

        self.layers = nn.ModuleList([ResLayer(hidden_channels) for _ in range(layers)])

        self.priors = nn.Sequential(
            nn.Conv2d(hidden_channels, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )

        self.priors = nn.Sequential(
            nn.Flatten(), nn.Linear(32 * columns * rows, 2 * columns * rows + 4)
        )

        self.log_output = True

        self.value = nn.Sequential(
            nn.Conv2d(hidden_channels, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * columns * rows, 1),
            nn.Tanh(),
        )

    def forward(self, x):
        x = self.start.forward(x)

        for layer in self.layers:
            x = layer.forward(x)

        priors = self.priors.forward(x)
        if self.log_output:
            priors = fn.log_softmax(priors, dim=1)
        else:
            priors = fn.softmax(priors, dim=1)

        value = self.value.forward(x)

        return priors, value


class ResLayer(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        x = fn.relu(self.bn1.forward(self.conv1.forward(x)))
        x = fn.relu(residual + self.bn2.forward(self.conv2.forward(x)))
        return x
