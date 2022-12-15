# It's a convolutional neural network with three convolutional layers,
# followed by a fully connected layer
import torch
from torch import nn


class ConvNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(
            # input shape (128, 3, 240, 135)
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            # input shape (128, 32, 240, 135)
            nn.ReLU(),
            # input shape (128, 32, 240, 135)
            nn.MaxPool2d(kernel_size=2)
        )

        self.layer2 = nn.Sequential(
            # input shape (128, 32, 120, 67)
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            # input shape (128, 64, 120, 67)
            nn.ReLU(),
            # input shape (128, 64, 120, 67)
            nn.MaxPool2d(kernel_size=2),
            # input shape (128, 64, 60, 33)
        )

        self.layer3 = nn.Sequential(
            # input shape (128, 64, 60, 33)
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            # input shape (128, 64, 60, 33)
            nn.ReLU(),
            # input shape (128, 64, 60, 33)
            nn.MaxPool2d(kernel_size=2),
            # input shape (128, 64, 30, 16)
            nn.Dropout()
        )

        self.flatten = nn.Flatten()

        self.fc = nn.Sequential(
            nn.Linear(in_features=64 * 30 * 16, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=10),
        )

    def forward(self, input):
        x = self.layer1(input)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x
