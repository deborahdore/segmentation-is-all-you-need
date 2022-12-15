import torch
from torch import nn


# FINAL HYPERTUNED MODEL
class ConvNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(
            # input shape (128, 3, 240, 135)
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1),
            # input shape (128, 32, 240, 135)
            nn.ReLU(),
            # input shape (128, 32, 240, 135)
            nn.MaxPool2d(kernel_size=2)
        )

        self.layer2 = nn.Sequential(
            # input shape (128, 32, 120, 67)
            nn.Conv2d(in_channels=16, out_channels=128, kernel_size=3, padding=1),
            # input shape (128, 64, 120, 67)
            nn.ReLU(),
            # input shape (128, 64, 120, 67)
            nn.MaxPool2d(kernel_size=2),
            # input shape (128, 64, 60, 33)
            nn.Dropout(0.5)
        )

        self.flatten = nn.Flatten()

        self.fc = nn.Sequential(
            nn.Linear(in_features=128 * 60 * 33, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=10),
        )

    def forward(self, input):
        x = self.layer1(input)
        x = self.layer2(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x
