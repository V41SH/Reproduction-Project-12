import torch
import torch.nn as nn
import torch.nn.functional as F

class ComparableCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            # Layer 1: 24 channels, 9x9
            nn.Conv2d(1, 24, kernel_size=9, padding=4),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True),

            # Layer 2: 32 channels, 7x7
            nn.Conv2d(24, 32, kernel_size=7, padding=3),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 14x14

            # Layer 3 & 4: 36 channels, 7x7
            nn.Conv2d(32, 36, kernel_size=7, padding=3),
            nn.BatchNorm2d(36),
            nn.ReLU(inplace=True),
            nn.Conv2d(36, 36, kernel_size=7, padding=3),
            nn.BatchNorm2d(36),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 7x7

            # Layer 5 & 6: 64 and 96 channels
            nn.Conv2d(36, 64, kernel_size=7, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 96, kernel_size=5, padding=2),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),

            # Global average pooling (7x7 → 1x1)
            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(96, 96),
            nn.ReLU(),
            nn.Linear(96, 96),
            nn.ReLU(),
            nn.Linear(96, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
