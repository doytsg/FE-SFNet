import torch
import torch.nn as nn


class WDCNN(nn.Module):
    """
    Wide Deep Convolutional Neural Network for Bearing Fault Diagnosis.

    Input : (B, 1, 2048)
    Output: (B, num_classes)
    """
    def __init__(self, num_classes: int = 10, use_dropout: bool = True, dropout: float = 0.5):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=64, stride=16, padding=24),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),

            nn.Conv1d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),

            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),

            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),

            nn.Conv1d(64, 64, kernel_size=3),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )

        if use_dropout:
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(192, 100),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(100, num_classes),
            )
        else:
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(192, 100),
                nn.ReLU(inplace=True),
                nn.Linear(100, num_classes),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)
