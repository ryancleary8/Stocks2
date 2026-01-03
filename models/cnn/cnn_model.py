# models/cnn/cnn_model.py
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN1DPredictor(nn.Module):
    """
    Input:  x (B, C, T)  where C=features, T=seq_len
    Output:
      - regression: (B,)
      - binary classification logits: (B,)
    """
    def __init__(
        self,
        in_channels: int,
        conv_channels: int = 64,
        kernel_size: int = 5,
        dropout: float = 0.15,
        task: str = "reg",
    ):
        super().__init__()
        self.task = task

        self.conv1 = nn.Conv1d(in_channels, conv_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        self.conv2 = nn.Conv1d(conv_channels, conv_channels, kernel_size=kernel_size, padding=kernel_size // 2)

        self.bn1 = nn.BatchNorm1d(conv_channels)
        self.bn2 = nn.BatchNorm1d(conv_channels)

        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(conv_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.drop(x)
        x = F.relu(self.bn2(self.conv2(x)))

        # Global pooling over time (reduces T -> 1)
        x = F.adaptive_max_pool1d(x, output_size=1).squeeze(-1)  # (B, conv_channels)
        x = self.drop(x)

        out = self.fc(x).squeeze(-1)  # (B,)
        return out