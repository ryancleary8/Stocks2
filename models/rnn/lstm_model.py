# models/rnn/lstm_model.py
from __future__ import annotations

import torch
import torch.nn as nn


class StockLSTM(nn.Module):
    """
    Input:  (B, T, F)
    Output: (B,) regression value OR (B,) logits for binary classification
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 1,
        dropout: float = 0.20,
        task: str = "reg",  # "reg" or "cls"
    ):
        super().__init__()
        self.task = task

        # dropout only applies between layers when num_layers>1 in PyTorch
        lstm_dropout = dropout if num_layers > 1 else 0.0

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=lstm_dropout,
        )

        self.out_drop = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, F)
        out, _ = self.lstm(x)          # out: (B, T, H)
        last = out[:, -1, :]           # last time step hidden
        last = self.out_drop(last)
        y = self.fc(last).squeeze(-1)  # (B,)
        return y