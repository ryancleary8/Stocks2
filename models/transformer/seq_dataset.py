# models/transformer/seq_dataset.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, List, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import StandardScaler


@dataclass
class SplitConfig:
    train_ratio: float = 0.75
    val_ratio: float = 0.10
    test_ratio: float = 0.15


class SequenceDataset(Dataset):
    """
    Builds sliding windows for seq->one prediction.
    Each item:
      X: (seq_len, n_features)
      y: scalar
    """
    def __init__(self, X: np.ndarray, y: np.ndarray, seq_len: int):
        assert len(X) == len(y)
        self.X = X
        self.y = y
        self.seq_len = seq_len

        if len(X) < seq_len + 1:
            raise ValueError(f"Not enough rows ({len(X)}) for seq_len={seq_len}")

    def __len__(self):
        return len(self.X) - self.seq_len + 1

    def __getitem__(self, idx: int):
        X_seq = self.X[idx : idx + self.seq_len]
        y_t = self.y[idx + self.seq_len - 1]
        return torch.tensor(X_seq, dtype=torch.float32), torch.tensor(y_t, dtype=torch.float32)


def load_dataset_csv(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "date" not in df.columns or "y" not in df.columns:
        raise ValueError("Dataset must contain 'date' and 'y' columns")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    return df


def infer_feature_cols(df: pd.DataFrame) -> List[str]:
    # everything except date and y
    cols = [c for c in df.columns if c not in ("date", "y")]
    return cols


def time_split_indices(n: int, split: SplitConfig) -> Tuple[slice, slice, slice]:
    tr = int(n * split.train_ratio)
    va = int(n * (split.train_ratio + split.val_ratio))
    # remainder is test
    return slice(0, tr), slice(tr, va), slice(va, n)


def build_dataloaders(
    dataset_csv: str | Path,
    seq_len: int = 30,
    batch_size: int = 64,
    split: SplitConfig = SplitConfig(),
    task: str = "reg",  # "reg" or "cls"
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader, DataLoader, StandardScaler, List[str]]:
    """
    Returns train/val/test DataLoaders + fitted scaler + feature_cols.
    Scaling is fit on TRAIN ONLY and applied to val/test.
    """
    df = load_dataset_csv(dataset_csv)
    feature_cols = infer_feature_cols(df)

    X = df[feature_cols].to_numpy(dtype=np.float32)
    y = df["y"].to_numpy(dtype=np.float32)

    # For classification we store y as float32 but it should be 0/1.
    if task == "cls":
        # enforce 0/1
        y = (y > 0.5).astype(np.float32)

    # Split BEFORE scaling
    tr_sl, va_sl, te_sl = time_split_indices(len(df), split)

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X[tr_sl])
    X_va = scaler.transform(X[va_sl])
    X_te = scaler.transform(X[te_sl])

    y_tr = y[tr_sl]
    y_va = y[va_sl]
    y_te = y[te_sl]

    ds_tr = SequenceDataset(X_tr, y_tr, seq_len=seq_len)
    ds_va = SequenceDataset(X_va, y_va, seq_len=seq_len)
    ds_te = SequenceDataset(X_te, y_te, seq_len=seq_len)

    dl_tr = DataLoader(ds_tr, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=False)
    dl_va = DataLoader(ds_va, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=False)
    dl_te = DataLoader(ds_te, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=False)

    return dl_tr, dl_va, dl_te, scaler, feature_cols