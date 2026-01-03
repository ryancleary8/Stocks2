# models/cnn/seq_dataset.py
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


class CNNSequenceDataset(Dataset):
    """
    Outputs:
      X: (channels, seq_len)  for Conv1d
      y: scalar
    """
    def __init__(self, X_cnn: np.ndarray, y: np.ndarray):
        # X_cnn: (samples, channels, seq_len)
        assert len(X_cnn) == len(y)
        self.X = X_cnn
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx: int):
        return (
            torch.tensor(self.X[idx], dtype=torch.float32),
            torch.tensor(self.y[idx], dtype=torch.float32),
        )


def load_dataset_csv(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "date" not in df.columns or "y" not in df.columns:
        raise ValueError("Dataset must contain 'date' and 'y' columns")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    return df


def infer_feature_cols(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if c not in ("date", "y")]


def time_split_indices(n: int, split: SplitConfig) -> Tuple[slice, slice, slice]:
    tr = int(n * split.train_ratio)
    va = int(n * (split.train_ratio + split.val_ratio))
    return slice(0, tr), slice(tr, va), slice(va, n)


def build_windows(df: pd.DataFrame, feature_cols: List[str], seq_len: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Builds:
      X: (samples, seq_len, n_features)
      y: (samples,)
    Each label corresponds to the last row in its window (index i+seq_len-1).
    """
    feats = df[feature_cols].to_numpy(dtype=np.float32)
    y = df["y"].to_numpy(dtype=np.float32)

    if len(df) < seq_len:
        raise ValueError(f"Not enough rows ({len(df)}) for seq_len={seq_len}")

    X_list = []
    y_list = []
    for i in range(len(df) - seq_len + 1):
        X_list.append(feats[i : i + seq_len])
        y_list.append(y[i + seq_len - 1])

    X = np.array(X_list, dtype=np.float32)  # (samples, seq_len, n_features)
    y_out = np.array(y_list, dtype=np.float32)
    return X, y_out


def scale_3d_fit_train_only(X_tr: np.ndarray, X_va: np.ndarray, X_te: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    """
    Paper-style scaling: flatten 3D -> 2D, fit scaler on train, transform all, reshape back.
     [oai_citation:4‡Building a 1D CNN for Stock Prediction Using OHLCV Time-Series Features.pdf](sediment://file_000000002dd8720c983443cd1778dcb6)
    """
    ntr, T, F = X_tr.shape
    scaler = StandardScaler()

    X_tr_2d = X_tr.reshape(ntr * T, F)
    scaler.fit(X_tr_2d)

    def transform(X: np.ndarray) -> np.ndarray:
        n, T2, F2 = X.shape
        X2d = X.reshape(n * T2, F2)
        Xs = scaler.transform(X2d).reshape(n, T2, F2)
        return Xs.astype(np.float32)

    return transform(X_tr), transform(X_va), transform(X_te), scaler


def to_cnn_format(X: np.ndarray) -> np.ndarray:
    """
    Convert (samples, seq_len, features) -> (samples, channels=features, seq_len)
    because Conv1d expects (batch, channels, seq_len).
     [oai_citation:5‡Building a 1D CNN for Stock Prediction Using OHLCV Time-Series Features.pdf](sediment://file_000000002dd8720c983443cd1778dcb6)
    """
    return np.transpose(X, (0, 2, 1)).astype(np.float32)


def build_dataloaders(
    dataset_csv: str | Path,
    seq_len: int = 30,
    batch_size: int = 128,
    split: SplitConfig = SplitConfig(),
    task: str = "reg",  # "reg" or "cls"
    num_workers: int = 0,
    shuffle_train: bool = True,
):
    df = load_dataset_csv(dataset_csv)
    feature_cols = infer_feature_cols(df)

    tr_sl, va_sl, te_sl = time_split_indices(len(df), split)

    X_tr, y_tr = build_windows(df.iloc[tr_sl].reset_index(drop=True), feature_cols, seq_len)
    X_va, y_va = build_windows(df.iloc[va_sl].reset_index(drop=True), feature_cols, seq_len)
    X_te, y_te = build_windows(df.iloc[te_sl].reset_index(drop=True), feature_cols, seq_len)

    # classification labels must be 0/1 float for BCEWithLogitsLoss
    if task == "cls":
        y_tr = (y_tr > 0.5).astype(np.float32)
        y_va = (y_va > 0.5).astype(np.float32)
        y_te = (y_te > 0.5).astype(np.float32)

    X_tr, X_va, X_te, scaler = scale_3d_fit_train_only(X_tr, X_va, X_te)

    X_tr_cnn = to_cnn_format(X_tr)
    X_va_cnn = to_cnn_format(X_va)
    X_te_cnn = to_cnn_format(X_te)

    ds_tr = CNNSequenceDataset(X_tr_cnn, y_tr)
    ds_va = CNNSequenceDataset(X_va_cnn, y_va)
    ds_te = CNNSequenceDataset(X_te_cnn, y_te)

    dl_tr = DataLoader(ds_tr, batch_size=batch_size, shuffle=shuffle_train, num_workers=num_workers, drop_last=False)
    dl_va = DataLoader(ds_va, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=False)
    dl_te = DataLoader(ds_te, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=False)

    return dl_tr, dl_va, dl_te, scaler, feature_cols