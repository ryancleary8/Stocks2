# models/transformer/ts_transformer.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Tuple, Optional, List
from pathlib import Path
import json

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score, roc_auc_score, precision_score

from joblib import dump as joblib_dump

from .seq_dataset import build_dataloaders, SplitConfig


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        T = x.size(1)
        return x + self.pe[:, :T, :]


class TimeSeriesTransformer(nn.Module):
    """
    Encoder-only transformer for seq->one.
    Input:  (B, T, F)
    Output: (B,) regression or (B,) logits for binary classification
    """
    def __init__(
        self,
        n_features: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dropout: float = 0.10,
        task: str = "reg",  # "reg" or "cls"
    ):
        super().__init__()
        self.task = task

        self.proj = nn.Linear(n_features, d_model)
        self.pos = PositionalEncoding(d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)

        self.head = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, F)
        h = self.proj(x)
        h = self.pos(h)
        h = self.encoder(h)
        h = self.norm(h)

        # take last token (represents most recent day in the window)
        last = h[:, -1, :]
        out = self.head(last).squeeze(-1)  # (B,)

        return out


@dataclass
class TrainConfig:
    seq_len: int = 30
    batch_size: int = 64
    epochs: int = 30
    lr: float = 1e-4
    weight_decay: float = 1e-2
    grad_clip: float = 1.0
    patience: int = 5  # early stop
    device: str = "auto"  # "auto" | "cpu" | "cuda"


def _device(cfg: TrainConfig) -> torch.device:
    if cfg.device == "cpu":
        return torch.device("cpu")
    if cfg.device == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def _eval_epoch(model: nn.Module, dl, task: str, device: torch.device) -> Tuple[float, Dict[str, float]]:
    model.eval()
    ys, ps = [], []
    total_loss = 0.0
    n = 0

    for X, y in dl:
        X = X.to(device)
        y = y.to(device)

        pred = model(X)

        if task == "reg":
            loss = F.mse_loss(pred, y)
        else:
            loss = F.binary_cross_entropy_with_logits(pred, y)

        bs = X.size(0)
        total_loss += loss.item() * bs
        n += bs

        ys.append(y.detach().cpu().numpy())
        ps.append(pred.detach().cpu().numpy())

    y_true = np.concatenate(ys)
    y_pred = np.concatenate(ps)

    metrics: Dict[str, float] = {}
    avg_loss = total_loss / max(n, 1)

    if task == "reg":
        mae = float(mean_absolute_error(y_true, y_pred))
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        # directional accuracy on returns: sign(pred) vs sign(true)
        dir_acc = float(np.mean(np.sign(y_pred) == np.sign(y_true)))
        metrics.update({"mae": mae, "rmse": rmse, "dir_acc": dir_acc})
    else:
        prob = 1.0 / (1.0 + np.exp(-y_pred))
        y_hat = (prob >= 0.5).astype(int)
        acc = float(accuracy_score(y_true.astype(int), y_hat))
        # AUC can fail if only one class in y_true
        try:
            auc = float(roc_auc_score(y_true.astype(int), prob))
        except Exception:
            auc = float("nan")
        # precision at a higher confidence threshold (useful for trading)
        y_hat_60 = (prob >= 0.60).astype(int)
        try:
            prec60 = float(precision_score(y_true.astype(int), y_hat_60, zero_division=0))
        except Exception:
            prec60 = float("nan")
        metrics.update({"acc": acc, "auc": auc, "prec_at_0.60": prec60})

    return avg_loss, metrics


def train_transformer(
    dataset_csv: str | Path,
    out_dir: str | Path,
    task: str,  # "reg" or "cls"
    cfg: TrainConfig = TrainConfig(),
    model_kwargs: Optional[Dict[str, Any]] = None,
    split: SplitConfig = SplitConfig(),
) -> Dict[str, Any]:
    """
    Trains a transformer on ONE dataset file and saves:
      - model .pt
      - scaler.joblib
      - feature_cols.json
      - metrics.json

    Returns a dict with paths + final metrics.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = _device(cfg)

    dl_tr, dl_va, dl_te, scaler, feature_cols = build_dataloaders(
        dataset_csv=dataset_csv,
        seq_len=cfg.seq_len,
        batch_size=cfg.batch_size,
        split=split,
        task=task,
    )

    n_features = len(feature_cols)
    mk = model_kwargs or {}
    model = TimeSeriesTransformer(n_features=n_features, task=task, **mk).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    best_val = float("inf")
    best_state = None
    bad = 0

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        total = 0.0
        n = 0

        for X, y in dl_tr:
            X = X.to(device)
            y = y.to(device)

            pred = model(X)
            if task == "reg":
                loss = F.mse_loss(pred, y)
            else:
                loss = F.binary_cross_entropy_with_logits(pred, y)

            opt.zero_grad()
            loss.backward()
            if cfg.grad_clip:
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()

            bs = X.size(0)
            total += loss.item() * bs
            n += bs

        train_loss = total / max(n, 1)
        val_loss, val_metrics = _eval_epoch(model, dl_va, task=task, device=device)

        # Early stopping on val_loss
        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= cfg.patience:
                break

    # restore best
    if best_state is not None:
        model.load_state_dict(best_state)

    test_loss, test_metrics = _eval_epoch(model, dl_te, task=task, device=device)

    # Save artifacts
    model_path = out_dir / "model.pt"
    torch.save(model.state_dict(), model_path)

    scaler_path = out_dir / "scaler.joblib"
    joblib_dump(scaler, scaler_path)

    feat_path = out_dir / "feature_cols.json"
    feat_path.write_text(json.dumps(feature_cols, indent=2))

    metrics = {
        "task": task,
        "dataset": str(dataset_csv),
        "seq_len": cfg.seq_len,
        "batch_size": cfg.batch_size,
        "epochs_ran": epoch,
        "best_val_loss": best_val,
        "test_loss": test_loss,
        "test_metrics": test_metrics,
        "val_metrics_last": val_metrics,
    }
    metrics_path = out_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))

    return {
        "model_path": str(model_path),
        "scaler_path": str(scaler_path),
        "feature_cols_path": str(feat_path),
        "metrics_path": str(metrics_path),
        "test_metrics": test_metrics,
    }