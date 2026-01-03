# models/rnn/train_lstm.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Optional
import json
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from joblib import dump as joblib_dump
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score, roc_auc_score, precision_score

from .seq_dataset import build_dataloaders, SplitConfig
from .lstm_model import StockLSTM


@dataclass
class TrainConfig:
    seq_len: int = 20
    batch_size: int = 128
    epochs: int = 40
    lr: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    patience: int = 6
    device: str = "auto"  # "auto" | "cpu" | "cuda"


def _device(cfg: TrainConfig) -> torch.device:
    if cfg.device == "cpu":
        return torch.device("cpu")
    if cfg.device == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def _eval(model: nn.Module, dl, task: str, device: torch.device):
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
    avg_loss = total_loss / max(n, 1)

    metrics: Dict[str, float] = {}
    if task == "reg":
        mae = float(mean_absolute_error(y_true, y_pred))
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        dir_acc = float(np.mean(np.sign(y_pred) == np.sign(y_true)))
        metrics.update({"mae": mae, "rmse": rmse, "dir_acc": dir_acc})
    else:
        prob = 1.0 / (1.0 + np.exp(-y_pred))
        y_hat = (prob >= 0.5).astype(int)
        acc = float(accuracy_score(y_true.astype(int), y_hat))
        try:
            auc = float(roc_auc_score(y_true.astype(int), prob))
        except Exception:
            auc = float("nan")
        y_hat_60 = (prob >= 0.60).astype(int)
        prec60 = float(precision_score(y_true.astype(int), y_hat_60, zero_division=0))
        metrics.update({"acc": acc, "auc": auc, "prec_at_0.60": prec60})

    return avg_loss, metrics


def train_lstm(
    dataset_csv: str | Path,
    out_dir: str | Path,
    task: str,  # "reg" or "cls"
    cfg: TrainConfig = TrainConfig(),
    model_kwargs: Optional[Dict[str, Any]] = None,
    split: SplitConfig = SplitConfig(),
):
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

    mk = model_kwargs or {}
    model = StockLSTM(
        input_dim=len(feature_cols),
        task=task,
        **mk,
    ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    best_val = float("inf")
    best_state = None
    bad = 0
    last_epoch = 0

    for epoch in range(1, cfg.epochs + 1):
        last_epoch = epoch
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

        val_loss, val_metrics = _eval(model, dl_va, task=task, device=device)

        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= cfg.patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    test_loss, test_metrics = _eval(model, dl_te, task=task, device=device)

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
        "epochs_ran": last_epoch,
        "best_val_loss": best_val,
        "test_loss": test_loss,
        "test_metrics": test_metrics,
        "val_metrics_last": val_metrics,
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    return {
        "model_path": str(model_path),
        "scaler_path": str(scaler_path),
        "feature_cols_path": str(feat_path),
        "test_metrics": test_metrics,
    }