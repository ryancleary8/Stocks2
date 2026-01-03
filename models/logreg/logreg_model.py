# models/logreg/logreg_model.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
import json

import numpy as np
import pandas as pd
from joblib import dump as joblib_dump

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)


@dataclass
class TrainConfig:
    # Chronological split (no shuffling)  [oai_citation:2‡Building a Logistic Regression Pipeline for Next-Day Stock Direction Prediction.pdf](sediment://file_000000001878722f9fa3bc8483499fea)
    train_ratio: float = 0.80

    # Regularization grid (smaller C = stronger regularization)  [oai_citation:3‡Building a Logistic Regression Pipeline for Next-Day Stock Direction Prediction.pdf](sediment://file_000000001878722f9fa3bc8483499fea)
    C_grid: Tuple[float, ...] = (0.01, 0.05, 0.1, 0.5, 1.0)

    # TimeSeriesSplit folds for tuning  [oai_citation:4‡Building a Logistic Regression Pipeline for Next-Day Stock Direction Prediction.pdf](sediment://file_000000001878722f9fa3bc8483499fea)
    tscv_splits: int = 5

    # Handle imbalance safely via weights  [oai_citation:5‡Building a Logistic Regression Pipeline for Next-Day Stock Direction Prediction.pdf](sediment://file_000000001878722f9fa3bc8483499fea)
    class_weight: Optional[str] = "balanced"  # "balanced" or None

    # Convergence
    max_iter: int = 2000

    # Trading-friendly evaluation threshold (can change later)
    proba_threshold: float = 0.50


def _load_dataset(dataset_csv: str | Path) -> pd.DataFrame:
    df = pd.read_csv(dataset_csv)
    if "date" not in df.columns or "y" not in df.columns:
        raise ValueError("Dataset must contain 'date' and 'y' columns")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    # Logistic regression expects 0/1 target for direction prediction  [oai_citation:6‡Building a Logistic Regression Pipeline for Next-Day Stock Direction Prediction.pdf](sediment://file_000000001878722f9fa3bc8483499fea)
    df["y"] = (df["y"] > 0.5).astype(int)
    return df


def _feature_cols(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if c not in ("date", "y")]


def _time_split(df: pd.DataFrame, train_ratio: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
    n = len(df)
    cut = int(n * train_ratio)
    return df.iloc[:cut].copy(), df.iloc[cut:].copy()


def _build_pipeline(cfg: TrainConfig) -> Pipeline:
    # Scale -> LogisticRegression pipeline (recommended) 
    return Pipeline([
        ("scaler", StandardScaler()),
        ("logreg", LogisticRegression(
            penalty="l2",
            solver="lbfgs",
            C=0.1,  # will be tuned
            max_iter=cfg.max_iter,
            class_weight=cfg.class_weight,
        )),
    ])


def _metrics(y_true: np.ndarray, proba: np.ndarray, thr: float) -> Dict[str, Any]:
    y_pred = (proba >= thr).astype(int)

    out: Dict[str, Any] = {}
    out["accuracy"] = float(accuracy_score(y_true, y_pred))
    out["precision"] = float(precision_score(y_true, y_pred, zero_division=0))
    out["recall"] = float(recall_score(y_true, y_pred, zero_division=0))
    out["f1"] = float(f1_score(y_true, y_pred, zero_division=0))

    # AUC measures ranking skill; can be informative even if accuracy ~50%  [oai_citation:7‡Building a Logistic Regression Pipeline for Next-Day Stock Direction Prediction.pdf](sediment://file_000000001878722f9fa3bc8483499fea)
    try:
        out["auc"] = float(roc_auc_score(y_true, proba))
    except Exception:
        out["auc"] = float("nan")

    cm = confusion_matrix(y_true, y_pred)
    out["confusion_matrix"] = cm.tolist()
    out["threshold"] = float(thr)
    return out


def train_logreg(
    dataset_csv: str | Path,
    out_dir: str | Path,
    cfg: TrainConfig = TrainConfig(),
) -> Dict[str, Any]:
    """
    Trains logistic regression on one dataset file and saves:
      - model.joblib (Pipeline with scaler + model)
      - feature_cols.json
      - metrics.json
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = _load_dataset(dataset_csv)
    feats = _feature_cols(df)

    df_tr, df_te = _time_split(df, cfg.train_ratio)

    X_tr = df_tr[feats].to_numpy(dtype=float)
    y_tr = df_tr["y"].to_numpy(dtype=int)

    X_te = df_te[feats].to_numpy(dtype=float)
    y_te = df_te["y"].to_numpy(dtype=int)

    pipe = _build_pipeline(cfg)

    # Time-series CV tuning for C (regularization) 
    tscv = TimeSeriesSplit(n_splits=cfg.tscv_splits)

    grid = {
        "logreg__C": list(cfg.C_grid),
    }

    gs = GridSearchCV(
        estimator=pipe,
        param_grid=grid,
        scoring="f1",  # aligns with precision/recall balance  [oai_citation:8‡Building a Logistic Regression Pipeline for Next-Day Stock Direction Prediction.pdf](sediment://file_000000001878722f9fa3bc8483499fea)
        cv=tscv,
        n_jobs=-1,
        refit=True,
    )
    gs.fit(X_tr, y_tr)

    best_pipe: Pipeline = gs.best_estimator_
    best_params = gs.best_params_

    proba_te = best_pipe.predict_proba(X_te)[:, 1]
    test_metrics = _metrics(y_te, proba_te, cfg.proba_threshold)

    # Save artifacts for production loading with joblib  [oai_citation:9‡Building a Logistic Regression Pipeline for Next-Day Stock Direction Prediction.pdf](sediment://file_000000001878722f9fa3bc8483499fea)
    model_path = out_dir / "model.joblib"
    joblib_dump(best_pipe, model_path)

    feat_path = out_dir / "feature_cols.json"
    feat_path.write_text(json.dumps(feats, indent=2))

    metrics_path = out_dir / "metrics.json"
    payload = {
        "dataset": str(dataset_csv),
        "best_params": best_params,
        "train_ratio": cfg.train_ratio,
        "tscv_splits": cfg.tscv_splits,
        "class_weight": cfg.class_weight,
        "test_metrics": test_metrics,
    }
    metrics_path.write_text(json.dumps(payload, indent=2))

    return {
        "model_path": str(model_path),
        "feature_cols_path": str(feat_path),
        "metrics_path": str(metrics_path),
        "test_metrics": test_metrics,
        "best_params": best_params,
    }