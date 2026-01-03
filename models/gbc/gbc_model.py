# models/gbc/gbc_model.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import json

import numpy as np
import pandas as pd
from joblib import dump as joblib_dump

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)


@dataclass
class TrainConfig:
    # Chronological train/test split (no shuffle)  [oai_citation:2‡Predicting Next-Day Open Direction with Gradient Boosting (Daily U.S. Equities).pdf](sediment://file_00000000a9bc722f8f6865ba418ea385)
    train_ratio: float = 0.80

    # Time series CV (forward / expanding windows)  [oai_citation:3‡Predicting Next-Day Open Direction with Gradient Boosting (Daily U.S. Equities).pdf](sediment://file_00000000a9bc722f8f6865ba418ea385)
    tscv_splits: int = 4

    # Probability thresholds for trading signals  [oai_citation:4‡Predicting Next-Day Open Direction with Gradient Boosting (Daily U.S. Equities).pdf](sediment://file_00000000a9bc722f8f6865ba418ea385)
    thr_long: float = 0.60
    thr_short: float = 0.40  # if you want long/short; long-only can ignore this

    # Parameter grid (keep small; you’ll run this across many tickers)
    # Typical depth 3–5, learning_rate ~0.05–0.1, subsample 0.7–1.0 
    n_estimators_grid: Tuple[int, ...] = (200, 400)
    learning_rate_grid: Tuple[float, ...] = (0.05, 0.1)
    max_depth_grid: Tuple[int, ...] = (3, 4)
    subsample_grid: Tuple[float, ...] = (0.8, 1.0)
    min_samples_leaf_grid: Tuple[int, ...] = (3, 5)

    random_state: int = 42


def _load_dataset(dataset_csv: str | Path) -> pd.DataFrame:
    df = pd.read_csv(dataset_csv)
    if "date" not in df.columns or "y" not in df.columns:
        raise ValueError("Dataset must contain 'date' and 'y' columns")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    # Direction target: 1 if up else 0  [oai_citation:5‡Predicting Next-Day Open Direction with Gradient Boosting (Daily U.S. Equities).pdf](sediment://file_00000000a9bc722f8f6865ba418ea385)
    df["y"] = (df["y"] > 0.5).astype(int)
    return df


def _feature_cols(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if c not in ("date", "y")]


def _time_split(df: pd.DataFrame, train_ratio: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    n = len(df)
    cut = int(n * train_ratio)
    return df.iloc[:cut].copy(), df.iloc[cut:].copy()


def _metrics(y_true: np.ndarray, proba_up: np.ndarray, thr_long: float) -> Dict[str, Any]:
    y_pred = (proba_up >= thr_long).astype(int)

    out: Dict[str, Any] = {}
    out["accuracy"] = float(accuracy_score(y_true, y_pred))
    out["precision"] = float(precision_score(y_true, y_pred, zero_division=0))
    out["recall"] = float(recall_score(y_true, y_pred, zero_division=0))
    out["f1"] = float(f1_score(y_true, y_pred, zero_division=0))
    try:
        out["auc"] = float(roc_auc_score(y_true, proba_up))
    except Exception:
        out["auc"] = float("nan")

    out["confusion_matrix"] = confusion_matrix(y_true, y_pred).tolist()
    out["thr_long"] = float(thr_long)
    return out


def signals_from_proba(proba_up: np.ndarray, thr_long: float = 0.60, thr_short: float = 0.40) -> np.ndarray:
    """
    Convert probabilities into trading signals:
      +1 long if p_up > thr_long
      -1 short if p_up < thr_short
       0 otherwise
     [oai_citation:6‡Predicting Next-Day Open Direction with Gradient Boosting (Daily U.S. Equities).pdf](sediment://file_00000000a9bc722f8f6865ba418ea385)
    """
    sig = np.zeros_like(proba_up, dtype=int)
    sig[proba_up > thr_long] = 1
    sig[proba_up < thr_short] = -1
    return sig


def train_gbc(
    dataset_csv: str | Path,
    out_dir: str | Path,
    cfg: TrainConfig = TrainConfig(),
    scoring: str = "roc_auc",  # accuracy or roc_auc are common  [oai_citation:7‡Predicting Next-Day Open Direction with Gradient Boosting (Daily U.S. Equities).pdf](sediment://file_00000000a9bc722f8f6865ba418ea385)
) -> Dict[str, Any]:
    """
    Trains a GradientBoostingClassifier using time-series CV and saves:
      - model.joblib
      - feature_cols.json
      - feature_importances.json
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

    # Base model (depth controls interactions; subsample adds stochastic regularization) 
    base = GradientBoostingClassifier(random_state=cfg.random_state)

    param_grid = {
        "n_estimators": list(cfg.n_estimators_grid),
        "learning_rate": list(cfg.learning_rate_grid),
        "max_depth": list(cfg.max_depth_grid),
        "subsample": list(cfg.subsample_grid),
        "min_samples_leaf": list(cfg.min_samples_leaf_grid),
    }

    tscv = TimeSeriesSplit(n_splits=cfg.tscv_splits)  # no shuffling  [oai_citation:8‡Predicting Next-Day Open Direction with Gradient Boosting (Daily U.S. Equities).pdf](sediment://file_00000000a9bc722f8f6865ba418ea385)

    gs = GridSearchCV(
        estimator=base,
        param_grid=param_grid,
        scoring=scoring,
        cv=tscv,
        n_jobs=-1,
        refit=True,
    )
    gs.fit(X_tr, y_tr)

    best_model: GradientBoostingClassifier = gs.best_estimator_
    best_params = gs.best_params_
    best_cv = float(gs.best_score_)

    proba_up = best_model.predict_proba(X_te)[:, 1]
    test_metrics = _metrics(y_te, proba_up, cfg.thr_long)

    # Save model + metadata
    model_path = out_dir / "model.joblib"
    joblib_dump(best_model, model_path)

    feat_path = out_dir / "feature_cols.json"
    feat_path.write_text(json.dumps(feats, indent=2))

    # Feature importance can help sanity-check what the model uses  [oai_citation:9‡Predicting Next-Day Open Direction with Gradient Boosting (Daily U.S. Equities).pdf](sediment://file_00000000a9bc722f8f6865ba418ea385)
    importances = {f: float(v) for f, v in sorted(zip(feats, best_model.feature_importances_), key=lambda x: -x[1])}
    (out_dir / "feature_importances.json").write_text(json.dumps(importances, indent=2))

    payload = {
        "dataset": str(dataset_csv),
        "train_ratio": cfg.train_ratio,
        "tscv_splits": cfg.tscv_splits,
        "scoring": scoring,
        "best_params": best_params,
        "best_cv_score": best_cv,
        "thresholds": {"thr_long": cfg.thr_long, "thr_short": cfg.thr_short},
        "test_metrics": test_metrics,
    }
    metrics_path = out_dir / "metrics.json"
    metrics_path.write_text(json.dumps(payload, indent=2))

    return {
        "model_path": str(model_path),
        "feature_cols_path": str(feat_path),
        "feature_importances_path": str(out_dir / "feature_importances.json"),
        "metrics_path": str(metrics_path),
        "best_params": best_params,
        "best_cv_score": best_cv,
        "test_metrics": test_metrics,
    }