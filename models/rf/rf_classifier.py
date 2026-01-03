# models/rf/rf_classifier.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import json

import numpy as np
import pandas as pd
from joblib import dump as joblib_dump

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)


@dataclass
class TrainConfig:
    # strict chronological split + time-series CV (no shuffling)  [oai_citation:2‡Next-Day Stock Return Prediction with Random Forests.pdf](sediment://file_00000000bdf0722f88e4e7744c6d15ac)
    train_ratio: float = 0.80
    tscv_splits: int = 4

    # “noise filter” thresholds: only act when probability is high/low  [oai_citation:3‡Next-Day Stock Return Prediction with Random Forests.pdf](sediment://file_00000000bdf0722f88e4e7744c6d15ac)
    thr_long: float = 0.60
    thr_short: float = 0.40

    # RF hyperparameter grid (kept small because you’ll train many tickers)
    n_estimators_grid: Tuple[int, ...] = (300, 600)
    max_depth_grid: Tuple[Optional[int], ...] = (6, 10, None)
    min_samples_leaf_grid: Tuple[int, ...] = (1, 3, 5)
    max_features_grid: Tuple[str, ...] = ("sqrt", "log2")
    class_weight: Optional[str] = None  # or "balanced"

    random_state: int = 42
    n_jobs: int = -1


def _load_dataset(dataset_csv: str | Path) -> pd.DataFrame:
    df = pd.read_csv(dataset_csv)
    if "date" not in df.columns or "y" not in df.columns:
        raise ValueError("Dataset must contain 'date' and 'y' columns")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    # classification target must be 0/1
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
    +1 long if p_up > thr_long
    -1 short if p_up < thr_short
     0 otherwise (noise)
     [oai_citation:4‡Next-Day Stock Return Prediction with Random Forests.pdf](sediment://file_00000000bdf0722f88e4e7744c6d15ac)
    """
    sig = np.zeros_like(proba_up, dtype=int)
    sig[proba_up > thr_long] = 1
    sig[proba_up < thr_short] = -1
    return sig


def train_rf_classifier(
    dataset_csv: str | Path,
    out_dir: str | Path,
    cfg: TrainConfig = TrainConfig(),
    scoring: str = "roc_auc",  # good for ranking; you can also use "f1"
) -> Dict[str, Any]:
    """
    Train RF classifier (direction) on one dataset file and save:
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

    base = RandomForestClassifier(
        random_state=cfg.random_state,
        n_jobs=cfg.n_jobs,
        class_weight=cfg.class_weight,
        bootstrap=True,
    )

    param_grid = {
        "n_estimators": list(cfg.n_estimators_grid),
        "max_depth": list(cfg.max_depth_grid),
        "min_samples_leaf": list(cfg.min_samples_leaf_grid),
        "max_features": list(cfg.max_features_grid),
    }

    # time-series CV (forward splits)
    tscv = TimeSeriesSplit(n_splits=cfg.tscv_splits)

    gs = GridSearchCV(
        estimator=base,
        param_grid=param_grid,
        scoring=scoring,
        cv=tscv,
        n_jobs=cfg.n_jobs,
        refit=True,
    )
    gs.fit(X_tr, y_tr)

    best_model: RandomForestClassifier = gs.best_estimator_
    best_params = gs.best_params_
    best_cv = float(gs.best_score_)

    proba_up = best_model.predict_proba(X_te)[:, 1]
    test_metrics = _metrics(y_te, proba_up, cfg.thr_long)

    # Save artifacts
    model_path = out_dir / "model.joblib"
    joblib_dump(best_model, model_path)

    feat_path = out_dir / "feature_cols.json"
    feat_path.write_text(json.dumps(feats, indent=2))

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