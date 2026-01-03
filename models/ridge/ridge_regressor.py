# models/ridge/ridge_regressor.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import json

import numpy as np
import pandas as pd
from joblib import dump as joblib_dump

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error


@dataclass
class TrainConfig:
    # Chronological train/test split (no shuffling)  [oai_citation:1‡Predicting Next-Day Stock Returns with Ridge Regression_ A Comprehensive Guide.pdf](sediment://file_000000001a0c722fb26f98b637b5dead)
    train_ratio: float = 0.80

    # Time-series CV for alpha tuning  [oai_citation:2‡Predicting Next-Day Stock Returns with Ridge Regression_ A Comprehensive Guide.pdf](sediment://file_000000001a0c722fb26f98b637b5dead)
    tscv_splits: int = 5
    alpha_grid: Tuple[float, ...] = (0.1, 1.0, 10.0, 100.0)

    # Ridge is sensitive to feature scale; scale using train-only fit  [oai_citation:3‡Predicting Next-Day Stock Returns with Ridge Regression_ A Comprehensive Guide.pdf](sediment://file_000000001a0c722fb26f98b637b5dead)
    with_scaler: bool = True

    # Evaluation
    baseline_zero: bool = True  # compare vs predicting 0 return baseline  [oai_citation:4‡Predicting Next-Day Stock Returns with Ridge Regression_ A Comprehensive Guide.pdf](sediment://file_000000001a0c722fb26f98b637b5dead)


def _load_dataset(dataset_csv: str | Path) -> pd.DataFrame:
    df = pd.read_csv(dataset_csv)
    if "date" not in df.columns or "y" not in df.columns:
        raise ValueError("Dataset must contain 'date' and 'y' columns")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    # y is next-day open->open return (float)  [oai_citation:5‡Predicting Next-Day Stock Returns with Ridge Regression_ A Comprehensive Guide.pdf](sediment://file_000000001a0c722fb26f98b637b5dead)
    df["y"] = pd.to_numeric(df["y"], errors="coerce")
    df = df.dropna(subset=["y"]).reset_index(drop=True)
    return df


def _feature_cols(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if c not in ("date", "y")]


def _time_split(df: pd.DataFrame, train_ratio: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    n = len(df)
    cut = int(n * train_ratio)
    return df.iloc[:cut].copy(), df.iloc[cut:].copy()


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    mse = float(mean_squared_error(y_true, y_pred))
    rmse = float(np.sqrt(mse))
    mae = float(mean_absolute_error(y_true, y_pred))

    # Directional accuracy: sign correctness  [oai_citation:6‡Predicting Next-Day Stock Returns with Ridge Regression_ A Comprehensive Guide.pdf](sediment://file_000000001a0c722fb26f98b637b5dead)
    dir_acc = float(np.mean(np.sign(y_pred) == np.sign(y_true)))
    return {"mse": mse, "rmse": rmse, "mae": mae, "dir_acc": dir_acc}


def train_ridge_regressor(
    dataset_csv: str | Path,
    out_dir: str | Path,
    cfg: TrainConfig = TrainConfig(),
) -> Dict[str, Any]:
    """
    Train Ridge regression on one dataset CSV and save:
      - model.joblib (Pipeline: scaler + ridge)
      - feature_cols.json
      - metrics.json
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = _load_dataset(dataset_csv)
    feats = _feature_cols(df)

    df_tr, df_te = _time_split(df, cfg.train_ratio)
    X_tr = df_tr[feats].to_numpy(dtype=float)
    y_tr = df_tr["y"].to_numpy(dtype=float)
    X_te = df_te[feats].to_numpy(dtype=float)
    y_te = df_te["y"].to_numpy(dtype=float)

    # Pipeline: scaler (fit on train only) + Ridge 
    if cfg.with_scaler:
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("ridge", Ridge()),
        ])
        param_grid = {"ridge__alpha": list(cfg.alpha_grid)}
    else:
        pipe = Pipeline([
            ("ridge", Ridge()),
        ])
        param_grid = {"ridge__alpha": list(cfg.alpha_grid)}

    # Time series CV for alpha (no shuffle)  [oai_citation:7‡Predicting Next-Day Stock Returns with Ridge Regression_ A Comprehensive Guide.pdf](sediment://file_000000001a0c722fb26f98b637b5dead)
    tscv = TimeSeriesSplit(n_splits=cfg.tscv_splits)
    gs = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        cv=tscv,
        scoring="neg_mean_squared_error",
        n_jobs=-1,
        refit=True,
    )
    gs.fit(X_tr, y_tr)

    best_pipe: Pipeline = gs.best_estimator_
    best_params = gs.best_params_
    best_cv = float(gs.best_score_)

    y_pred = best_pipe.predict(X_te)
    test_metrics = _metrics(y_te, y_pred)

    payload: Dict[str, Any] = {
        "dataset": str(dataset_csv),
        "train_ratio": cfg.train_ratio,
        "tscv_splits": cfg.tscv_splits,
        "alpha_grid": list(cfg.alpha_grid),
        "with_scaler": cfg.with_scaler,
        "best_params": best_params,
        "best_cv_score": best_cv,
        "test_metrics": test_metrics,
    }

    # Baseline: predict 0 return (no-change)  [oai_citation:8‡Predicting Next-Day Stock Returns with Ridge Regression_ A Comprehensive Guide.pdf](sediment://file_000000001a0c722fb26f98b637b5dead)
    if cfg.baseline_zero:
        zero_pred = np.zeros_like(y_te)
        payload["baseline_zero_metrics"] = _metrics(y_te, zero_pred)

    # Save artifacts
    model_path = out_dir / "model.joblib"
    joblib_dump(best_pipe, model_path)

    (out_dir / "feature_cols.json").write_text(json.dumps(feats, indent=2))
    (out_dir / "metrics.json").write_text(json.dumps(payload, indent=2))

    # Optional: save coefficients for interpretability (Ridge keeps small non-zero weights)  [oai_citation:9‡Predicting Next-Day Stock Returns with Ridge Regression_ A Comprehensive Guide.pdf](sediment://file_000000001a0c722fb26f98b637b5dead)
    try:
        ridge = best_pipe.named_steps["ridge"]
        coefs = {f: float(c) for f, c in sorted(zip(feats, ridge.coef_), key=lambda x: -abs(x[1]))}
        (out_dir / "coefficients.json").write_text(json.dumps(coefs, indent=2))
    except Exception:
        pass

    return {
        "model_path": str(model_path),
        "feature_cols_path": str(out_dir / "feature_cols.json"),
        "metrics_path": str(out_dir / "metrics.json"),
        "best_params": best_params,
        "best_cv_score": best_cv,
        "test_metrics": test_metrics,
    }