# models/rf/rf_regressor.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import json

import numpy as np
import pandas as pd
from joblib import dump as joblib_dump

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error


@dataclass
class TrainConfig:
    # Strict chronological split: train on past, test on future  [oai_citation:1‡Predicting Next-Day Open Returns with a Random Forest Regressor.pdf](sediment://file_00000000395c722fb29821cafccf81e1)
    train_ratio: float = 0.80

    # Walk-forward style CV via TimeSeriesSplit  [oai_citation:2‡Predicting Next-Day Open Returns with a Random Forest Regressor.pdf](sediment://file_00000000395c722fb29821cafccf81e1)
    tscv_splits: int = 3

    # RF tuning grid (depth + min leaf are the main anti-overfit knobs) 
    n_estimators_grid: Tuple[int, ...] = (200, 400)
    max_depth_grid: Tuple[Optional[int], ...] = (6, 10, None)
    min_samples_leaf_grid: Tuple[int, ...] = (1, 5, 10)
    max_features_grid: Tuple[str, ...] = ("sqrt", "log2")  # optional to tune

    random_state: int = 42
    n_jobs: int = -1


def _load_dataset(dataset_csv: str | Path) -> pd.DataFrame:
    df = pd.read_csv(dataset_csv)
    if "date" not in df.columns or "y" not in df.columns:
        raise ValueError("Dataset must contain 'date' and 'y' columns")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    # Ensure y is numeric float target (next-day open return in decimals)  [oai_citation:3‡Predicting Next-Day Open Returns with a Random Forest Regressor.pdf](sediment://file_00000000395c722fb29821cafccf81e1)
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

    # Directional accuracy (sign match) is useful for trading evaluation  [oai_citation:4‡Predicting Next-Day Open Returns with a Random Forest Regressor.pdf](sediment://file_00000000395c722fb29821cafccf81e1)
    dir_acc = float(np.mean(np.sign(y_pred) == np.sign(y_true)))

    return {"mse": mse, "rmse": rmse, "mae": mae, "dir_acc": dir_acc}


def train_rf_regressor(
    dataset_csv: str | Path,
    out_dir: str | Path,
    cfg: TrainConfig = TrainConfig(),
    scoring: str = "neg_mean_squared_error",  # minimize MSE  [oai_citation:5‡Predicting Next-Day Open Returns with a Random Forest Regressor.pdf](sediment://file_00000000395c722fb29821cafccf81e1)
) -> Dict[str, Any]:
    """
    Train RandomForestRegressor on one dataset file and save:
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
    y_tr = df_tr["y"].to_numpy(dtype=float)

    X_te = df_te[feats].to_numpy(dtype=float)
    y_te = df_te["y"].to_numpy(dtype=float)

    base = RandomForestRegressor(
        n_jobs=cfg.n_jobs,
        random_state=cfg.random_state,
        bootstrap=True,
    )

    param_grid = {
        "n_estimators": list(cfg.n_estimators_grid),
        "max_depth": list(cfg.max_depth_grid),
        "min_samples_leaf": list(cfg.min_samples_leaf_grid),
        "max_features": list(cfg.max_features_grid),
    }

    # Walk-forward CV (no shuffle)  [oai_citation:6‡Predicting Next-Day Open Returns with a Random Forest Regressor.pdf](sediment://file_00000000395c722fb29821cafccf81e1)
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

    best_model: RandomForestRegressor = gs.best_estimator_
    best_params = gs.best_params_
    best_cv = float(gs.best_score_)

    y_pred = best_model.predict(X_te)
    test_metrics = _metrics(y_te, y_pred)

    # Save model
    model_path = out_dir / "model.joblib"
    joblib_dump(best_model, model_path)

    # Save feature list (needed at inference time)
    feat_path = out_dir / "feature_cols.json"
    feat_path.write_text(json.dumps(feats, indent=2))

    # Feature importances help you prune redundant indicators  [oai_citation:7‡Predicting Next-Day Open Returns with a Random Forest Regressor.pdf](sediment://file_00000000395c722fb29821cafccf81e1)
    importances = {f: float(v) for f, v in sorted(zip(feats, best_model.feature_importances_), key=lambda x: -x[1])}
    (out_dir / "feature_importances.json").write_text(json.dumps(importances, indent=2))

    payload = {
        "dataset": str(dataset_csv),
        "train_ratio": cfg.train_ratio,
        "tscv_splits": cfg.tscv_splits,
        "scoring": scoring,
        "best_params": best_params,
        "best_cv_score": best_cv,
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