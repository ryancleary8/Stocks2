# models/xgb/xgb_regressor.py
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from xgboost import XGBRegressor
except ImportError as e:
    raise ImportError(
        "Missing dependency: xgboost. Install with: pip install xgboost"
    ) from e


@dataclass
class XGBRegressorConfig:
    # Core
    n_estimators: int = 4000          # large + early stopping
    learning_rate: float = 0.02
    max_depth: int = 5
    min_child_weight: float = 3.0
    subsample: float = 0.8            # recommended randomness for generalization
    colsample_bytree: float = 0.8     # recommended randomness for generalization
    reg_alpha: float = 0.1            # L1
    reg_lambda: float = 1.0           # L2
    gamma: float = 0.0

    # Objective / eval
    objective: str = "reg:squarederror"
    eval_metric: str = "rmse"

    # Training
    random_state: int = 42
    n_jobs: int = -1
    early_stopping_rounds: int = 100

    # Data / split
    val_fraction: float = 0.15        # last 15% as validation (time-ordered)
    min_train_rows: int = 252         # ~1 trading year
    min_val_rows: int = 60            # ~3 months


class XGBOpenReturnRegressor:
    """
    XGBoost regressor for next-day open return prediction.

    Assumptions:
      - df is time-sorted by date ascending
      - target column is numeric return (e.g. (open[t+1]-open[t])/open[t])
      - features contain ONLY allowed (leak-free) inputs at prediction time
    """

    def __init__(self, config: Optional[XGBRegressorConfig] = None):
        self.cfg = config or XGBRegressorConfig()
        self.model: Optional[XGBRegressor] = None
        self.feature_cols: Optional[list[str]] = None

    def _time_split(
        self, X: pd.DataFrame, y: pd.Series
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        n = len(X)
        val_n = max(int(n * self.cfg.val_fraction), self.cfg.min_val_rows)
        val_n = min(val_n, n // 3)  # keep some training
        split = n - val_n

        X_train, X_val = X.iloc[:split], X.iloc[split:]
        y_train, y_val = y.iloc[:split], y.iloc[split:]

        if len(X_train) < self.cfg.min_train_rows:
            raise ValueError(
                f"Not enough training rows ({len(X_train)}). "
                f"Need at least {self.cfg.min_train_rows}."
            )
        if len(X_val) < self.cfg.min_val_rows:
            raise ValueError(
                f"Not enough validation rows ({len(X_val)}). "
                f"Need at least {self.cfg.min_val_rows}."
            )

        return X_train, X_val, y_train, y_val

    def fit(self, df: pd.DataFrame, target_col: str, feature_cols: Optional[list[str]] = None):
        if feature_cols is None:
            # Default: everything except target/date/symbol-ish columns
            drop = {target_col, "date", "symbol", "ticker"}
            feature_cols = [c for c in df.columns if c not in drop]
        self.feature_cols = feature_cols

        X = df[feature_cols].copy()
        y = df[target_col].astype(float).copy()

        # Replace inf, drop NaNs (after your rolling windows)
        X = X.replace([np.inf, -np.inf], np.nan)
        mask = ~(X.isna().any(axis=1) | y.isna())
        X, y = X.loc[mask], y.loc[mask]

        X_train, X_val, y_train, y_val = self._time_split(X, y)

        self.model = XGBRegressor(
            n_estimators=self.cfg.n_estimators,
            learning_rate=self.cfg.learning_rate,
            max_depth=self.cfg.max_depth,
            min_child_weight=self.cfg.min_child_weight,
            subsample=self.cfg.subsample,
            colsample_bytree=self.cfg.colsample_bytree,
            reg_alpha=self.cfg.reg_alpha,
            reg_lambda=self.cfg.reg_lambda,
            gamma=self.cfg.gamma,
            objective=self.cfg.objective,
            eval_metric=self.cfg.eval_metric,
            random_state=self.cfg.random_state,
            n_jobs=self.cfg.n_jobs,
        )

        # Early stopping API varies across xgboost versions.
        # Try the modern sklearn signature first; if unavailable, fall back to callbacks;
        # and if neither is supported, train without early stopping.
        fit_kwargs = {
            "eval_set": [(X_val, y_val)],
            "verbose": False,
        }

        try:
            # Many versions support this directly
            self.model.fit(
                X_train,
                y_train,
                early_stopping_rounds=self.cfg.early_stopping_rounds,
                **fit_kwargs,
            )
        except TypeError:
            try:
                # Some versions require callbacks instead
                from xgboost.callback import EarlyStopping  # type: ignore

                self.model.fit(
                    X_train,
                    y_train,
                    callbacks=[EarlyStopping(rounds=self.cfg.early_stopping_rounds, save_best=True)],
                    **fit_kwargs,
                )
            except Exception:
                # As a last resort, fit without early stopping
                self.model.fit(
                    X_train,
                    y_train,
                    **fit_kwargs,
                )

        return self

    def predict(self, df_or_X: pd.DataFrame) -> np.ndarray:
        if self.model is None or self.feature_cols is None:
            raise RuntimeError("Model not fitted yet.")
        X = df_or_X[self.feature_cols].copy() if set(self.feature_cols).issubset(df_or_X.columns) else df_or_X
        X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return self.model.predict(X)

    def save(self, path: str | Path):
        if self.model is None:
            raise RuntimeError("No model to save.")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save_model(str(path))

        meta_path = path.with_suffix(".meta.json")
        meta = {
            "type": "xgb_regressor",
            "feature_cols": self.feature_cols,
            "config": asdict(self.cfg),
        }
        meta_path.write_text(pd.Series(meta).to_json(), encoding="utf-8")

    def load(self, path: str | Path):
        path = Path(path)
        self.model = XGBRegressor()
        self.model.load_model(str(path))

        meta_path = path.with_suffix(".meta.json")
        meta = pd.read_json(meta_path.read_text(encoding="utf-8"), typ="series")
        self.feature_cols = list(meta["feature_cols"])
        # config rehydration is optional; keep current cfg
        return self