# models/xgb/xgb_classifier.py
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

try:
    from xgboost import XGBClassifier
except ImportError as e:
    raise ImportError(
        "Missing dependency: xgboost. Install with: pip install xgboost"
    ) from e


@dataclass
class XGBClassifierConfig:
    n_estimators: int = 4000
    learning_rate: float = 0.03
    max_depth: int = 4
    min_child_weight: float = 3.0
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    reg_alpha: float = 0.1
    reg_lambda: float = 1.0
    gamma: float = 0.0

    objective: str = "binary:logistic"
    eval_metric: str = "logloss"

    random_state: int = 42
    n_jobs: int = -1
    early_stopping_rounds: int = 100

    val_fraction: float = 0.15
    min_train_rows: int = 252
    min_val_rows: int = 60

    # Trading/decision threshold is separate; this is just for evaluation defaults
    default_threshold: float = 0.5


class XGBOpenDirectionClassifier:
    """
    XGBoost classifier for next-day open direction (up/down).

    y should be 0/1 where:
      1 = next-day open return > 0 (or > threshold)
      0 = otherwise
    """

    def __init__(self, config: Optional[XGBClassifierConfig] = None):
        self.cfg = config or XGBClassifierConfig()
        self.model: Optional[XGBClassifier] = None
        self.feature_cols: Optional[list[str]] = None
        # Provide sklearn-style estimator hint for compatibility with callbacks/helpers
        self._estimator_type = "classifier"

    def _time_split(
        self, X: pd.DataFrame, y: pd.Series
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        n = len(X)
        val_n = max(int(n * self.cfg.val_fraction), self.cfg.min_val_rows)
        val_n = min(val_n, n // 3)
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
            drop = {target_col, "date", "symbol", "ticker"}
            feature_cols = [c for c in df.columns if c not in drop]
        self.feature_cols = feature_cols

        X = df[feature_cols].copy()
        y = df[target_col].astype(int).copy()

        X = X.replace([np.inf, -np.inf], np.nan)
        mask = ~(X.isna().any(axis=1) | y.isna())
        X, y = X.loc[mask], y.loc[mask]

        X_train, X_val, y_train, y_val = self._time_split(X, y)

        # Handle imbalance (optional but good): scale_pos_weight = neg/pos in train
        pos = (y_train == 1).sum()
        neg = (y_train == 0).sum()
        scale_pos_weight = float(neg / max(pos, 1))

        self.model = XGBClassifier(
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
            scale_pos_weight=scale_pos_weight,
        )

        fit_kwargs = {
            "eval_set": [(X_val, y_val)],
            "verbose": False,
        }

        try:
            # xgboost<2.0 supported early_stopping_rounds directly in fit
            self.model.fit(
                X_train,
                y_train,
                early_stopping_rounds=self.cfg.early_stopping_rounds,
                **fit_kwargs,
            )
        except TypeError:
            try:
                # xgboost>=2.0 requires callbacks for early stopping
                from xgboost.callback import EarlyStopping  # type: ignore

                self.model.fit(
                    X_train,
                    y_train,
                    callbacks=[
                        EarlyStopping(
                            rounds=self.cfg.early_stopping_rounds, save_best=True
                        )
                    ],
                    **fit_kwargs,
                )
            except Exception:
                # Fallback: train without early stopping if callbacks are unavailable
                self.model.fit(
                    X_train,
                    y_train,
                    **fit_kwargs,
                )

        return self

    def predict_proba(self, df_or_X: pd.DataFrame) -> np.ndarray:
        if self.model is None or self.feature_cols is None:
            raise RuntimeError("Model not fitted yet.")
        X = df_or_X[self.feature_cols].copy() if set(self.feature_cols).issubset(df_or_X.columns) else df_or_X
        X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return self.model.predict_proba(X)[:, 1]  # P(up)

    def predict(self, df_or_X: pd.DataFrame, threshold: Optional[float] = None) -> np.ndarray:
        th = self.cfg.default_threshold if threshold is None else threshold
        p = self.predict_proba(df_or_X)
        return (p >= th).astype(int)

    def save(self, path: str | Path):
        if self.model is None:
            raise RuntimeError("No model to save.")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save_model(str(path))

        meta_path = path.with_suffix(".meta.json")
        meta = {
            "type": "xgb_classifier",
            "feature_cols": self.feature_cols,
            "config": asdict(self.cfg),
        }
        meta_path.write_text(pd.Series(meta).to_json(), encoding="utf-8")

    def load(self, path: str | Path):
        path = Path(path)
        self.model = XGBClassifier()
        self.model.load_model(str(path))

        meta_path = path.with_suffix(".meta.json")
        meta = pd.read_json(meta_path.read_text(encoding="utf-8"), typ="series")
        self.feature_cols = list(meta["feature_cols"])
        return self
