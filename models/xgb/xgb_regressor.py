# models/xgb/xgb_regressor.py
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin

try:
    from xgboost import XGBRegressor
except ImportError as e:
    raise ImportError(
        "Missing dependency: xgboost. Install with: pip install xgboost"
    ) from e


def _ensure_estimator_type(obj: object, est_type: str) -> None:
    """Set `_estimator_type` on an object and its class if possible."""

    if obj is None:
        return

    for target in (obj, getattr(obj, "__class__", None)):
        if target is None:
            continue
        try:
            setattr(target, "_estimator_type", est_type)
        except Exception:
            # Some objects like xgboost.Booster don't allow attribute setting
            pass

    # Also patch an attached Booster if present; some sklearn utilities
    # inspect nested estimators.
    try:
        booster = obj.get_booster()
        for target in (booster, getattr(booster, "__class__", None)):
            try:
                setattr(target, "_estimator_type", est_type)
            except Exception:
                pass
    except Exception:
        pass


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


class XGBOpenReturnRegressor(BaseEstimator, RegressorMixin):
    """
    XGBoost regressor for next-day open return prediction.

    Assumptions:
      - df is time-sorted by date ascending
      - target column is numeric return (e.g. (open[t+1]-open[t])/open[t])
      - features contain ONLY allowed (leak-free) inputs at prediction time
    """

    # Explicitly set for sklearn meta-estimators that rely on this attribute
    _estimator_type: str = "regressor"

    def __init__(self, config: Optional[XGBRegressorConfig] = None):
        self.cfg = config or XGBRegressorConfig()
        self.model: Optional[XGBRegressor] = None
        self.feature_cols: Optional[list[str]] = None
        _ensure_estimator_type(self, "regressor")

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

    def fit(
        self,
        X,
        y: Optional[Union[pd.Series, np.ndarray, str]] = None,
        feature_cols: Optional[list[str]] = None,
        target_col: Optional[str] = None,
    ):
        """
        Train the regressor.

        Supports both the legacy signature ``fit(df, target_col, feature_cols=None)``
        and the sklearn-style ``fit(X, y)``.
        """

        # Legacy signature: fit(df, target_col, feature_cols=None)
        if isinstance(X, pd.DataFrame) and (isinstance(y, str) or target_col):
            target_col = target_col or y  # type: ignore[assignment]
            if target_col is None:
                raise ValueError("target_col must be provided for dataframe training")
            if feature_cols is None:
                drop = {target_col, "date", "symbol", "ticker"}
                feature_cols = [c for c in X.columns if c not in drop]
            self.feature_cols = feature_cols

            X_df = X[feature_cols].copy()
            y_series = X[target_col].astype(float).copy()

            X_df = X_df.replace([np.inf, -np.inf], np.nan)
            mask = ~(X_df.isna().any(axis=1) | y_series.isna())
            X_df, y_series = X_df.loc[mask], y_series.loc[mask]

            X_train, X_val, y_train, y_val = self._time_split(X_df, y_series)

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
        else:
            # Sklearn-style signature: fit(X, y)
            if y is None:
                raise ValueError("y must be provided when using sklearn-style fit")

            X_df = pd.DataFrame(X)
            if feature_cols is None:
                feature_cols = list(X_df.columns)
            self.feature_cols = feature_cols
            X_df = X_df[self.feature_cols]

            y_series = pd.Series(y).astype(float)
            X_train, X_val, y_train, y_val = self._time_split(X_df, y_series)

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

        _ensure_estimator_type(self.model, "regressor")

        fit_kwargs: Dict[str, object] = {
            "eval_set": [(X_val, y_val)],
            "verbose": False,
        }

        try:
            self.model.fit(
                X_train,
                y_train,
                early_stopping_rounds=self.cfg.early_stopping_rounds,
                **fit_kwargs,
            )
        except TypeError:
            try:
                from xgboost.callback import EarlyStopping  # type: ignore

                self.model.fit(
                    X_train,
                    y_train,
                    callbacks=[EarlyStopping(rounds=self.cfg.early_stopping_rounds, save_best=True)],
                    **fit_kwargs,
                )
            except Exception:
                self.model.fit(
                    X_train,
                    y_train,
                    **fit_kwargs,
                )

        return self

    def predict(self, df_or_X: pd.DataFrame) -> np.ndarray:
        if self.model is None or self.feature_cols is None:
            raise RuntimeError("Model not fitted yet.")
        _ensure_estimator_type(self.model, "regressor")
        _ensure_estimator_type(self, "regressor")
        X = df_or_X[self.feature_cols].copy() if set(self.feature_cols).issubset(df_or_X.columns) else df_or_X
        X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return self.model.predict(X)

    def save(self, path: Union[str, Path]):
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

    def load(self, path: Union[str, Path]):
        path = Path(path)
        self.model = XGBRegressor()
        self.model.load_model(str(path))

        _ensure_estimator_type(self.model, "regressor")

        meta_path = path.with_suffix(".meta.json")
        meta = pd.read_json(meta_path.read_text(encoding="utf-8"), typ="series")
        self.feature_cols = list(meta["feature_cols"])
        # config rehydration is optional; keep current cfg
        _ensure_estimator_type(self, "regressor")
        return self
