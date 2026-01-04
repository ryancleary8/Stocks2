# models/xgb/xgb_classifier.py
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Tuple, Union, Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin


def _get_xgb_classifier() -> Any:
    """Lazily import :class:`xgboost.XGBClassifier` with a clear error."""

    try:
        from xgboost import XGBClassifier  # type: ignore
    except ImportError as exc:  # pragma: no cover - runtime guard
        raise ImportError(
            "Missing dependency: xgboost. Install with: pip install xgboost"
        ) from exc

    # Some environments strip `_estimator_type` during serialization or ship
    # XGBoost builds without the sklearn mixin metadata wired up. Patch the
    # class attribute defensively so any instance advertises itself as a
    # classifier to sklearn helpers.
    try:
        if getattr(XGBClassifier, "_estimator_type", None) != "classifier":
            XGBClassifier._estimator_type = "classifier"  # type: ignore[attr-defined]
    except Exception:
        pass

    return XGBClassifier


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


class XGBOpenDirectionClassifier(BaseEstimator, ClassifierMixin):
    """
    XGBoost classifier for next-day open direction (up/down).

    y should be 0/1 where:
      1 = next-day open return > 0 (or > threshold)
      0 = otherwise
    """

    # Explicitly set for sklearn meta-estimators that rely on this attribute
    _estimator_type: str = "classifier"


    def __init__(self, config: Optional[XGBClassifierConfig] = None):
        self.cfg = config or XGBClassifierConfig()
        self.model: Optional[Any] = None
        self.feature_cols: Optional[list[str]] = None
        _ensure_estimator_type(self, "classifier")

    @property
    def estimator_type(self) -> str:
        """Expose estimator type for sklearn utilities."""

        return "classifier"

    def __sklearn_tags__(self) -> dict[str, object]:
        """
        Expose correct estimator type for scikit-learn tooling.
        """
        # Base tags from ClassifierMixin if available
        try:
            tags = super().__sklearn_tags__()  # type: ignore[attr-defined]
        except Exception:
            tags = {}

        # Explicitly set classifier type
        tags["estimator_type"] = "classifier"
        return tags

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

    def fit(
        self,
        X,
        y: Optional[Union[pd.Series, np.ndarray, str]] = None,
        feature_cols: Optional[list[str]] = None,
        target_col: Optional[str] = None,
    ):
        """
        Train the classifier.

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
            y_series = X[target_col].astype(int).copy()

            X_df = X_df.replace([np.inf, -np.inf], np.nan)
            mask = ~(X_df.isna().any(axis=1) | y_series.isna())
            X_df, y_series = X_df.loc[mask], y_series.loc[mask]

            X_train, X_val, y_train, y_val = self._time_split(X_df, y_series)

            pos = (y_train == 1).sum()
            neg = (y_train == 0).sum()
            scale_pos_weight = float(neg / max(pos, 1))

            XGBClassifier = _get_xgb_classifier()

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
        else:
            # Sklearn-style signature: fit(X, y)
            if y is None:
                raise ValueError("y must be provided when using sklearn-style fit")

            X_df = pd.DataFrame(X)
            if feature_cols is None:
                feature_cols = list(X_df.columns)
            self.feature_cols = feature_cols
            X_df = X_df[self.feature_cols]

            y_series = pd.Series(y).astype(int)
            X_train, X_val, y_train, y_val = self._time_split(X_df, y_series)

            pos = (y_train == 1).sum()
            neg = (y_train == 0).sum()
            scale_pos_weight = float(neg / max(pos, 1))

            XGBClassifier = _get_xgb_classifier()

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

        _ensure_estimator_type(self.model, "classifier")

        fit_kwargs = {
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
                    callbacks=[
                        EarlyStopping(
                            rounds=self.cfg.early_stopping_rounds, save_best=True
                        )
                    ],
                    **fit_kwargs,
                )
            except Exception:
                self.model.fit(
                    X_train,
                    y_train,
                    **fit_kwargs,
                )

        return self

    def predict_proba(self, df_or_X: pd.DataFrame) -> np.ndarray:
        if self.model is None or self.feature_cols is None:
            raise RuntimeError("Model not fitted yet.")
        _ensure_estimator_type(self.model, "classifier")
        _ensure_estimator_type(self, "classifier")
        X = (
            df_or_X[self.feature_cols].copy()
            if isinstance(df_or_X, pd.DataFrame)
            and set(self.feature_cols).issubset(df_or_X.columns)
            else pd.DataFrame(df_or_X, columns=self.feature_cols)
        )
        X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        proba = self.model.predict_proba(X)
        if getattr(proba, "ndim", 1) == 1:
            proba = np.column_stack([1 - proba, proba])
        return proba

    def predict(self, df_or_X: pd.DataFrame, threshold: Optional[float] = None) -> np.ndarray:
        th = self.cfg.default_threshold if threshold is None else threshold
        proba = self.predict_proba(df_or_X)
        positive = proba[:, 1] if proba.ndim > 1 else proba
        return (positive >= th).astype(int)

    def save(self, path: Union[str, Path]):
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

    def load(self, path: Union[str, Path]):
        path = Path(path)

        # Load underlying XGBClassifier model
        XGBClassifier = _get_xgb_classifier()

        self.model = XGBClassifier()
        self.model.load_model(str(path))

        # Ensure sklearn sees both wrapper & internal model as classifiers
        _ensure_estimator_type(self, "classifier")
        _ensure_estimator_type(self.model, "classifier")
        try:
            booster = self.model.get_booster()
            _ensure_estimator_type(booster, "classifier")
        except Exception:
            pass

        # Load feature columns from metadata
        meta_path = path.with_suffix(".meta.json")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        self.feature_cols = list(meta.get("feature_cols", []))

        # Tag self again after loading metadata
        _ensure_estimator_type(self, "classifier")
        return self
