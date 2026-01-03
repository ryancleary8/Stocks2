"""Predict next-day open price/direction for all symbols in ./current.

The script expects one CSV per symbol inside the ``current`` directory
with filename ``{SYMBOL}.{YYYY-MM-DD}.csv`` containing columns
``timestamp_utc,symbol,price`` produced by ``scraper.py``.  Historical
OHLCV data must already exist (e.g., in ``processed/`` or
``data/clean/``) so features can be computed up to the latest open
price. Models are loaded from ``trained_models/``.

Outputs
-------
- ``predict_results/{YYYY-MM-DD}.csv`` with symbols as rows and model
  predictions as columns.  It also includes the latest open price and a
  boolean flag indicating whether the latest open is above the prior
  close.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import joblib
import pandas as pd

from features import calculate_features
from models.xgb.xgb_classifier import XGBOpenDirectionClassifier
from models.xgb.xgb_regressor import XGBOpenReturnRegressor

CURRENT_DIR = Path("current")
TRAINED_MODELS_DIR = Path("trained_models")
RESULTS_DIR = Path("predict_results")
HISTORY_LOCATIONS = [Path("processed"), Path("data/clean"), Path("data"), Path(".")]


@dataclass
class CurrentQuote:
    symbol: str
    date: pd.Timestamp
    price: float
    source_path: Path


def _read_current_quotes() -> list[CurrentQuote]:
    quotes: list[CurrentQuote] = []
    if not CURRENT_DIR.exists():
        raise FileNotFoundError(f"Missing current directory: {CURRENT_DIR}")

    for path in CURRENT_DIR.glob("*.csv"):
        try:
            symbol, date_str, _ext = path.name.split(".")
        except ValueError:
            print(f"⚠️  Skipping unexpected filename format: {path.name}")
            continue

        df = pd.read_csv(path)
        if "price" not in df.columns:
            print(f"⚠️  Skipping {path.name}: missing 'price' column")
            continue

        price = float(df.iloc[0]["price"])
        quotes.append(
            CurrentQuote(
                symbol=symbol.upper(),
                date=pd.to_datetime(date_str, errors="coerce"),
                price=price,
                source_path=path,
            )
        )

    return quotes


def _find_history_path(symbol: str) -> Optional[Path]:
    candidates = [
        f"{symbol}.csv",
        f"{symbol}.clean.csv",
        f"{symbol.upper()}.csv",
        f"{symbol.upper()}.clean.csv",
    ]

    for base in HISTORY_LOCATIONS:
        for cand in candidates:
            p = base / cand
            if p.exists():
                return p
    return None


def _load_history_with_today(symbol: str, quote: CurrentQuote) -> pd.DataFrame:
    hist_path = _find_history_path(symbol)
    if hist_path is None:
        raise FileNotFoundError(
            f"No historical file found for {symbol}. Checked: {[str(p) for p in HISTORY_LOCATIONS]}"
        )

    df = pd.read_csv(hist_path)
    if "date" not in df.columns:
        raise ValueError(f"Historical file for {symbol} missing 'date' column: {hist_path}")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    # If the latest date is already present, overwrite open with scraped value
    if (df["date"] == quote.date).any():
        idx = df[df["date"] == quote.date].index[-1]
        df.loc[idx, "open"] = quote.price
    else:
        # Append placeholder row for day T so we can compute features for T
        last_row = df.iloc[-1]
        new_row = {
            "date": quote.date,
            "open": quote.price,
            # Use previous close/high/low/volume as placeholders to avoid NaNs
            "high": last_row.get("close", last_row.get("open", quote.price)),
            "low": last_row.get("close", last_row.get("open", quote.price)),
            "close": last_row.get("close", quote.price),
            "volume": last_row.get("volume", 0),
        }
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

    return df


def _latest_feature_row(df: pd.DataFrame, as_of: pd.Timestamp) -> pd.Series:
    feat = calculate_features(df)
    feat = feat.dropna().sort_values("date").reset_index(drop=True)
    if feat.empty:
        raise ValueError("Feature dataframe is empty after cleaning")

    # Use the feature row matching the target date; fallback to last available
    match = feat[feat["date"] == as_of]
    if not match.empty:
        return match.iloc[-1]
    return feat.iloc[-1]


def _load_model_meta(model_dir: Path) -> dict:
    """Load feature columns + type metadata when available."""

    meta: dict = {}

    feature_path = model_dir / "feature_cols.json"
    if feature_path.exists():
        try:
            meta["feature_cols"] = json.loads(feature_path.read_text())
        except Exception:
            pass

    meta_path = model_dir / "model.meta.json"
    if meta_path.exists():
        try:
            meta.update(json.loads(meta_path.read_text()))
        except Exception:
            pass

    return meta


def _predict_joblib(model_path: Path, features: pd.DataFrame) -> float:
    model = joblib.load(model_path)
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(features)
        # Return probability of class 1 when available
        if proba.ndim > 1 and proba.shape[1] > 1:
            return float(proba[:, 1][0])
    preds = model.predict(features)
    return float(preds[0])


def _predict_xgb_reg(model_path: Path, features: pd.DataFrame) -> float:
    model = XGBOpenReturnRegressor()
    model.load(model_path)
    return float(model.predict(features)[0])


def _predict_xgb_cls(model_path: Path, features: pd.DataFrame) -> float:
    model = XGBOpenDirectionClassifier()
    model.load(model_path)
    proba = model.predict_proba(features)
    return float(proba[0])


def _gather_model_paths(symbol: str) -> dict[str, Path]:
    """Return mapping of model-key -> path for a symbol.

    Supports both directory layouts:
    - trained_models/{symbol}/{model}/{task}/model.*
    - trained_models/{model}/{symbol}/{task}/model.*
    """

    paths: dict[str, Path] = {}

    def scan(root: Path) -> Iterable[Path]:
        if not root.exists():
            return []
        return root.rglob("model.*")

    for p in scan(TRAINED_MODELS_DIR / symbol):
        key = f"{p.parent.parent.name}-{p.parent.name}"  # model-task
        paths[key] = p

    for p in scan(TRAINED_MODELS_DIR):
        try:
            sym_component = p.parents[1].name
        except Exception:
            continue
        if sym_component.upper() != symbol.upper():
            continue
        key = f"{p.parent.parent.name}-{p.parent.name}"
        paths.setdefault(key, p)

    return paths


def _make_feature_frame(row: pd.Series, feature_cols: Optional[list[str]]) -> pd.DataFrame:
    if feature_cols:
        missing = [c for c in feature_cols if c not in row.index]
        if missing:
            raise ValueError(f"Missing required features: {missing}")
        X = row[feature_cols].to_frame().T
    else:
        drop_cols = {"date", "open", "high", "low", "close", "volume"}
        X = row.drop(labels=[c for c in row.index if c in drop_cols]).to_frame().T
    return X.replace([pd.NA, float("inf"), float("-inf")], 0).fillna(0.0)


def predict_for_symbol(quote: CurrentQuote) -> dict[str, float]:
    history = _load_history_with_today(quote.symbol, quote)
    feat_row = _latest_feature_row(history, quote.date)

    prev_close = history.loc[history["date"] < quote.date, "close"].iloc[-1]
    gap_up_flag = bool(quote.price > prev_close)

    preds: Dict[str, float] = {
        "open_price": quote.price,
        "open_gt_prev_close": gap_up_flag,
    }

    model_paths = _gather_model_paths(quote.symbol)
    if not model_paths:
        print(f"⚠️  No models found for {quote.symbol}")
        return preds

    for key, model_path in sorted(model_paths.items()):
        meta = _load_model_meta(model_path.parent)
        feature_cols = meta.get("feature_cols")
        try:
            X = _make_feature_frame(feat_row, feature_cols)
        except Exception as exc:
            print(f"⚠️  Skipping {quote.symbol} {key}: {exc}")
            continue

        try:
            model_type = str(meta.get("type", "")).lower()
            if "xgb_cls" in key or "classifier" in model_type:
                preds[key] = _predict_xgb_cls(model_path, X)
            elif "xgb_reg" in key or model_path.suffix == ".json" or "regressor" in model_type:
                preds[key] = _predict_xgb_reg(model_path, X)
            else:
                preds[key] = _predict_joblib(model_path, X)
        except Exception as exc:
            print(f"⚠️  Prediction failed for {quote.symbol} {key}: {exc}")
            continue

    return preds


def main() -> None:
    quotes = _read_current_quotes()
    if not quotes:
        print("No current quotes found in ./current")
        sys.exit(1)

    dates = {q.date.normalize() for q in quotes}
    if len(dates) != 1:
        raise ValueError(f"Mixed dates found in current folder: {sorted(dates)}")
    run_date = dates.pop()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"{run_date.date()}.csv"

    records: list[dict[str, object]] = []
    for quote in quotes:
        try:
            preds = predict_for_symbol(quote)
            record = {"symbol": quote.symbol, **preds}
            records.append(record)
        except Exception as exc:
            print(f"⚠️  Skipping {quote.symbol} due to error: {exc}")

    if not records:
        print("No predictions generated.")
        sys.exit(1)

    # Normalize columns so every model appears as a separate column
    all_keys: set[str] = set()
    for rec in records:
        all_keys.update(rec.keys())
    ordered_keys = ["symbol", "open_price", "open_gt_prev_close"] + sorted(k for k in all_keys if k not in {"symbol", "open_price", "open_gt_prev_close"})

    df_out = pd.DataFrame(records)
    df_out = df_out.reindex(columns=ordered_keys)
    df_out.to_csv(out_path, index=False)
    print(f"✓ Wrote predictions to {out_path}")


if __name__ == "__main__":
    main()
