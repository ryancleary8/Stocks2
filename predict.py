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
import numpy as np
import pandas as pd
import torch

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

    # Also check metrics.json for metadata
    metrics_path = model_dir / "metrics.json"
    if metrics_path.exists():
        try:
            metrics_data = json.loads(metrics_path.read_text())
            meta.setdefault("seq_len", metrics_data.get("seq_len"))
            meta.setdefault("task", metrics_data.get("task"))
        except Exception:
            pass

    return meta


def _ensure_estimator_type(model: object, is_classifier: bool) -> None:
    """Best-effort patching of ``_estimator_type`` on estimators and containers.

    This mirrors the helper inside the XGBoost wrappers but is defensive for
    arbitrary sklearn objects, pipelines, and nested boosters. Any failures are
    swallowed so prediction can continue.
    """

    if model is None:
        return

    est_type = "classifier" if is_classifier else "regressor"

    def _set(target: object) -> None:
        if target is None:
            return
        for attr_target in (target, getattr(target, "__class__", None)):
            if attr_target is None:
                continue
            try:
                setattr(attr_target, "_estimator_type", est_type)
            except Exception:
                pass

    _set(model)

    # Pipelines store steps in named_steps/steps; patch each component so
    # downstream "is_classifier" checks don't choke on internals.
    if hasattr(model, "named_steps"):
        try:
            steps = list(getattr(model, "named_steps").values())
            for step in steps:
                _ensure_estimator_type(step, is_classifier)
        except Exception:
            pass
    elif hasattr(model, "steps"):
        try:
            for _name, step in getattr(model, "steps"):
                _ensure_estimator_type(step, is_classifier)
        except Exception:
            pass

    # Some XGBoost sklearn classes expose a booster that also needs patching.
    try:
        booster = model.get_booster()
        _set(booster)
    except Exception:
        pass


def _predict_joblib(model_path: Path, features: pd.DataFrame, is_classifier: bool) -> float:
    """Load and predict from a joblib model, handling _estimator_type issues."""
    model = joblib.load(model_path)

    # Fix _estimator_type before any operations
    _ensure_estimator_type(model, is_classifier)
    
    # Many sklearn estimators/pipelines were fitted without feature names.
    # Passing a DataFrame then triggers noisy warnings like:
    # "X has feature names, but <Estimator> was fitted without feature names".
    # To keep output clean and behavior consistent, detect that case and
    # pass a NumPy array instead.
    X_in = features
    try:
        # If estimator exposes feature_names_in_, it was fitted with names.
        # Pipelines may not, so we check the final estimator when possible.
        fitted_with_names = False
        if hasattr(model, "feature_names_in_"):
            fitted_with_names = True
        elif hasattr(model, "named_steps"):
            steps = list(getattr(model, "named_steps").values())
            if steps and hasattr(steps[-1], "feature_names_in_"):
                fitted_with_names = True

        if not fitted_with_names and isinstance(features, pd.DataFrame):
            X_in = features.to_numpy(dtype=float, copy=False)
    except Exception:
        # If anything goes wrong, just keep the original input.
        X_in = features

    # For classifiers, prefer predict_proba
    if is_classifier and hasattr(model, "predict_proba"):
        try:
            proba = model.predict_proba(X_in)
            if hasattr(proba, "ndim") and proba.ndim > 1 and proba.shape[1] > 1:
                return float(proba[:, 1][0])
        except Exception:
            pass

    # Fall back to predict
    preds = model.predict(X_in)
    return float(preds[0])


def _predict_xgb_reg(model_path: Path, features: pd.DataFrame) -> float:
    model = XGBOpenReturnRegressor()
    model.load(model_path)
    _ensure_estimator_type(model, is_classifier=False)
    _ensure_estimator_type(getattr(model, "model", None), is_classifier=False)
    return float(model.predict(features)[0])


def _predict_xgb_cls(model_path: Path, features: pd.DataFrame) -> float:
    model = XGBOpenDirectionClassifier()
    model.load(model_path)
    _ensure_estimator_type(model, is_classifier=True)
    _ensure_estimator_type(getattr(model, "model", None), is_classifier=True)
    proba = model.predict_proba(features)
    return float(proba[0])


def _load_pytorch_model(model_path: Path, model_dir: Path, model_type: str) -> torch.nn.Module:
    """Load a PyTorch model (CNN, LSTM, or Transformer)."""
    
    # Load metadata to get model architecture info
    meta = _load_model_meta(model_dir)
    task = meta.get("task", "reg")
    
    # Determine model type and import the appropriate class
    if "cnn" in model_type:
        from models.cnn.cnn_model import CNN1DPredictor
        
        n_features = len(meta.get("feature_cols", []))
        model = CNN1DPredictor(
            in_channels=n_features,
            conv_channels=64,
            kernel_size=5,
            dropout=0.15,
            task=task
        )
    elif "lstm" in model_type:
        from models.rnn.lstm_model import StockLSTM
        
        n_features = len(meta.get("feature_cols", []))
        model = StockLSTM(
            input_dim=n_features,
            hidden_dim=64,
            num_layers=1,
            dropout=0.25,
            task=task
        )
    elif "transformer" in model_type:
        from models.transformer.ts_transformer import TimeSeriesTransformer
        
        n_features = len(meta.get("feature_cols", []))
        model = TimeSeriesTransformer(
            n_features=n_features,
            d_model=64,
            nhead=4,
            num_layers=2,
            dropout=0.10,
            task=task
        )
    else:
        raise ValueError(f"Unknown PyTorch model type: {model_type}")
    
    # Load the state dict
    state_dict = torch.load(model_path, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    model.eval()
    
    return model


def _prepare_sequence_input(
    history: pd.DataFrame, 
    feat_row: pd.Series, 
    feature_cols: List[str],
    seq_len: int,
    model_type: str
) -> np.ndarray:
    """Prepare sequence input for PyTorch models (CNN, LSTM, Transformer).
    
    Returns:
        For CNN: (1, channels, seq_len) 
        For LSTM/Transformer: (1, seq_len, features)
    """
    # Calculate features for entire history
    feat_df = calculate_features(history)
    feat_df = feat_df.dropna().sort_values("date").reset_index(drop=True)
    
    if len(feat_df) < seq_len:
        raise ValueError(f"Not enough history ({len(feat_df)} rows) for seq_len={seq_len}")
    
    # Get the last seq_len rows of features
    seq_data = feat_df[feature_cols].iloc[-seq_len:].values.astype(np.float32)
    
    # Load scaler if it exists
    scaler_path = feat_row.name if hasattr(feat_row, 'name') else None
    if scaler_path and isinstance(scaler_path, Path):
        scaler_path = scaler_path.parent / "scaler.joblib"
    else:
        # Try to find scaler in model directory
        # This will be passed in context
        scaler_path = None
    
    # For now, we'll handle scaling inline
    # In production, you'd want to load the saved scaler
    
    if "cnn" in model_type:
        # CNN expects (batch, channels, seq_len)
        # Input is (seq_len, features), transpose to (features, seq_len)
        seq_data = seq_data.T  # (features, seq_len)
        seq_data = np.expand_dims(seq_data, 0)  # (1, features, seq_len)
    else:
        # LSTM/Transformer expect (batch, seq_len, features)
        seq_data = np.expand_dims(seq_data, 0)  # (1, seq_len, features)
    
    return seq_data


def _predict_pytorch(
    model_path: Path, 
    model_dir: Path,
    history: pd.DataFrame,
    feat_row: pd.Series,
    feature_cols: List[str],
    model_type: str,
    task: str,
    seq_len: int = 30
) -> float:
    """Predict using a PyTorch model (CNN, LSTM, or Transformer)."""
    
    # Load the model
    model = _load_pytorch_model(model_path, model_dir, model_type)
    
    # Prepare sequence input
    seq_data = _prepare_sequence_input(history, feat_row, feature_cols, seq_len, model_type)
    
    # Convert to tensor
    x_tensor = torch.tensor(seq_data, dtype=torch.float32)
    
    # Make prediction
    with torch.no_grad():
        output = model(x_tensor)
        
        # Handle output based on task
        if task == "cls":
            # For classification, apply sigmoid to get probability
            prob = torch.sigmoid(output).item()
            return float(prob)
        else:
            # For regression, return the raw output
            return float(output.item())


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

        # IMPORTANT:
        # rglob("model.*") also matches files like "model.meta.json" which are
        # metadata (not the actual model). Feeding those into XGBoost's loader
        # causes crashes like: "Invalid cast, from Null to Object".
        allowed_suffixes = {".joblib", ".pkl", ".pickle", ".json", ".ubj", ".pt", ".pth"}
        for p in root.rglob("model.*"):
            name = p.name.lower()
            # Exclude known metadata files
            if name in {"model.meta.json", "model.meta"}:
                continue
            # Allow model artifacts only
            if p.suffix.lower() not in allowed_suffixes:
                continue
            # Exclude any other "*.meta.json" patterns defensively
            if name.endswith(".meta.json"):
                continue
            yield p

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

    # Clean feature frame while opting into pandas' future downcasting behavior
    with pd.option_context("future.no_silent_downcasting", True):
        cleaned = X.replace([pd.NA, float("inf"), float("-inf")], 0)
    cleaned = cleaned.infer_objects(copy=False)
    return cleaned.fillna(0.0)


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
        
        # Skip if no feature columns found
        if not feature_cols:
            print(f"⚠️  Skipping {quote.symbol} {key}: no feature_cols.json found")
            continue
            
        try:
            model_type = str(meta.get("type", "")).lower()
            task = meta.get("task", "reg")
            
            # Determine if this is a classifier or regressor
            is_classifier = ("cls" in key or "classifier" in model_type or task == "cls")
            
            # Handle PyTorch models
            if model_path.suffix.lower() in {".pt", ".pth"}:
                # Determine model type from key
                if "cnn" in key:
                    model_type_name = "cnn"
                elif "lstm" in key:
                    model_type_name = "lstm"
                elif "transformer" in key:
                    model_type_name = "transformer"
                else:
                    print(f"⚠️  Skipping {quote.symbol} {key}: unknown PyTorch model type")
                    continue
                
                # Get sequence length from metadata (default to reasonable values)
                seq_len = meta.get("seq_len", 30 if model_type_name == "cnn" else 20)
                
                preds[key] = _predict_pytorch(
                    model_path, 
                    model_path.parent,
                    history,
                    feat_row,
                    feature_cols,
                    model_type_name,
                    task,
                    seq_len
                )
            # Handle XGBoost models
            elif "xgb_cls" in key or model_type == "xgb_classifier":
                X = _make_feature_frame(feat_row, feature_cols)

                # Some "xgb_*" models may have been saved via joblib/pickle (sklearn style).
                # Those require _estimator_type to be set, so route through _predict_joblib.
                if model_path.suffix.lower() in {".joblib", ".pkl", ".pickle"}:
                    preds[key] = _predict_joblib(model_path, X, is_classifier=True)
                else:
                    preds[key] = _predict_xgb_cls(model_path, X)

            elif "xgb_reg" in key or model_type == "xgb_regressor" or model_path.suffix.lower() in {".json", ".ubj"}:
                X = _make_feature_frame(feat_row, feature_cols)

                # Same deal: if stored as sklearn-serialized artifact, predict via joblib path.
                if model_path.suffix.lower() in {".joblib", ".pkl", ".pickle"}:
                    preds[key] = _predict_joblib(model_path, X, is_classifier=False)
                else:
                    preds[key] = _predict_xgb_reg(model_path, X)

            # Handle sklearn models
            else:
                X = _make_feature_frame(feat_row, feature_cols)
                preds[key] = _predict_joblib(model_path, X, is_classifier)
                
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