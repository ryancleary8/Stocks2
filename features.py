"""Feature engineering (OPEN → OPEN leak-free)

Goal:
- Predict tomorrow's OPEN using only information known BEFORE today's OPEN.

Inputs:
- data/clean/{SYMBOL}.clean.csv  (from valdiate.py)

Outputs:
- data/features/{SYMBOL}.features.csv

Key rule:
- Any value that didn’t exist before today’s open is illegal as a feature.
  So we use yesterday's completed OHLCV (shifted by 1) and allow today's open
  only for the overnight gap feature.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


REQUIRED_COLS = ["date", "open", "high", "low", "close", "volume"]


def clean_numeric(s: pd.Series) -> pd.Series:
    """Robust numeric cleaner: strips $, commas, whitespace; converts to float."""
    s = s.astype(str).str.replace("$", "", regex=False).str.replace(",", "", regex=False).str.strip()
    return pd.to_numeric(s, errors="coerce")


def calculate_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate leak-free features for OPEN → OPEN prediction.

    All predictive features at day T use ONLY data available
    BEFORE the market open at day T.

    Assumes df is sorted by date ascending.
    """
    df = df.copy()

    # -----------------------------
    # Clean numeric columns
    # -----------------------------
    for col in ["open", "high", "low", "close", "volume"]:
        if col in df.columns:
            df[col] = clean_numeric(df[col])

    # -----------------------------
    # Shift OHLCV by 1 day
    # (yesterday's completed session)
    # -----------------------------
    df["open_prev"] = df["open"].shift(1)
    df["high_prev"] = df["high"].shift(1)
    df["low_prev"] = df["low"].shift(1)
    df["close_prev"] = df["close"].shift(1)
    df["volume_prev"] = df["volume"].shift(1)

    # -----------------------------
    # Core price features (PAST ONLY)
    # -----------------------------
    df["prev_daily_return"] = df["close_prev"].pct_change()
    df["prev_log_return"] = np.log(df["close_prev"] / df["close_prev"].shift(1))

    df["prev_range_pct"] = (df["high_prev"] - df["low_prev"]) / df["close_prev"]
    df["prev_close_to_high"] = (df["high_prev"] - df["close_prev"]) / df["high_prev"]
    df["prev_close_to_low"] = (df["close_prev"] - df["low_prev"]) / df["low_prev"]

    # -----------------------------
    # Overnight gap into today's open (ALLOWED)
    # -----------------------------
    df["gap_open_pct"] = (df["open"] - df["close_prev"]) / df["close_prev"]

    # -----------------------------
    # Moving averages (PAST ONLY)
    # -----------------------------
    df["ma_5"] = df["close_prev"].rolling(5).mean()
    df["ma_10"] = df["close_prev"].rolling(10).mean()
    df["ma_20"] = df["close_prev"].rolling(20).mean()

    df["price_to_ma5"] = df["close_prev"] / df["ma_5"]
    df["price_to_ma20"] = df["close_prev"] / df["ma_20"]

    # -----------------------------
    # Volatility (PAST ONLY)
    # -----------------------------
    df["volatility_5"] = df["prev_daily_return"].rolling(5).std()
    df["volatility_20"] = df["prev_daily_return"].rolling(20).std()

    # -----------------------------
    # EMA-based trend (PAST ONLY)
    # -----------------------------
    df["ema_9"] = df["close_prev"].ewm(span=9, adjust=False).mean()
    df["ema_21"] = df["close_prev"].ewm(span=21, adjust=False).mean()

    df["ema_9_21_diff"] = df["ema_9"] - df["ema_21"]
    df["ema_9_21_ratio"] = df["ema_9"] / df["ema_21"]

    # -----------------------------
    # RSI (14) — PAST ONLY
    # -----------------------------
    delta = df["close_prev"].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df["rsi_14"] = 100 - (100 / (1 + rs))

    # -----------------------------
    # Bollinger Bands (PAST ONLY)
    # -----------------------------
    bb_mid = df["close_prev"].rolling(20).mean()
    bb_std = df["close_prev"].rolling(20).std()

    df["bb_width_20"] = (2 * bb_std) / bb_mid
    df["price_bb_zscore_20"] = (df["close_prev"] - bb_mid) / bb_std

    # -----------------------------
    # ATR (14) — PAST ONLY
    # -----------------------------
    prev_close_shifted = df["close_prev"].shift(1)

    tr1 = df["high_prev"] - df["low_prev"]
    tr2 = (df["high_prev"] - prev_close_shifted).abs()
    tr3 = (df["low_prev"] - prev_close_shifted).abs()

    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df["atr_14"] = true_range.rolling(14).mean()

    # -----------------------------
    # Volume features (PAST ONLY)
    # -----------------------------
    df["volume_change"] = df["volume_prev"].pct_change()
    df["volume_ma_5"] = df["volume_prev"].rolling(5).mean()
    df["volume_ratio"] = df["volume_prev"] / df["volume_ma_5"]

    # -----------------------------
    # Momentum (PAST ONLY)
    # -----------------------------
    df["momentum_5"] = df["close_prev"] - df["close_prev"].shift(5)
    df["momentum_10"] = df["close_prev"] - df["close_prev"].shift(10)

    # -----------------------------
    # Higher-order return stats (PAST ONLY)
    # -----------------------------
    df["return_skew_20"] = df["prev_daily_return"].rolling(20).skew()
    df["return_kurtosis_20"] = df["prev_daily_return"].rolling(20).kurt()

    return df


def build_features_for_symbol(symbol: str, clean_dir: str = "data/clean", out_dir: str = "data/features") -> Path:
    """Load clean CSV, compute features, drop NaNs created by shifting/rolling, save."""
    symbol = symbol.upper().strip()
    in_path = Path(clean_dir) / f"{symbol}.clean.csv"
    if not in_path.exists():
        raise FileNotFoundError(f"Missing clean file: {in_path}")

    df = pd.read_csv(in_path)

    # Ensure schema
    for c in REQUIRED_COLS:
        if c not in df.columns:
            df[c] = pd.NA

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    feat = calculate_features(df)

    # Drop NaNs AFTER shifting/rolling (as your note says)
    # Keep date + today's open (allowed reference) + engineered features
    feat = feat.dropna().reset_index(drop=True)

    out_path = Path(out_dir) / f"{symbol}.features.csv"
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    feat.to_csv(out_path, index=False)
    return out_path


def build_features_all(stocks_file: str = "stocks.txt") -> None:
    """Build features for every symbol in stocks.txt (fallback stock.txt)."""
    symbols = []
    for fname in (stocks_file, "stock.txt"):
        p = Path(fname)
        if p.exists():
            with p.open() as f:
                symbols = [line.strip().upper() for line in f if line.strip()]
            break
    if not symbols:
        raise FileNotFoundError("Couldn't find stocks.txt or stock.txt")

    ok = 0
    failed = []
    for sym in symbols:
        try:
            out = build_features_for_symbol(sym)
            print(f"✓ {sym} -> {out}")
            ok += 1
        except Exception as e:
            print(f"✗ {sym} failed: {e}")
            failed.append(sym)

    print(f"\nDone: {ok}/{len(symbols)} feature files written")
    if failed:
        print("Failed:")
        print(", ".join(failed))


if __name__ == "__main__":
    # Usage:
    #   python features.py           -> build for all tickers in stocks.txt
    #   python features.py AAPL      -> build for one ticker
    arg = sys.argv[1].strip().upper() if len(sys.argv) > 1 else "ALL"
    if arg == "ALL":
        build_features_all()
    else:
        out = build_features_for_symbol(arg)
        print(f"✓ {arg} -> {out}")
