

"""Dataset builder for OPEN → OPEN modeling (regression + classification).

Inputs:
- data/features/{SYMBOL}.features.csv   (built by features.py)

Outputs:
- data/datasets/{SYMBOL}.reg_open_next.dataset.csv
- data/datasets/{SYMBOL}.reg_ret_next.dataset.csv
- data/datasets/{SYMBOL}.cls_up.dataset.csv
- data/datasets/{SYMBOL}.cls_bigmove.dataset.csv

Notes:
- This is leak-safe for your OPEN→OPEN setup.
- We intentionally DROP same-day OHLCV (high/low/close/volume) because those are not known at the open.
- We also drop same-day open itself; the only “today open” signal we keep is gap_open_pct which already uses it.

Run:
- python datasets.py            # all tickers in stocks.txt (fallback stock.txt)
- python datasets.py AAPL       # one ticker
- python datasets.py ALL 0.01   # all tickers, big-move threshold = 1%
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


FEATURES_DIR = Path("data/features")
OUT_DIR = Path("data/datasets")

# Columns that are not allowed as features at the open (same-day session data).
DROP_SAME_DAY_COLS = {"open", "high", "low", "close", "volume"}


def load_symbols() -> list[str]:
    """Loads symbols from stocks.txt, falling back to stock.txt."""
    for fname in ("stocks.txt", "stock.txt"):
        p = Path(fname)
        if p.exists():
            with p.open() as f:
                return [line.strip().upper() for line in f if line.strip()]
    raise FileNotFoundError("Couldn't find stocks.txt or stock.txt in the project root")


def _infer_feature_columns(df: pd.DataFrame) -> list[str]:
    """Infer model feature columns from a features dataframe.

    Rule:
    - Keep numeric columns except same-day OHLCV.
    - Always drop label columns if they exist.
    - Keep engineered features like *_prev, rolling stats, gap_open_pct, etc.
    """
    cols = []
    for c in df.columns:
        if c == "date":
            continue
        if c in DROP_SAME_DAY_COLS:
            continue
        if c.startswith("y_"):
            continue
        # keep only numeric-ish columns (after coercion below)
        cols.append(c)
    return cols


def build_datasets_for_symbol(symbol: str, big_move_threshold: float = 0.01) -> dict[str, Path]:
    """Build regression + classification datasets for one symbol."""
    symbol = symbol.upper().strip()
    in_path = FEATURES_DIR / f"{symbol}.features.csv"
    if not in_path.exists():
        raise FileNotFoundError(f"Missing features file: {in_path}")

    df = pd.read_csv(in_path)
    if "date" not in df.columns:
        raise ValueError(f"{symbol} features missing 'date' column")

    # Parse / sort
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    # Ensure numeric for all non-date columns
    for c in df.columns:
        if c == "date":
            continue
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Labels (tomorrow open)
    if "open" not in df.columns:
        raise ValueError(f"{symbol} features missing raw 'open' needed to create labels")

    open_t = df["open"]
    open_next = open_t.shift(-1)

    df["y_open_next"] = open_next
    df["y_ret_next"] = (open_next - open_t) / open_t
    df["y_up"] = (open_next > open_t).astype("int64")
    df["y_bigmove"] = (df["y_ret_next"].abs() >= float(big_move_threshold)).astype("int64")

    # Build feature matrix columns
    feature_cols = _infer_feature_columns(df)

    # Drop rows where features or labels are missing (last row will be missing labels)
    base = df[["date"] + feature_cols + ["y_open_next", "y_ret_next", "y_up", "y_bigmove"]].copy()

    # IMPORTANT: remove any rows with NaNs after shifting/rolling + label shift
    base = base.dropna().reset_index(drop=True)

    # Output dir
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    outputs: dict[str, Path] = {}

    # Regression: predict next open
    reg_open = base[["date"] + feature_cols + ["y_open_next"]].rename(columns={"y_open_next": "y"})
    p1 = OUT_DIR / f"{symbol}.reg_open_next.dataset.csv"
    reg_open.to_csv(p1, index=False)
    outputs["reg_open_next"] = p1

    # Regression: predict next open return
    reg_ret = base[["date"] + feature_cols + ["y_ret_next"]].rename(columns={"y_ret_next": "y"})
    p2 = OUT_DIR / f"{symbol}.reg_ret_next.dataset.csv"
    reg_ret.to_csv(p2, index=False)
    outputs["reg_ret_next"] = p2

    # Classification: up/down
    cls_up = base[["date"] + feature_cols + ["y_up"]].rename(columns={"y_up": "y"})
    p3 = OUT_DIR / f"{symbol}.cls_up.dataset.csv"
    cls_up.to_csv(p3, index=False)
    outputs["cls_up"] = p3

    # Classification: big move
    cls_big = base[["date"] + feature_cols + ["y_bigmove"]].rename(columns={"y_bigmove": "y"})
    p4 = OUT_DIR / f"{symbol}.cls_bigmove.dataset.csv"
    cls_big.to_csv(p4, index=False)
    outputs["cls_bigmove"] = p4

    return outputs


def build_all(big_move_threshold: float = 0.01) -> None:
    symbols = load_symbols()
    ok = 0
    failed: list[str] = []

    for sym in symbols:
        try:
            outs = build_datasets_for_symbol(sym, big_move_threshold=big_move_threshold)
            # print a compact one-liner
            print(f"✓ {sym} datasets -> {len(outs)} files")
            ok += 1
        except Exception as e:
            print(f"✗ {sym} failed: {e}")
            failed.append(sym)

    print(f"\nDone: {ok}/{len(symbols)} symbols -> datasets in {OUT_DIR}/")
    if failed:
        print("Failed:")
        print(", ".join(failed))


if __name__ == "__main__":
    # Usage:
    #   python datasets.py            -> all tickers
    #   python datasets.py AAPL       -> one ticker
    #   python datasets.py ALL 0.01   -> all tickers, big-move threshold

    arg = sys.argv[1].strip().upper() if len(sys.argv) > 1 else "ALL"
    thr = float(sys.argv[2]) if len(sys.argv) > 2 else 0.01

    if arg == "ALL":
        build_all(big_move_threshold=thr)
    else:
        outs = build_datasets_for_symbol(arg, big_move_threshold=thr)
        for k, p in outs.items():
            print(f"✓ {arg} {k} -> {p}")