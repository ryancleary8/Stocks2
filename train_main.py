"""Pipeline orchestrator for scraping, cleaning, feature building, dataset creation, and training.

Usage:
    python train_main.py                # run full pipeline for tickers in stocks.txt
    python train_main.py AAPL MSFT      # run pipeline for specific tickers
    python train_main.py --thr 0.02     # override big-move threshold for classification label

This script wires together existing modules:
- scraper1.update_latest_row (scrape / append raw history)
- valdiate.validate_and_clean (clean + validate raw CSVs)
- features.build_features_for_symbol (engineer leak-safe features)
- datasets.build_datasets_for_symbol (prepare regression/classification datasets)
- train.train_for_symbol (train all available models)
"""

from __future__ import annotations

import argparse
from typing import Iterable

from datasets import build_datasets_for_symbol, load_symbols
from features import build_features_for_symbol
from scraper1 import update_latest_row
from train import train_for_symbol
from valdiate import validate_and_clean


def run_pipeline_for_symbol(symbol: str, big_move_threshold: float) -> None:
    """Run the full data→model pipeline for a single ticker."""
    symbol = symbol.upper().strip()
    print(f"\n===== {symbol}: pipeline start =====")

    # 1) Scrape / update raw history
    try:
        update_latest_row(symbol)
    except Exception as exc:  # noqa: BLE001
        print(f"❌ {symbol}: scraping failed -> {exc}")
        return

    # 2) Clean + validate
    ok, _ = validate_and_clean(symbol)
    if not ok:
        print(f"❌ {symbol}: validation failed; skipping downstream steps")
        return

    # 3) Feature engineering
    try:
        feat_path = build_features_for_symbol(symbol)
        print(f"✓ {symbol}: features -> {feat_path}")
    except Exception as exc:  # noqa: BLE001
        print(f"❌ {symbol}: feature build failed -> {exc}")
        return

    # 4) Dataset creation
    try:
        dataset_paths = build_datasets_for_symbol(symbol, big_move_threshold=big_move_threshold)
        dataset_summary = ", ".join(f"{k}:{v.name}" for k, v in dataset_paths.items())
        print(f"✓ {symbol}: datasets -> {dataset_summary}")
    except Exception as exc:  # noqa: BLE001
        print(f"❌ {symbol}: dataset build failed -> {exc}")
        return

    # 5) Model training
    try:
        train_for_symbol(symbol)
    except Exception as exc:  # noqa: BLE001
        print(f"❌ {symbol}: training failed -> {exc}")
        return

    print(f"===== {symbol}: pipeline complete =====\n")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run full training pipeline for stocks.")
    parser.add_argument(
        "symbols",
        nargs="*",
        help="Ticker symbols to process. Defaults to stocks.txt list when omitted.",
    )
    parser.add_argument(
        "--thr",
        type=float,
        default=0.01,
        help="Big-move threshold (absolute return) for classification label (default: 0.01).",
    )
    return parser.parse_args()


def main(symbols: Iterable[str] | None = None, big_move_threshold: float = 0.01) -> None:
    symbol_list = list(symbols) if symbols is not None else []
    if not symbol_list:
        symbol_list = load_symbols()
    else:
        symbol_list = [s.upper().strip() for s in symbol_list]

    for sym in symbol_list:
        run_pipeline_for_symbol(sym, big_move_threshold=big_move_threshold)


if __name__ == "__main__":
    args = _parse_args()
    main(symbols=args.symbols, big_move_threshold=args.thr)
