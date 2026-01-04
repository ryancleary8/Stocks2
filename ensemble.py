"""Aggregate daily prediction results into a buy/hold ranking."""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

RESULTS_DIR = Path("predict_results")
SIGNALS_DIR = Path("Signals")


@dataclass
class EnsembleArgs:
    results_file: Path | None
    threshold: float
    output: Path | None


@dataclass
class StockDecision:
    symbol: str
    prob_up: float
    action: str


def parse_args(argv: Sequence[str] | None = None) -> EnsembleArgs:
    parser = argparse.ArgumentParser(
        description=(
            "Rank stocks by ensemble probability of going up and suggest buy/hold."
        )
    )
    parser.add_argument(
        "--results-file",
        type=Path,
        help=(
            "Path to a predict_results CSV. Defaults to the most recent file in "
            "predict_results/."
        ),
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Probability threshold above which to mark a stock as 'buy'.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help=(
            "Optional output CSV path. Defaults to Signals/ensemble_{date}.csv"
        ),
    )

    args = parser.parse_args(argv)
    return EnsembleArgs(
        results_file=args.results_file,
        threshold=args.threshold,
        output=args.output,
    )


def _most_recent_results_file() -> Path:
    csv_files = sorted(RESULTS_DIR.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(
            "No prediction result files found in predict_results/."
        )
    return csv_files[-1]


def _load_rows(path: Path) -> tuple[List[str], List[dict]]:
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        raw_header = reader.fieldnames or []
        header = [h.strip() for h in raw_header if h is not None]
        rows = []
        for row in reader:
            if row is None:
                continue
            # Strip whitespace from keys to avoid header-mismatch issues.
            cleaned = {str(k).strip(): v for k, v in row.items() if k is not None}
            rows.append(cleaned)

    if not header or "symbol" not in {h.lower() for h in header}:
        raise ValueError("Prediction results must include a 'symbol' column.")

    # Ensure the canonical 'symbol' key exists in rows even if header used different casing.
    for r in rows:
        if "symbol" not in r:
            for k in list(r.keys()):
                if k.lower() == "symbol":
                    r["symbol"] = r.get(k)
                    break

    return header, rows


def _up_probability_columns(headers: Iterable[str]) -> List[str]:
    headers_list = [str(h).strip() for h in headers if h is not None]

    # Primary: our convention (any column containing cls_up)
    up_cols = [col for col in headers_list if "cls_up" in col.lower()]

    # Fallbacks: accept common alternative names used by other scripts
    if not up_cols:
        up_cols = [
            col
            for col in headers_list
            if any(tok in col.lower() for tok in ("prob_up", "p_up", "up_proba", "up_probability"))
        ]

    if not up_cols:
        preview = ", ".join(headers_list[:25])
        raise ValueError(
            "No up-probability columns found in results file. "
            "Expected something containing 'cls_up' (preferred). "
            f"Headers seen (first 25): {preview}"
        )
    return up_cols


def _average_probability(row: dict, columns: Sequence[str]) -> float:
    values: List[float] = []
    for col in columns:
        try:
            val = float(row.get(col, ""))
        except (TypeError, ValueError):
            continue
        values.append(val)
    if not values:
        return 0.0
    return sum(values) / len(values)


def build_ensemble(
    *, results_path: Path, threshold: float
) -> List[StockDecision]:
    headers, rows = _load_rows(results_path)
    up_cols = _up_probability_columns(headers)

    decisions: List[StockDecision] = []
    for row in rows:
        prob_up = _average_probability(row, up_cols)
        action = "buy" if prob_up >= threshold else "hold"
        decisions.append(
            StockDecision(symbol=row["symbol"], prob_up=prob_up, action=action)
        )

    decisions.sort(key=lambda d: d.prob_up, reverse=True)
    return decisions


def _default_output_path(results_path: Path) -> Path:
    date_part = results_path.stem
    return SIGNALS_DIR / f"ensemble_{date_part}.csv"


def save_decisions(decisions: Sequence[StockDecision], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["symbol", "prob_up", "action"])
        for d in decisions:
            writer.writerow([d.symbol, f"{d.prob_up:.6f}", d.action])


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    results_path = args.results_file or _most_recent_results_file()
    decisions = build_ensemble(results_path=results_path, threshold=args.threshold)
    output_path = args.output or _default_output_path(results_path)
    save_decisions(decisions, output_path)
    print(f"Ensemble ranking saved to {output_path}")


if __name__ == "__main__":
    main()
