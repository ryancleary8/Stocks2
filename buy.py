# buy.py
"""
Reads the most recent ensemble_YYYY-MM-DD.csv from ./Signals (or ./signals),
takes the first N rows where action == "buy", and places $5 notional market buys
for each symbol using Alpaca.

Env vars required:
  ALPACA_API_KEY
  ALPACA_API_SECRET
Optional:
  ALPACA_PAPER   (default "true")  -> "true"/"false"
  ALPACA_BASE_URL (optional override; usually not needed)

Install:
  pip install alpaca-py pandas
Run:
  python buy.py
  python buy.py --limit 20 --notional 5
  python buy.py --dry-run
"""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd

try:
    from alpaca.trading.client import TradingClient
    from alpaca.trading.requests import MarketOrderRequest
    from alpaca.trading.enums import OrderSide, TimeInForce
except ImportError as e:
    raise SystemExit(
        "Missing dependency: alpaca-py\n"
        "Install with: pip install alpaca-py pandas\n"
        f"Original error: {e}"
    )


DATE_RE = re.compile(r"ensemble_(\d{4}-\d{2}-\d{2})\.csv$")


def _find_signals_dir() -> Path:
    # Prefer ./Signals if it exists, else ./signals
    here = Path.cwd()
    for name in ("Signals", "signals"):
        p = here / name
        if p.exists() and p.is_dir():
            return p
    raise FileNotFoundError("Could not find a ./Signals or ./signals folder in the current directory.")


def _pick_latest_ensemble_csv(signals_dir: Path) -> Path:
    candidates = []
    for p in signals_dir.glob("ensemble_*.csv"):
        m = DATE_RE.search(p.name)
        if m:
            candidates.append((m.group(1), p))
        else:
            # fallback: allow odd names but sort by mtime later
            candidates.append(("", p))

    if not candidates:
        raise FileNotFoundError(f"No ensemble_*.csv files found in {signals_dir}")

    # Prefer date in filename, else mtime
    dated = [c for c in candidates if c[0]]
    if dated:
        dated.sort(key=lambda x: x[0])  # YYYY-MM-DD sorts naturally
        return dated[-1][1]

    # fallback: newest modified
    candidates.sort(key=lambda x: x[1].stat().st_mtime)
    return candidates[-1][1]


def _load_buys(csv_path: Path, limit: int) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    # Expected columns: symbol, action (and possibly prob/confidence)
    # Make it robust:
    cols = {c.lower(): c for c in df.columns}
    if "symbol" not in cols or "action" not in cols:
        raise ValueError(
            f"{csv_path.name} must include at least columns: symbol, action. Found: {list(df.columns)}"
        )

    symbol_col = cols["symbol"]
    action_col = cols["action"]

    buys = df[df[action_col].astype(str).str.lower().eq("buy")].copy()
    buys[symbol_col] = buys[symbol_col].astype(str).str.upper().str.strip()
    buys = buys[buys[symbol_col] != ""]

    return buys.head(limit)


def _get_trading_client() -> TradingClient:
    key = os.getenv("ALPACA_API_KEY", "").strip()
    secret = os.getenv("ALPACA_API_SECRET", "").strip()
    if not key or not secret:
        raise EnvironmentError(
            "Missing ALPACA_API_KEY or ALPACA_API_SECRET in environment variables."
        )

    paper_str = os.getenv("ALPACA_PAPER", "true").strip().lower()
    paper = paper_str in ("1", "true", "yes", "y", "on")

    base_url = os.getenv("ALPACA_BASE_URL", "").strip()
    if base_url:
        # alpaca-py supports overriding base_url
        return TradingClient(api_key=key, secret_key=secret, paper=paper, base_url=base_url)

    return TradingClient(api_key=key, secret_key=secret, paper=paper)


def place_orders(
    client: TradingClient,
    buys: pd.DataFrame,
    notional: float,
    dry_run: bool,
) -> None:
    cols = {c.lower(): c for c in buys.columns}
    symbol_col = cols["symbol"]

    print(f"Placing orders for {len(buys)} symbols. Notional per symbol: ${notional:.2f}")
    if dry_run:
        print("DRY RUN enabled: no orders will be sent.\n")

    for i, row in enumerate(buys.itertuples(index=False), start=1):
        symbol = getattr(row, symbol_col)
        try:
            order_req = MarketOrderRequest(
                symbol=symbol,
                notional=float(notional),     # fractional buy by dollars
                side=OrderSide.BUY,
                time_in_force=TimeInForce.DAY,  # DAY is typical for market orders
            )

            if dry_run:
                print(f"[{i:02d}] would BUY ${notional:.2f} of {symbol}")
                continue

            order = client.submit_order(order_data=order_req)
            print(f"[{i:02d}] ✅ submitted BUY ${notional:.2f} of {symbol} | order_id={order.id}")

        except Exception as e:
            print(f"[{i:02d}] ❌ failed {symbol}: {e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=19, help="How many buy rows to take (default: 20)")
    ap.add_argument("--notional", type=float, default=5.0, help="Dollar amount per buy (default: 5.0)")
    ap.add_argument("--csv", type=str, default="", help="Optional explicit path to an ensemble CSV")
    ap.add_argument("--dry-run", action="store_true", help="Print what would be bought without placing orders")
    args = ap.parse_args()

    if args.limit < 1:
        raise SystemExit("--limit must be >= 1")
    if args.notional <= 0:
        raise SystemExit("--notional must be > 0")

    if args.csv:
        csv_path = Path(args.csv).expanduser().resolve()
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV not found: {csv_path}")
    else:
        signals_dir = _find_signals_dir()
        csv_path = _pick_latest_ensemble_csv(signals_dir)

    print(f"Using signals file: {csv_path}")

    buys = _load_buys(csv_path, limit=args.limit)
    if buys.empty:
        print("No action == 'buy' rows found. Nothing to do.")
        return

    client = _get_trading_client()
    place_orders(client, buys, notional=args.notional, dry_run=args.dry_run)


if __name__ == "__main__":
    main()