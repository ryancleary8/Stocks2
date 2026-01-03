"""Nasdaq historical daily scraper (full + incremental updates)

- Reads tickers from stocks.txt (fallback: stock.txt)
- If data/raw/{SYMBOL}.raw.csv does not exist: full scrape from Nasdaq and save
- If it exists: fetch latest daily row and append only if it's newer

Data source:
  https://api.nasdaq.com/api/quote/{symbol}/historical
Page reference:
  https://www.nasdaq.com/market-activity/stocks/{symbol}/historical
"""

from __future__ import annotations

import os
from pathlib import Path
from datetime import datetime

import requests
import pandas as pd


# -------------------------
# CONFIG
# -------------------------

DATA_DIR = Path("data/raw")
TIMEOUT_S = 20


# -------------------------
# HELPERS
# -------------------------

def load_symbols() -> list[str]:
    """Loads symbols from stocks.txt, falling back to stock.txt."""
    for fname in ("stocks.txt", "stock.txt"):
        p = Path(fname)
        if p.exists():
            with p.open() as f:
                return [line.strip().upper() for line in f if line.strip()]
    raise FileNotFoundError("Couldn't find stocks.txt or stock.txt in the project root")


def nasdaq_headers(symbol: str) -> dict[str, str]:
    sym = symbol.lower()
    return {
        "User-Agent": "Mozilla/5.0",
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": f"https://www.nasdaq.com/market-activity/stocks/{sym}/historical",
        "Origin": "https://www.nasdaq.com",
    }


def parse_num(x):
    """Parse Nasdaq number strings like '$123.45', '1,234', 'N/A'."""
    if x is None:
        return None
    s = str(x).strip()
    if not s or s.lower() in {"n/a", "na", "null", "none", "--"}:
        return None
    s = s.replace("$", "").replace(",", "")
    try:
        return float(s)
    except ValueError:
        return None


def ensure_data_dir() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def csv_path(symbol: str) -> Path:
    return DATA_DIR / f"{symbol}.raw.csv"


# -------------------------
# NASDAQ FETCH
# -------------------------

def _get_historical_json(symbol: str, limit: int) -> dict:
    symbol = symbol.upper().strip()
    url = f"https://api.nasdaq.com/api/quote/{symbol}/historical"

    params = {
        "assetclass": "stocks",
        "limit": limit,
        "fromdate": "1900-01-01",
        "todate": "2100-01-01",
    }

    r = requests.get(url, headers=nasdaq_headers(symbol), params=params, timeout=TIMEOUT_S)
    r.raise_for_status()
    return r.json()


def _extract_rows(j: dict) -> list[dict]:
    data = (j.get("data") or {})
    trades = (data.get("tradesTable") or {})
    rows = trades.get("rows") or []
    return rows


def get_latest_historical_row(symbol: str) -> dict:
    """Returns newest available historical daily row from Nasdaq as a dict."""
    j = _get_historical_json(symbol, limit=1)
    rows = _extract_rows(j)

    if not rows:
        status = j.get("status")
        message = j.get("message")
        raise RuntimeError(f"No rows returned. status={status} message={message}")

    latest = rows[0]
    return {"symbol": symbol.upper().strip(), **latest}


def get_full_history(symbol: str, limit: int = 10000) -> list[dict]:
    """Fetch full historical data for a symbol."""
    j = _get_historical_json(symbol, limit=limit)
    rows = _extract_rows(j)
    if not rows:
        raise RuntimeError(f"No historical data for {symbol}")
    return rows


# -------------------------
# SAVE / UPDATE
# -------------------------

def scrape_and_save(symbol: str, data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Scrape full history and save to CSV."""
    symbol = symbol.upper().strip()
    ensure_data_dir()

    rows = get_full_history(symbol)
    df = pd.DataFrame(rows)

    # Keep only desired columns; Nasdaq keys are usually: date, open, high, low, close, volume
    cols = ["date", "open", "high", "low", "close", "volume"]
    for c in cols:
        if c not in df.columns:
            df[c] = None
    df = df[cols]

    # Parse / normalize
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    for c in ("open", "high", "low", "close", "volume"):
        df[c] = df[c].apply(parse_num)

    # Sort oldest -> newest
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    path = data_dir / f"{symbol}.raw.csv"
    df.to_csv(path, index=False)
    print(f"✅ {symbol} full scrape -> {path} ({len(df)} rows)")
    return df


def update_latest_row(symbol: str, data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Fetch latest row and append to existing CSV if new."""
    symbol = symbol.upper().strip()
    ensure_data_dir()
    path = data_dir / f"{symbol}.raw.csv"

    if not path.exists():
        print(f"No existing data for {symbol}, doing full scrape...")
        return scrape_and_save(symbol, data_dir)

    # Load existing
    df = pd.read_csv(path)
    if "date" not in df.columns:
        print(f"⚠️  {symbol} missing date column; re-scraping...")
        return scrape_and_save(symbol, data_dir)

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    if df.empty:
        print(f"⚠️  {symbol} empty/invalid file; re-scraping...")
        return scrape_and_save(symbol, data_dir)

    latest_date = df["date"].max()

    # Fetch newest row
    latest = get_latest_historical_row(symbol)
    new_date = pd.to_datetime(latest.get("date"), errors="coerce")

    if pd.isna(new_date):
        print(f"❌ {symbol} latest row has invalid date")
        return df

    if new_date <= latest_date:
        print(f"✓ {symbol} already up to date ({latest_date.date()})")
        return df

    new_df = pd.DataFrame([
        {
            "date": new_date,
            "open": parse_num(latest.get("open")),
            "high": parse_num(latest.get("high")),
            "low": parse_num(latest.get("low")),
            "close": parse_num(latest.get("close")),
            "volume": parse_num(latest.get("volume")),
        }
    ])

    df = pd.concat([df, new_df], ignore_index=True)
    df = df.drop_duplicates(subset=["date"], keep="last").sort_values("date").reset_index(drop=True)
    df.to_csv(path, index=False)

    print(f"✅ {symbol} appended {new_date.date()} -> {path} (now {len(df)} rows)")
    return df


# -------------------------
# MAIN
# -------------------------

if __name__ == "__main__":
    symbols = load_symbols()
    ensure_data_dir()

    print(f"📥 Nasdaq historical scraper | {len(symbols)} tickers")

    ok = 0
    failed: list[str] = []

    for sym in symbols:
        try:
            df = update_latest_row(sym)
            if df is None or df.empty:
                failed.append(sym)
            else:
                ok += 1
        except Exception as e:
            failed.append(sym)
            print(f"✗ Error for {sym}: {e}")

    print(f"\nDone: {ok}/{len(symbols)} tickers processed. Data in {DATA_DIR}/")
    if failed:
        print("\nTickers with errors/no data:")
        print(", ".join(failed))