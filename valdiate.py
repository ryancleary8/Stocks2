# pipeline/validate.py
import os
import sys
from pathlib import Path

import pandas as pd


REQUIRED_COLS = ["date", "open", "high", "low", "close", "volume"]


def _clean_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Remove formatting characters (commas, currency symbols) before numeric checks."""
    numeric_cols = ["open", "high", "low", "close", "volume"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = (
                df[col]
                .astype(str)
                .str.replace("$", "", regex=False)
                .str.replace(",", "", regex=False)
                .str.strip()
            )
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def validate_data(df: pd.DataFrame, symbol: str) -> bool:
    """Validate data quality. Prints issues and returns True/False."""
    errors = []

    # Required columns
    missing = set(REQUIRED_COLS) - set(df.columns)
    if missing:
        errors.append(f"Missing columns: {sorted(missing)}")
        # If required columns missing, many other checks don't make sense
        _print_result(symbol, errors)
        return False

    # Expect date already parsed
    if not pd.api.types.is_datetime64_any_dtype(df["date"]):
        errors.append("'date' column is not datetime")

    # Clean numeric columns
    df = _clean_numeric_columns(df)

    # Missing values in required fields
    if df[REQUIRED_COLS].isnull().any().any():
        errors.append("Contains NaN values in required columns")

    # Date sorting
    if not df["date"].is_monotonic_increasing:
        errors.append("Dates not sorted ascending")

    # Duplicates
    if df["date"].duplicated().any():
        errors.append("Contains duplicate dates")

    # Volume > 0
    if (df["volume"] <= 0).any():
        errors.append("Contains zero or negative volume")

    # Minimum rows
    if len(df) < 50:
        errors.append(f"Insufficient data: only {len(df)} rows")

    _print_result(symbol, errors)
    return len(errors) == 0


def _print_result(symbol: str, errors: list[str]) -> None:
    if errors:
        print(f"✗ Validation failed for {symbol}:")
        for e in errors:
            print(f"  - {e}")
    else:
        print(f"✓ Validation passed for {symbol}")


def validate_and_clean(symbol: str, raw_dir: str = "data/raw", clean_dir: str = "data/clean"):
    """Load raw data, clean/standardize it for modeling, validate, and write clean CSV."""
    symbol = symbol.upper().strip()

    raw_path = Path(raw_dir) / f"{symbol}.raw.csv"
    if not raw_path.exists():
        print(f"✗ Missing raw file for {symbol}: {raw_path}")
        return False, None

    df = pd.read_csv(raw_path)

    # Ensure required columns exist (create if missing so downstream logic is predictable)
    for c in REQUIRED_COLS:
        if c not in df.columns:
            df[c] = pd.NA

    # Parse date
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # Clean numeric columns
    df = _clean_numeric_columns(df)

    # Standardize ordering
    df = df[REQUIRED_COLS]

    # Sort, de-dup, drop invalid rows
    df = df.dropna(subset=["date"]).sort_values("date")
    df = df.drop_duplicates(subset=["date"], keep="last")

    # Drop rows with NaNs in required numeric fields
    df = df.dropna(subset=["open", "high", "low", "close", "volume"])

    # Enforce volume positive
    df = df[df["volume"] > 0]

    # Final validation
    ok = validate_data(df.copy(), symbol)
    if not ok:
        return False, None

    # Save cleaned output for your model
    out_dir = Path(clean_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{symbol}.clean.csv"
    df.to_csv(out_path, index=False)
    print(f"✓ Wrote clean file: {out_path}")

    return True, df


if __name__ == "__main__":
    # Usage:
    #   python valdiate.py            -> validate all tickers in stocks.txt (fallback stock.txt)
    #   python valdiate.py AAPL       -> validate only AAPL
    #   python valdiate.py all        -> validate all tickers in stocks.txt (fallback stock.txt)

    arg = sys.argv[1].strip().upper() if len(sys.argv) > 1 else "ALL"

    # Build symbol list
    if arg == "ALL":
        # Read stocks from stocks.txt (fallback stock.txt)
        symbols = []
        for fname in ("stocks.txt", "stock.txt"):
            p = Path(fname)
            if p.exists():
                with p.open() as f:
                    symbols = [line.strip().upper() for line in f if line.strip()]
                break
        if not symbols:
            print("✗ Could not find stocks.txt or stock.txt to validate")
            sys.exit(1)
    else:
        symbols = [arg]

    ok_count = 0
    failed = []

    for sym in symbols:
        ok, _ = validate_and_clean(sym)
        if ok:
            ok_count += 1
        else:
            failed.append(sym)

    print(f"\nDone: {ok_count}/{len(symbols)} validated and cleaned")
    if failed:
        print("Failed:")
        print(", ".join(failed))
        sys.exit(1)
