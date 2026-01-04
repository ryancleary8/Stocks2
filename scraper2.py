import asyncio
from time import perf_counter
from pathlib import Path
from datetime import datetime, timezone
import csv

import aiohttp
from bs4 import BeautifulSoup

HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Connection": "keep-alive",
}

# Tune these for speed vs. being polite / avoiding blocks
CONCURRENCY = 40          # how many symbols to fetch at once
REQUEST_TIMEOUT = 5       # seconds
RETRIES = 1               # retry once on transient errors

OUTPUT_DIR = Path("current")
DATE_STR = datetime.now(timezone.utc).strftime("%Y-%m-%d")  # UTC date for filenames


def load_symbols(path="stocks.txt"):
    with open(path) as f:
        return [line.strip().upper() for line in f if line.strip()]


def clean_output_dir():
    """Ensure OUTPUT_DIR exists and remove any existing .csv files inside it."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for p in OUTPUT_DIR.glob("*.csv"):
        try:
            p.unlink()
        except OSError:
            # If a file can't be removed, skip it rather than crashing the run.
            pass


def write_price_csv(symbol: str, price: float):
    """Write one-row CSV for a symbol to current/SYMBOL.YYYY-MM-DD.csv."""
    path = OUTPUT_DIR / f"{symbol}.{DATE_STR}.csv"
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["timestamp_utc", "symbol", "price"])
        w.writerow([datetime.now(timezone.utc).isoformat(), symbol, price])


# ---------------------------
# PARSERS
# ---------------------------

def parse_yahoo_price(html: str):
    soup = BeautifulSoup(html, "lxml")
    node = soup.find("fin-streamer", {"data-field": "regularMarketPrice"})
    if node and node.text:
        try:
            return float(node.text.replace(",", ""))
        except ValueError:
            return None
    return None


def parse_google_price(html: str):
    soup = BeautifulSoup(html, "lxml")
    node = soup.find("div", class_="YMlKec fxKbKc")
    if node and node.text:
        try:
            return float(node.text.replace("$", "").replace(",", ""))
        except ValueError:
            return None
    return None


# ---------------------------
# FETCH HELPERS
# ---------------------------

async def fetch_text(session: aiohttp.ClientSession, url: str) -> str | None:
    for attempt in range(RETRIES + 1):
        try:
            async with session.get(url, headers=HEADERS) as resp:
                resp.raise_for_status()
                return await resp.text()
        except (aiohttp.ClientError, asyncio.TimeoutError):
            if attempt >= RETRIES:
                return None
            await asyncio.sleep(0.15 * (attempt + 1))
    return None


async def scrape_symbol(session: aiohttp.ClientSession, symbol: str) -> tuple[str, float | None]:
    # Yahoo first (usually fastest / most consistent)
    yahoo_url = f"https://finance.yahoo.com/quote/{symbol}"
    html = await fetch_text(session, yahoo_url)
    if html:
        price = parse_yahoo_price(html)
        if price is not None:
            return symbol, price

    # Fallback to Google Finance
    google_url = f"https://www.google.com/finance/quote/{symbol}:NASDAQ"
    html = await fetch_text(session, google_url)
    if html:
        price = parse_google_price(html)
        if price is not None:
            return symbol, price

    return symbol, None


async def bounded_scrape(sem: asyncio.Semaphore, session: aiohttp.ClientSession, symbol: str):
    async with sem:
        return await scrape_symbol(session, symbol)


# ---------------------------
# MAIN
# ---------------------------

async def run_async():
    symbols = load_symbols()
    clean_output_dir()
    sem = asyncio.Semaphore(CONCURRENCY)

    timeout = aiohttp.ClientTimeout(total=REQUEST_TIMEOUT)
    connector = aiohttp.TCPConnector(limit=CONCURRENCY, ttl_dns_cache=300)

    t0 = perf_counter()
    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        tasks = [asyncio.create_task(bounded_scrape(sem, session, s)) for s in symbols]

        # Print results as they finish (fast feedback)
        ok = 0
        failed = []
        for coro in asyncio.as_completed(tasks):
            symbol, price = await coro
            if price is None:
                failed.append(symbol)
                continue
            ok += 1
            print(f"{symbol:<6} | ${price}")
            write_price_csv(symbol, price)

    dt = perf_counter() - t0
    print(f"\nDone: {ok}/{len(symbols)} prices in {dt:.2f}s (concurrency={CONCURRENCY})")
    if failed:
        print("\nSymbols with no price retrieved:")
        print(", ".join(failed))


def run():
    asyncio.run(run_async())


if __name__ == "__main__":
    run()
