#!/usr/bin/env python3
"""
update_universe.py — Build TW top-150 market cap universe.

Sources:
  • FinMind TaiwanStockInfo  → list of all TWSE 4-digit common stocks
  • yfinance fast_info       → market cap per stock (parallel, ~2-3 min)
  → Writes to GSheets tab `universe_tw`

Usage:
  python update_universe.py               # fetch & push top 150
  python update_universe.py --top 200     # change count
  python update_universe.py --dry-run     # print list, no GSheets write
  python update_universe.py --workers 40  # tune parallelism (default 30)
"""
import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
from typing import Dict, List, Optional

import gspread
import requests
import yfinance as yf
from google.oauth2.service_account import Credentials

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

FINMIND_TOKEN = os.environ.get("FINMIND_TOKEN", "")
FINMIND_BASE  = "https://api.finmindtrade.com/api/v4/data"
GOOGLE_CREDS  = os.environ.get("GOOGLE_CREDENTIALS_JSON", "")
_SHEET_NAME   = "gigi-war-room-watchlist"
_UNIVERSE_TAB = "universe_tw"
_UNIVERSE_HDR = ["rank", "code", "name", "industry", "market_cap_b_twd", "updated_at"]
_SCOPES       = ["https://www.googleapis.com/auth/spreadsheets",
                 "https://www.googleapis.com/auth/drive.readonly"]

# ETF / warrant / preferred share patterns to exclude
_EXCLUDE_PREFIXES = ("00", "020", "021")  # ETFs mostly start with 00xx
_EXCLUDE_SUFFIXES = ("P", "A", "B")       # preferred / warrants


def _get_gc():
    raw = GOOGLE_CREDS
    if not raw and os.path.exists("credentials.json"):
        with open("credentials.json") as f:
            raw = f.read()
    if not raw:
        raise RuntimeError("No GOOGLE_CREDENTIALS_JSON env var or credentials.json found")
    creds = Credentials.from_service_account_info(json.loads(raw), scopes=_SCOPES)
    return gspread.authorize(creds)


def fetch_twse_stock_ids() -> List[dict]:
    """Return all TWSE-listed 4-digit common stocks from FinMind (deduplicated)."""
    print("  [1/3] Fetching stock list from FinMind TaiwanStockInfo...")
    r = requests.get(FINMIND_BASE, params={
        "dataset": "TaiwanStockInfo",
        "token":   FINMIND_TOKEN,
    }, timeout=60)
    r.raise_for_status()
    rows = r.json().get("data", [])

    seen: set = set()
    stocks = []
    for s in rows:
        sid = str(s.get("stock_id", ""))
        typ = s.get("type", "")
        if typ != "twse":
            continue
        if len(sid) != 4:
            continue
        if sid.startswith(_EXCLUDE_PREFIXES):
            continue
        if sid[-1].upper() in _EXCLUDE_SUFFIXES:
            continue
        if sid in seen:
            continue  # deduplicate — FinMind lists some stocks multiple times with different industries
        seen.add(sid)
        stocks.append({
            "stock_id": sid,
            "name":     s.get("stock_name", ""),
            "industry": s.get("industry_category", ""),
        })

    print(f"     → {len(stocks)} unique TWSE common stocks (ETFs/warrants excluded)")
    return stocks


def _fetch_one_mc(stock: dict) -> Optional[dict]:
    """Fetch market cap for one stock via yfinance fast_info (with retry)."""
    sym = stock["stock_id"] + ".TW"
    for attempt in range(2):
        try:
            mc = yf.Ticker(sym).fast_info.market_cap
            if mc and mc > 0:
                return {
                    "code":             sym,
                    "name":             stock["name"],
                    "industry":         stock["industry"],
                    "market_cap_b_twd": round(mc / 1e9, 1),
                }
            break  # got None/0 — not a data error, stop retrying
        except Exception:
            if attempt == 0:
                time.sleep(0.5)
    return None


def fetch_market_caps_parallel(stocks: List[dict], workers: int) -> List[dict]:
    """Parallel-fetch market caps in batches to avoid yfinance rate-limit."""
    total   = len(stocks)
    results = []
    t0      = time.time()
    BATCH   = 150  # pause after each batch to avoid throttling

    print(f"  [2/3] Fetching market caps for {total} stocks ({workers} workers, {BATCH}-stock batches)...")
    for batch_start in range(0, total, BATCH):
        batch = stocks[batch_start:batch_start + BATCH]
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = {ex.submit(_fetch_one_mc, s): s for s in batch}
            for fut in as_completed(futures):
                res = fut.result()
                if res:
                    results.append(res)

        done    = min(batch_start + BATCH, total)
        elapsed = time.time() - t0
        eta     = (elapsed / done * (total - done)) if done < total else 0
        print(f"     {done}/{total} done  |  {len(results)} with data  |  ETA {int(eta//60)}m{int(eta%60):02d}s",
              flush=True)
        if done < total:
            time.sleep(1)  # brief pause between batches

    results.sort(key=lambda x: -x["market_cap_b_twd"])
    return results


def push_to_gsheets(universe: List[dict]) -> None:
    today = date.today().isoformat()
    gc = _get_gc()
    sh = gc.open(_SHEET_NAME)

    try:
        ws = sh.worksheet(_UNIVERSE_TAB)
        ws.clear()
    except gspread.exceptions.WorksheetNotFound:
        ws = sh.add_worksheet(title=_UNIVERSE_TAB, rows=200, cols=len(_UNIVERSE_HDR))

    rows = [_UNIVERSE_HDR]
    for i, u in enumerate(universe, 1):
        rows.append([i, u["code"], u["name"], u["industry"], u["market_cap_b_twd"], today])

    ws.update("A1", rows)
    print(f"  ↑ Pushed {len(universe)} stocks → GSheets tab '{_UNIVERSE_TAB}'")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top",     type=int, default=150, help="How many stocks to keep (default 150)")
    ap.add_argument("--workers", type=int, default=15,  help="Parallel workers per batch (default 15)")
    ap.add_argument("--dry-run", action="store_true",   help="Print list, skip GSheets push")
    args = ap.parse_args()

    if not FINMIND_TOKEN:
        sys.exit("ERROR: FINMIND_TOKEN not set.")

    print(f"\n=== TW Top-{args.top} Market Cap Universe Builder ===\n")
    t0 = time.time()

    stocks   = fetch_twse_stock_ids()
    universe = fetch_market_caps_parallel(stocks, args.workers)

    top_n = universe[:args.top]

    print(f"\n  [3/3] Top {args.top} selected. Top 20 preview:")
    for u in top_n[:20]:
        rank = universe.index(u) + 1
        print(f"    #{rank:>3}  {u['code']:<10}  {u['name']:<20}  {u['market_cap_b_twd']:>8.1f}B TWD  {u['industry']}")

    elapsed = time.time() - t0
    print(f"\n  Total time: {int(elapsed//60)}m{int(elapsed%60):02d}s  |  {len(universe)} stocks have market cap data")

    if args.dry_run:
        print("\n  [dry-run] Skipping GSheets push.")
        return

    push_to_gsheets(top_n)
    print(f"\nDone. Run backtest with:  python simulate.py --mode both --market tw --universe")


if __name__ == "__main__":
    main()
