#!/usr/bin/env python3
"""
update_outcomes_local.py — v11.0 本機 signal_outcomes 更新器

取代雲端 /api/outcomes/update（該端點呼叫的 _price_n_days_later 從未被定義過，
每次執行都是 NameError，被 try/except 吞掉，signal_outcomes 從來沒真的被填過）。

用法:
  python3 update_outcomes_local.py --market tw
  python3 update_outcomes_local.py --market both
  python3 update_outcomes_local.py --market us --dry-run   # 只印出，不寫入 GSheets
"""
import argparse
import datetime as _dt

import simulate as sim

_OUTCOME_TAB = "signal_outcomes"
_OUTCOME_HDR = ["signal_date", "ticker", "market", "close_signal",
                "rsi14", "adx14", "bias", "macd_cross", "vol_expansion", "confirmed",
                "close_5d", "return_5d", "close_10d", "return_10d", "win"]


def _closes_n_days_later(ws_row_ticker: str, market: str, signal_date: _dt.date):
    clean = ws_row_ticker.split(".")[0] if market == "tw" else ws_row_ticker
    fetch_start = signal_date.isoformat()
    fetch_end = (signal_date + _dt.timedelta(days=30)).isoformat()
    df = sim._fetch_ohlcv(clean, fetch_start, fetch_end, market=market)
    if df is None or df.empty:
        return None, None
    dates = df["date"].dt.date
    idx = df.index[dates >= signal_date]
    if len(idx) == 0:
        return None, None
    base = idx[0]
    c5 = float(df.loc[base + 5, "Close"]) if base + 5 < len(df) else None
    c10 = float(df.loc[base + 10, "Close"]) if base + 10 < len(df) else None
    return c5, c10


def update_market(market: str, dry_run: bool) -> int:
    ws = sim._get_or_create_tab(_OUTCOME_TAB, _OUTCOME_HDR)
    all_rows = ws.get_all_values()
    if len(all_rows) < 2:
        print(f"[{market}] signal_outcomes 空的，跳過")
        return 0

    today = _dt.date.today()
    updated = 0
    for i, row in enumerate(all_rows[1:], 2):
        if len(row) < 10 or row[2] != market:
            continue
        if len(row) > 10 and row[10]:
            continue  # close_5d already filled
        try:
            sig_date = _dt.datetime.strptime(row[0], "%Y-%m-%d").date()
            if (today - sig_date).days < 15:
                continue
            close_sig = float(row[3]) if row[3] else None
            if not close_sig:
                continue
            c5, c10 = _closes_n_days_later(row[1], market, sig_date)
            if c5 is None and c10 is None:
                continue
            r5 = round((c5 / close_sig - 1) * 100, 2) if c5 else ""
            r10 = round((c10 / close_sig - 1) * 100, 2) if c10 else ""
            win = 1 if (isinstance(r10, float) and r10 >= 3.0) else 0
            print(f"  [{market}] {row[1]:8s} {row[0]}  close_5d={c5} r5={r5}  close_10d={c10} r10={r10} win={win}")
            if not dry_run:
                ws.update(f"K{i}:O{i}", [[c5 or "", r5, c10 or "", r10, win]])
            updated += 1
        except Exception as e:
            print(f"  row {i} ({row[1] if len(row) > 1 else '?'}): {e}")
    print(f"[{market}] updated {updated} rows" + (" (dry-run, not written)" if dry_run else ""))
    return updated


def main():
    ap = argparse.ArgumentParser(description="Fill 5d/10d actual returns for aged signal_outcomes rows")
    ap.add_argument("--market", default="tw", choices=["tw", "us", "both"])
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    markets = ["tw", "us"] if args.market == "both" else [args.market]
    for m in markets:
        update_market(m, args.dry_run)


if __name__ == "__main__":
    main()
