#!/usr/bin/env python3
"""
Gigi Stock War Room — Local Simulation Engine (v10.0)

Run on your local machine to:
  1) Collect 2-year candidate signals across full watchlist (no timeout constraints)
  2) Grid-search optimal RSI/ADX/Bias/MACD_H params on historical data
  3) Train XGBoost with 200 estimators and more depth than cloud allows
  4) Push best params + model to Google Sheets
  5) Notify cloud backend to reload cache (instant sync)

Usage:
  python simulate.py                    # TW market, 2 years
  python simulate.py --market us        # US stocks
  python simulate.py --market both      # TW + US sequentially
  python simulate.py --years 3          # 3-year window
  python simulate.py --no-push          # dry run, don't write to GSheets
  python simulate.py --no-reload        # push to GSheets but skip cloud reload
"""
import argparse
import json
import os
import sys
import pickle
import base64 as _b64
import time
from datetime import datetime, timedelta
from typing import List, Optional, Any

import numpy as np
import pandas as pd
import yfinance as yf
import requests
import gspread
from google.oauth2.service_account import Credentials

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ── Config ─────────────────────────────────────────────────────────────────────
CLOUD_BACKEND    = "https://gigi-stock-war-room-2.onrender.com"
_SHEET_NAME      = "gigi-war-room-watchlist"
_SHEET_TABS      = {"tw": "gigi-war-room-watchlist", "us": "gigi-us-watchlist"}
_TICKER_COL      = "ticker"
_SHEETS_SCOPES   = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive.readonly",
]
FINMIND_TOKEN    = os.environ.get("FINMIND_TOKEN", "")
FINMIND_BASE     = "https://api.finmindtrade.com/api/v4/data"
GOOGLE_CREDS     = os.environ.get("GOOGLE_CREDENTIALS_JSON", "")

_MODEL_PARAMS_TAB = "model_params"
_MODEL_PARAMS_HDR = ["market","rsi_lo","rsi_hi","adx_lo","adx_hi","bias_lo","bias_hi",
                     "macd_h_pct_min","win_rate","sharpe","updated"]
_MODEL_STORE_TAB  = "model_store"
_MODEL_STORE_HDR  = ["market","model_b64","trained_at","n_samples","accuracy"]

TW_BEST_PARAMS   = {"rsi_lo":52,"rsi_hi":60,"bias_lo":4,"bias_hi":8,"adx_lo":18,"adx_hi":35,"macd_h_pct_min":60}
US_DEFAULT_PARAMS = {"rsi_lo":60,"rsi_hi":65,"bias_lo":4,"bias_hi":8,"adx_lo":18,"adx_hi":30,"macd_h_pct_min":60}
_MACD_H_COLS     = {33:"MACD_H_p33",40:"MACD_H_p40",50:"MACD_H_p50",60:"MACD_H_p60",66:"MACD_H_p66"}


# ── GSheets helpers ────────────────────────────────────────────────────────────

def _get_gc():
    if not GOOGLE_CREDS:
        sys.exit("ERROR: GOOGLE_CREDENTIALS_JSON not set.\n"
                 "Create a .env file with GOOGLE_CREDENTIALS_JSON='...' or export it in your shell.")
    return gspread.authorize(
        Credentials.from_service_account_info(json.loads(GOOGLE_CREDS), scopes=_SHEETS_SCOPES)
    )

def _get_or_create_tab(tab_name: str, headers: list):
    gc = _get_gc()
    sh = gc.open(_SHEET_NAME)
    try:
        return sh.worksheet(tab_name)
    except Exception:
        ws = sh.add_worksheet(title=tab_name, rows=5000, cols=len(headers))
        ws.update("A1", [headers])
        return ws

def _get_watchlist(market: str) -> List[str]:
    try:
        gc = _get_gc()
        sh = gc.open(_SHEET_NAME)
        ws = sh.worksheet(_SHEET_TABS[market])
        rows = ws.get_all_values()
        if not rows:
            return []
        hdr = rows[0]
        col = hdr.index(_TICKER_COL) if _TICKER_COL in hdr else 0
        return [r[col].strip() for r in rows[1:] if r and r[col].strip()]
    except Exception as e:
        print(f"  Error reading watchlist: {e}")
        return []

def _save_live_params(market: str, params: dict, win_rate: float, sharpe: float):
    ws = _get_or_create_tab(_MODEL_PARAMS_TAB, _MODEL_PARAMS_HDR)
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    new_row = [
        market,
        params.get("rsi_lo"), params.get("rsi_hi"),
        params.get("adx_lo"), params.get("adx_hi"),
        params.get("bias_lo"), params.get("bias_hi"),
        params.get("macd_h_pct_min"),
        round(win_rate, 4), round(sharpe, 4), ts,
    ]
    rows = ws.get_all_values()
    for i, r in enumerate(rows[1:], 2):
        if r and r[0] == market:
            ws.update(f"A{i}", [new_row])
            print(f"  ↑ Updated model_params [{market}]")
            return
    ws.append_rows([new_row], value_input_option="RAW")
    print(f"  + Inserted model_params [{market}]")

def _save_xgb_model(market: str, model: Any, n_samples: int, accuracy: float):
    ws = _get_or_create_tab(_MODEL_STORE_TAB, _MODEL_STORE_HDR)
    b64 = _b64.b64encode(pickle.dumps(model)).decode()
    ts  = datetime.now().strftime("%Y-%m-%d %H:%M")
    new_row = [market, b64, ts, n_samples, round(accuracy, 4)]
    rows = ws.get_all_values()
    for i, r in enumerate(rows[1:], 2):
        if r and r[0] == market:
            ws.update(f"A{i}", [new_row])
            print(f"  ↑ Updated model_store [{market}] ({n_samples} samples, acc={accuracy:.3f})")
            return
    ws.append_rows([new_row], value_input_option="RAW")
    print(f"  + Inserted model_store [{market}] ({n_samples} samples, acc={accuracy:.3f})")

def _notify_cloud_reload(market: str):
    """Tell cloud to clear its param/model cache so new GSheets data is used immediately."""
    try:
        r = requests.post(f"{CLOUD_BACKEND}/api/model/reload?market={market}", timeout=90)
        print(f"  Cloud reload: HTTP {r.status_code}")
    except Exception as e:
        print(f"  Cloud reload failed (will auto-sync within 1 hour): {e}")


# ── Indicator computation ──────────────────────────────────────────────────────

def _compute_bt_indicators(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["MA10"]  = d["Close"].rolling(10).mean()
    d["MA20"]  = d["Close"].rolling(20).mean()
    d["MA60"]  = d["Close"].rolling(60).mean()
    d["MA120"] = d["Close"].rolling(120).mean()
    d["VMA20"] = d["Volume"].rolling(20).mean()

    delta = d["Close"].diff()
    ag = delta.clip(lower=0).ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    al = (-delta).clip(lower=0).ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    d["RSI14"] = 100.0 - 100.0 / (1.0 + ag / al.replace(0, 1e-10))
    d["Bias"]  = (d["Close"] - d["MA20"]) / d["MA20"] * 100

    e12 = d["Close"].ewm(span=12, adjust=False).mean()
    e26 = d["Close"].ewm(span=26, adjust=False).mean()
    d["MACD"]     = e12 - e26
    d["MACD_Sig"] = d["MACD"].ewm(span=9, adjust=False).mean()
    d["MACD_H"]   = d["MACD"] - d["MACD_Sig"]
    for _p in [33, 40, 50, 60, 66]:
        d[f"MACD_H_p{_p}"] = d["MACD_H"].rolling(50).quantile(_p / 100)

    tr = pd.concat([
        d["High"] - d["Low"],
        (d["High"] - d["Close"].shift()).abs(),
        (d["Low"]  - d["Close"].shift()).abs(),
    ], axis=1).max(axis=1)
    up, dn = d["High"].diff(), -d["Low"].diff()
    dm_p = up.where((up > dn) & (up > 0), 0.0)
    dm_m = dn.where((dn > up) & (dn > 0), 0.0)
    atr  = tr.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    d["ATR14"] = atr
    dip = dm_p.ewm(alpha=1/14, min_periods=14, adjust=False).mean() / atr.replace(0, np.nan) * 100
    dim = dm_m.ewm(alpha=1/14, min_periods=14, adjust=False).mean() / atr.replace(0, np.nan) * 100
    dx  = (dip - dim).abs() / (dip + dim).replace(0, np.nan) * 100
    d["ADX14"] = dx.ewm(alpha=1/14, min_periods=14, adjust=False).mean().fillna(0)

    obv = (d["Volume"] * np.sign(d["Close"].diff()).fillna(0)).cumsum()
    d["OBV"]      = obv
    d["OBV_MA20"] = obv.rolling(20).mean()

    d["ret5d"]         = d["Close"].pct_change(5) * 100
    d["is_extended"]   = d["ret5d"] > 8
    d["monthly_trend"] = (d["MA20"] > d["MA60"]) & (d["MA60"] > d["MA120"])
    return d


# ── Data fetching ──────────────────────────────────────────────────────────────

def _fetch_ohlcv(code: str, fetch_start: str, fetch_end: str, market: str = "tw") -> Optional[pd.DataFrame]:
    if market == "tw" and FINMIND_TOKEN:
        try:
            r = requests.get(FINMIND_BASE, params={
                "dataset": "TaiwanStockPrice", "data_id": code,
                "start_date": fetch_start, "end_date": fetch_end,
                "token": FINMIND_TOKEN,
            }, timeout=30)
            raw = r.json().get("data", []) if r.ok else []
            if raw:
                df = pd.DataFrame(raw)
                df["date"] = pd.to_datetime(df["date"])
                df = df.sort_values("date").reset_index(drop=True)
                for src, dst in [("open","Open"),("max","High"),("min","Low"),
                                  ("close","Close"),("Trading_Volume","Volume")]:
                    if src in df.columns:
                        df[dst] = pd.to_numeric(df[src], errors="coerce")
                for col in ["Open","High","Low","Close","Volume"]:
                    if col not in df.columns:
                        return None
                    df[col] = pd.to_numeric(df[col], errors="coerce")
                df = df.dropna(subset=["Close"]).reset_index(drop=True)
                df["Volume"] = df["Volume"].fillna(0)
                return df
        except Exception:
            pass

    sym = f"{code}.TW" if market == "tw" else code
    try:
        yf_df = yf.download(sym, start=fetch_start, end=fetch_end,
                             auto_adjust=True, progress=False)
        if yf_df is None or len(yf_df) < 60:
            return None
        yf_df = yf_df.reset_index()
        if isinstance(yf_df.columns, pd.MultiIndex):
            yf_df.columns = [c[0] for c in yf_df.columns]
        yf_df = yf_df.rename(columns={"Date": "date"})
        df = yf_df[["date","Open","High","Low","Close","Volume"]].copy()
        df["date"] = pd.to_datetime(df["date"])
        if hasattr(df["date"], "dt") and df["date"].dt.tz is not None:
            df["date"] = df["date"].dt.tz_localize(None)
        df = df.sort_values("date").reset_index(drop=True)
        for col in ["Open","High","Low","Close","Volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna(subset=["Close"]).reset_index(drop=True)
        df["Volume"] = df["Volume"].fillna(0)
        return df
    except Exception:
        return None


# ── Wide signal collection (base conditions only, all RSI/ADX values kept) ────

def _collect_wide_signals(code: str, start_date: str, end_date: str,
                           hold_days: int = 10, market: str = "tw") -> List[dict]:
    fetch_start = (datetime.strptime(start_date, "%Y-%m-%d") - timedelta(days=400)).strftime("%Y-%m-%d")
    fetch_end   = (datetime.strptime(end_date,   "%Y-%m-%d") + timedelta(days=hold_days + 15)).strftime("%Y-%m-%d")
    df = _fetch_ohlcv(code, fetch_start, fetch_end, market=market)
    if df is None:
        return []
    df = _compute_bt_indicators(df)

    s_dt = pd.to_datetime(start_date)
    e_dt = pd.to_datetime(end_date)
    in_range = df[(df["date"] >= s_dt) & (df["date"] <= e_dt)]

    signals = []
    for idx, row in in_range.iterrows():
        try:
            for col in ["MA20","VMA20","RSI14","Bias","MACD","MACD_Sig","ADX14","MACD_H"]:
                if pd.isna(row[col]):
                    raise ValueError()
            if not (
                row["Close"]  > row["MA20"]    and
                row["Volume"] > row["VMA20"]   and
                row["MACD"]   > row["MACD_Sig"] and
                row["OBV"]    > row["OBV_MA20"] and
                bool(row["monthly_trend"])      and
                not bool(row["is_extended"])
            ):
                continue
        except Exception:
            continue

        past_mh = df["MACD_H"].iloc[max(0, idx - 50):idx].dropna()
        mh_pct  = float((past_mh < float(row["MACD_H"])).mean() * 100) if len(past_mh) >= 5 else 50.0

        future = df[df.index > idx].head(hold_days)
        if len(future) < hold_days:
            continue
        entry  = float(row["Close"])
        exit_p = float(future.iloc[-1]["Close"])
        if entry <= 0:
            continue

        ret_pct = (exit_p - entry) / entry * 100
        signals.append({
            "ticker":     code,
            "date":       row["date"].strftime("%Y-%m-%d"),
            "rsi14":      round(float(row["RSI14"]), 1),
            "adx14":      round(float(row["ADX14"]), 1),
            "bias":       round(float(row["Bias"]),  2),
            "macd_h":     round(float(row["MACD_H"]), 4),
            "macd_h_pct": round(mh_pct, 1),
            "return_pct": round(ret_pct, 2),
            "won":        ret_pct > 0,
            # XGBoost features — base buy conditions are all True here
            "macd_cross":       "above",
            "vol_expansion":    "1",
            "monthly_trend":    "1",
            "obv_trend":        "rising",
            "is_breakout20":    "0",
            "rs_score":         0,
            "confirmed_signal": "0",
        })
    return signals


# ── XGBoost feature vector ─────────────────────────────────────────────────────

def _xgb_features(row: dict) -> List[float]:
    macd_map = {"golden":2.0,"above":1.0,"none":0.0,"below":-1.0,"death":-2.0}
    return [
        float(row.get("rsi14") or 50),
        float(row.get("adx14") or 20),
        float(row.get("bias")  or 0),
        macd_map.get(str(row.get("macd_cross") or "none"), 0.0),
        1.0 if str(row.get("vol_expansion"))    in ("True","true","1") else 0.0,
        1.0 if str(row.get("is_breakout20"))    in ("True","true","1") else 0.0,
        1.0 if str(row.get("monthly_trend"))    in ("True","true","1") else 0.0,
        1.0 if str(row.get("obv_trend")) == "rising" else (
           -1.0 if str(row.get("obv_trend")) == "falling" else 0.0),
        min(float(row.get("rs_score") or 0), 3.0),
        1.0 if str(row.get("confirmed_signal")) in ("True","true","1") else 0.0,
    ]


# ── Grid search ────────────────────────────────────────────────────────────────

def run_grid_search(all_signals: List[dict], market: str, hold_days: int = 10) -> Optional[dict]:
    is_tw    = market == "tw"
    rsi_los  = [50,52,54]      if is_tw else [45,50,55,60]
    rsi_his  = [58,60,62]      if is_tw else [65,70,75,80]
    adx_los  = [18,20,22,24]   if is_tw else [10,15,18,20]
    adx_his  = [28,30,35]      if is_tw else [30,35,40,45]
    mh_pcts  = [50,60,66]      if is_tw else [33,40,50,60]
    bias_los = [0,2,4]
    bias_his = [8]             if is_tw else [8,12,15]

    grid = [
        {"rsi_lo":rsl,"rsi_hi":rsh,"adx_lo":adl,"adx_hi":adh,
         "macd_h_pct_min":mh,"bias_lo":blo,"bias_hi":bhi}
        for rsl in rsi_los for rsh in rsi_his if rsl < rsh
        for adl in adx_los for adh in adx_his if adl < adh
        for mh  in mh_pcts
        for blo in bias_los for bhi in bias_his
    ]
    print(f"  Grid: {len(grid)} combos × {len(all_signals)} candidates")

    results = []
    for p in grid:
        sub = [s for s in all_signals
               if p["rsi_lo"] <= s["rsi14"] <= p["rsi_hi"]
               and p["adx_lo"] <= s["adx14"] <= p["adx_hi"]
               and s["macd_h_pct"] >= p["macd_h_pct_min"]
               and p["bias_lo"] <= s["bias"] <= p["bias_hi"]]
        if len(sub) < 5:
            continue
        rets   = [s["return_pct"] for s in sub]
        wins   = sum(1 for r in rets if r > 0)
        avg    = float(np.mean(rets))
        std    = float(np.std(rets)) if len(rets) > 1 else 0.0
        sharpe = round(avg / std * (252 / hold_days) ** 0.5, 2) if std > 0 else None
        if sharpe is None:
            continue
        results.append({
            "params":     p,
            "n_signals":  len(sub),
            "win_rate":   round(wins / len(sub), 3),
            "avg_return": round(avg, 2),
            "sharpe":     sharpe,
        })

    if not results:
        print("  No valid combos found.")
        return None

    results.sort(key=lambda x: x["sharpe"], reverse=True)
    best = results[0]
    print(f"  Best params  : {best['params']}")
    print(f"  Win rate     : {best['win_rate']:.1%}  |  Sharpe: {best['sharpe']}  |  n={best['n_signals']}")
    print(f"\n  Top 5:")
    for r in results[:5]:
        print(f"    WR={r['win_rate']:.1%} sharpe={r['sharpe']:5.2f} n={r['n_signals']:3d}  {r['params']}")
    return best


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Gigi Stock War Room — Local Simulation Engine")
    parser.add_argument("--market",    default="tw",  choices=["tw","us","both"])
    parser.add_argument("--years",     default=2,     type=int, help="Years of history (default: 2)")
    parser.add_argument("--hold",      default=10,    type=int, help="Hold days for backtest (default: 10)")
    parser.add_argument("--no-push",   action="store_true", help="Dry run: don't write to GSheets")
    parser.add_argument("--no-reload", action="store_true", help="Don't notify cloud to reload cache")
    args = parser.parse_args()

    markets    = ["tw","us"] if args.market == "both" else [args.market]
    end_date   = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=365 * args.years)).strftime("%Y-%m-%d")

    for market in markets:
        print(f"\n{'='*60}")
        print(f"  {market.upper()} | {start_date} → {end_date} ({args.years}yr) | hold={args.hold}d")
        print(f"{'='*60}")

        # ── 1. Watchlist ──────────────────────────────────────────────────────
        print("\n[1/4] Loading watchlist...")
        tickers = _get_watchlist(market)
        if not tickers:
            print("  No tickers. Check GSheets or GOOGLE_CREDENTIALS_JSON.")
            continue
        print(f"  {len(tickers)} tickers: {', '.join(tickers[:8])}{'...' if len(tickers)>8 else ''}")

        # ── 2. Collect signals ────────────────────────────────────────────────
        print(f"\n[2/4] Collecting candidates ({len(tickers)} tickers)...")
        all_signals: List[dict] = []
        t0 = time.time()
        for i, ticker in enumerate(tickers):
            code = ticker.split(".")[0] if market == "tw" else ticker
            sigs = _collect_wide_signals(code, start_date, end_date, args.hold, market=market)
            all_signals.extend(sigs)
            won = sum(1 for s in sigs if s["won"])
            elapsed = time.time() - t0
            eta_sec = elapsed / (i + 1) * (len(tickers) - i - 1)
            print(f"  [{i+1:3d}/{len(tickers)}] {ticker:8s}: {len(sigs):3d} signals "
                  f"({won} wins)  ETA {int(eta_sec//60)}m{int(eta_sec%60):02d}s", flush=True)

        if not all_signals:
            print("  No signals found. Try --years 3 or check data source.")
            continue

        total_won  = sum(1 for s in all_signals if s["won"])
        base_wr    = total_won / len(all_signals)
        print(f"\n  Total candidates : {len(all_signals)}")
        print(f"  Base win rate    : {base_wr:.1%}  (all base conditions, no RSI/ADX filter)")

        # ── 3. Grid search ────────────────────────────────────────────────────
        print(f"\n[3/4] Grid search...")
        best = run_grid_search(all_signals, market, args.hold)

        # ── 4. XGBoost ───────────────────────────────────────────────────────
        print(f"\n[4/4] Training XGBoost ({len(all_signals)} samples)...")
        model: Optional[Any] = None
        acc: float = 0.0
        try:
            from xgboost import XGBClassifier
            from sklearn.metrics import accuracy_score
            X  = [_xgb_features(s) for s in all_signals]
            y  = [1 if s["won"] else 0 for s in all_signals]
            Xn = np.array(X)
            yn = np.array(y)
            model = XGBClassifier(
                n_estimators=200, max_depth=5, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                eval_metric="logloss", random_state=42, verbosity=0,
            )
            model.fit(Xn, yn)
            acc = float(accuracy_score(yn, model.predict(Xn)))
            print(f"  Accuracy         : {acc:.1%}")
            print(f"  Train win rate   : {float(np.mean(yn)):.1%}")
        except ImportError:
            print("  xgboost not installed. Run: pip install xgboost scikit-learn")

        # ── Summary ───────────────────────────────────────────────────────────
        print(f"\n{'─'*60}")
        print(f"  RESULT SUMMARY — {market.upper()}")
        print(f"  Candidates        : {len(all_signals)}")
        print(f"  Base win rate     : {base_wr:.1%}")
        if best:
            print(f"  Optimized win rate: {best['win_rate']:.1%}  (Sharpe={best['sharpe']}, n={best['n_signals']})")
            print(f"  Best params       : {best['params']}")
        if model:
            print(f"  XGBoost accuracy  : {acc:.1%}")
        print(f"{'─'*60}")

        if args.no_push:
            print("\n  [--no-push] Skipping GSheets write.")
            continue

        # ── Push to GSheets ───────────────────────────────────────────────────
        print("\n  Pushing to Google Sheets...")
        if best:
            _save_live_params(market, best["params"], best["win_rate"], best["sharpe"])
        if model:
            _save_xgb_model(market, model, len(X), acc)

        if not args.no_reload:
            print("  Notifying cloud to reload...")
            _notify_cloud_reload(market)

    total_time = int(time.time() - (t0 if 'market' in dir() and 't0' in dir() else time.time()))
    print(f"\nDone! Total time: {total_time//60}m{total_time%60:02d}s")


if __name__ == "__main__":
    main()
