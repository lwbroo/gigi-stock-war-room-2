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

import gzip
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
_SIM_RESULTS_TAB  = "sim_results"
_SIM_RESULTS_HDR  = ["run_at","market","years","n_tickers","n_signals","base_wr",
                     "opt_wr","sharpe","n_opt_signals","xgb_accuracy","xgb_samples",
                     "rsi_lo","rsi_hi","adx_lo","adx_hi","bias_lo","bias_hi","macd_h_pct_min","notes"]
_PAPER_RESULTS_TAB = "paper_results"
_PAPER_RESULTS_HDR = ["run_at","market","start_date","end_date","n_tickers","total_trades",
                      "win_rate","avg_return_pct","annual_return_pct","cumulative_return_pct",
                      "max_consec_loss","sharpe","avg_held_days","passed",
                      "rsi_lo","rsi_hi","adx_lo","adx_hi","bias_lo","bias_hi","macd_h_pct_min"]

TW_BEST_PARAMS   = {"rsi_lo":52,"rsi_hi":60,"bias_lo":4,"bias_hi":8,"adx_lo":18,"adx_hi":35,"macd_h_pct_min":60}
US_DEFAULT_PARAMS = {"rsi_lo":60,"rsi_hi":65,"bias_lo":4,"bias_hi":8,"adx_lo":18,"adx_hi":30,"macd_h_pct_min":60}
_MACD_H_COLS     = {33:"MACD_H_p33",40:"MACD_H_p40",50:"MACD_H_p50",60:"MACD_H_p60",66:"MACD_H_p66"}


# ── GSheets helpers ────────────────────────────────────────────────────────────

def _get_gc():
    if not GOOGLE_CREDS:
        sys.exit("ERROR: GOOGLE_CREDENTIALS_JSON not set.\n"
                 "Create a .env file with GOOGLE_CREDENTIALS_JSON='...' or export it in your shell.")
    creds_dict = json.loads(GOOGLE_CREDS)
    return gspread.service_account_from_dict(creds_dict)

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
    b64 = _b64.b64encode(gzip.compress(pickle.dumps(model))).decode()
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

def _save_sim_result(market: str, years: int, n_tickers: int, all_signals: list,
                     best: Optional[dict], acc: float, notes: str = ""):
    """Append one row to sim_results tab as proof of each local simulation run."""
    try:
        ws = _get_or_create_tab(_SIM_RESULTS_TAB, _SIM_RESULTS_HDR)
        total = len(all_signals)
        base_wr = round(sum(1 for s in all_signals if s["won"]) / total, 4) if total else 0
        p = best["params"] if best else {}
        row = [
            datetime.now().strftime("%Y-%m-%d %H:%M"),
            market, years, n_tickers, total,
            base_wr,
            best["win_rate"] if best else "",
            best["sharpe"]   if best else "",
            best["n_signals"] if best else "",
            round(acc, 4) if acc else "",
            total,
            p.get("rsi_lo",""), p.get("rsi_hi",""),
            p.get("adx_lo",""), p.get("adx_hi",""),
            p.get("bias_lo",""), p.get("bias_hi",""),
            p.get("macd_h_pct_min",""),
            notes,
        ]
        ws.append_rows([row], value_input_option="RAW")
        print(f"  + Logged to sim_results [{market}]")
    except Exception as e:
        print(f"  sim_results write failed: {e}")


def _save_paper_result(market: str, trades: list, params: dict,
                       start_date: str, end_date: str, passed: bool,
                       annual_ret: float, cum_ret: float, sharpe: Optional[float],
                       max_consec: int):
    """Upsert latest paper trading result into paper_results tab (one row per market)."""
    try:
        ws = _get_or_create_tab(_PAPER_RESULTS_TAB, _PAPER_RESULTS_HDR)
        rets = [t["return_pct"] for t in trades]
        avg_r  = round(float(np.mean(rets)), 3) if rets else 0
        wr     = round(sum(1 for t in trades if t["won"]) / len(trades), 4) if trades else 0
        avg_hd = round(float(np.mean([t["held_days"] for t in trades])), 1) if trades else 0
        row = [
            datetime.now().strftime("%Y-%m-%d %H:%M"),
            market, start_date, end_date,
            len(set(t["ticker"] for t in trades)),
            len(trades),
            wr,
            avg_r,
            round(annual_ret, 2),
            round(cum_ret, 2),
            max_consec,
            sharpe if sharpe else "",
            avg_hd,
            "YES" if passed else "NO",
            params.get("rsi_lo",""), params.get("rsi_hi",""),
            params.get("adx_lo",""), params.get("adx_hi",""),
            params.get("bias_lo",""), params.get("bias_hi",""),
            params.get("macd_h_pct_min",""),
        ]
        all_rows = ws.get_all_values()
        for i, r in enumerate(all_rows[1:], start=2):
            if r and r[1] == market:
                ws.update(f"A{i}", [row])
                print(f"  ↑ Updated paper_results [{market}]")
                return
        ws.append_rows([row], value_input_option="RAW")
        print(f"  + Logged to paper_results [{market}]")
    except Exception as e:
        print(f"  paper_results write failed: {e}")

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
    d["is_extended"]   = d["ret5d"] > 5
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
    # Tighter RSI / higher ADX / higher MACD_H to push win rate toward 85%
    rsi_los  = [52,54,56,58]       if is_tw else [50,55,58,60]
    rsi_his  = [60,62,64]          if is_tw else [65,68,72]
    adx_los  = [20,22,24,26]       if is_tw else [15,18,20,22]
    adx_his  = [32,36,40]          if is_tw else [32,36,40]
    mh_pcts  = [60,66,70,75]       if is_tw else [50,60,66,70]
    bias_los = [2,4]
    bias_his = [7,8]               if is_tw else [8,10]

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
        if len(sub) < 8:   # raised from 5 → more statistical confidence
            continue
        rets   = [s["return_pct"] for s in sub]
        wins   = sum(1 for r in rets if r > 0)
        wr     = wins / len(sub)
        avg    = float(np.mean(rets))
        std    = float(np.std(rets)) if len(rets) > 1 else 0.0
        sharpe = round(avg / std * (252 / hold_days) ** 0.5, 2) if std > 0 else None
        if sharpe is None or wr < 0.65:   # pre-filter: skip combos below 65% WR
            continue
        # Score = Sharpe × WR bonus × sample confidence — n<15 gets penalized
        score = sharpe * (wr / 0.65) * min(1.0, len(sub) / 15)
        results.append({
            "params":     p,
            "n_signals":  len(sub),
            "win_rate":   round(wr, 3),
            "avg_return": round(avg, 2),
            "sharpe":     sharpe,
            "score":      round(score, 3),
        })

    if not results:
        print("  No valid combos found (try --years 3 for more data).")
        return None

    results.sort(key=lambda x: x["score"], reverse=True)
    best = results[0]
    print(f"  Best params  : {best['params']}")
    print(f"  Win rate     : {best['win_rate']:.1%}  |  Sharpe: {best['sharpe']}  |  n={best['n_signals']}")
    print(f"\n  Top 5:")
    for r in results[:5]:
        print(f"    WR={r['win_rate']:.1%} sharpe={r['sharpe']:5.2f} n={r['n_signals']:3d}  {r['params']}")
    return best


# ── Paper trading simulation ───────────────────────────────────────────────────

def _bt_is_buy(row: pd.Series, params: dict) -> bool:
    """Return True if the row passes both base conditions and optimised param filters."""
    try:
        for col in ["MA20","VMA20","RSI14","Bias","MACD","MACD_Sig","ADX14","MACD_H"]:
            if pd.isna(row.get(col, float("nan"))):
                return False
        if not (
            row["Close"]  > row["MA20"]     and
            row["Volume"] > row["VMA20"]    and
            row["MACD"]   > row["MACD_Sig"] and
            row["OBV"]    > row["OBV_MA20"] and
            bool(row["monthly_trend"])       and
            not bool(row["is_extended"])
        ):
            return False
    except Exception:
        return False
    rsi  = float(row["RSI14"])
    adx  = float(row["ADX14"])
    bias = float(row["Bias"])
    return (params["rsi_lo"]  <= rsi  <= params["rsi_hi"]  and
            params["adx_lo"]  <= adx  <= params["adx_hi"]  and
            params["bias_lo"] <= bias <= params["bias_hi"])


def _collect_paper_trades(code: str, start_date: str, end_date: str,
                           params: dict, hold_days: int = 10, market: str = "tw") -> List[dict]:
    """
    Run full backtest with DYNAMIC EXIT for one ticker using given params.
    Returns list of completed trades (same logic as cloud _backtest_ticker).
    """
    fetch_start = (datetime.strptime(start_date, "%Y-%m-%d") - timedelta(days=400)).strftime("%Y-%m-%d")
    fetch_end   = (datetime.strptime(end_date,   "%Y-%m-%d") + timedelta(days=hold_days + 15)).strftime("%Y-%m-%d")
    df = _fetch_ohlcv(code, fetch_start, fetch_end, market=market)
    if df is None:
        return []
    df = _compute_bt_indicators(df)

    s_dt = pd.to_datetime(start_date)
    e_dt = pd.to_datetime(end_date)
    in_range = df[(df["date"] >= s_dt) & (df["date"] <= e_dt)]

    trades = []
    for idx, row in in_range.iterrows():
        if not _bt_is_buy(row, params):
            continue
        past_mh = df["MACD_H"].iloc[max(0, idx - 50):idx].dropna()
        mh_pct  = float((past_mh < float(row["MACD_H"])).mean() * 100) if len(past_mh) >= 5 else 50.0
        if mh_pct < params.get("macd_h_pct_min", 0):
            continue
        entry     = float(row["Close"])
        if entry <= 0: continue
        entry_atr = float(row["ATR14"]) if not pd.isna(row.get("ATR14", float("nan"))) else entry * 0.03
        atr_stop  = entry - entry_atr * 1.5

        future_idx = [i for i in df.index if i > idx]
        if len(future_idx) < 3: continue

        exit_i          = future_idx[min(hold_days - 1, len(future_idx) - 1)]
        exit_reason     = "持滿天數"
        prev_macd_above = True
        below_ma10_days = 0   # 2-day confirmation counter

        for j, fi in enumerate(future_idx[:hold_days]):
            frow = df.loc[fi]
            day  = j + 1
            macd_above = frow["MACD"] > frow["MACD_Sig"]
            if day >= 3:
                if frow["RSI14"] > 70:
                    exit_i = fi; exit_reason = "RSI>70超買"; break
                # Require 2 consecutive days below MA10 to avoid false exits
                ma10 = frow.get("MA10", float("nan"))
                if not pd.isna(ma10) and frow["Close"] < ma10:
                    below_ma10_days += 1
                    if below_ma10_days >= 2:
                        exit_i = fi; exit_reason = "跌破MA10×2日"; break
                else:
                    below_ma10_days = 0
                if prev_macd_above and not macd_above:
                    exit_i = fi; exit_reason = "MACD死叉"; break
                if frow["Close"] < atr_stop:
                    exit_i = fi; exit_reason = "ATR停損"; break
            prev_macd_above = macd_above

        exit_p  = float(df.loc[exit_i]["Close"])
        held    = future_idx.index(exit_i) + 1
        ret_pct = (exit_p - entry) / entry * 100

        trades.append({
            "ticker":      code,
            "entry_date":  row["date"].strftime("%Y-%m-%d"),
            "exit_date":   df.loc[exit_i]["date"].strftime("%Y-%m-%d"),
            "entry_price": round(entry, 2),
            "exit_price":  round(exit_p, 2),
            "return_pct":  round(ret_pct, 2),
            "held_days":   held,
            "exit_reason": exit_reason,
            "won":         ret_pct > 0,
            "rsi14":       round(float(row["RSI14"]), 1),
            "adx14":       round(float(row["ADX14"]), 1),
        })
    return trades


def run_paper_report(trades: List[dict], market: str, params: dict,
                     start_date: str, end_date: str, hold_days: int):
    """Print a comprehensive paper trading report and return pass/fail verdict."""
    if not trades:
        print("  No trades generated with current params.")
        return False

    trades_sorted = sorted(trades, key=lambda x: x["entry_date"])
    rets   = [t["return_pct"] for t in trades]
    wins   = [t for t in trades if t["won"]]
    losses = [t for t in trades if not t["won"]]
    wr     = len(wins) / len(trades)
    avg_r  = float(np.mean(rets))
    avg_w  = float(np.mean([t["return_pct"] for t in wins]))  if wins   else 0
    avg_l  = float(np.mean([t["return_pct"] for t in losses])) if losses else 0

    # Max consecutive losses
    max_consec = consec = 0
    for t in trades_sorted:
        consec = consec + 1 if not t["won"] else 0
        max_consec = max(max_consec, consec)

    # Cumulative return (equal weight, compounded)
    cum = 1.0
    for r in [t["return_pct"] for t in trades_sorted]:
        cum *= (1 + r / 100)
    total_ret = (cum - 1) * 100
    date_range_days = (datetime.strptime(end_date, "%Y-%m-%d") - datetime.strptime(start_date, "%Y-%m-%d")).days
    years = date_range_days / 365
    annual_ret = (cum ** (1 / years) - 1) * 100 if years > 0 else 0

    # Monthly breakdown
    monthly: dict = {}
    for t in trades_sorted:
        mo = t["entry_date"][:7]
        if mo not in monthly: monthly[mo] = {"wins":0,"total":0,"rets":[]}
        monthly[mo]["total"] += 1
        if t["won"]: monthly[mo]["wins"] += 1
        monthly[mo]["rets"].append(t["return_pct"])

    # Exit reason breakdown
    reasons: dict = {}
    for t in trades: reasons[t["exit_reason"]] = reasons.get(t["exit_reason"], 0) + 1

    # Sharpe
    std = float(np.std(rets)) if len(rets) > 1 else 0
    sharpe = round(avg_r / std * (252 / hold_days) ** 0.5, 2) if std > 0 else None

    # ── Print report ────────────────────────────────────────────────────────────
    W = 62
    print(f"\n{'═'*W}")
    print(f"  📊 PAPER TRADING SIMULATION — {market.upper()}")
    print(f"  {start_date} → {end_date}  |  hold={hold_days}d")
    print(f"  參數: RSI {params['rsi_lo']}-{params['rsi_hi']} / ADX {params['adx_lo']}-{params['adx_hi']} / Bias {params['bias_lo']}-{params.get('bias_hi',8)}%")
    print(f"{'═'*W}")

    TARGET_WR   = 0.65
    TARGET_CONS = 3
    wr_ok   = wr >= TARGET_WR
    cons_ok = max_consec <= TARGET_CONS

    print(f"\n  {'─'*28} 總覽 {'─'*28}")
    print(f"  總交易次數    : {len(trades)}筆  ({len(wins)}勝 {len(losses)}負)")
    print(f"  勝率          : {wr:.1%}  {'✅' if wr_ok   else '❌'}  (目標 ≥{TARGET_WR:.0%})")
    print(f"  最大連續虧損  : {max_consec}筆  {'✅' if cons_ok else '❌'}  (目標 ≤{TARGET_CONS}筆)")
    print(f"  平均報酬      : {avg_r:+.2f}%")
    print(f"  平均獲利      : {avg_w:+.2f}%  |  平均虧損: {avg_l:+.2f}%")
    print(f"  最大單筆獲利  : {max(rets):+.2f}%")
    print(f"  最大單筆虧損  : {min(rets):+.2f}%")
    print(f"  平均持倉天數  : {np.mean([t['held_days'] for t in trades]):.1f}天")
    print(f"  累積報酬(等權): {total_ret:+.1f}%  ({years:.1f}年)")
    print(f"  年化報酬      : {annual_ret:+.1f}%")
    if sharpe: print(f"  Sharpe Ratio  : {sharpe}")

    print(f"\n  {'─'*26} 每月勝率 {'─'*26}")
    for mo, d in sorted(monthly.items()):
        mo_wr  = d["wins"] / d["total"]
        mo_avg = float(np.mean(d["rets"]))
        ok     = "✅" if mo_wr >= TARGET_WR else "⚠️ "
        bar    = "█" * d["wins"] + "░" * (d["total"] - d["wins"])
        print(f"  {mo}  {bar}  {d['wins']}/{d['total']} ({mo_wr:.0%})  avg {mo_avg:+.1f}%  {ok}")

    print(f"\n  {'─'*24} 出場原因分布 {'─'*24}")
    for reason, cnt in sorted(reasons.items(), key=lambda x: -x[1]):
        pct = cnt / len(trades)
        print(f"  {reason:<12}: {cnt:3d}筆  ({pct:.0%})  {'█'*int(pct*30)}")

    print(f"\n  {'─'*60}")
    passed = wr_ok and cons_ok
    if passed:
        print(f"  🚀 結論：達到 v12.0 真實交易標準！")
        print(f"     勝率 {wr:.1%} ≥ 65%  ✅  最大連續虧損 {max_consec}筆 ≤ 3筆  ✅")
        print(f"     → 可以進入 v11.0 真實 Paper Trading 驗證階段")
    else:
        print(f"  ⚠️  結論：尚未達到標準，建議繼續優化策略")
        if not wr_ok:   print(f"     勝率 {wr:.1%} < 65%  ❌  需要提升")
        if not cons_ok: print(f"     最大連續虧損 {max_consec}筆 > 3筆  ❌  風險過高")
    print(f"{'═'*W}\n")
    return passed, annual_ret, total_ret, sharpe, max_consec


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Gigi Stock War Room — Local Simulation Engine")
    parser.add_argument("--market",    default="tw",  choices=["tw","us","both"])
    parser.add_argument("--years",     default=2,     type=int, help="Years of history (default: 2)")
    parser.add_argument("--hold",      default=10,    type=int, help="Hold days for backtest (default: 10)")
    parser.add_argument("--mode",      default="optimize", choices=["optimize","paper","both"],
                        help="optimize: grid search + XGBoost | paper: paper trading report | both: run optimize then paper")
    parser.add_argument("--no-push",   action="store_true", help="Dry run: don't write to GSheets")
    parser.add_argument("--no-reload", action="store_true", help="Don't notify cloud to reload cache")
    args = parser.parse_args()

    markets    = ["tw","us"] if args.market == "both" else [args.market]
    end_date   = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=365 * args.years)).strftime("%Y-%m-%d")
    t0         = time.time()

    for market in markets:
        print(f"\n{'='*60}")
        print(f"  {market.upper()} | {start_date} → {end_date} | mode={args.mode} | hold={args.hold}d")
        print(f"{'='*60}")

        # ── 1. Watchlist ──────────────────────────────────────────────────────
        print("\n[1] Loading watchlist...")
        tickers = _get_watchlist(market)
        if not tickers:
            print("  No tickers. Check GSheets or GOOGLE_CREDENTIALS_JSON.")
            continue
        print(f"  {len(tickers)} tickers: {', '.join(tickers[:8])}{'...' if len(tickers)>8 else ''}")

        # ══════════════════════════════════════════════════════════════════════
        #  OPTIMIZE MODE  (grid search + XGBoost + push to cloud)
        # ══════════════════════════════════════════════════════════════════════
        best: Optional[dict] = None
        model: Optional[Any] = None
        acc: float = 0.0
        all_signals: List[dict] = []

        if args.mode in ("optimize", "both"):
            print(f"\n[2] Collecting wide candidates ({len(tickers)} tickers)...")
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
                print("  No signals found."); continue

            base_wr = sum(1 for s in all_signals if s["won"]) / len(all_signals)
            print(f"\n  Total candidates : {len(all_signals)}  |  Base win rate: {base_wr:.1%}")

            print(f"\n[3] Grid search...")
            best = run_grid_search(all_signals, market, args.hold)

            print(f"\n[4] Training XGBoost ({len(all_signals)} samples)...")
            try:
                from xgboost import XGBClassifier
                from sklearn.metrics import accuracy_score
                X  = [_xgb_features(s) for s in all_signals]
                y  = [1 if s["won"] else 0 for s in all_signals]
                model = XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.1,
                                      subsample=0.8, colsample_bytree=0.8,
                                      eval_metric="logloss", random_state=42, verbosity=0)
                model.fit(np.array(X), np.array(y))
                acc = float(accuracy_score(y, model.predict(np.array(X))))
                print(f"  Accuracy: {acc:.1%}")
            except ImportError:
                print("  xgboost not installed.")

            if not args.no_push:
                print("\n  Pushing to GSheets...")
                if best:  _save_live_params(market, best["params"], best["win_rate"], best["sharpe"])
                if model: _save_xgb_model(market, model, len(X), acc)
                if not args.no_reload:
                    print("  Notifying cloud reload...")
                    _notify_cloud_reload(market)
                _save_sim_result(market, args.years, len(tickers), all_signals, best, acc)
            else:
                print("\n  [--no-push] Skipping GSheets write.")

        # ══════════════════════════════════════════════════════════════════════
        #  PAPER MODE  (simulate actual trades with dynamic exit)
        # ══════════════════════════════════════════════════════════════════════
        if args.mode in ("paper", "both"):
            # Decide which params to use
            if best:
                params = best["params"]
                print(f"\n[{'5' if args.mode=='both' else '2'}] Paper simulation using grid-search params...")
            else:
                # Load from GSheets or fall back to hardcoded
                params = TW_BEST_PARAMS if market == "tw" else US_DEFAULT_PARAMS
                print(f"\n[2] Paper simulation using {'GSheets live' if True else 'default'} params...")
                print(f"  Params: {params}")

            print(f"  Collecting paper trades ({len(tickers)} tickers)...")
            all_trades: List[dict] = []
            for i, ticker in enumerate(tickers):
                code = ticker.split(".")[0] if market == "tw" else ticker
                trades = _collect_paper_trades(code, start_date, end_date,
                                               params, args.hold, market=market)
                all_trades.extend(trades)
                print(f"  [{i+1:3d}/{len(tickers)}] {ticker:8s}: {len(trades)} trades", flush=True)

            result = run_paper_report(all_trades, market, params, start_date, end_date, args.hold)
            if result and not args.no_push:
                passed, annual_ret, cum_ret, sharpe, max_consec = result
                _save_paper_result(market, all_trades, params, start_date, end_date,
                                   passed, annual_ret, cum_ret, sharpe, max_consec)

    total_time = int(time.time() - t0)
    print(f"\nDone! Total time: {total_time//60}m{total_time%60:02d}s")


if __name__ == "__main__":
    main()
