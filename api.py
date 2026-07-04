import json
import os
import time
# REMOVED (heavy computation moved to local Mac):
# import numpy as np
# import pandas as pd
# import yfinance as yf
from datetime import datetime, timedelta, timezone
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import requests
import gspread
from google.oauth2.service_account import Credentials

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_BUNDLE_PATH   = os.path.join(os.path.dirname(__file__), "tw_names.json")
FINMIND_TOKEN  = os.environ.get("FINMIND_TOKEN", "")
FINMIND_BASE   = "https://api.finmindtrade.com/api/v4/data"
_SHEET_NAME    = "gigi-war-room-watchlist"
_SHEET_TABS    = {"tw": "gigi-war-room-watchlist", "us": "gigi-us-watchlist"}
_TICKER_COL    = "ticker"
_SHEETS_SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive.readonly",
]

import datetime as _dt

SCAN_LOG_TAB   = "scan_log"
REG_COEFFS_TAB = "regression_coeffs"

# ── GSheets tab names & headers (used by lightweight read endpoints) ───────────
_OUTCOME_TAB      = "signal_outcomes"
_OUTCOME_HDR      = ["signal_date","ticker","market","close_signal",
                     "rsi14","adx14","bias","macd_cross","vol_expansion","confirmed",
                     "close_5d","return_5d","close_10d","return_10d","win"]
_MODEL_PARAMS_TAB = "model_params"
_MODEL_PARAMS_HDR = ["market","rsi_lo","rsi_hi","adx_lo","adx_hi",
                     "bias_lo","bias_hi","macd_h_pct_min","win_rate","sharpe","updated"]
_MODEL_STORE_TAB  = "model_store"
_MODEL_STORE_HDR  = ["market","model_b64","trained_at","n_samples","accuracy"]
_PAPER_RESULTS_TAB = "paper_results"
_PAPER_RESULTS_HDR = ["run_at","market","start_date","end_date","n_tickers","total_trades",
                      "win_rate","avg_return_pct","annual_return_pct","cumulative_return_pct",
                      "max_consec_loss","sharpe","avg_held_days","passed",
                      "rsi_lo","rsi_hi","adx_lo","adx_hi","bias_lo","bias_hi","macd_h_pct_min",
                      "avg_win_pct","avg_loss_pct","max_win_pct","max_loss_pct",
                      "monthly_wr","exit_reasons"]
_LIVE_PARAMS_CACHE: Dict[str, dict] = {}
_LIVE_PARAMS_TS:    Dict[str, float] = {}

def _get_live_params(market: str) -> Optional[dict]:
    now = time.time()
    if market in _LIVE_PARAMS_CACHE and now - _LIVE_PARAMS_TS.get(market, 0) < 3600:
        return _LIVE_PARAMS_CACHE[market]
    try:
        ws = _get_or_create_tab(_MODEL_PARAMS_TAB, _MODEL_PARAMS_HDR)
        if not ws: return None
        for row in ws.get_all_values()[1:]:
            if len(row) >= 8 and row[0] == market:
                p = {"rsi_lo":float(row[1]),"rsi_hi":float(row[2]),
                     "adx_lo":float(row[3]),"adx_hi":float(row[4]),
                     "bias_lo":float(row[5]),"bias_hi":float(row[6]),
                     "macd_h_pct_min":float(row[7])}
                _LIVE_PARAMS_CACHE[market] = p
                _LIVE_PARAMS_TS[market] = now
                return p
    except Exception as e:
        print(f"_get_live_params: {e}")
    return None

FEATURE_NAMES = [
    "macd_num", "adx_norm", "obv_num", "monthly_num",
    "breakout20", "vol_exp", "inst_fgn_k", "inst_tst_k",
    "rs_clip", "weekly_num", "rsi_norm", "bias_pct",
]

SCAN_LOG_HEADERS = [
    "scan_date", "ticker", "close",
    "macd_cross", "adx14", "obv_trend", "monthly_trend",
    "is_breakout20", "vol_expansion", "inst_foreign", "inst_trust",
    "rs_score", "weekly_trend", "rsi14", "bias", "is_buy",
]

# ── Caches ────────────────────────────────────────────────────────────────────
_INDEX_CACHE:   dict = {}
_INST_CACHE:    dict = {}
_INFO_CACHE:    dict = {}        # ticker -> {"info": {...}, "ts": float}
_FINMIND_CACHE: dict = {}        # code   -> {"data": {...}, "ts": float}
_PREV_SIGNALS:  dict = {"date": "", "signals": {}}
_SCAN_CACHE:    dict = {}        # market -> {"data": [...], "scanned_at": str, "ts": float}


# ── Company names ─────────────────────────────────────────────────────────────

def _load_tw_names() -> dict:
    result = {}
    try:
        with open(_BUNDLE_PATH, encoding="utf-8") as f:
            result = json.load(f)
        print(f"Loaded {len(result)} TW names from bundle.")
    except Exception as e:
        print(f"Warning: {_BUNDLE_PATH}: {e}")
    for url, cf, nf in [
        ("https://openapi.twse.com.tw/v1/opendata/t187ap03_L", "公司代號", "公司簡稱"),
        ("https://www.tpex.org.tw/openapi/v1/mopsfin_t187ap03_O", "SecuritiesCompanyCode", "CompanyAbbreviation"),
    ]:
        try:
            r = requests.get(url, timeout=8, headers={"Accept": "application/json"})
            if r.ok:
                for item in r.json():
                    c, n = item.get(cf,"").strip(), item.get(nf,"").strip()
                    if c and n:
                        result[c] = n
        except Exception:
            pass
    return result

_TW_NAME_MAP = _load_tw_names()

def get_company_name(ticker: str) -> str:
    return _TW_NAME_MAP.get(ticker.split(".")[0], ticker)


# ── Market index ──────────────────────────────────────────────────────────────

def _get_index_df(market: str) -> Optional[pd.DataFrame]:
    sym = "^TWII" if market == "tw" else "^GSPC"
    c = _INDEX_CACHE.get(market)
    if c and time.time() - c[1] < 600:
        return c[0]
    try:
        df = yf.Ticker(sym).history(period="1y")
        if not df.empty:
            _INDEX_CACHE[market] = (df, time.time())
            return df
    except Exception:
        pass
    return None


# ── Institutional data ─────────────────────────────────────────────────────────

def _load_inst_data() -> dict:
    now = time.time()
    if _INST_CACHE.get("data") and now - _INST_CACHE.get("ts", 0) < 3600:
        return _INST_CACHE["data"]
    for days_back in range(6):
        d = datetime.now() - timedelta(days=days_back)
        if d.weekday() >= 5:
            continue
        date_str = d.strftime("%Y%m%d")
        try:
            r = requests.get(
                f"https://www.twse.com.tw/rwd/zh/fund/T86?response=json&date={date_str}&selectType=ALL",
                timeout=12, headers={"Accept":"application/json","User-Agent":"Mozilla/5.0"},
            )
            if not r.ok: continue
            payload = r.json()
            if payload.get("stat") != "OK" or not payload.get("data"): continue
            fields = payload.get("fields", [])
            def _fi(kws, ex=""):
                for i,f in enumerate(fields):
                    if all(k in f for k in kws) and (not ex or ex not in f): return i
                return None
            fi_f = _fi(["外陸資買賣超"],"自營") or 4
            fi_t = _fi(["投信買賣超"]) or 10
            def _n(s):
                try: return int(str(s).replace(",","").replace("+","").strip() or "0")
                except: return 0
            result = {}
            for row in payload["data"]:
                try:
                    code = str(row[0]).strip()
                    result[code] = {"foreign": _n(row[fi_f]), "trust": _n(row[fi_t])}
                except Exception: pass
            if result:
                _INST_CACHE.update({"data": result, "ts": now, "date": date_str})
                return result
        except Exception as e:
            print(f"TWSE inst ({date_str}): {e}")
    return {}


# ── 8-Step EPS Forecast (pure function, unit-agnostic) ────────────────────────

def _forecast_eps_8step(
    current_ytd_rev: float,
    last_ytd_rev: float,
    last_total_rev: float,
    ttm_net_income_rate: float,
    outstanding_shares: float,
    past_3y_payout_rates: list,
) -> dict:
    """
    8-step annual EPS & dividend forecast.

    Revenue params: same unit (千元 TWD from FinMind).
    outstanding_shares: actual shares (not thousands).
    Returns EPS & dividend in TWD/share.

    Unit proof:
      est_net_income [千元] × 1000 [元/千元] ÷ shares [股] = EPS [元/股] ✓
    """
    if last_ytd_rev == 0:
        raise ValueError("last_ytd_rev cannot be 0")
    if outstanding_shares == 0:
        raise ValueError("outstanding_shares cannot be 0")
    if not past_3y_payout_rates:
        raise ValueError("past_3y_payout_rates cannot be empty")

    # Phase 1 — Revenue
    growth_yoy  = (current_ytd_rev - last_ytd_rev) / last_ytd_rev     # Step 1
    est_revenue = last_total_rev * (1 + growth_yoy)                    # Steps 2-3

    # Phase 2 — EPS
    est_net_income = est_revenue * ttm_net_income_rate                 # Steps 4-5
    est_eps = (est_net_income * 1000) / outstanding_shares             # Step 6

    # Phase 3 — Dividend
    avg_payout   = sum(past_3y_payout_rates) / len(past_3y_payout_rates)  # Step 7
    est_dividend = est_eps * avg_payout                                     # Step 8

    return {
        "revenue_growth_yoy":       round(growth_yoy * 100, 2),
        "estimated_annual_revenue": round(est_revenue, 0),
        "estimated_eps":            round(est_eps, 2),
        "estimated_cash_dividend":  round(est_dividend, 2),
    }


# ── FinMind fundamentals (TW stocks) ─────────────────────────────────────────

def _get_finmind_fundamentals(code: str, shares_actual: int = 0) -> dict:
    """
    Fetch from FinMind:
      - eps_growth    : quarterly YoY EPS growth %
      - revenue_growth: monthly YoY revenue growth %
      - est_eps       : 8-step forecasted annual EPS (TWD/share)
      - est_dividend  : forecasted cash dividend (TWD/share)
      - est_rev_growth: forecast-implied YTD revenue growth %
    """
    result = {
        "eps_growth": None, "revenue_growth": None,
        "est_eps": None, "est_dividend": None, "est_rev_growth": None,
    }
    if not FINMIND_TOKEN or not code:
        return result

    cached = _FINMIND_CACHE.get(code)
    if cached and time.time() - cached["ts"] < 86400:
        return cached["data"]

    now        = datetime.now()
    this_year  = now.year
    last_year  = this_year - 1
    start      = (now - timedelta(days=450)).strftime("%Y-%m-%d")
    div_start  = (now - timedelta(days=4 * 365)).strftime("%Y-%m-%d")

    # ── 1. Monthly Revenue ────────────────────────────────────────────────────
    rev_rows = []
    try:
        r = requests.get(FINMIND_BASE, params={
            "dataset": "TaiwanStockMonthRevenue", "data_id": code,
            "start_date": start, "token": FINMIND_TOKEN,
        }, timeout=12)
        if r.ok:
            rev_rows = sorted(
                r.json().get("data", []),
                key=lambda x: (int(x.get("revenue_year", 0)), int(x.get("revenue_month", 0)))
            )
    except Exception as e:
        print(f"FinMind Rev {code}: {e}")

    cur_ytd = last_ytd = last_total = ttm_rev = 0.0
    if rev_rows:
        try:
            last_rec = rev_rows[-1]
            cur_m    = int(last_rec.get("revenue_month", 0))

            # YoY: latest month vs same month last year
            same_m_ly = [x for x in rev_rows
                         if int(x.get("revenue_year", 0)) == last_year
                         and int(x.get("revenue_month", 0)) == cur_m]
            if same_m_ly:
                lr = float(same_m_ly[0]["revenue"])
                if lr != 0:
                    result["revenue_growth"] = round(
                        (float(last_rec["revenue"]) - lr) / lr * 100, 1)

            # Forecast components
            cur_ytd    = sum(float(x["revenue"]) for x in rev_rows
                            if int(x.get("revenue_year", 0)) == this_year
                            and int(x.get("revenue_month", 0)) <= cur_m)
            last_ytd   = sum(float(x["revenue"]) for x in rev_rows
                            if int(x.get("revenue_year", 0)) == last_year
                            and int(x.get("revenue_month", 0)) <= cur_m)
            last_total = sum(float(x["revenue"]) for x in rev_rows
                            if int(x.get("revenue_year", 0)) == last_year)
            ttm_rev    = sum(float(x["revenue"]) for x in rev_rows[-12:]) \
                         if len(rev_rows) >= 12 else 0.0
        except Exception as e:
            print(f"FinMind Rev calc {code}: {e}")

    # ── 2. Quarterly EPS ──────────────────────────────────────────────────────
    ttm_eps    = 0.0
    annual_eps: dict = {}
    try:
        r = requests.get(FINMIND_BASE, params={
            "dataset": "TaiwanStockFinancialStatements", "data_id": code,
            "start_date": start, "token": FINMIND_TOKEN,
        }, timeout=12)
        if r.ok:
            eps_rows = sorted(
                [d for d in r.json().get("data", []) if d.get("type") == "EPS"],
                key=lambda x: x["date"]
            )
            if eps_rows:
                # YoY same-quarter EPS growth
                if len(eps_rows) >= 5:
                    ne = float(eps_rows[-1]["value"])
                    ye = float(eps_rows[-5]["value"])
                    if ye != 0:
                        result["eps_growth"] = round((ne - ye) / abs(ye) * 100, 1)
                # TTM EPS (last 4 quarters)
                if len(eps_rows) >= 4:
                    ttm_eps = sum(float(x["value"]) for x in eps_rows[-4:])
                # Annual EPS by year (for payout ratio later)
                for rec in eps_rows:
                    yr = rec["date"][:4]
                    annual_eps[yr] = annual_eps.get(yr, 0.0) + float(rec["value"])
    except Exception as e:
        print(f"FinMind EPS {code}: {e}")

    # ── 3. Dividend payout rates (past 3 years) ───────────────────────────────
    payout_rates: list = []
    try:
        r = requests.get(FINMIND_BASE, params={
            "dataset": "TaiwanStockDividend", "data_id": code,
            "start_date": div_start, "token": FINMIND_TOKEN,
        }, timeout=12)
        if r.ok:
            cash_by_year: dict = {}
            for d in r.json().get("data", []):
                yr = str(d.get("year", "") or "")
                # FinMind TaiwanStockDividend: cash items have "現金" in dividend_item
                item = str(d.get("dividend_item", "") or "")
                cash = 0.0
                if "現金" in item:
                    # primary field is "dividend"
                    try:
                        cash = float(d.get("dividend", 0) or 0)
                    except Exception:
                        cash = 0.0
                # also try legacy field names as fallback
                if cash == 0:
                    for field in ("CashDividend", "cash_dividend",
                                  "CashEarningsDistribution", "cash_earnings_distribution"):
                        v = d.get(field)
                        if v not in (None, "", "0", 0):
                            try:
                                cash = float(v); break
                            except Exception:
                                pass
                if cash > 0 and yr:
                    cash_by_year[yr] = cash_by_year.get(yr, 0.0) + cash

            for yr, div in sorted(cash_by_year.items(), reverse=True)[:3]:
                eps_yr = annual_eps.get(yr, 0.0)
                if eps_yr > 0 and div > 0:
                    payout_rates.append(round(div / eps_yr, 4))
    except Exception as e:
        print(f"FinMind Div {code}: {e}")

    if not payout_rates:
        payout_rates = [0.50]   # conservative 50% default

    # ── 4. 8-Step EPS Forecast ────────────────────────────────────────────────
    print(f"Forecast inputs {code}: cur_ytd={cur_ytd:.0f} last_ytd={last_ytd:.0f} "
          f"last_total={last_total:.0f} ttm_rev={ttm_rev:.0f} "
          f"ttm_eps={ttm_eps:.2f} shares={shares_actual} payouts={payout_rates}")
    try:
        if (cur_ytd > 0 and last_ytd > 0 and last_total > 0
                and ttm_rev > 0 and ttm_eps != 0 and shares_actual > 0):
            # TTM net income rate: (TTM_EPS × shares → 元) → 千元 ÷ TTM_Rev (千元)
            ttm_net_income_千元 = ttm_eps * shares_actual / 1000
            ttm_rate = ttm_net_income_千元 / ttm_rev
            fc = _forecast_eps_8step(
                current_ytd_rev      = cur_ytd,
                last_ytd_rev         = last_ytd,
                last_total_rev       = last_total,
                ttm_net_income_rate  = ttm_rate,
                outstanding_shares   = float(shares_actual),
                past_3y_payout_rates = payout_rates[:3],
            )
            result["est_eps"]       = fc["estimated_eps"]
            result["est_dividend"]  = fc["estimated_cash_dividend"]
            result["est_rev_growth"]= fc["revenue_growth_yoy"]
    except Exception as e:
        print(f"Forecast {code}: {e}")

    _FINMIND_CACHE[code] = {"data": result, "ts": time.time()}
    print(f"FinMind {code}: eps_g={result.get('eps_growth')}% "
          f"rev_g={result.get('revenue_growth')}% "
          f"est_eps={result.get('est_eps')} est_div={result.get('est_dividend')}")
    return result


# ── Company fundamentals (EPS, revenue, earnings date) ────────────────────────

def _get_info_cached(stock: yf.Ticker) -> dict:
    ticker = stock.ticker
    c = _INFO_CACHE.get(ticker)
    if c and time.time() - c["ts"] < 21600:  # 6-hour cache
        return c["info"]
    try:
        info = stock.info
        _INFO_CACHE[ticker] = {"info": info, "ts": time.time()}
        return info
    except Exception:
        return {}


def _get_fundamentals(stock: yf.Ticker) -> dict:
    """Returns eps_growth, revenue_growth, earnings_date, near_earnings."""
    result = {
        "eps_growth": None, "revenue_growth": None,
        "earnings_date": None, "near_earnings": False,
    }
    try:
        info = _get_info_cached(stock)
        # EPS growth (quarterly YoY)
        eg = info.get("earningsQuarterlyGrowth")
        if eg is not None:
            result["eps_growth"] = round(float(eg) * 100, 1)
        # Revenue growth (YoY)
        rg = info.get("revenueGrowth")
        if rg is not None:
            result["revenue_growth"] = round(float(rg) * 100, 1)
        # Earnings date from earningsTimestamp
        ets = info.get("earningsTimestamp")
        if ets:
            ed = datetime.fromtimestamp(int(ets))
            result["earnings_date"] = ed.strftime("%Y-%m-%d")
            days = (ed - datetime.now()).days
            result["near_earnings"] = -1 <= days <= 7
    except Exception:
        pass

    # Also try calendar for earnings date if not found via info
    if not result["earnings_date"]:
        try:
            cal = stock.calendar
            if cal is not None:
                dates = []
                if isinstance(cal, dict):
                    dates = cal.get("Earnings Date", [])
                elif hasattr(cal, "columns"):
                    dates = list(cal.columns)
                if dates:
                    first = dates[0]
                    ds = first.strftime("%Y-%m-%d") if hasattr(first,"strftime") else str(first)[:10]
                    result["earnings_date"] = ds
                    try:
                        days = (datetime.strptime(ds, "%Y-%m-%d") - datetime.now()).days
                        result["near_earnings"] = -1 <= days <= 7
                    except Exception:
                        pass
        except Exception:
            pass
    return result


# ── Previous signals cache (for 2-day confirmation) ───────────────────────────

def _load_prev_signals() -> dict:
    today = datetime.now().strftime("%Y-%m-%d")
    if _PREV_SIGNALS["date"] == today and _PREV_SIGNALS["signals"]:
        return _PREV_SIGNALS["signals"]
    try:
        ws = _get_or_create_tab(SCAN_LOG_TAB, SCAN_LOG_HEADERS)
        if not ws:
            return {}
        records = ws.get_all_records()
        cutoff = (datetime.now() - timedelta(days=4)).strftime("%Y-%m-%d")
        signals = {}
        for r in records:
            d = str(r.get("scan_date", ""))
            if d >= cutoff and d < today and r.get("ticker"):
                if str(r.get("is_buy", "")).lower() in ("true", "1", "yes"):
                    signals[r["ticker"]] = True
        _PREV_SIGNALS["date"] = today
        _PREV_SIGNALS["signals"] = signals
        print(f"Loaded {len(signals)} prev buy signals")
        return signals
    except Exception as e:
        print(f"_load_prev_signals: {e}")
        return {}


# ── Sheets helpers ─────────────────────────────────────────────────────────────

def _get_gc():
    raw = os.environ.get("GOOGLE_CREDENTIALS_JSON")
    if not raw:
        raise HTTPException(503, "GOOGLE_CREDENTIALS_JSON not configured")
    return gspread.authorize(
        Credentials.from_service_account_info(json.loads(raw), scopes=_SHEETS_SCOPES)
    )

def _get_sheet_tab(market: str = "tw"):
    gc = _get_gc()
    sh = gc.open(_SHEET_NAME)
    tab = _SHEET_TABS.get(market, _SHEET_TABS["tw"])
    try:
        return sh.worksheet(tab)
    except Exception:
        ws = sh.add_worksheet(title=tab, rows=200, cols=1)
        ws.update("A1", [[_TICKER_COL]])
        return ws

def _get_or_create_tab(tab_name: str, headers: list):
    try:
        raw = os.environ.get("GOOGLE_CREDENTIALS_JSON")
        if not raw: return None
        gc = gspread.authorize(
            Credentials.from_service_account_info(json.loads(raw), scopes=_SHEETS_SCOPES)
        )
        sh = gc.open(_SHEET_NAME)
        try:
            return sh.worksheet(tab_name)
        except Exception:
            ws = sh.add_worksheet(title=tab_name, rows=5000, cols=len(headers))
            ws.update("A1", [headers])
            return ws
    except Exception as e:
        print(f"_get_or_create_tab({tab_name}): {e}")
        return None


# ── Regression helpers ─────────────────────────────────────────────────────────

def _row_to_features(r: dict) -> list:
    macd_map = {"golden":2.0,"above":1.0,"none":0.0,"below":-1.0,"death":-2.0}
    monthly = r.get("monthly_trend")
    weekly  = r.get("weekly_trend")
    return [
        macd_map.get(str(r.get("macd_cross") or "none"), 0.0),
        min(float(r.get("adx14") or 25), 50.0) / 50.0,
        1.0 if str(r.get("obv_trend"))=="rising" else (-1.0 if str(r.get("obv_trend"))=="falling" else 0.0),
        1.0 if monthly in (True,"True","true","1") else (-1.0 if monthly in (False,"False","false","0") else 0.0),
        1.0 if str(r.get("is_breakout20")) in ("True","true","1") or r.get("is_breakout20") is True else 0.0,
        1.0 if str(r.get("vol_expansion")) in ("True","true","1") or r.get("vol_expansion") is True else 0.0,
        max(-10.0, min(10.0, float(r.get("inst_foreign") or 0)/1000.0)),
        max(-5.0,  min(5.0,  float(r.get("inst_trust")   or 0)/1000.0)),
        max(-3.0,  min(3.0,  float(r.get("rs_score")     or 0))),
        1.0 if weekly in (True,"True","true","1") else (-1.0 if weekly in (False,"False","false","0") else 0.0),
        (float(r.get("rsi14") or 50)-50.0)/50.0,
        float(r.get("bias") or 0),
    ]

def _load_reg_coeffs() -> Optional[dict]:
    try:
        ws = _get_or_create_tab(REG_COEFFS_TAB, ["feature","value"])
        if not ws: return None
        records = ws.get_all_records()
        if not records: return None
        meta, coeffs = {}, {}
        for rec in records:
            feat = str(rec.get("feature",""))
            val  = str(rec.get("value",""))
            if feat.startswith("_"): meta[feat[1:]] = val
            elif feat:
                try: coeffs[feat] = float(val)
                except: pass
        if not coeffs: return None
        return {
            "intercept":  float(meta.get("intercept",0)),
            "r2":         float(meta.get("r2",0)),
            "n_samples":  int(float(meta.get("n",0))),
            "updated":    meta.get("updated",""),
            "coefficients": [coeffs.get(n,0.0) for n in FEATURE_NAMES],
        }
    except Exception:
        return None

def _apply_reg(row: dict, reg: Optional[dict]) -> Optional[float]:
    if not reg: return None
    try:
        pred = reg["intercept"] + sum(f*c for f,c in zip(_row_to_features(row), reg["coefficients"]))
        return round(pred*100, 2)
    except Exception:
        return None


# ── Candlestick patterns ───────────────────────────────────────────────────────

def _detect_pattern(df: pd.DataFrame) -> str:
    if len(df) < 3: return ""
    o,c,h,l = df["Open"].values, df["Close"].values, df["High"].values, df["Low"].values
    body  = lambda i: abs(c[i]-o[i])
    uw    = lambda i: h[i]-max(c[i],o[i])
    lw    = lambda i: min(c[i],o[i])-l[i]
    rng   = lambda i: h[i]-l[i]
    bull  = lambda i: c[i]>o[i]
    bear  = lambda i: c[i]<o[i]
    if bull(-3) and bull(-2) and bull(-1) and c[-2]>c[-3] and c[-1]>c[-2] and o[-2]>o[-3] and o[-1]>o[-2]: return "紅三兵"
    if bear(-3) and bear(-2) and bear(-1) and c[-2]<c[-3] and c[-1]<c[-2]: return "黑三兵"
    if bear(-2) and bull(-1) and o[-1]<=c[-2] and c[-1]>=o[-2]: return "多頭吞噬"
    if bull(-2) and bear(-1) and o[-1]>=c[-2] and c[-1]<=o[-2]: return "空頭吞噬"
    if rng(-1)>0 and body(-1)>0 and lw(-1)>=2*body(-1) and uw(-1)<=body(-1)*0.3: return "錘子線"
    if rng(-1)>0 and body(-1)>0 and uw(-1)>=2*body(-1) and lw(-1)<=body(-1)*0.3: return "射擊之星"
    if rng(-1)>0 and body(-1)<=rng(-1)*0.1: return "十字星"
    return ""


# ── Scan log ───────────────────────────────────────────────────────────────────

def _append_scan_log(results: list, market: str):
    if market != "tw": return
    try:
        ws = _get_or_create_tab(SCAN_LOG_TAB, SCAN_LOG_HEADERS)
        if not ws: return
        today = datetime.now().strftime("%Y-%m-%d")
        rows = []
        for r in results:
            if r.get("signal") in ("NO_DATA","ERROR") or r.get("close") is None: continue
            rows.append([
                today, r["ticker"], r.get("close",""),
                r.get("macd_cross",""), r.get("adx14",""), r.get("obv_trend",""),
                str(r.get("monthly_trend","")), str(r.get("is_breakout20","")),
                str(r.get("vol_expansion","")), r.get("inst_foreign",""),
                r.get("inst_trust",""), r.get("rs_score",""),
                str(r.get("weekly_trend","")), r.get("rsi14",""), r.get("bias",""),
                str(r.get("signal","") == "YES"),
            ])
        if rows:
            ws.append_rows(rows, value_input_option="RAW")
            print(f"Logged {len(rows)} rows to scan_log")
    except Exception as e:
        print(f"scan_log append: {e}")


# ── Watchlist endpoints ────────────────────────────────────────────────────────

@app.get("/api/watchlist")
async def get_watchlist(market: str = "tw"):
    try:
        ws = _get_sheet_tab(market)
        records = ws.get_all_records()
        return {"tickers": [r[_TICKER_COL] for r in records if r.get(_TICKER_COL,"").strip()]}
    except HTTPException: raise
    except Exception as e: raise HTTPException(500, f"{type(e).__name__}: {e}")

class WatchlistUpdate(BaseModel):
    tickers: List[str]

@app.put("/api/watchlist")
async def put_watchlist(body: WatchlistUpdate, market: str = "tw"):
    try:
        ws = _get_sheet_tab(market)
        ws.clear()
        ws.update("A1", [[_TICKER_COL]] + [[t] for t in body.tickers])
    except HTTPException: raise
    except Exception as e: raise HTTPException(500, f"{type(e).__name__}: {e}")
    return {"status": "ok", "count": len(body.tickers)}


# ── Scan endpoint ──────────────────────────────────────────────────────────────

class ScanRequest(BaseModel):
    tickers: List[str]
    market: str = "tw"
    tg_bot_token: str = ""
    tg_chat_id: str = ""

def send_telegram(msg: str, bot_token: str, chat_id: str):
    if not bot_token or not chat_id: return
    try:
        requests.post(
            f"https://api.telegram.org/bot{bot_token}/sendMessage",
            json={"chat_id": chat_id, "text": msg},
            timeout=10
        )
    except Exception: pass

SCAN_CACHE_TAB = "scan_cache"

@app.get("/api/scan/cache")
async def get_scan_cache(market: str = "tw"):
    """Return pre-computed scan results from GSheets scan_cache (written by scan_local.py)."""
    # Serve from in-memory cache if fresh (< 6 hours)
    cached = _SCAN_CACHE.get(market)
    if cached and time.time() - cached["ts"] < 21600:
        return {"status":"ok","market":market,
                "scanned_at":cached["scanned_at"],"data":cached["data"]}
    try:
        ws = _get_or_create_tab(SCAN_CACHE_TAB, ["scanned_at","market","ticker"])
        if not ws:
            return {"status":"no_data","market":market,"data":[]}
        all_rows = ws.get_all_values()
        if len(all_rows) < 2:
            return {"status":"no_data","market":market,"data":[]}
        hdr = all_rows[0]
        def _ci(col):
            return hdr.index(col) if col in hdr else None
        def _f(r, col):
            i = _ci(col)
            if i is None or i >= len(r) or r[i] == "": return None
            try: return float(r[i])
            except: return r[i]
        def _b(r, col):
            v = _f(r, col)
            return str(v).lower() in ("true","1","yes") if v is not None else None
        rows = [r for r in all_rows[1:] if r and len(r) > 2 and r[1] == market]
        scanned_at = rows[0][0] if rows else None
        results = []
        for r in rows:
            try:
                results.append({
                    "ticker":         r[_ci("ticker")] if _ci("ticker") is not None else "",
                    "companyName":    r[_ci("company_name")] if _ci("company_name") is not None else "",
                    "close":          _f(r,"close"),  "open": _f(r,"open"),
                    "high":           _f(r,"high"),   "low":  _f(r,"low"),
                    "volume":         _f(r,"volume"),  "vol_ma20": _f(r,"vol_ma20"),
                    "ma20":           _f(r,"ma20"),   "ma60": _f(r,"ma60"), "ma120": _f(r,"ma120"),
                    "rsi14":          _f(r,"rsi14"),  "bias": _f(r,"bias"),
                    "adx14":          _f(r,"adx14"),  "di_plus": _f(r,"di_plus"), "di_minus": _f(r,"di_minus"),
                    "macd_cross":     _f(r,"macd_cross"),
                    "macd_line":      _f(r,"macd_line"), "macd_signal": _f(r,"macd_signal"), "macd_hist": _f(r,"macd_hist"),
                    "obv_trend":      _f(r,"obv_trend"),
                    "monthly_trend":  _b(r,"monthly_trend"), "weekly_trend": _b(r,"weekly_trend"),
                    "is_breakout20":  _b(r,"is_breakout20"), "vol_expansion": _b(r,"vol_expansion"),
                    "week52_high":    _f(r,"week52_high"), "week52_low": _f(r,"week52_low"),
                    "pct_from_52high":_f(r,"pct_from_52high"),
                    "rs_score":       _f(r,"rs_score"), "max_drawdown_1y": _f(r,"max_drawdown_1y"),
                    "stop_loss":      _f(r,"stop_loss"), "target_price": _f(r,"target_price"),
                    "pattern":        _f(r,"pattern") or "",
                    "inst_foreign":   _f(r,"inst_foreign"), "inst_trust": _f(r,"inst_trust"),
                    "week5d_return":  _f(r,"week5d_return"), "is_extended": _b(r,"is_extended"),
                    "eps_growth":     _f(r,"eps_growth"), "revenue_growth": _f(r,"revenue_growth"),
                    "earnings_date":  _f(r,"earnings_date"), "near_earnings": _b(r,"near_earnings"),
                    "est_eps":        _f(r,"est_eps"), "est_dividend": _f(r,"est_dividend"),
                    "est_rev_growth": _f(r,"est_rev_growth"),
                    "market_regime_bull":  _b(r,"market_regime_bull"),
                    "market_week_return":  _f(r,"market_week_return"),
                    "market_week_rising":  _b(r,"market_week_rising"),
                    "signal":              _f(r,"signal") or "NO",
                    "confirmed_signal":    _b(r,"confirmed_signal"),
                    "conds": {
                        "price":  _b(r,"conds_price"),  "volume": _b(r,"conds_volume"),
                        "trend":  _b(r,"conds_trend"),  "candle": _b(r,"conds_candle"),
                        "rsi":    _b(r,"conds_rsi"),    "bias":   _b(r,"conds_bias"),
                    },
                    "sell_flags": {
                        "is_trend_broken":       _b(r,"sell_trend_broken"),
                        "is_momentum_lost":      _b(r,"sell_momentum_lost"),
                        "is_heavy_distribution": _b(r,"sell_heavy_dist"),
                    },
                    "xgb_prob": _f(r,"xgb_prob"),
                    "predicted_return": None,
                })
            except Exception:
                pass
        # Store in memory for fast subsequent reads
        _SCAN_CACHE[market] = {"data": results, "scanned_at": scanned_at, "ts": time.time()}
        return {"status":"ok","market":market,"scanned_at":scanned_at,"data":results}
    except Exception as e:
        return {"status":"error","reason":str(e),"data":[]}


# Invalidate in-memory scan cache when new local results are pushed
@app.post("/api/scan/cache/reload")
async def reload_scan_cache(market: str = "tw"):
    _SCAN_CACHE.pop(market, None)
    return {"status":"ok","reloaded":market}


@app.post("/api/scan")
async def scan_stocks(request: ScanRequest):
    return {"status": "run_locally", "message": "此功能已移至本機執行。請在 Mac 上執行 scan_local.py / simulate.py。"}
@app.post("/api/notify")
async def send_notify(req: NotifyRequest):
    if not req.tg_bot_token or not req.tg_chat_id: return {"status":"skipped"}
    send_telegram(req.message, req.tg_bot_token, req.tg_chat_id)
    return {"status":"ok"}


# ── Scheduled Notifications ────────────────────────────────────────────────────

_TG_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
_TG_CHAT_ID   = os.environ.get("TELEGRAM_CHAT_ID", "")
_TW_TZ = timezone(timedelta(hours=8))

@app.post("/api/scheduled/scan")
async def scheduled_scan(market: str = "tw"):
    return {"status": "run_locally", "message": "此功能已移至本機執行。請在 Mac 上執行 scan_local.py / simulate.py。"}
@app.post("/api/scheduled/summary")
async def midnight_summary():
    return {"status": "run_locally", "message": "此功能已移至本機執行。請在 Mac 上執行 scan_local.py / simulate.py。"}
@app.post("/api/outcomes/update")
async def update_outcomes(market: str = "tw"):
    """Fill 5d/10d actual returns for signals that are 15+ calendar days old."""
    try:
        ws = _get_or_create_tab(_OUTCOME_TAB, _OUTCOME_HDR)
        if not ws: return {"status":"error","reason":"sheet unavailable"}
        all_rows = ws.get_all_values()
        if len(all_rows) < 2: return {"status":"ok","updated":0}
        today = datetime.now(_TW_TZ).date()
        updated = 0
        for i, row in enumerate(all_rows[1:], 2):
            if len(row) < 10 or row[2] != market: continue
            if row[10]: continue  # already has close_5d
            try:
                sig_dt = datetime.strptime(row[0], "%Y-%m-%d").date()
                if (today - sig_dt).days < 15: continue
                close_sig = float(row[3]) if row[3] else None
                if not close_sig: continue
                c5  = _price_n_days_later(row[1], market, row[0], 5)
                c10 = _price_n_days_later(row[1], market, row[0], 10)
                r5  = round((c5 / close_sig - 1) * 100, 2) if c5 else ""
                r10 = round((c10 / close_sig - 1) * 100, 2) if c10 else ""
                win = 1 if (isinstance(r10, float) and r10 >= 3.0) else 0
                ws.update(f"K{i}:O{i}", [[c5 or "", r5, c10 or "", r10, win]])
                updated += 1
            except Exception as e:
                print(f"outcome row {i}: {e}")
        return {"status":"ok","market":market,"updated":updated}
    except Exception as e:
        return {"status":"error","reason":str(e)}

# ── XGBoost model ─────────────────────────────────────────────────────────────

def _xgb_features(row: dict) -> List[float]:
    macd_map = {"golden":2.0,"above":1.0,"none":0.0,"below":-1.0,"death":-2.0}
    return [
        float(row.get("rsi14") or 50),
        float(row.get("adx14") or 20),
        float(row.get("bias") or 0),
        macd_map.get(str(row.get("macd_cross") or "none"), 0.0),
        1.0 if str(row.get("vol_expansion")) in ("True","true","1") else 0.0,
        1.0 if str(row.get("is_breakout20")) in ("True","true","1") else 0.0,
        1.0 if str(row.get("monthly_trend")) in ("True","true","1","True") else 0.0,
        1.0 if str(row.get("obv_trend"))=="rising" else (-1.0 if str(row.get("obv_trend"))=="falling" else 0.0),
        min(float(row.get("rs_score") or 0), 3.0),
        1.0 if str(row.get("confirmed_signal")) in ("True","true","1") else 0.0,
    ]

def _save_xgb_model(market: str, model: Any, n_samples: int, accuracy: float):
    try:
        ws = _get_or_create_tab(_MODEL_STORE_TAB, _MODEL_STORE_HDR)
        if not ws: return
        b64 = _b64.b64encode(_gzip.compress(pickle.dumps(model))).decode()
        ts = datetime.now(_TW_TZ).strftime("%Y-%m-%d %H:%M")
        new_row = [market, b64, ts, n_samples, round(accuracy, 4)]
        rows = ws.get_all_values()
        for i, r in enumerate(rows[1:], 2):
            if r and r[0] == market:
                ws.update(f"A{i}", [new_row]); _XGB_MODEL_CACHE[market] = model; return
        ws.append_rows([new_row], value_input_option="RAW")
        _XGB_MODEL_CACHE[market] = model
    except Exception as e:
        print(f"_save_xgb_model: {e}")

def _load_xgb_model(market: str) -> Optional[Any]:
    if market in _XGB_MODEL_CACHE: return _XGB_MODEL_CACHE[market]
    try:
        ws = _get_or_create_tab(_MODEL_STORE_TAB, _MODEL_STORE_HDR)
        if not ws: return None
        for row in ws.get_all_values()[1:]:
            if len(row) >= 2 and row[0] == market:
                raw = _b64.b64decode(row[1])
                model = pickle.loads(_gzip.decompress(raw) if raw[:2] == b'\x1f\x8b' else raw)
                _XGB_MODEL_CACHE[market] = model
                return model
    except Exception as e:
        print(f"_load_xgb_model: {e}")
    return None

def _xgb_predict_prob(row: dict, market: str) -> Optional[float]:
    model = _load_xgb_model(market)
    if model is None: return None
    try:
# REMOVED:         import numpy as np
        prob = float(model.predict_proba(np.array([_xgb_features(row)]))[0][1])
        return round(prob, 3)
    except Exception:
        return None

@app.post("/api/model/train-xgb")
async def train_xgb(market: str = "tw", seed_from_scanlog: bool = False):
    return {"status": "run_locally", "message": "此功能已移至本機執行。請在 Mac 上執行 scan_local.py / simulate.py。"}
@app.post("/api/model/walk-forward")
async def walk_forward(market: str = "tw"):
    return {"status": "run_locally", "message": "此功能已移至本機執行。請在 Mac 上執行 scan_local.py / simulate.py。"}
@app.get("/api/model/stats")
async def model_stats(market: str = "tw"):
    """Real win rate from tracked outcomes + current model info."""
    try:
        out: Dict[str, Any] = {}
        live = _get_live_params(market)
        out["live_params"] = live or "defaults"
        # also pull win_rate / sharpe / updated stored alongside params
        try:
            ws0 = _get_or_create_tab(_MODEL_PARAMS_TAB, _MODEL_PARAMS_HDR)
            for r0 in (ws0.get_all_values()[1:] if ws0 else []):
                if r0 and r0[0] == market:
                    out["opt_win_rate"]   = float(r0[8])  if len(r0) > 8  and r0[8]  else None
                    out["opt_sharpe"]     = float(r0[9])  if len(r0) > 9  and r0[9]  else None
                    out["params_updated"] = r0[10]        if len(r0) > 10 else None
                    break
        except Exception: pass
        ws = _get_or_create_tab(_OUTCOME_TAB, _OUTCOME_HDR)
        rows = [r for r in (ws.get_all_values()[1:] if ws else [])
                if len(r) >= 15 and r[2] == market and r[14]]
        if rows:
            wins = sum(1 for r in rows if r[14] == "1")
            rets = [float(r[13]) for r in rows if r[13]]
            out["real_win_rate"]  = round(wins / len(rows), 3)
            out["total_signals"]  = len(rows)
            out["avg_return_10d"] = round(sum(rets) / len(rets), 2) if rets else None
        ws2 = _get_or_create_tab(_MODEL_STORE_TAB, _MODEL_STORE_HDR)
        for r in (ws2.get_all_values()[1:] if ws2 else []):
            if r and r[0] == market:
                out["xgb_trained_at"] = r[2] if len(r) > 2 else None
                out["xgb_samples"]    = r[3] if len(r) > 3 else None
                out["xgb_accuracy"]   = r[4] if len(r) > 4 else None
                break
        return {"status":"ok","market":market,**out}
    except Exception as e:
        return {"status":"error","reason":str(e)}

@app.post("/api/model/reload")
async def model_reload(market: str = "tw"):
    """
    Clear in-memory param + model caches so the next request re-reads from GSheets.
    Called by simulate.py after pushing local optimization results.
    """
    _LIVE_PARAMS_CACHE.pop(market, None)
    _LIVE_PARAMS_TS.pop(market, None)
    _XGB_MODEL_CACHE.pop(market, None)
    return {"status": "ok", "market": market, "message": "Cache cleared — next scan uses latest GSheets params/model"}


@app.get("/api/paper-report")
async def paper_report(market: str = "tw"):
    """Return latest paper trading backtest result from GSheets paper_results tab."""
    try:
        ws = _get_or_create_tab(_PAPER_RESULTS_TAB, _PAPER_RESULTS_HDR)
        all_rows = ws.get_all_values() if ws else []
        hdr = _PAPER_RESULTS_HDR
        target = None
        for r in all_rows[1:]:
            if len(r) >= 2 and r[1] == market:
                target = r
        if not target:
            return {"status": "no_data", "market": market}
        def _f(val):
            try: return float(val)
            except: return None
        import json as _json
        def _j(val):
            try: return _json.loads(val)
            except: return {}
        return {
            "status":               "ok",
            "market":               market,
            "run_at":               target[0]  if len(target) > 0  else None,
            "start_date":           target[2]  if len(target) > 2  else None,
            "end_date":             target[3]  if len(target) > 3  else None,
            "n_tickers":            target[4]  if len(target) > 4  else None,
            "total_trades":         _f(target[5])  if len(target) > 5  else None,
            "win_rate":             _f(target[6])  if len(target) > 6  else None,
            "avg_return_pct":       _f(target[7])  if len(target) > 7  else None,
            "annual_return_pct":    _f(target[8])  if len(target) > 8  else None,
            "cumulative_return_pct":_f(target[9])  if len(target) > 9  else None,
            "max_consec_loss":      _f(target[10]) if len(target) > 10 else None,
            "sharpe":               _f(target[11]) if len(target) > 11 else None,
            "avg_held_days":        _f(target[12]) if len(target) > 12 else None,
            "passed":               (target[13] == "YES") if len(target) > 13 else False,
            "params": {
                "rsi_lo":        _f(target[14]) if len(target) > 14 else None,
                "rsi_hi":        _f(target[15]) if len(target) > 15 else None,
                "adx_lo":        _f(target[16]) if len(target) > 16 else None,
                "adx_hi":        _f(target[17]) if len(target) > 17 else None,
                "bias_lo":       _f(target[18]) if len(target) > 18 else None,
                "bias_hi":       _f(target[19]) if len(target) > 19 else None,
                "macd_h_pct_min":_f(target[20]) if len(target) > 20 else None,
            },
            "avg_win_pct":          _f(target[21]) if len(target) > 21 else None,
            "avg_loss_pct":         _f(target[22]) if len(target) > 22 else None,
            "max_win_pct":          _f(target[23]) if len(target) > 23 else None,
            "max_loss_pct":         _f(target[24]) if len(target) > 24 else None,
            "monthly_wr":           _j(target[25]) if len(target) > 25 else {},
            "exit_reasons":         _j(target[26]) if len(target) > 26 else {},
        }
    except Exception as e:
        return {"status": "error", "reason": str(e)}


# ── Backtest ───────────────────────────────────────────────────────────────────

@app.post("/api/backtest")
async def backtest(request: ScanRequest):
    return {"status": "run_locally", "message": "此功能已移至本機執行。請在 Mac 上執行 scan_local.py / simulate.py。"}
@app.post("/api/regression/train")
async def regression_train(market: str = "tw"):
    return {"status": "run_locally", "message": "此功能已移至本機執行。請在 Mac 上執行 scan_local.py / simulate.py。"}
@app.get("/api/regression/coeffs")
async def regression_coeffs_get(market: str = "tw"):
    reg=_load_reg_coeffs()
    if not reg: return {"status":"no_data"}
    return {"status":"ok","intercept":reg["intercept"],"r2":reg["r2"],
            "n_samples":reg["n_samples"],"updated":reg["updated"],
            "feature_names":FEATURE_NAMES,"coefficients":reg["coefficients"]}

# ── FinMind Historical Backtest ───────────────────────────────────────────────

def _compute_bt_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Vectorized indicator computation for backtesting (full DataFrame at once)."""
    d = df.copy()
    d["MA10"]  = d["Close"].rolling(10).mean()
    d["MA20"]  = d["Close"].rolling(20).mean()
    d["MA60"]  = d["Close"].rolling(60).mean()
    d["MA120"] = d["Close"].rolling(120).mean()
    d["VMA20"] = d["Volume"].rolling(20).mean()

    # RSI-14
    delta = d["Close"].diff()
    ag = delta.clip(lower=0).ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    al = (-delta).clip(lower=0).ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    d["RSI14"] = 100.0 - 100.0 / (1.0 + ag / al.replace(0, 1e-10))
    d["Bias"]  = (d["Close"] - d["MA20"]) / d["MA20"] * 100

    # MACD 12/26/9
    e12 = d["Close"].ewm(span=12, adjust=False).mean()
    e26 = d["Close"].ewm(span=26, adjust=False).mean()
    d["MACD"]       = e12 - e26
    d["MACD_Sig"]   = d["MACD"].ewm(span=9, adjust=False).mean()
    d["MACD_H"]     = d["MACD"] - d["MACD_Sig"]
    for _p in [33, 40, 50, 60, 66]:
        d[f"MACD_H_p{_p}"] = d["MACD_H"].rolling(50).quantile(_p / 100)
    d["MACD_H_Med"] = d["MACD_H_p60"]  # backward compat

    # ADX-14 (vectorized)
    tr = pd.concat([
        d["High"] - d["Low"],
        (d["High"] - d["Close"].shift()).abs(),
        (d["Low"]  - d["Close"].shift()).abs(),
    ], axis=1).max(axis=1)
    up   = d["High"].diff()
    down = -d["Low"].diff()
    dm_p = up.where((up > down) & (up > 0), 0.0)
    dm_m = down.where((down > up) & (down > 0), 0.0)
    atr  = tr.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    d["ATR14"] = atr  # absolute ATR for dynamic stop
    dip  = dm_p.ewm(alpha=1/14, min_periods=14, adjust=False).mean() / atr.replace(0, np.nan) * 100
    dim  = dm_m.ewm(alpha=1/14, min_periods=14, adjust=False).mean() / atr.replace(0, np.nan) * 100
    dx   = (dip - dim).abs() / (dip + dim).replace(0, np.nan) * 100
    d["ADX14"]  = dx.ewm(alpha=1/14, min_periods=14, adjust=False).mean().fillna(0)
    d["DI_plus"]= dip.fillna(0)
    d["DI_minus"]= dim.fillna(0)

    # OBV
    obv = (d["Volume"] * np.sign(d["Close"].diff()).fillna(0)).cumsum()
    d["OBV"]     = obv
    d["OBV_MA20"]= obv.rolling(20).mean()

    # 5-day return (extended check)
    d["ret5d"]      = d["Close"].pct_change(5) * 100
    d["is_extended"]= d["ret5d"] > 5

    # Monthly trend proxy: MA20 > MA60 > MA120
    d["monthly_trend"] = (d["MA20"] > d["MA60"]) & (d["MA60"] > d["MA120"])

    return d


# ── Market-specific signal parameters ────────────────────────────────────────
TW_BEST_PARAMS: dict = {
    "rsi_lo": 52, "rsi_hi": 60,
    "bias_lo": 4,  "bias_hi": 8,
    "adx_lo": 18,  "adx_hi": 35,
    "macd_h_pct_min": 60,
}
US_DEFAULT_PARAMS: dict = {   # Grid Search BEST: RSI 60-65, ADX 18-30, MACD_H ≥60%, Bias 4-8%
    "rsi_lo": 60, "rsi_hi": 65,
    "bias_lo": 4,  "bias_hi": 8,
    "adx_lo": 18,  "adx_hi": 30,
    "macd_h_pct_min": 60,
}

def _get_market_params(market: str) -> dict:
    live = _get_live_params(market)
    if live: return live
    return TW_BEST_PARAMS if market == "tw" else US_DEFAULT_PARAMS

_MACD_H_COLS = {33: "MACD_H_p33", 40: "MACD_H_p40",
                50: "MACD_H_p50", 60: "MACD_H_p60", 66: "MACD_H_p66"}

def _bt_is_buy(row, params: Optional[dict] = None) -> bool:
    """Apply core signal logic to one row of computed indicators."""
    if params is None:
        params = TW_BEST_PARAMS
    pct_min   = params.get("macd_h_pct_min", 60)
    pct_col   = _MACD_H_COLS[min(_MACD_H_COLS, key=lambda x: abs(x - pct_min))]
    try:
        for col in ["MA20", "VMA20", "RSI14", "Bias", "MACD", "MACD_Sig", "ADX14", pct_col]:
            if pd.isna(row[col]):
                return False
        return (
            row["Close"]  > row["MA20"]                                      and
            row["Volume"] > row["VMA20"]                                     and
            params["rsi_lo"]  <= row["RSI14"] <= params["rsi_hi"]           and
            params["bias_lo"] <= row["Bias"]  <= params.get("bias_hi", 8)   and
            row["MACD"]   > row["MACD_Sig"]                                  and
            row["MACD_H"] > row[pct_col]                                     and
            params["adx_lo"]  <= row["ADX14"] <= params["adx_hi"]           and
            row["OBV"]    > row["OBV_MA20"]                                  and
            bool(row["monthly_trend"])                                        and
            not bool(row["is_extended"])
        )
    except Exception:
        return False


def _fetch_ohlcv(code: str, fetch_start: str, fetch_end: str, market: str = "tw") -> Optional[pd.DataFrame]:
    """Fetch OHLCV as standardised DataFrame (columns: date,Open,High,Low,Close,Volume)."""
    if market == "tw":
        try:
            r = requests.get(FINMIND_BASE, params={
                "dataset": "TaiwanStockPrice", "data_id": code,
                "start_date": fetch_start, "end_date": fetch_end,
                "token": FINMIND_TOKEN,
            }, timeout=30)
            raw = r.json().get("data", []) if r.ok else []
        except Exception:
            return None
        if len(raw) < 60:
            return None
        df = pd.DataFrame(raw)
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)
        for src, dst in [("open","Open"),("max","High"),("min","Low"),
                         ("close","Close"),("Trading_Volume","Volume")]:
            if src in df.columns:
                df[dst] = pd.to_numeric(df[src], errors="coerce")
    else:
        # US stocks via yfinance
        try:
            ticker_sym = code if "." not in code else code
            yf_df = yf.download(ticker_sym, start=fetch_start, end=fetch_end,
                                auto_adjust=True, progress=False)
            if yf_df.empty or len(yf_df) < 60:
                return None
            yf_df = yf_df.reset_index()
            # yfinance returns MultiIndex columns sometimes
            if isinstance(yf_df.columns, pd.MultiIndex):
                yf_df.columns = [c[0] for c in yf_df.columns]
            yf_df = yf_df.rename(columns={"Date":"date","Open":"Open","High":"High",
                                           "Low":"Low","Close":"Close","Volume":"Volume"})
            df = yf_df[["date","Open","High","Low","Close","Volume"]].copy()
            df["date"] = pd.to_datetime(df["date"])
            df = df.sort_values("date").reset_index(drop=True)
        except Exception:
            return None

    for col in ["Open","High","Low","Close","Volume"]:
        if col not in df.columns:
            return None
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["Close"]).reset_index(drop=True)
    df["Volume"] = df["Volume"].fillna(0)
    return df


def _backtest_ticker(
    code: str, start_date: str, end_date: str,
    hold_days: int = 10, company_name: str = "", market: str = "tw",
) -> dict:
    """
    Single-stock backtest. Uses FinMind for TW stocks, yfinance for US stocks.
    Fetches full OHLCV history, computes indicators vectorized, finds all
    signal days in [start_date, end_date], measures forward returns.
    """
    base = {
        "ticker": code, "company_name": company_name,
        "total_signals": 0, "win_rate": None,
        "avg_return": None, "avg_win": None,
        "avg_loss": None, "sharpe": None, "signals": [],
    }

    # Need 250-day warm-up before start, plus hold_days buffer after end
    fetch_start = (datetime.strptime(start_date, "%Y-%m-%d") - timedelta(days=400)).strftime("%Y-%m-%d")
    fetch_end   = (datetime.strptime(end_date,   "%Y-%m-%d") + timedelta(days=hold_days + 15)).strftime("%Y-%m-%d")

    df = _fetch_ohlcv(code, fetch_start, fetch_end, market=market)
    if df is None:
        return {**base, "error": "Insufficient data"}

    # Compute all indicators
    df = _compute_bt_indicators(df)

    # Restrict signal detection to user date range
    s_dt = pd.to_datetime(start_date)
    e_dt = pd.to_datetime(end_date)
    in_range = df[(df["date"] >= s_dt) & (df["date"] <= e_dt)]

    params = _get_market_params(market)
    signals = []
    for idx, row in in_range.iterrows():
        if not _bt_is_buy(row, params):
            continue

        entry = float(row["Close"])
        if entry <= 0:
            continue
        entry_atr = float(row["ATR14"]) if not pd.isna(row.get("ATR14", float("nan"))) else entry * 0.03
        atr_stop  = entry - entry_atr * 1.5

        # Dynamic exit: min 3 days hold, check conditions from day 3 onward
        future_idx = [i for i in df.index if i > idx]
        if len(future_idx) < 3:
            continue

        exit_i          = future_idx[min(hold_days - 1, len(future_idx) - 1)]
        exit_reason     = "持滿天數"
        prev_macd_above = True
        below_ma10_days = 0

        for j, fi in enumerate(future_idx[:hold_days]):
            frow = df.loc[fi]
            day  = j + 1
            macd_above = frow["MACD"] > frow["MACD_Sig"]

            if day >= 3:
                if frow["RSI14"] > 70:
                    exit_i = fi; exit_reason = "RSI>70超買"; break
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
        segment = df.loc[future_idx[0]:exit_i]
        max_dd  = float((segment["Low"].min() - entry) / entry * 100)
        ret_pct = (exit_p - entry) / entry * 100

        signals.append({
            "date":        row["date"].strftime("%Y-%m-%d"),
            "entry_price": round(entry,   2),
            "exit_price":  round(exit_p,  2),
            "return_pct":  round(ret_pct, 2),
            "max_dd":      round(max_dd,  2),
            "won":         ret_pct > 0,
            "held_days":   held,
            "exit_reason": exit_reason,
            "atr_stop":    round(atr_stop, 2),
            "rsi14":  round(float(row["RSI14"]),  1) if not pd.isna(row["RSI14"])  else None,
            "adx14":  round(float(row["ADX14"]),  1) if not pd.isna(row["ADX14"])  else None,
            "macd_h": round(float(row["MACD_H"]), 4) if not pd.isna(row["MACD_H"]) else None,
            "bias":   round(float(row["Bias"]),   1) if not pd.isna(row["Bias"])   else None,
        })

    if not signals:
        return base

    rets   = [s["return_pct"] for s in signals]
    wins   = [r for r in rets if r > 0]
    losses = [r for r in rets if r <= 0]
    avg    = float(np.mean(rets))
    std    = float(np.std(rets)) if len(rets) > 1 else 0.0
    sharpe = round(avg / std * (252 / hold_days) ** 0.5, 2) if std > 0 else None

    return {
        **base,
        "total_signals": len(signals),
        "win_rate":  round(len(wins) / len(signals), 3),
        "avg_return": round(avg, 2),
        "avg_win":   round(float(np.mean(wins)),   2) if wins   else None,
        "avg_loss":  round(float(np.mean(losses)), 2) if losses else None,
        "sharpe":    sharpe,
        "signals":   signals,
    }


def _analyze_signals(all_signals: list) -> dict:
    """
    Condition contribution analysis: bucket each signal by indicator value at
    entry time, compare win_rate + avg_return across buckets to reveal which
    indicator ranges actually produce better outcomes.
    """
    if len(all_signals) < 5:
        return {}

    def _bucket(signals, key, bins, labels):
        out = []
        for i, lbl in enumerate(labels):
            lo, hi = bins[i], bins[i + 1]
            sub = [s for s in signals
                   if s.get(key) is not None and lo <= s[key] < hi]
            if not sub:
                out.append({"label": lbl, "count": 0, "win_rate": None, "avg_return": None})
                continue
            wins = sum(1 for s in sub if s["won"])
            rets = [s["return_pct"] for s in sub]
            out.append({
                "label":      lbl,
                "count":      len(sub),
                "win_rate":   round(wins / len(sub), 3),
                "avg_return": round(float(np.mean(rets)), 2),
            })
        return out

    # MACD_H: use data-driven tertile split so it adapts to each watchlist's price scale
    mh_vals = sorted(s["macd_h"] for s in all_signals if s.get("macd_h") is not None)
    n = len(mh_vals)
    if n >= 6:
        p33 = mh_vals[n // 3]
        p67 = mh_vals[n * 2 // 3]
        mh_buckets = _bucket(
            all_signals, "macd_h",
            [-1e9, p33, p67, 1e9],
            [f"MACD_H 弱 (≤{p33:.3f})", f"MACD_H 中 ({p33:.3f}~{p67:.3f})", f"MACD_H 強 (>{p67:.3f})"],
        )
    else:
        mh_buckets = []

    return {
        "total_analyzed": len(all_signals),
        "rsi_buckets":  _bucket(all_signals, "rsi14",
            [40, 50, 60, 75.01],
            ["RSI 40–50", "RSI 50–60", "RSI 60–75"]),
        "adx_buckets":  _bucket(all_signals, "adx14",
            [20, 25, 30, 999],
            ["ADX 20–25 弱", "ADX 25–30 中", "ADX 30+ 強"]),
        "bias_buckets": _bucket(all_signals, "bias",
            [-8.01, -4, 0, 4, 8.01],
            ["Bias -8~-4%", "Bias -4~0%", "Bias 0~+4%", "Bias +4~+8%"]),
        "macd_h_buckets": mh_buckets,
    }


def _collect_wide_signals(
    code: str, start_date: str, end_date: str,
    hold_days: int = 10, company_name: str = "", market: str = "tw",
) -> list:
    """
    Collect all candidate signals passing BASE conditions only (no RSI/ADX/MACD_H filters).
    Each signal carries its indicator values so grid search can filter in memory.
    Base: Close>MA20, Vol>VMA20, MACD>Signal, OBV>OBV_MA20, monthly_trend, not extended.
    Supports TW (FinMind) and US (yfinance).
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

    signals = []
    for idx, row in in_range.iterrows():
        try:
            for col in ["MA20","VMA20","RSI14","Bias","MACD","MACD_Sig","ADX14","MACD_H"]:
                if pd.isna(row[col]):
                    raise ValueError()
            # Base conditions only — RSI/ADX/MACD_H not filtered here
            if not (row["Close"] > row["MA20"] and
                    row["Volume"] > row["VMA20"] and
                    row["MACD"] > row["MACD_Sig"] and
                    row["OBV"] > row["OBV_MA20"] and
                    bool(row["monthly_trend"]) and
                    not bool(row["is_extended"])):
                continue
        except Exception:
            continue

        # MACD_H rolling percentile (0-100) vs past 50 bars
        past_mh = df["MACD_H"].iloc[max(0, idx - 50):idx].dropna()
        mh_pct = float((past_mh < float(row["MACD_H"])).mean() * 100) if len(past_mh) >= 5 else 50.0

        future = df[df.index > idx].head(hold_days)
        if len(future) < hold_days:
            continue
        entry  = float(row["Close"])
        exit_p = float(future.iloc[-1]["Close"])
        if entry <= 0:
            continue

        ret_pct = (exit_p - entry) / entry * 100
        signals.append({
            "ticker":       code,
            "company_name": company_name,
            "date":         row["date"].strftime("%Y-%m-%d"),
            "rsi14":        round(float(row["RSI14"]), 1),
            "adx14":        round(float(row["ADX14"]), 1),
            "bias":         round(float(row["Bias"]),  2),
            "macd_h":       round(float(row["MACD_H"]), 4),
            "macd_h_pct":   round(mh_pct, 1),
            "return_pct":   round(ret_pct, 2),
            "won":          ret_pct > 0,
        })
    return signals


class BacktestFullRequest(BaseModel):
    tickers:    List[str]
    market:     str = "tw"
    start_date: Optional[str] = None
    end_date:   Optional[str] = None
    hold_days:  int = 10


@app.post("/api/backtest/full")
async def backtest_full(request: BacktestFullRequest):
    return {"status": "run_locally", "message": "此功能已移至本機執行。請在 Mac 上執行 scan_local.py / simulate.py。"}
@app.post("/api/backtest/gridsearch")
async def backtest_gridsearch(request: BacktestFullRequest):
    return {"status": "run_locally", "message": "此功能已移至本機執行。請在 Mac 上執行 scan_local.py / simulate.py。"}
@app.get("/api/forecast/{code}")
async def forecast_eps(code: str):
    return {"status": "run_locally", "message": "此功能已移至本機執行。請在 Mac 上執行 scan_local.py / simulate.py。"}
@app.get("/api/sentiment")
async def get_sentiment():
    if not GROK_API_KEY:
        return {"sentiment_score": 0.0, "key_reason": "GROK_API_KEY 未設定",
                "target_sectors": [], "error": "no_key"}
    try:
        return _get_or_refresh_sentiment()
    except Exception as e:
        return {"sentiment_score": 0.0, "key_reason": str(e),
                "target_sectors": [], "error": "fetch_failed"}


@app.post("/api/sentiment/refresh")
async def refresh_sentiment():
    if not GROK_API_KEY:
        return {"error": "no_key"}
    try:
        return _get_or_refresh_sentiment(force=True)
    except Exception as e:
        return {"error": str(e)}


# ─── v9.0 VIX Market Regime ───────────────────────────────────────────────────
_VIX_CACHE: dict = {"date": None, "vix": None}

def _get_vix() -> float:
    today = _dt.date.today().isoformat()
    if _VIX_CACHE["date"] == today and _VIX_CACHE["vix"] is not None:
        return _VIX_CACHE["vix"]
    try:
        r = requests.get(
            "https://query1.finance.yahoo.com/v8/finance/chart/%5EVIX",
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=10,
        )
        data = r.json()
        level = float(data["chart"]["result"][0]["meta"]["regularMarketPrice"])
        _VIX_CACHE["date"] = today
        _VIX_CACHE["vix"]  = level
        return level
    except Exception:
        return 20.0

@app.get("/api/vix")
async def get_vix_endpoint():
    try:
        level = _get_vix()
        if level > 30:   status = "極度恐慌"; score = -20
        elif level > 25: status = "恐慌";     score = -10
        elif level > 20: status = "警戒";     score =  -3
        elif level > 15: status = "中性";     score =   0
        else:            status = "樂觀";     score =  +5
        return {"vix": round(level, 2), "status": status, "score": score}
    except Exception as e:
        return {"vix": None, "status": "未知", "score": 0, "error": str(e)}


# ─── v8.0 Claude Chat Agent ───────────────────────────────────────────────────
import base64 as _b64
import anthropic as _anthropic

ANTHROPIC_API_KEY  = os.environ.get("ANTHROPIC_API_KEY", "")
GITHUB_TOKEN       = os.environ.get("GITHUB_TOKEN", "")
GITHUB_REPO        = "lwbroo/gigi-stock-war-room-2"
VERCEL_DEPLOY_HOOK = os.environ.get("VERCEL_DEPLOY_HOOK", "")

_CHAT_TOOLS = [
    {
        "name": "read_file",
        "description": "Read a file from the GitHub repository. Use this to understand current code before making changes.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File path, e.g. 'frontend/index.html' or 'api.py'"}
            },
            "required": ["path"]
        }
    },
    {
        "name": "write_file",
        "description": "Write/update a file in the GitHub repository and commit it. After writing frontend/index.html, call deploy_frontend to go live.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path":           {"type": "string", "description": "File path to write"},
                "content":        {"type": "string", "description": "Full new file content"},
                "commit_message": {"type": "string", "description": "Git commit message (concise, imperative)"}
            },
            "required": ["path", "content", "commit_message"]
        }
    },
    {
        "name": "patch_file",
        "description": "Replace a specific string in a file with a new string, then commit. Much more efficient than write_file for small edits — you only need to output the changed portion, not the entire file. Always read_file first to get the exact current text to replace.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path":           {"type": "string", "description": "File path, e.g. 'frontend/index.html'"},
                "old_str":        {"type": "string", "description": "The exact string to find and replace (must be unique in the file)"},
                "new_str":        {"type": "string", "description": "The replacement string"},
                "commit_message": {"type": "string", "description": "Git commit message"}
            },
            "required": ["path", "old_str", "new_str", "commit_message"]
        }
    },
    {
        "name": "deploy_frontend",
        "description": "Trigger Vercel deployment for the frontend. Call this after patching or writing frontend/index.html.",
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": []
        }
    },
    {
        "name": "list_files",
        "description": "List files in a directory of the repository.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Directory path (empty string for root)"}
            },
            "required": []
        }
    }
]

def _gh_headers():
    return {"Authorization": f"token {GITHUB_TOKEN}", "Accept": "application/vnd.github.v3+json"}

def _gh_read(path: str) -> tuple:
    url = f"https://api.github.com/repos/{GITHUB_REPO}/contents/{path}"
    r = requests.get(url, headers=_gh_headers(), timeout=15)
    if r.status_code != 200:
        return None, None
    data = r.json()
    content = _b64.b64decode(data["content"].replace("\n","")).decode("utf-8")
    return content, data["sha"]

def _gh_write(path: str, content: str, message: str, sha: Optional[str] = None) -> str:
    url = f"https://api.github.com/repos/{GITHUB_REPO}/contents/{path}"
    if sha is None:
        _, sha = _gh_read(path)
    payload: dict = {
        "message": message,
        "content": _b64.b64encode(content.encode("utf-8")).decode("utf-8"),
    }
    if sha:
        payload["sha"] = sha
    r = requests.put(url, json=payload, headers=_gh_headers(), timeout=20)
    if r.status_code in (200, 201):
        return f"✅ Committed: {path} — \"{message}\""
    return f"❌ GitHub write failed ({r.status_code}): {r.text[:200]}"

def _gh_list(path: str = "") -> str:
    url = f"https://api.github.com/repos/{GITHUB_REPO}/contents/{path}"
    r = requests.get(url, headers=_gh_headers(), timeout=10)
    if r.status_code != 200:
        return f"Error: {r.status_code}"
    items = r.json()
    return "\n".join(f"{'📁' if i['type']=='dir' else '📄'} {i['path']}" for i in items)

def _run_tool(name: str, inp: dict, actions: list) -> str:
    if name == "read_file":
        content, _ = _gh_read(inp["path"])
        if content is None:
            return f"File not found: {inp['path']}"
        actions.append(f"📖 Read `{inp['path']}` ({len(content):,} chars)")
        return content
    if name == "write_file":
        result = _gh_write(inp["path"], inp["content"], inp["commit_message"])
        actions.append(f"✏️ {result}")
        return result
    if name == "patch_file":
        content, sha = _gh_read(inp["path"])
        if content is None:
            return f"File not found: {inp['path']}"
        old_str, new_str = inp["old_str"], inp["new_str"]
        if old_str not in content:
            return f"❌ old_str not found in {inp['path']}. Read the file again to get the exact current text."
        count = content.count(old_str)
        if count > 1:
            return f"❌ old_str appears {count} times — make it more specific (add more surrounding context)."
        new_content = content.replace(old_str, new_str, 1)
        result = _gh_write(inp["path"], new_content, inp["commit_message"], sha)
        actions.append(f"✏️ {result}")
        return result
    if name == "deploy_frontend":
        if not VERCEL_DEPLOY_HOOK:
            actions.append("⚠️ VERCEL_DEPLOY_HOOK not set — skipping deploy")
            return "VERCEL_DEPLOY_HOOK not configured. Code was committed to GitHub. Set VERCEL_DEPLOY_HOOK in Render env vars to enable auto-deploy."
        r = requests.post(VERCEL_DEPLOY_HOOK, timeout=10)
        if r.status_code in (200, 201):
            actions.append("🚀 Vercel deployment triggered — live in ~30s")
            return "Vercel deployment triggered successfully."
        actions.append(f"❌ Vercel deploy failed ({r.status_code})")
        return f"Vercel deploy failed: {r.status_code}"
    if name == "list_files":
        result = _gh_list(inp.get("path", ""))
        actions.append(f"📂 Listed `{inp.get('path','root')}`")
        return result
    return f"Unknown tool: {name}"

_CHAT_SYSTEM = """You are the AI coding assistant built into Gigi Stock War Room v8.0.

You have direct access to the codebase and can read, edit, commit, and deploy it.

## Project
- Frontend: `frontend/index.html` — single-file React 18 (CDN Babel). Deployed on Vercel.
- Backend:  `api.py` — FastAPI. Deployed on Render (auto-deploys on git push).
- Repo: lwbroo/gigi-stock-war-room-2

## Rules
1. Always read the relevant file BEFORE making any edit, so you have the exact current text.
2. PREFER patch_file over write_file for ALL edits. patch_file only outputs the changed portion — much faster and token-efficient. Only use write_file when creating a brand new file.
3. For frontend changes: read_file → patch_file → deploy_frontend. Safe to do directly.
4. For backend changes (api.py): describe the change and confirm with the user BEFORE writing. If the backend breaks, the chat breaks too.
4. Keep the Premium Dark Glass design system (dark bg, glassmorphism, Inter font, indigo accent).
5. Never remove existing features. Additions only, unless the user explicitly asks to remove.
6. Write concise commit messages in imperative form.
7. After deploying, tell the user the live URL: https://gigi-frontend-mu.vercel.app

## Tech notes
- Python 3.9 on Render: use Optional[X] not X|None
- PatternBadge must be defined OUTSIDE App() component
- sectorHeat useMemo must come AFTER isRowBuy/Sell/Warn declarations
- @babel/standalone@7.23.10, data-presets="react,env"
"""

class ChatRequest(BaseModel):
    messages: List[Dict[str, Any]]

@app.post("/api/chat")
async def chat_agent(body: ChatRequest):
    if not ANTHROPIC_API_KEY:
        return {"response": "ANTHROPIC_API_KEY 未設定。請在 Render 環境變數中加入。", "actions": []}
    if not GITHUB_TOKEN:
        return {"response": "GITHUB_TOKEN 未設定。請在 Render 環境變數中加入。", "actions": []}

    try:
        client = _anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        messages = [{"role": m["role"], "content": m["content"]} for m in body.messages]
        actions: list = []

        for _ in range(12):  # max agentic iterations
            resp = client.messages.create(
                model="claude-opus-4-8",
                max_tokens=16000,
                system=_CHAT_SYSTEM,
                tools=_CHAT_TOOLS,
                messages=messages
            )

            if resp.stop_reason == "end_turn":
                text = next((b.text for b in resp.content if hasattr(b, "text")), "")
                return {"response": text, "actions": actions}

            if resp.stop_reason == "tool_use":
                # Convert content blocks to dicts for serialization
                assistant_content = []
                for b in resp.content:
                    if b.type == "text":
                        assistant_content.append({"type": "text", "text": b.text})
                    elif b.type == "tool_use":
                        assistant_content.append({"type": "tool_use", "id": b.id, "name": b.name, "input": b.input})
                messages.append({"role": "assistant", "content": assistant_content})

                tool_results = []
                for block in resp.content:
                    if block.type == "tool_use":
                        result = _run_tool(block.name, block.input, actions)
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": str(result)
                        })
                messages.append({"role": "user", "content": tool_results})
            else:
                break

        return {"response": "Agent reached max iterations.", "actions": actions}

    except Exception as e:
        import traceback
        return {"response": f"❌ Error: {str(e)}\n\n```\n{traceback.format_exc()[-800:]}\n```", "actions": []}


if __name__=="__main__":
    import uvicorn; uvicorn.run(app,host="0.0.0.0",port=8000)
