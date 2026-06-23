import json
import os
import time
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
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

SCAN_LOG_TAB   = "scan_log"
REG_COEFFS_TAB = "regression_coeffs"

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
                cash = 0.0
                for field in ("CashDividend", "cash_dividend",
                              "CashEarningsDistribution", "cash_earnings_distribution"):
                    v = d.get(field)
                    if v not in (None, "", "0", 0):
                        try:
                            cash += float(v); break
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
    line_token: str = ""

def send_line_notify(msg: str, token: str):
    if not token: return
    try:
        requests.post("https://notify-api.line.me/api/notify",
                      headers={"Authorization": f"Bearer {token}"}, data={"message": msg})
    except Exception: pass

@app.post("/api/scan")
async def scan_stocks(request: ScanRequest):
    results = []
    triggered = []

    # ── Market-level data ─────────────────────────────────────────────────────
    index_df = _get_index_df(request.market)
    index_20d_return = None
    market_regime_bull = None
    market_week_return = None
    market_week_rising = None

    if index_df is not None and len(index_df) >= 21:
        idx_c = index_df["Close"]
        index_20d_return = float((idx_c.iloc[-1] - idx_c.iloc[-21]) / idx_c.iloc[-21])
        idx_ma20 = idx_c.rolling(20).mean().iloc[-1]
        market_regime_bull = bool(idx_c.iloc[-1] > idx_ma20)

    if index_df is not None and len(index_df) >= 6:
        idx_c = index_df["Close"]
        market_week_return = round(float((idx_c.iloc[-1] - idx_c.iloc[-6]) / idx_c.iloc[-6]) * 100, 2)
        market_week_rising = market_week_return > 0

    inst_data = _load_inst_data() if request.market == "tw" else {}

    # ── Load yesterday's buy signals (for 2-day confirmation) ────────────────
    prev_signals = {}
    try:
        prev_signals = _load_prev_signals()
    except Exception:
        pass

    # ── Load regression model ─────────────────────────────────────────────────
    reg_coeffs = None
    try:
        reg_coeffs = _load_reg_coeffs()
    except Exception:
        pass

    def _no_data(ticker):
        return {
            "ticker": ticker, "companyName": get_company_name(ticker),
            "close":None,"open":None,"high":None,"low":None,
            "ma20":None,"ma60":None,"ma120":None,"volume":None,"vol_ma20":None,
            "rsi14":None,"bias":None,
            "week52_high":None,"week52_low":None,"pct_from_52high":None,
            "rs_score":None,"weekly_trend":None,
            "max_drawdown_1y":None,"stop_loss":None,"target_price":None,"pattern":"",
            "macd_line":None,"macd_signal":None,"macd_hist":None,"macd_cross":None,
            "adx14":None,"di_plus":None,"di_minus":None,
            "obv_trend":None,"monthly_trend":None,
            "is_breakout20":None,"vol_expansion":None,"inst_foreign":None,"inst_trust":None,
            "market_regime_bull":market_regime_bull,
            # v6 new
            "week5d_return":None,"is_extended":None,
            "confirmed_signal":False,
            "earnings_date":None,"near_earnings":False,
            "eps_growth":None,"revenue_growth":None,
            "est_eps":None,"est_dividend":None,"est_rev_growth":None,
            "market_week_return":market_week_return,"market_week_rising":market_week_rising,
            "predicted_return":None,
            "conds":{},"sell_flags":{},"signal":"NO_DATA",
        }

    for ticker in request.tickers:
        try:
            stock = yf.Ticker(ticker)
            df    = stock.history(period="1y")

            company_name = get_company_name(ticker)
            if company_name == ticker:
                try:
                    info = _get_info_cached(stock)
                    company_name = info.get("shortName") or info.get("longName") or ticker
                except Exception:
                    pass

            if df.empty or len(df) < 136:
                row = _no_data(ticker); row["companyName"] = company_name
                results.append(row); continue

            # ── Moving averages ───────────────────────────────────────────────
            df["MA20"]  = df["Close"].rolling(20).mean()
            df["MA60"]  = df["Close"].rolling(60).mean()
            df["MA120"] = df["Close"].rolling(120).mean()
            df["VMA20"] = df["Volume"].rolling(20).mean()

            # ── RSI-14 ───────────────────────────────────────────────────────
            delta = df["Close"].diff()
            ag = delta.clip(lower=0).ewm(alpha=1/14,min_periods=14,adjust=False).mean()
            al = (-delta).clip(lower=0).ewm(alpha=1/14,min_periods=14,adjust=False).mean()
            sl = al.copy(); sl[sl==0] = 1e-10
            df["RSI14"] = 100.0 - 100.0/(1.0+ag/sl)
            df["Bias"]  = (df["Close"]-df["MA20"])/df["MA20"]

            # ── MACD ─────────────────────────────────────────────────────────
            e12 = df["Close"].ewm(span=12,adjust=False).mean()
            e26 = df["Close"].ewm(span=26,adjust=False).mean()
            df["MACD"]     = e12-e26
            df["MACD_Sig"] = df["MACD"].ewm(span=9,adjust=False).mean()
            df["MACD_H"]   = df["MACD"]-df["MACD_Sig"]

            # ── ADX-14 ───────────────────────────────────────────────────────
            df["TR"]  = df[[(df["High"]-df["Low"]),(df["High"]-df["Close"].shift(1)).abs(),(df["Low"]-df["Close"].shift(1)).abs()]].max(axis=1) if False else \
                        pd.concat([df["High"]-df["Low"],(df["High"]-df["Close"].shift(1)).abs(),(df["Low"]-df["Close"].shift(1)).abs()],axis=1).max(axis=1)
            hd = df["High"]-df["High"].shift(1); ld = df["Low"].shift(1)-df["Low"]
            df["DM+"] = np.where((hd>ld)&(hd>0),hd,0.0)
            df["DM-"] = np.where((ld>hd)&(ld>0),ld,0.0)
            atr    = df["TR"].ewm(alpha=1/14,min_periods=14,adjust=False).mean()
            dip    = 100*df["DM+"].ewm(alpha=1/14,min_periods=14,adjust=False).mean()/atr.replace(0,np.nan)
            dim    = 100*df["DM-"].ewm(alpha=1/14,min_periods=14,adjust=False).mean()/atr.replace(0,np.nan)
            df["ADX"] = (100*(dip-dim).abs()/(dip+dim).replace(0,np.nan)).ewm(alpha=1/14,min_periods=14,adjust=False).mean()
            df["DI+"] = dip; df["DI-"] = dim

            # ── OBV ──────────────────────────────────────────────────────────
            df["OBV"]    = (np.sign(df["Close"].diff())*df["Volume"]).cumsum()
            df["OBV_MA"] = df["OBV"].rolling(20).mean()

            # ── 52-week, drawdown ─────────────────────────────────────────────
            w52h = float(df["High"].max())
            w52l = float(df["Low"].min())
            md1y = float(((df["Close"]-df["Close"].cummax())/df["Close"].cummax()).min())

            # ── Weekly trend ──────────────────────────────────────────────────
            wk = df["Close"].resample("W").last().dropna()
            wma5,wma10,wma20 = wk.rolling(5).mean(),wk.rolling(10).mean(),wk.rolling(20).mean()
            weekly_trend = None
            if len(wk)>=20 and not any(pd.isna([wma5.iloc[-1],wma10.iloc[-1],wma20.iloc[-1]])):
                weekly_trend = bool(wma5.iloc[-1]>wma10.iloc[-1]>wma20.iloc[-1])

            # ── Monthly trend ─────────────────────────────────────────────────
            mn = df["Close"].resample("MS").last().dropna()
            mm5,mm10 = mn.rolling(5).mean(),mn.rolling(10).mean()
            monthly_trend = None
            if len(mn)>=10 and not any(pd.isna([mm5.iloc[-1],mm10.iloc[-1]])):
                monthly_trend = bool(mm5.iloc[-1]>mm10.iloc[-1])

            # ── Extract latest ────────────────────────────────────────────────
            last   = df.iloc[-1]; prev = df.iloc[-2]
            c_close,c_open,c_high,c_low,c_vol = last["Close"],last["Open"],last["High"],last["Low"],last["Volume"]
            ma20,ma60,ma120,vol_ma20 = last["MA20"],last["MA60"],last["MA120"],last["VMA20"]
            rsi14,bias,prev_rsi = last["RSI14"],last["Bias"],df["RSI14"].iloc[-2]

            if any(pd.isna(v) for v in [ma20,ma60,ma120,vol_ma20,rsi14,bias,prev_rsi]):
                row = _no_data(ticker); row["companyName"] = company_name
                if not pd.isna(c_close): row["close"] = round(float(c_close),2)
                results.append(row); continue

            # ── v6: 5-day return / extended check ─────────────────────────────
            week5d_return = None
            is_extended   = False
            if len(df) >= 6:
                p5 = float(df["Close"].iloc[-6])
                week5d_return = round((float(c_close)-p5)/p5*100, 2)
                is_extended   = week5d_return > 5.0

            # ── MACD cross ───────────────────────────────────────────────────
            mn_m,pr_m = last["MACD"],prev["MACD"]
            mn_s,pr_s = last["MACD_Sig"],prev["MACD_Sig"]
            macd_cross = "none"
            if not any(pd.isna([mn_m,pr_m,mn_s,pr_s])):
                if pr_m<pr_s and mn_m>mn_s: macd_cross="golden"
                elif pr_m>pr_s and mn_m<mn_s: macd_cross="death"
                elif mn_m>mn_s: macd_cross="above"
                else: macd_cross="below"

            # ── ADX ──────────────────────────────────────────────────────────
            adx14v = None if pd.isna(last["ADX"]) else round(float(last["ADX"]),1)
            dip_v  = None if pd.isna(last["DI+"]) else round(float(last["DI+"]),1)
            dim_v  = None if pd.isna(last["DI-"]) else round(float(last["DI-"]),1)

            # ── OBV ──────────────────────────────────────────────────────────
            obv_trend = None
            if not pd.isna(last["OBV"]) and not pd.isna(last["OBV_MA"]):
                obv_trend = "rising" if last["OBV"]>last["OBV_MA"] else "falling"

            # ── Breakout & vol expansion ──────────────────────────────────────
            is_breakout20 = False
            if len(df)>=21: is_breakout20 = bool(float(c_close)>float(df["High"].iloc[-21:-1].max()))
            vol_expansion = False
            if len(df)>=3:
                v1,v2,v3 = df["Volume"].iloc[-3],df["Volume"].iloc[-2],df["Volume"].iloc[-1]
                vol_expansion = bool(v3>v2>v1 and v3>vol_ma20)

            # ── RS ────────────────────────────────────────────────────────────
            rs_score = None
            if len(df)>=21 and index_20d_return:
                s20 = float((c_close-df["Close"].iloc[-21])/df["Close"].iloc[-21])
                if index_20d_return != 0:
                    rs_score = round(s20/abs(index_20d_return),2)

            # ── Institutional ─────────────────────────────────────────────────
            inst = inst_data.get(ticker.split(".")[0],{})
            inst_foreign = inst.get("foreign") if inst else None
            inst_trust   = inst.get("trust")   if inst else None

            # ── 6 buy conditions ──────────────────────────────────────────────
            mid = (c_high+c_low)/2.0
            conds = {
                "price":  bool(c_close>ma20),
                "volume": bool(c_vol>1.2*vol_ma20),
                "trend":  bool(ma20>ma60 and ma60>ma120),
                "candle": bool(c_close>c_open and c_close>mid),
                "rsi":    bool(60.0<rsi14<70.0 and rsi14>prev_rsi),
                "bias":   bool(bias<0.03),
            }
            sell_flags = {
                "is_trend_broken":       bool(c_close<ma20),
                "is_momentum_lost":      bool(rsi14<50.0),
                "is_heavy_distribution": bool(c_close<c_open and c_vol>vol_ma20),
            }
            is_buy = all(conds.values())

            # ── v6: Confirmed signal (buy yesterday too) ──────────────────────
            confirmed_signal = prev_signals.get(ticker, False)

            # ── v6: Fundamentals (EPS, revenue, earnings) ─────────────────────
            fundamentals = _get_fundamentals(stock)  # earnings date from yfinance
            fm_est: dict = {}
            # Override EPS & revenue with FinMind for TW stocks (more accurate)
            if request.market == "tw":
                try:
                    _info = _get_info_cached(stock)
                    _shares = int(_info.get("sharesOutstanding", 0))
                except Exception:
                    _shares = 0
                fm = _get_finmind_fundamentals(ticker.split(".")[0], shares_actual=_shares)
                if fm["eps_growth"] is not None:
                    fundamentals["eps_growth"] = fm["eps_growth"]
                if fm["revenue_growth"] is not None:
                    fundamentals["revenue_growth"] = fm["revenue_growth"]
                fm_est = {
                    "est_eps":        fm.get("est_eps"),
                    "est_dividend":   fm.get("est_dividend"),
                    "est_rev_growth": fm.get("est_rev_growth"),
                }

            pattern       = _detect_pattern(df.tail(5))
            stop_loss     = round(float(ma20)*0.97,2)
            target_price  = round(float(c_close)*1.15,2)
            pct_from_52h  = round((float(c_close)-w52h)/w52h*100,1)

            result_row = {
                "ticker": ticker, "companyName": company_name,
                "close": round(float(c_close),2), "open": round(float(c_open),2),
                "high":  round(float(c_high),2),  "low":  round(float(c_low),2),
                "ma20":  round(float(ma20),2),     "ma60": round(float(ma60),2),
                "ma120": round(float(ma120),2),
                "volume": int(c_vol), "vol_ma20": int(vol_ma20),
                "rsi14": round(float(rsi14),1), "bias": round(float(bias)*100,2),
                "week52_high": round(w52h,2), "week52_low": round(w52l,2),
                "pct_from_52high": pct_from_52h,
                "rs_score": rs_score, "weekly_trend": weekly_trend,
                "max_drawdown_1y": round(md1y*100,1),
                "stop_loss": stop_loss, "target_price": target_price, "pattern": pattern,
                "macd_line":   round(float(mn_m),4) if not pd.isna(mn_m) else None,
                "macd_signal": round(float(mn_s),4) if not pd.isna(mn_s) else None,
                "macd_hist":   round(float(last["MACD_H"]),4) if not pd.isna(last["MACD_H"]) else None,
                "macd_cross":  macd_cross,
                "adx14": adx14v, "di_plus": dip_v, "di_minus": dim_v,
                "obv_trend": obv_trend, "monthly_trend": monthly_trend,
                "is_breakout20": is_breakout20, "vol_expansion": vol_expansion,
                "inst_foreign": inst_foreign, "inst_trust": inst_trust,
                "market_regime_bull": market_regime_bull,
                # v6 new fields
                "week5d_return":   week5d_return,
                "is_extended":     is_extended,
                "confirmed_signal": confirmed_signal,
                "earnings_date":    fundamentals["earnings_date"],
                "near_earnings":    fundamentals["near_earnings"],
                "eps_growth":       fundamentals["eps_growth"],
                "revenue_growth":   fundamentals["revenue_growth"],
                "est_eps":          fm_est.get("est_eps"),
                "est_dividend":     fm_est.get("est_dividend"),
                "est_rev_growth":   fm_est.get("est_rev_growth"),
                "market_week_return": market_week_return,
                "market_week_rising": market_week_rising,
                "predicted_return": None,
                "conds": conds, "sell_flags": sell_flags,
                "signal": "YES" if is_buy else "NO",
            }
            result_row["predicted_return"] = _apply_reg(result_row, reg_coeffs)
            results.append(result_row)

            if is_buy:
                triggered.append(
                    f"🚀 {ticker} {company_name} 買進訊號！"
                    f" Close:{round(float(c_close),2)} MACD:{macd_cross}"
                    + (" ✅確認" if confirmed_signal else "")
                    + (" ⚠️財報即將" if fundamentals["near_earnings"] else "")
                )

        except Exception as e:
            print(f"Error {ticker}: {e}")
            row = _no_data(ticker); row["signal"] = "ERROR"
            results.append(row)

    if triggered and request.line_token:
        send_line_notify("\n"+"\n".join(triggered), request.line_token)

    try:
        _append_scan_log(results, request.market)
    except Exception as e:
        print(f"scan_log (non-fatal): {e}")

    return {"status":"success","data":results}


# ── Notify ─────────────────────────────────────────────────────────────────────

class NotifyRequest(BaseModel):
    message: str
    line_token: str = ""

@app.post("/api/notify")
async def send_notify(req: NotifyRequest):
    if not req.line_token: return {"status":"skipped"}
    send_line_notify(req.message, req.line_token)
    return {"status":"ok"}


# ── Backtest ───────────────────────────────────────────────────────────────────

@app.post("/api/backtest")
async def backtest(request: ScanRequest):
    results = []
    for ticker in request.tickers:
        try:
            df = yf.Ticker(ticker).history(period="2y")
            if df.empty or len(df)<200:
                results.append({"ticker":ticker,"error":"insufficient data"}); continue
            df["MA20"]=df["Close"].rolling(20).mean(); df["MA60"]=df["Close"].rolling(60).mean()
            df["MA120"]=df["Close"].rolling(120).mean(); df["VMA20"]=df["Volume"].rolling(20).mean()
            d=df["Close"].diff(); ag=d.clip(lower=0).ewm(alpha=1/14,min_periods=14,adjust=False).mean()
            al=(-d).clip(lower=0).ewm(alpha=1/14,min_periods=14,adjust=False).mean()
            sl=al.copy(); sl[sl==0]=1e-10; df["RSI14"]=100-100/(1+ag/sl)
            df["Bias"]=(df["Close"]-df["MA20"])/df["MA20"]
            e12=df["Close"].ewm(span=12,adjust=False).mean(); e26=df["Close"].ewm(span=26,adjust=False).mean()
            df["MACD"]=e12-e26; df["MACD_S"]=df["MACD"].ewm(span=9,adjust=False).mean()
            signals=[]
            for i in range(max(136,len(df)-252), len(df)-10):
                r=df.iloc[i]; p=df.iloc[i-1]
                if any(pd.isna([r["MA20"],r["MA60"],r["MA120"],r["VMA20"],r["RSI14"],r["Bias"],p["RSI14"]])): continue
                mid=(r["High"]+r["Low"])/2
                if not(r["Close"]>r["MA20"] and r["Volume"]>1.2*r["VMA20"] and
                       r["MA20"]>r["MA60"]>r["MA120"] and r["Close"]>r["Open"]>mid and
                       60<r["RSI14"]<70 and r["RSI14"]>p["RSI14"] and r["Bias"]<0.03): continue
                entry=float(r["Close"]); ep=float(df["Close"].iloc[i+10])
                for j in range(1,11):
                    if float(df["Close"].iloc[i+j])<float(df["MA20"].iloc[i+j]):
                        ep=float(df["Close"].iloc[i+j]); break
                ret=(ep-entry)/entry
                signals.append({"date":df.index[i].strftime("%Y-%m-%d"),"entry":round(entry,2),
                                 "exit10d":round(ep,2),"return10d":round(ret*100,2),"win":ret>0,
                                 "macd_ok":not pd.isna(r["MACD"]) and r["MACD"]>r["MACD_S"]})
            if signals:
                wr=sum(1 for s in signals if s["win"])/len(signals)
                ar=sum(s["return10d"] for s in signals)/len(signals)
                ms=[s for s in signals if s["macd_ok"]]
                mwr=sum(1 for s in ms if s["win"])/len(ms) if ms else None
                results.append({"ticker":ticker,"companyName":get_company_name(ticker),
                                 "total_signals":len(signals),"win_rate":round(wr*100,1),
                                 "avg_return_10d":round(ar,2),
                                 "macd_filtered_wr":round(mwr*100,1) if mwr else None,
                                 "signals":signals[-20:]})
            else:
                results.append({"ticker":ticker,"companyName":get_company_name(ticker),
                                 "total_signals":0,"win_rate":None,"avg_return_10d":None,
                                 "macd_filtered_wr":None,"signals":[]})
        except Exception as e:
            results.append({"ticker":ticker,"error":str(e)})
    return {"status":"success","data":results}


# ── Regression endpoints ───────────────────────────────────────────────────────

@app.post("/api/regression/train")
async def regression_train(market: str = "tw"):
    ws = _get_or_create_tab(SCAN_LOG_TAB, SCAN_LOG_HEADERS)
    if not ws: raise HTTPException(503, "Google Sheets not configured")
    records = ws.get_all_records()
    if not records:
        return {"status":"insufficient_data","message":"No scan log yet. Scan daily to build data.","eligible":0}
    cutoff = (datetime.now()-timedelta(days=10)).strftime("%Y-%m-%d")
    eligible = [r for r in records if str(r.get("scan_date",""))<=cutoff and r.get("ticker")]
    if len(eligible)<20:
        return {"status":"insufficient_data",
                "message":f"Need 20+ points older than 10 days (have {len(eligible)}). Keep scanning!",
                "eligible":len(eligible)}
    seen,deduped = set(),[]
    for r in eligible:
        k=(r["scan_date"],r["ticker"])
        if k not in seen: seen.add(k); deduped.append(r)
    tickers = list(set(r["ticker"] for r in deduped))
    pcache = {}
    for t in tickers:
        try:
            h=yf.Ticker(t).history(period="1y")
            if not h.empty:
                s=h["Close"].copy(); s.index=pd.to_datetime(s.index).normalize().tz_localize(None)
                pcache[t]=s
        except Exception: pass
    Xr,yv=[],[]
    for r in deduped:
        t=r["ticker"]
        if t not in pcache: continue
        try:
            prices=pcache[t]; sd=pd.Timestamp(r["scan_date"]).tz_localize(None)
            fut=prices[prices.index>=sd]
            if len(fut)<11: continue
            ret=(float(fut.iloc[10])-float(fut.iloc[0]))/float(fut.iloc[0])
            Xr.append(_row_to_features(r)); yv.append(ret)
        except Exception: pass
    n=len(Xr)
    if n<20:
        return {"status":"insufficient_data","message":f"Only {n} valid samples (need 20+).","eligible":n}
    X=np.array(Xr,dtype=float); y=np.array(yv,dtype=float)
    Xb=np.column_stack([np.ones(n),X])
    coeffs,_,_,_=np.linalg.lstsq(Xb,y,rcond=None)
    yp=Xb@coeffs; ssr=float(np.sum((y-yp)**2)); sst=float(np.sum((y-y.mean())**2))
    r2=float(1-ssr/sst) if sst>0 else 0.0
    intercept=float(coeffs[0]); fc=[float(c) for c in coeffs[1:]]
    ws_reg=_get_or_create_tab(REG_COEFFS_TAB,["feature","value"])
    if ws_reg:
        ws_reg.clear()
        ws_reg.update("A1",[["feature","value"]]+
            [["_r2",str(round(r2,6))],["_n",str(n)],
             ["_updated",datetime.now().strftime("%Y-%m-%d %H:%M")],
             ["_intercept",str(intercept)]]+
            [[nm,str(c)] for nm,c in zip(FEATURE_NAMES,fc)])
    return {"status":"ok","r2":round(r2,4),"n_samples":n,"intercept":round(intercept,6),
            "feature_names":FEATURE_NAMES,
            "coefficients":{nm:round(c,6) for nm,c in zip(FEATURE_NAMES,fc)}}

@app.get("/api/regression/coeffs")
async def regression_coeffs_get(market: str = "tw"):
    reg=_load_reg_coeffs()
    if not reg: return {"status":"no_data"}
    return {"status":"ok","intercept":reg["intercept"],"r2":reg["r2"],
            "n_samples":reg["n_samples"],"updated":reg["updated"],
            "feature_names":FEATURE_NAMES,"coefficients":reg["coefficients"]}

@app.get("/api/forecast/{code}")
async def forecast_eps(code: str):
    """
    Direct 8-step EPS forecast for a TW stock code (e.g. 2330).
    Fetches all needed data from FinMind + yfinance and returns the forecast.
    """
    if not FINMIND_TOKEN:
        raise HTTPException(status_code=503, detail="FINMIND_TOKEN not set")
    try:
        stock = yf.Ticker(f"{code}.TW")
        info  = _get_info_cached(stock)
        shares = int(info.get("sharesOutstanding", 0))
    except Exception:
        shares = 0

    # Force-refresh by evicting cache entry so this endpoint always returns fresh data
    _FINMIND_CACHE.pop(code, None)
    fm = _get_finmind_fundamentals(code, shares_actual=shares)

    return {
        "code": code,
        "companyName": info.get("shortName") or info.get("longName") or code if shares else code,
        "shares_actual": shares,
        "eps_growth":     fm.get("eps_growth"),
        "revenue_growth": fm.get("revenue_growth"),
        "est_eps":        fm.get("est_eps"),
        "est_dividend":   fm.get("est_dividend"),
        "est_rev_growth": fm.get("est_rev_growth"),
    }


if __name__=="__main__":
    import uvicorn; uvicorn.run(app,host="0.0.0.0",port=8000)
