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
            df["MACD"]       = e12-e26
            df["MACD_Sig"]   = df["MACD"].ewm(span=9,adjust=False).mean()
            df["MACD_H"]     = df["MACD"]-df["MACD_Sig"]
            df["MACD_H_Med"] = df["MACD_H"].rolling(50).quantile(0.6)  # 60th percentile：Grid Search 最優

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
                "rsi":    bool(52.0 < rsi14 <= 60.0 and rsi14 > prev_rsi),
                "bias":   bool(0.04 <= bias <= 0.08),
            }
            sell_flags = {
                "is_trend_broken":       bool(c_close < ma20),
                "is_momentum_lost":      bool(rsi14 < 50.0),
                "is_heavy_distribution": bool(c_close < c_open and c_vol > vol_ma20),
            }
            # MACD_H 強動能：當前值須高於自身 50 日中位數
            _mh     = last["MACD_H"]
            _mh_med = last["MACD_H_Med"]
            macd_h_strong = (not pd.isna(_mh) and not pd.isna(_mh_med)
                             and float(_mh) > float(_mh_med))
            # ADX 20-30：有趨勢但未過熱（ADX>30 回測勝率僅 33%）
            adx_ok = (adx14v is not None and 18 <= adx14v <= 35)
            is_buy = all(conds.values()) and macd_h_strong and adx_ok

            # ── v6: Confirmed signal (buy yesterday too) ──────────────────────
            confirmed_signal = prev_signals.get(ticker, False)

            # ── v6: Fundamentals (EPS, revenue, earnings) ─────────────────────
            fundamentals = _get_fundamentals(stock)  # earnings date from yfinance
            fm_est: dict = {}
            # Override EPS & revenue with FinMind for TW stocks (more accurate)
            if request.market == "tw":
                try:
                    _info = _get_info_cached(stock)
                    _shares = int(_info.get("sharesOutstanding", 0) or 0)
                    # yfinance often omits sharesOutstanding for TW stocks
                    # fall back to marketCap / close price (both in TWD for .TW)
                    if _shares == 0:
                        _mktcap = int(_info.get("marketCap", 0) or 0)
                        _price  = float(c_close) if c_close else 0
                        if _mktcap > 0 and _price > 0:
                            _shares = int(_mktcap / _price)
                    # last resort: fast_info.shares
                    if _shares == 0:
                        try:
                            _shares = int(stock.fast_info.shares or 0)
                        except Exception:
                            pass
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

# ── FinMind Historical Backtest ───────────────────────────────────────────────

def _compute_bt_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Vectorized indicator computation for backtesting (full DataFrame at once)."""
    d = df.copy()
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
    d["is_extended"]= d["ret5d"] > 8

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

        # Exit: hold_days trading-day rows after entry
        future = df[df.index > idx].head(hold_days)
        if len(future) < hold_days:
            continue

        entry  = float(row["Close"])
        exit_p = float(future.iloc[-1]["Close"])
        if entry <= 0:
            continue

        ret_pct = (exit_p - entry) / entry * 100
        max_dd  = float((future["Low"].min() - entry) / entry * 100)

        signals.append({
            "date":        row["date"].strftime("%Y-%m-%d"),
            "entry_price": round(entry,   2),
            "exit_price":  round(exit_p,  2),
            "return_pct":  round(ret_pct, 2),
            "max_dd":      round(max_dd,  2),
            "won":         ret_pct > 0,
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
    """
    Full historical backtest using FinMind TaiwanStockPrice.
    Replays indicator logic on every past trading day and measures forward returns.
    Returns per-ticker stats + aggregate summary.
    """
    if not FINMIND_TOKEN:
        raise HTTPException(503, "FINMIND_TOKEN not configured")

    now   = datetime.now()
    start = request.start_date or (now - timedelta(days=730)).strftime("%Y-%m-%d")
    end   = request.end_date   or now.strftime("%Y-%m-%d")
    hold  = max(1, min(int(request.hold_days), 60))

    results = []
    for ticker in request.tickers[:60]:
        code = ticker.split(".")[0] if request.market == "tw" else ticker
        name = get_company_name(ticker)
        res  = _backtest_ticker(code, start, end, hold, company_name=name, market=request.market)
        results.append(res)
        print(f"BT {code}: signals={res.get('total_signals')} "
              f"WR={res.get('win_rate')} avg={res.get('avg_return')}")

    all_signals = [s for r in results for s in r.get("signals", [])]
    all_rets    = [s["return_pct"] for s in all_signals]
    total_sig   = sum(r.get("total_signals", 0) for r in results)
    avg_all     = float(np.mean(all_rets))   if all_rets else None
    std_all     = float(np.std(all_rets))    if len(all_rets) > 1 else None
    sharpe_all  = round(avg_all / std_all * (252 / hold) ** 0.5, 2) \
                  if avg_all is not None and std_all and std_all > 0 else None

    summary = {
        "total_signals": total_sig,
        "win_rate":   round(sum(1 for x in all_rets if x > 0) / len(all_rets), 3) if all_rets else None,
        "avg_return": round(avg_all, 2) if avg_all is not None else None,
        "sharpe":     sharpe_all,
        "start_date": start, "end_date": end, "hold_days": hold,
    }

    return {
        "results":            results,
        "summary":            summary,
        "condition_analysis": _analyze_signals(all_signals),
    }


@app.post("/api/backtest/gridsearch")
async def backtest_gridsearch(request: BacktestFullRequest):
    """
    Grid search over RSI / ADX / MACD_H-percentile parameters.
    Step 1 — collect ALL candidate signals (base conditions only) across all tickers.
    Step 2 — filter in-memory for each parameter combo and compute Sharpe/win_rate.
    Returns top 20 combos sorted by Sharpe, plus current-params rank.
    """
    if not FINMIND_TOKEN:
        raise HTTPException(503, "FINMIND_TOKEN not configured")

    now   = datetime.now()
    start = request.start_date or (now - timedelta(days=730)).strftime("%Y-%m-%d")
    end   = request.end_date   or now.strftime("%Y-%m-%d")
    hold  = max(1, min(int(request.hold_days), 60))

    # ── Step 1: collect candidate signals ──────────────────────────────────
    all_signals: list = []
    for ticker in request.tickers[:60]:
        code = ticker.split(".")[0] if request.market == "tw" else ticker
        name = get_company_name(ticker)
        sigs = _collect_wide_signals(code, start, end, hold, company_name=name, market=request.market)
        all_signals.extend(sigs)
        print(f"GS {code}: {len(sigs)} candidates")

    if len(all_signals) < 8:
        return {"error": "Not enough candidate signals", "total_candidates": len(all_signals)}

    # ── Step 2: build parameter grid (market-specific ranges) ──────────────
    is_tw = request.market == "tw"
    grid = []
    rsi_los  = [50, 52, 54]       if is_tw else [45, 50, 55, 60]
    rsi_his  = [58, 60, 62]       if is_tw else [65, 70, 75, 80]
    adx_los  = [18, 20, 22, 24]   if is_tw else [10, 15, 18, 20]
    adx_his  = [28, 30, 35]       if is_tw else [30, 35, 40, 45]
    mh_pcts  = [50, 60, 66]       if is_tw else [33, 40, 50, 60]
    bias_los = [0, 2, 4]          if is_tw else [0, 2, 4]
    bias_his = [8]                 if is_tw else [8, 12, 15]

    for rsi_lo in rsi_los:
        for rsi_hi in rsi_his:
            if rsi_lo >= rsi_hi:
                continue
            for adx_lo in adx_los:
                for adx_hi in adx_his:
                    if adx_lo >= adx_hi:
                        continue
                    for mh_pct in mh_pcts:
                        for bias_lo in bias_los:
                            for bias_hi in bias_his:
                                grid.append({
                                    "rsi_lo": rsi_lo, "rsi_hi": rsi_hi,
                                    "adx_lo": adx_lo, "adx_hi": adx_hi,
                                    "macd_h_pct_min": mh_pct,
                                    "bias_lo": bias_lo,
                                    "bias_hi": bias_hi,
                                })

    # ── Step 3: evaluate each combo (pure in-memory filter) ────────────────
    results = []
    for p in grid:
        sub = [s for s in all_signals
               if p["rsi_lo"] <= s["rsi14"] <= p["rsi_hi"]
               and p["adx_lo"] <= s["adx14"] <= p["adx_hi"]
               and s["macd_h_pct"] >= p["macd_h_pct_min"]
               and p["bias_lo"] <= s["bias"] <= p["bias_hi"]]

        if len(sub) < 5:
            continue

        rets  = [s["return_pct"] for s in sub]
        wins  = sum(1 for r in rets if r > 0)
        avg   = float(np.mean(rets))
        std   = float(np.std(rets)) if len(rets) > 1 else 0.0
        sharpe = round(avg / std * (252 / hold) ** 0.5, 2) if std > 0 else None

        results.append({
            "params":     p,
            "n_signals":  len(sub),
            "win_rate":   round(wins / len(sub), 3),
            "avg_return": round(avg, 2),
            "sharpe":     sharpe,
        })

    results.sort(key=lambda x: (x["sharpe"] or -99), reverse=True)

    # Current live params for this market
    live = _get_market_params(request.market)
    current = {
        "rsi_lo": live["rsi_lo"], "rsi_hi": live["rsi_hi"],
        "adx_lo": live["adx_lo"], "adx_hi": live["adx_hi"],
        "macd_h_pct_min": live["macd_h_pct_min"],
        "bias_lo": live["bias_lo"], "bias_hi": live.get("bias_hi", 8),
    }
    current_rank = next(
        (i + 1 for i, r in enumerate(results) if r["params"] == current), None
    )

    return {
        "market":           request.market,
        "total_candidates": len(all_signals),
        "combos_tested":    len(results),
        "current_rank":     current_rank,
        "current_params":   current,
        "top_results":      results[:20],
    }


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


# ══════════════════════════ SENTIMENT MODULE (v7.0) ══════════════════════════

import datetime as _dt

GROK_API_KEY = os.environ.get("GROK_API_KEY", "")
_SENTIMENT_CACHE: dict = {"date": None, "data": None}


def _fetch_news_playwright() -> Optional[str]:
    """Scrape Investing.com via Playwright. Returns None if env doesn't support it."""
    try:
        from playwright.sync_api import sync_playwright
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            page.goto("https://www.investing.com/news/latest-news",
                      wait_until="networkidle", timeout=25000)
            articles = page.locator("article.news-analysis-v2_item__6vEFA").all()
            news_list = []
            for idx, article in enumerate(articles[:5]):
                try:
                    title = article.locator("[data-testid='article-title']").inner_text()
                    desc  = article.locator("[data-testid='article-description']").inner_text()
                    news_list.append(f"【新聞 {idx+1}】\n標題: {title}\n摘要: {desc}")
                except Exception:
                    continue
            browser.close()
            return "\n".join(news_list) if news_list else None
    except Exception as e:
        print(f"[Sentiment] Playwright unavailable: {e}")
        return None


def _fetch_news_yfinance() -> str:
    """Fallback: pull market news from yfinance (SPY/QQQ/NVDA/AAPL)."""
    items, seen = [], set()
    for sym in ["SPY", "QQQ", "NVDA", "AAPL", "MSFT"]:
        try:
            t = yf.Ticker(sym)
            news = t.news or []
            for n in news[:4]:
                # yfinance 0.2.x: title is under content.title; older: n.title
                content = n.get("content") or {}
                title = (content.get("title") or n.get("title") or "").strip()
                if not title or title in seen:
                    continue
                seen.add(title)
                summary = (content.get("summary") or content.get("description")
                           or n.get("summary") or n.get("description") or "")
                items.append(f"標題: {title}\n摘要: {str(summary)[:200]}")
        except Exception as e:
            print(f"[Sentiment] yfinance news {sym}: {e}")
    if not items:
        # Last resort: give Grok today's date for a neutral assessment
        today = _dt.date.today().isoformat()
        return f"今天是 {today}，請根據你的最新知識對當前全球金融市場情緒做出評估。"
    return "\n\n".join(items[:8])


def _analyze_with_grok(news_text: str) -> dict:
    """Send news to Grok (xAI) and return structured sentiment dict."""
    from openai import OpenAI
    client = OpenAI(api_key=GROK_API_KEY, base_url="https://api.x.ai/v1")
    system_prompt = (
        "你是一位頂尖的量化交易員。請評估以下最新財經新聞，為今日市場情緒打分。\n"
        "嚴格回傳標準 JSON，包含三個欄位：\n"
        "1. 'sentiment_score': 浮點數，範圍 -1.0（極度悲觀）到 +1.0（極度樂觀）。\n"
        "2. 'key_reason': 繁體中文，一句話說明核心原因。\n"
        "3. 'target_sectors': 受影響最大的產業板塊陣列，如 ['半導體', 'AI']。\n"
        "不要包含任何 markdown 語法，直接輸出純 JSON 字串。"
    )
    resp = client.chat.completions.create(
        model="grok-3-mini-beta",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": f"今日新聞：\n{news_text}"},
        ],
        temperature=0.0,
    )
    raw = resp.choices[0].message.content
    try:
        return json.loads(raw)
    except Exception:
        print(f"[Sentiment] Grok parse error, raw: {raw[:200]}")
        return {"sentiment_score": 0.0, "key_reason": "解析失敗，預設中立", "target_sectors": []}


def _get_or_refresh_sentiment(force: bool = False) -> dict:
    today = _dt.date.today().isoformat()
    if not force and _SENTIMENT_CACHE["date"] == today and _SENTIMENT_CACHE["data"]:
        return {**_SENTIMENT_CACHE["data"], "cached": True}

    news = _fetch_news_playwright()
    source = "investing.com"
    if not news:
        news = _fetch_news_yfinance()
        source = "yfinance"

    if not news:
        return {"sentiment_score": 0.0, "key_reason": "無法獲取新聞", "target_sectors": [],
                "source": "none", "cached": False}

    result = _analyze_with_grok(news)
    result["fetched_at"] = _dt.datetime.now().isoformat()
    result["source"]     = source
    result["cached"]     = False
    _SENTIMENT_CACHE["date"] = today
    _SENTIMENT_CACHE["data"] = result
    return result


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
        "name": "deploy_frontend",
        "description": "Trigger Vercel deployment for the frontend. Call this after writing frontend/index.html.",
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

def _gh_write(path: str, content: str, message: str) -> str:
    url = f"https://api.github.com/repos/{GITHUB_REPO}/contents/{path}"
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
1. Always read the relevant file BEFORE making any edit, so you have full context.
2. For frontend changes: read → edit → write_file → deploy_frontend. Safe to do directly.
3. For backend changes (api.py): describe the change and confirm with the user BEFORE writing. If the backend breaks, the chat breaks too.
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
    messages: list[dict]

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
                max_tokens=8192,
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
