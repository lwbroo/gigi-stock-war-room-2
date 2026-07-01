#!/usr/bin/env python3
"""
scan_local.py — Run the full stock scan locally and push results to GSheets.

All heavy computation (yfinance downloads, indicator calculation, signal detection)
runs on your Mac. Render cloud only reads the pre-computed results from GSheets.

Usage:
  python3 scan_local.py                    # scan TW watchlist
  python3 scan_local.py --market us        # scan US watchlist
  python3 scan_local.py --market both      # scan both markets
  python3 scan_local.py --notify           # also send Telegram alert
  python3 scan_local.py --dry-run          # print results, skip GSheets write
"""
import argparse
import json
import os
import sys
import time
from datetime import datetime, timedelta
from typing import Optional

import gspread
import numpy as np
import pandas as pd
import requests
import yfinance as yf
from dotenv import load_dotenv
from google.oauth2.service_account import Credentials

load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

# ── Config ────────────────────────────────────────────────────────────────────
FINMIND_TOKEN = os.environ.get("FINMIND_TOKEN", "")
FINMIND_BASE  = "https://api.finmindtrade.com/api/v4/data"
GOOGLE_CREDS  = os.environ.get("GOOGLE_CREDENTIALS_JSON", "")
_SHEET_NAME   = "gigi-war-room-watchlist"
_SHEET_TABS   = {"tw": "gigi-war-room-watchlist", "us": "gigi-us-watchlist"}
_TICKER_COL   = "ticker"
_SCOPES       = ["https://www.googleapis.com/auth/spreadsheets",
                 "https://www.googleapis.com/auth/drive"]

TG_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TG_CHAT_ID   = os.environ.get("TELEGRAM_CHAT_ID", "")

SCAN_CACHE_TAB = "scan_cache"
SCAN_LOG_TAB   = "scan_log"
SCAN_LOG_HEADERS = [
    "scan_date", "ticker", "close",
    "macd_cross", "adx14", "obv_trend", "monthly_trend",
    "is_breakout20", "vol_expansion", "inst_foreign", "inst_trust",
    "rs_score", "weekly_trend", "rsi14", "bias", "is_buy",
]
SCAN_CACHE_HEADERS = [
    "scanned_at", "market", "ticker", "company_name",
    "close", "open", "high", "low", "volume", "vol_ma20",
    "ma20", "ma60", "ma120",
    "rsi14", "bias", "adx14", "di_plus", "di_minus",
    "macd_cross", "macd_line", "macd_signal", "macd_hist",
    "obv_trend", "monthly_trend", "weekly_trend",
    "is_breakout20", "vol_expansion",
    "week52_high", "week52_low", "pct_from_52high",
    "rs_score", "max_drawdown_1y",
    "stop_loss", "target_price", "pattern",
    "inst_foreign", "inst_trust",
    "week5d_return", "is_extended",
    "eps_growth", "revenue_growth",
    "earnings_date", "near_earnings",
    "est_eps", "est_dividend", "est_rev_growth",
    "market_regime_bull", "market_week_return", "market_week_rising",
    "signal", "confirmed_signal",
    "conds_price", "conds_volume", "conds_trend", "conds_candle", "conds_rsi", "conds_bias",
    "sell_trend_broken", "sell_momentum_lost", "sell_heavy_dist",
    "xgb_prob",
]

# ── GSheets helpers ───────────────────────────────────────────────────────────

def _get_gc():
    raw = GOOGLE_CREDS
    if not raw and os.path.exists(os.path.join(os.path.dirname(__file__), "credentials.json")):
        with open(os.path.join(os.path.dirname(__file__), "credentials.json")) as f:
            raw = f.read()
    if not raw:
        sys.exit("ERROR: GOOGLE_CREDENTIALS_JSON not set")
    return gspread.authorize(Credentials.from_service_account_info(json.loads(raw), scopes=_SCOPES))


def _get_or_create_tab(sh, tab_name: str, headers: list):
    try:
        return sh.worksheet(tab_name)
    except gspread.exceptions.WorksheetNotFound:
        ws = sh.add_worksheet(title=tab_name, rows=5000, cols=len(headers))
        ws.update("A1", [headers])
        return ws


def _load_watchlist(sh, market: str) -> list:
    tab = _SHEET_TABS.get(market, _SHEET_TABS["tw"])
    try:
        ws = sh.worksheet(tab)
    except gspread.exceptions.WorksheetNotFound:
        return []
    vals = ws.col_values(1)
    return [t.strip() for t in vals if t.strip() and t.strip() != _TICKER_COL]


# ── Name map ──────────────────────────────────────────────────────────────────

def _load_tw_names() -> dict:
    result = {}
    bundle = os.path.join(os.path.dirname(__file__), "tw_names.json")
    try:
        with open(bundle, encoding="utf-8") as f:
            result = json.load(f)
    except Exception:
        pass
    for url, cf, nf in [
        ("https://openapi.twse.com.tw/v1/opendata/t187ap03_L", "公司代號", "公司簡稱"),
        ("https://www.tpex.org.tw/openapi/v1/mopsfin_t187ap03_O", "SecuritiesCompanyCode", "CompanyAbbreviation"),
    ]:
        try:
            r = requests.get(url, timeout=8, headers={"Accept": "application/json"})
            if r.ok:
                for item in r.json():
                    c, n = item.get(cf, "").strip(), item.get(nf, "").strip()
                    if c and n:
                        result[c] = n
        except Exception:
            pass
    return result


_TW_NAME_MAP: dict = {}


def get_company_name(ticker: str) -> str:
    return _TW_NAME_MAP.get(ticker.split(".")[0], ticker)


# ── Market index ──────────────────────────────────────────────────────────────

def _get_index_df(market: str) -> Optional[pd.DataFrame]:
    sym = "^TWII" if market == "tw" else "^GSPC"
    try:
        df = yf.Ticker(sym).history(period="1y")
        return df if not df.empty else None
    except Exception:
        return None


# ── Institutional data (TW only) ──────────────────────────────────────────────

def _load_inst_data() -> dict:
    for days_back in range(6):
        d = datetime.now() - timedelta(days=days_back)
        if d.weekday() >= 5:
            continue
        date_str = d.strftime("%Y%m%d")
        try:
            r = requests.get(
                f"https://www.twse.com.tw/rwd/zh/fund/T86?response=json&date={date_str}&selectType=ALL",
                timeout=12, headers={"Accept": "application/json", "User-Agent": "Mozilla/5.0"},
            )
            if not r.ok:
                continue
            payload = r.json()
            if payload.get("stat") != "OK" or not payload.get("data"):
                continue
            fields = payload.get("fields", [])
            def _fi(kws, ex=""):
                for i, f in enumerate(fields):
                    if all(k in f for k in kws) and (not ex or ex not in f):
                        return i
                return None
            fi_f = _fi(["外陸資買賣超"], "自營") or 4
            fi_t = _fi(["投信買賣超"]) or 10
            def _n(s):
                try:
                    return int(str(s).replace(",", "").replace("+", "").strip() or "0")
                except Exception:
                    return 0
            result = {}
            for row in payload["data"]:
                try:
                    code = str(row[0]).strip()
                    result[code] = {"foreign": _n(row[fi_f]), "trust": _n(row[fi_t])}
                except Exception:
                    pass
            if result:
                print(f"  Institutional data: {len(result)} stocks ({date_str})")
                return result
        except Exception as e:
            print(f"  TWSE inst ({date_str}): {e}")
    return {}


# ── Model params from GSheets ─────────────────────────────────────────────────

_MODEL_PARAMS_TAB = "model_params"

def _load_model_params(sh, market: str) -> dict:
    defaults = {"rsi_lo": 58, "rsi_hi": 62, "adx_lo": 22, "adx_hi": 32,
                "bias_lo": 4, "bias_hi": 7, "macd_h_pct_min": 70}
    try:
        ws = sh.worksheet(_MODEL_PARAMS_TAB)
        rows = ws.get_all_values()
        hdr = rows[0] if rows else []
        def _ci(col):
            return hdr.index(col) if col in hdr else None
        for r in rows[1:]:
            if r and r[0] == market:
                def _f(col):
                    i = _ci(col)
                    try:
                        return float(r[i]) if i is not None and r[i] else defaults.get(col)
                    except Exception:
                        return defaults.get(col)
                return {k: _f(k) for k in defaults}
    except Exception as e:
        print(f"  model_params load: {e}")
    return defaults


# ── FinMind fundamentals ──────────────────────────────────────────────────────

def _get_finmind_fundamentals(code: str, shares_actual: int = 0) -> dict:
    result = {"eps_growth": None, "revenue_growth": None,
              "est_eps": None, "est_dividend": None, "est_rev_growth": None}
    if not FINMIND_TOKEN or not code:
        return result
    now = datetime.now()
    this_year, last_year = now.year, now.year - 1
    start = (now - timedelta(days=450)).strftime("%Y-%m-%d")
    div_start = (now - timedelta(days=4 * 365)).strftime("%Y-%m-%d")
    try:
        r = requests.get(FINMIND_BASE, params={
            "dataset": "TaiwanStockMonthRevenue", "data_id": code,
            "start_date": start, "token": FINMIND_TOKEN,
        }, timeout=12)
        rev_rows = sorted(r.json().get("data", []) if r.ok else [],
                          key=lambda x: (int(x.get("revenue_year", 0)), int(x.get("revenue_month", 0))))
    except Exception:
        rev_rows = []
    cur_ytd = last_ytd = last_total = ttm_rev = 0.0
    if rev_rows:
        try:
            last_rec = rev_rows[-1]
            cur_m = int(last_rec.get("revenue_month", 0))
            same_m_ly = [x for x in rev_rows if int(x.get("revenue_year", 0)) == last_year
                         and int(x.get("revenue_month", 0)) == cur_m]
            if same_m_ly:
                lr = float(same_m_ly[0]["revenue"])
                if lr != 0:
                    result["revenue_growth"] = round((float(last_rec["revenue"]) - lr) / lr * 100, 1)
            cur_ytd  = sum(float(x["revenue"]) for x in rev_rows
                          if int(x.get("revenue_year", 0)) == this_year and int(x.get("revenue_month", 0)) <= cur_m)
            last_ytd = sum(float(x["revenue"]) for x in rev_rows
                          if int(x.get("revenue_year", 0)) == last_year and int(x.get("revenue_month", 0)) <= cur_m)
            last_total = sum(float(x["revenue"]) for x in rev_rows if int(x.get("revenue_year", 0)) == last_year)
            ttm_rev = sum(float(x["revenue"]) for x in rev_rows[-12:]) if len(rev_rows) >= 12 else 0.0
        except Exception:
            pass
    ttm_eps = 0.0
    annual_eps: dict = {}
    try:
        r = requests.get(FINMIND_BASE, params={
            "dataset": "TaiwanStockFinancialStatements", "data_id": code,
            "start_date": start, "token": FINMIND_TOKEN,
        }, timeout=12)
        if r.ok:
            eps_rows = sorted([d for d in r.json().get("data", []) if d.get("type") == "EPS"],
                              key=lambda x: x["date"])
            if eps_rows:
                if len(eps_rows) >= 5:
                    ne, ye = float(eps_rows[-1]["value"]), float(eps_rows[-5]["value"])
                    if ye != 0:
                        result["eps_growth"] = round((ne - ye) / abs(ye) * 100, 1)
                if len(eps_rows) >= 4:
                    ttm_eps = sum(float(x["value"]) for x in eps_rows[-4:])
                for rec in eps_rows:
                    yr = rec["date"][:4]
                    annual_eps[yr] = annual_eps.get(yr, 0.0) + float(rec["value"])
    except Exception:
        pass
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
                item = str(d.get("dividend_item", "") or "")
                cash = 0.0
                if "現金" in item:
                    try:
                        cash = float(d.get("dividend", 0) or 0)
                    except Exception:
                        pass
                if cash == 0:
                    for field in ("CashDividend", "cash_dividend", "CashEarningsDistribution"):
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
    except Exception:
        pass
    if not payout_rates:
        payout_rates = [0.50]
    try:
        if cur_ytd > 0 and last_ytd > 0 and last_total > 0 and ttm_rev > 0 and ttm_eps != 0 and shares_actual > 0:
            growth_yoy = (cur_ytd - last_ytd) / last_ytd
            est_revenue = last_total * (1 + growth_yoy)
            ttm_rate = (ttm_eps * shares_actual / 1000) / ttm_rev
            est_net_income = est_revenue * ttm_rate
            est_eps = (est_net_income * 1000) / shares_actual
            avg_payout = sum(payout_rates) / len(payout_rates)
            result["est_eps"] = round(est_eps, 2)
            result["est_dividend"] = round(est_eps * avg_payout, 2)
            result["est_rev_growth"] = round(growth_yoy * 100, 2)
    except Exception:
        pass
    return result


# ── Fundamentals (earnings date, EPS) ────────────────────────────────────────

def _get_fundamentals(stock: yf.Ticker) -> dict:
    result = {"eps_growth": None, "revenue_growth": None, "earnings_date": None, "near_earnings": False}
    try:
        info = stock.info
        eg = info.get("earningsQuarterlyGrowth")
        if eg is not None:
            result["eps_growth"] = round(float(eg) * 100, 1)
        rg = info.get("revenueGrowth")
        if rg is not None:
            result["revenue_growth"] = round(float(rg) * 100, 1)
        ets = info.get("earningsTimestamp")
        if ets:
            ed = datetime.fromtimestamp(int(ets))
            result["earnings_date"] = ed.strftime("%Y-%m-%d")
            result["near_earnings"] = -1 <= (ed - datetime.now()).days <= 7
    except Exception:
        pass
    if not result["earnings_date"]:
        try:
            cal = stock.calendar
            if cal is not None:
                dates = cal.get("Earnings Date", []) if isinstance(cal, dict) else list(cal.columns)
                if dates:
                    first = dates[0]
                    ds = first.strftime("%Y-%m-%d") if hasattr(first, "strftime") else str(first)[:10]
                    result["earnings_date"] = ds
                    try:
                        result["near_earnings"] = -1 <= (datetime.strptime(ds, "%Y-%m-%d") - datetime.now()).days <= 7
                    except Exception:
                        pass
        except Exception:
            pass
    return result


# ── Pattern detection ─────────────────────────────────────────────────────────

def _detect_pattern(df: pd.DataFrame) -> str:
    if len(df) < 3:
        return ""
    o, c, h, l = df["Open"].values, df["Close"].values, df["High"].values, df["Low"].values
    body = lambda i: abs(c[i] - o[i])
    uw   = lambda i: h[i] - max(c[i], o[i])
    lw   = lambda i: min(c[i], o[i]) - l[i]
    rng  = lambda i: h[i] - l[i]
    bull = lambda i: c[i] > o[i]
    bear = lambda i: c[i] < o[i]
    if bull(-3) and bull(-2) and bull(-1) and c[-2] > c[-3] and c[-1] > c[-2] and o[-2] > o[-3] and o[-1] > o[-2]:
        return "紅三兵"
    if bear(-3) and bear(-2) and bear(-1) and c[-2] < c[-3] and c[-1] < c[-2]:
        return "黑三兵"
    if bear(-2) and bull(-1) and o[-1] <= c[-2] and c[-1] >= o[-2]:
        return "多頭吞噬"
    if bull(-2) and bear(-1) and o[-1] >= c[-2] and c[-1] <= o[-2]:
        return "空頭吞噬"
    if rng(-1) > 0 and body(-1) > 0 and lw(-1) >= 2 * body(-1) and uw(-1) <= body(-1) * 0.3:
        return "錘子線"
    if rng(-1) > 0 and body(-1) > 0 and uw(-1) >= 2 * body(-1) and lw(-1) <= body(-1) * 0.3:
        return "射擊之星"
    if rng(-1) > 0 and body(-1) <= rng(-1) * 0.1:
        return "十字星"
    return ""


# ── Previous signals (for 2-day confirmation) ─────────────────────────────────

def _load_prev_signals(sh) -> dict:
    today = datetime.now().strftime("%Y-%m-%d")
    try:
        ws = _get_or_create_tab(sh, SCAN_LOG_TAB, SCAN_LOG_HEADERS)
        records = ws.get_all_records()
        cutoff = (datetime.now() - timedelta(days=4)).strftime("%Y-%m-%d")
        return {
            r["ticker"]: True for r in records
            if str(r.get("scan_date", "")) >= cutoff
            and str(r.get("scan_date", "")) < today
            and str(r.get("is_buy", "")).lower() in ("true", "1", "yes")
            and r.get("ticker")
        }
    except Exception:
        return {}


# ── Telegram ──────────────────────────────────────────────────────────────────

def send_telegram(msg: str, bot_token: str, chat_id: str):
    if not bot_token or not chat_id:
        return
    try:
        requests.post(
            f"https://api.telegram.org/bot{bot_token}/sendMessage",
            json={"chat_id": chat_id, "text": msg, "parse_mode": "HTML"},
            timeout=10,
        )
    except Exception as e:
        print(f"  Telegram: {e}")


# ── Core scan logic ───────────────────────────────────────────────────────────

def scan_ticker(ticker: str, market: str, params: dict, inst_data: dict,
                index_df: Optional[pd.DataFrame], prev_signals: dict) -> dict:
    """Compute all indicators and signal for one ticker. Returns result dict."""

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

    def _no_data(reason="NO_DATA"):
        return {
            "ticker": ticker, "company_name": get_company_name(ticker),
            "signal": reason, "close": None,
            "market_regime_bull": market_regime_bull,
            "market_week_return": market_week_return,
            "market_week_rising": market_week_rising,
        }

    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period="1y")
        company_name = get_company_name(ticker)
        if company_name == ticker:
            try:
                info = stock.info
                company_name = info.get("shortName") or info.get("longName") or ticker
            except Exception:
                pass

        if df.empty or len(df) < 136:
            return _no_data()

        # ── Indicators ────────────────────────────────────────────────────────
        df["MA20"]  = df["Close"].rolling(20).mean()
        df["MA60"]  = df["Close"].rolling(60).mean()
        df["MA120"] = df["Close"].rolling(120).mean()
        df["VMA20"] = df["Volume"].rolling(20).mean()

        delta = df["Close"].diff()
        ag = delta.clip(lower=0).ewm(alpha=1/14, min_periods=14, adjust=False).mean()
        al = (-delta).clip(lower=0).ewm(alpha=1/14, min_periods=14, adjust=False).mean()
        sl = al.copy(); sl[sl == 0] = 1e-10
        df["RSI14"] = 100.0 - 100.0 / (1.0 + ag / sl)
        df["Bias"]  = (df["Close"] - df["MA20"]) / df["MA20"]

        e12 = df["Close"].ewm(span=12, adjust=False).mean()
        e26 = df["Close"].ewm(span=26, adjust=False).mean()
        df["MACD"]     = e12 - e26
        df["MACD_Sig"] = df["MACD"].ewm(span=9, adjust=False).mean()
        df["MACD_H"]   = df["MACD"] - df["MACD_Sig"]
        macd_h_pct = params.get("macd_h_pct_min", 70) / 100
        df["MACD_H_Med"] = df["MACD_H"].rolling(50).quantile(macd_h_pct)

        tr = pd.concat([df["High"] - df["Low"],
                        (df["High"] - df["Close"].shift(1)).abs(),
                        (df["Low"] - df["Close"].shift(1)).abs()], axis=1).max(axis=1)
        hd = df["High"] - df["High"].shift(1)
        ld = df["Low"].shift(1) - df["Low"]
        df["DM+"] = np.where((hd > ld) & (hd > 0), hd, 0.0)
        df["DM-"] = np.where((ld > hd) & (ld > 0), ld, 0.0)
        atr = tr.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
        dip = 100 * df["DM+"].ewm(alpha=1/14, min_periods=14, adjust=False).mean() / atr.replace(0, np.nan)
        dim = 100 * df["DM-"].ewm(alpha=1/14, min_periods=14, adjust=False).mean() / atr.replace(0, np.nan)
        df["ADX"] = (100 * (dip - dim).abs() / (dip + dim).replace(0, np.nan)).ewm(alpha=1/14, min_periods=14, adjust=False).mean()
        df["DI+"] = dip; df["DI-"] = dim

        df["OBV"]    = (np.sign(df["Close"].diff()) * df["Volume"]).cumsum()
        df["OBV_MA"] = df["OBV"].rolling(20).mean()

        # ── Extract latest ────────────────────────────────────────────────────
        last = df.iloc[-1]; prev = df.iloc[-2]
        c_close, c_open, c_high, c_low, c_vol = last["Close"], last["Open"], last["High"], last["Low"], last["Volume"]
        ma20, ma60, ma120, vol_ma20 = last["MA20"], last["MA60"], last["MA120"], last["VMA20"]
        rsi14, bias, prev_rsi = last["RSI14"], last["Bias"], df["RSI14"].iloc[-2]

        if any(pd.isna(v) for v in [ma20, ma60, ma120, vol_ma20, rsi14, bias, prev_rsi]):
            return _no_data()

        # 52-week / drawdown
        w52h = float(df["High"].max())
        w52l = float(df["Low"].min())
        md1y = float(((df["Close"] - df["Close"].cummax()) / df["Close"].cummax()).min())

        # Weekly trend
        wk = df["Close"].resample("W").last().dropna()
        wma5, wma10, wma20w = wk.rolling(5).mean(), wk.rolling(10).mean(), wk.rolling(20).mean()
        weekly_trend = None
        if len(wk) >= 20 and not any(pd.isna([wma5.iloc[-1], wma10.iloc[-1], wma20w.iloc[-1]])):
            weekly_trend = bool(wma5.iloc[-1] > wma10.iloc[-1] > wma20w.iloc[-1])

        # Monthly trend
        mn = df["Close"].resample("MS").last().dropna()
        mm5, mm10 = mn.rolling(5).mean(), mn.rolling(10).mean()
        monthly_trend = None
        if len(mn) >= 10 and not any(pd.isna([mm5.iloc[-1], mm10.iloc[-1]])):
            monthly_trend = bool(mm5.iloc[-1] > mm10.iloc[-1])

        # MACD cross
        mn_m, pr_m = last["MACD"], prev["MACD"]
        mn_s, pr_s = last["MACD_Sig"], prev["MACD_Sig"]
        macd_cross = "none"
        if not any(pd.isna([mn_m, pr_m, mn_s, pr_s])):
            if pr_m < pr_s and mn_m > mn_s:   macd_cross = "golden"
            elif pr_m > pr_s and mn_m < mn_s: macd_cross = "death"
            elif mn_m > mn_s:                 macd_cross = "above"
            else:                             macd_cross = "below"

        adx14v = None if pd.isna(last["ADX"]) else round(float(last["ADX"]), 1)
        dip_v  = None if pd.isna(last["DI+"]) else round(float(last["DI+"]), 1)
        dim_v  = None if pd.isna(last["DI-"]) else round(float(last["DI-"]), 1)

        obv_trend = None
        if not pd.isna(last["OBV"]) and not pd.isna(last["OBV_MA"]):
            obv_trend = "rising" if last["OBV"] > last["OBV_MA"] else "falling"

        is_breakout20 = False
        if len(df) >= 21:
            is_breakout20 = bool(float(c_close) > float(df["High"].iloc[-21:-1].max()))
        vol_expansion = False
        if len(df) >= 3:
            v1, v2, v3 = df["Volume"].iloc[-3], df["Volume"].iloc[-2], df["Volume"].iloc[-1]
            vol_expansion = bool(v3 > v2 > v1 and v3 > vol_ma20)

        rs_score = None
        if len(df) >= 21 and index_20d_return:
            s20 = float((c_close - df["Close"].iloc[-21]) / df["Close"].iloc[-21])
            if index_20d_return != 0:
                rs_score = round(s20 / abs(index_20d_return), 2)

        inst = inst_data.get(ticker.split(".")[0], {})
        inst_foreign = inst.get("foreign") if inst else None
        inst_trust   = inst.get("trust")   if inst else None

        week5d_return = None
        is_extended = False
        if len(df) >= 6:
            p5 = float(df["Close"].iloc[-6])
            week5d_return = round((float(c_close) - p5) / p5 * 100, 2)
            is_extended = week5d_return > 5.0

        # ── Buy conditions (using optimized params) ───────────────────────────
        rsi_lo  = params.get("rsi_lo", 58)
        rsi_hi  = params.get("rsi_hi", 62)
        adx_lo  = params.get("adx_lo", 22)
        adx_hi  = params.get("adx_hi", 32)
        bias_lo = params.get("bias_lo", 4) / 100
        bias_hi = params.get("bias_hi", 7) / 100

        mid = (c_high + c_low) / 2.0
        conds = {
            "price":  bool(c_close > ma20),
            "volume": bool(c_vol > 1.2 * vol_ma20),
            "trend":  bool(ma20 > ma60 and ma60 > ma120),
            "candle": bool(c_close > c_open and c_close > mid),
            "rsi":    bool(rsi_lo < rsi14 <= rsi_hi and rsi14 > prev_rsi),
            "bias":   bool(bias_lo <= bias <= bias_hi),
        }
        sell_flags = {
            "is_trend_broken":       bool(c_close < ma20),
            "is_momentum_lost":      bool(rsi14 < 50.0),
            "is_heavy_distribution": bool(c_close < c_open and c_vol > vol_ma20),
        }

        _mh     = last["MACD_H"]
        _mh_med = last["MACD_H_Med"]
        macd_h_strong = (not pd.isna(_mh) and not pd.isna(_mh_med) and float(_mh) > float(_mh_med))
        adx_ok  = (adx14v is not None and adx_lo <= adx14v <= adx_hi)
        is_buy  = all(conds.values()) and macd_h_strong and adx_ok and not is_extended

        confirmed_signal = prev_signals.get(ticker, False)

        # ── Fundamentals ──────────────────────────────────────────────────────
        fundamentals = _get_fundamentals(stock)
        fm_est: dict = {}
        if market == "tw":
            try:
                info = stock.info
                shares = int(info.get("sharesOutstanding", 0) or 0)
                if shares == 0:
                    mktcap = int(info.get("marketCap", 0) or 0)
                    price = float(c_close) if c_close else 0
                    if mktcap > 0 and price > 0:
                        shares = int(mktcap / price)
                if shares == 0:
                    try:
                        shares = int(stock.fast_info.shares or 0)
                    except Exception:
                        pass
            except Exception:
                shares = 0
            fm = _get_finmind_fundamentals(ticker.split(".")[0], shares_actual=shares)
            if fm["eps_growth"] is not None:
                fundamentals["eps_growth"] = fm["eps_growth"]
            if fm["revenue_growth"] is not None:
                fundamentals["revenue_growth"] = fm["revenue_growth"]
            fm_est = {"est_eps": fm.get("est_eps"), "est_dividend": fm.get("est_dividend"),
                      "est_rev_growth": fm.get("est_rev_growth")}

        pattern    = _detect_pattern(df.tail(5))
        stop_loss  = round(float(ma20) * 0.97, 2)
        target     = round(float(c_close) * 1.15, 2)
        pct_from52 = round((float(c_close) - w52h) / w52h * 100, 1)

        return {
            "ticker": ticker, "company_name": company_name,
            "close": round(float(c_close), 2), "open": round(float(c_open), 2),
            "high":  round(float(c_high), 2),  "low": round(float(c_low), 2),
            "volume": int(c_vol), "vol_ma20": int(vol_ma20),
            "ma20": round(float(ma20), 2), "ma60": round(float(ma60), 2), "ma120": round(float(ma120), 2),
            "rsi14": round(float(rsi14), 1), "bias": round(float(bias) * 100, 2),
            "adx14": adx14v, "di_plus": dip_v, "di_minus": dim_v,
            "macd_cross":  macd_cross,
            "macd_line":   round(float(mn_m), 4) if not pd.isna(mn_m) else None,
            "macd_signal": round(float(mn_s), 4) if not pd.isna(mn_s) else None,
            "macd_hist":   round(float(last["MACD_H"]), 4) if not pd.isna(last["MACD_H"]) else None,
            "obv_trend": obv_trend, "monthly_trend": monthly_trend, "weekly_trend": weekly_trend,
            "is_breakout20": is_breakout20, "vol_expansion": vol_expansion,
            "week52_high": round(w52h, 2), "week52_low": round(w52l, 2),
            "pct_from_52high": pct_from52,
            "rs_score": rs_score, "max_drawdown_1y": round(md1y * 100, 1),
            "stop_loss": stop_loss, "target_price": target, "pattern": pattern,
            "inst_foreign": inst_foreign, "inst_trust": inst_trust,
            "week5d_return": week5d_return, "is_extended": is_extended,
            "eps_growth":    fundamentals["eps_growth"],
            "revenue_growth": fundamentals["revenue_growth"],
            "earnings_date": fundamentals["earnings_date"],
            "near_earnings": fundamentals["near_earnings"],
            "est_eps":       fm_est.get("est_eps"),
            "est_dividend":  fm_est.get("est_dividend"),
            "est_rev_growth":fm_est.get("est_rev_growth"),
            "market_regime_bull": market_regime_bull,
            "market_week_return": market_week_return,
            "market_week_rising": market_week_rising,
            "signal": "YES" if is_buy else "NO",
            "confirmed_signal": confirmed_signal,
            "conds": conds, "sell_flags": sell_flags,
            "xgb_prob": None,
        }

    except Exception as e:
        print(f"  ERROR {ticker}: {e}")
        r = _no_data("ERROR")
        r["_error"] = str(e)
        return r


# ── Save results to GSheets scan_cache ────────────────────────────────────────

def _save_scan_cache(sh, results: list, market: str):
    ws = _get_or_create_tab(sh, SCAN_CACHE_TAB, SCAN_CACHE_HEADERS)
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M")
    rows = []
    for r in results:
        if r.get("signal") in ("NO_DATA", "ERROR") or r.get("close") is None:
            continue
        conds = r.get("conds", {})
        sf    = r.get("sell_flags", {})
        rows.append([
            now_str, market, r["ticker"], r.get("company_name", ""),
            r.get("close", ""), r.get("open", ""), r.get("high", ""), r.get("low", ""),
            r.get("volume", ""), r.get("vol_ma20", ""),
            r.get("ma20", ""), r.get("ma60", ""), r.get("ma120", ""),
            r.get("rsi14", ""), r.get("bias", ""),
            r.get("adx14", ""), r.get("di_plus", ""), r.get("di_minus", ""),
            r.get("macd_cross", ""), r.get("macd_line", ""), r.get("macd_signal", ""), r.get("macd_hist", ""),
            r.get("obv_trend", ""), str(r.get("monthly_trend", "")), str(r.get("weekly_trend", "")),
            str(r.get("is_breakout20", "")), str(r.get("vol_expansion", "")),
            r.get("week52_high", ""), r.get("week52_low", ""), r.get("pct_from_52high", ""),
            r.get("rs_score", ""), r.get("max_drawdown_1y", ""),
            r.get("stop_loss", ""), r.get("target_price", ""), r.get("pattern", ""),
            r.get("inst_foreign", ""), r.get("inst_trust", ""),
            r.get("week5d_return", ""), str(r.get("is_extended", "")),
            r.get("eps_growth", ""), r.get("revenue_growth", ""),
            r.get("earnings_date", ""), str(r.get("near_earnings", "")),
            r.get("est_eps", ""), r.get("est_dividend", ""), r.get("est_rev_growth", ""),
            str(r.get("market_regime_bull", "")), r.get("market_week_return", ""),
            str(r.get("market_week_rising", "")),
            r.get("signal", ""), str(r.get("confirmed_signal", "")),
            str(conds.get("price", "")), str(conds.get("volume", "")),
            str(conds.get("trend", "")), str(conds.get("candle", "")),
            str(conds.get("rsi", "")), str(conds.get("bias", "")),
            str(sf.get("is_trend_broken", "")), str(sf.get("is_momentum_lost", "")),
            str(sf.get("is_heavy_distribution", "")),
            r.get("xgb_prob", ""),
        ])

    # Clear old rows for this market, keep header + other market rows
    all_rows = ws.get_all_values()
    header = all_rows[0] if all_rows else SCAN_CACHE_HEADERS
    keep = [r for r in all_rows[1:] if r and len(r) > 1 and r[1] != market]
    ws.clear()
    ws.update("A1", [header] + keep + rows)
    print(f"  ↑ scan_cache: {len(rows)} rows written [{market}]")
    # Notify cloud to clear in-memory cache so next fetch reads fresh data
    try:
        r = requests.post(
            f"https://gigi-stock-war-room-2.onrender.com/api/scan/cache/reload?market={market}",
            timeout=10,
        )
        print(f"  ↑ Cloud cache reloaded: HTTP {r.status_code}")
    except Exception:
        pass  # non-critical


def _save_scan_log(sh, results: list, market: str):
    if market != "tw":
        return
    ws = _get_or_create_tab(sh, SCAN_LOG_TAB, SCAN_LOG_HEADERS)
    today = datetime.now().strftime("%Y-%m-%d")
    rows = [
        [today, r["ticker"], r.get("close", ""),
         r.get("macd_cross", ""), r.get("adx14", ""), r.get("obv_trend", ""),
         str(r.get("monthly_trend", "")), str(r.get("is_breakout20", "")),
         str(r.get("vol_expansion", "")), r.get("inst_foreign", ""),
         r.get("inst_trust", ""), r.get("rs_score", ""),
         str(r.get("weekly_trend", "")), r.get("rsi14", ""), r.get("bias", ""),
         str(r.get("signal", "") == "YES")]
        for r in results
        if r.get("signal") not in ("NO_DATA", "ERROR") and r.get("close") is not None
    ]
    if rows:
        ws.append_rows(rows, value_input_option="RAW")
        print(f"  ↑ scan_log: {len(rows)} rows appended")


# ── Main ──────────────────────────────────────────────────────────────────────

def run_scan(market: str, notify: bool, dry_run: bool):
    print(f"\n{'='*60}")
    print(f"  Local Scan — {market.upper()} — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"{'='*60}\n")

    global _TW_NAME_MAP
    _TW_NAME_MAP = _load_tw_names()

    gc = _get_gc()
    sh = gc.open(_SHEET_NAME)

    tickers = _load_watchlist(sh, market)
    if not tickers:
        print(f"  No tickers in {market} watchlist — aborting")
        return

    print(f"[1] Watchlist: {len(tickers)} tickers")
    params = _load_model_params(sh, market)
    print(f"[2] Params: RSI {params['rsi_lo']:.0f}-{params['rsi_hi']:.0f} / ADX {params['adx_lo']:.0f}-{params['adx_hi']:.0f} / Bias {params['bias_lo']:.0f}-{params['bias_hi']:.0f}% / MACD_H {params['macd_h_pct_min']:.0f}th pct")

    print("[3] Fetching market index...")
    index_df = _get_index_df(market)

    print("[4] Loading institutional data...")
    inst_data = _load_inst_data() if market == "tw" else {}

    print("[5] Loading previous signals...")
    prev_signals = _load_prev_signals(sh)

    print(f"[6] Scanning {len(tickers)} tickers...\n")
    results = []
    buy_rows, sell_rows, warn_rows = [], [], []
    t0 = time.time()

    for i, ticker in enumerate(tickers, 1):
        r = scan_ticker(ticker, market, params, inst_data, index_df, prev_signals)
        results.append(r)

        sig = r.get("signal", "?")
        name = r.get("company_name", "")[:10]
        close = r.get("close", "—")
        rsi = r.get("rsi14", "—")
        label = "🟢 BUY" if sig == "YES" else ("❌" if sig in ("NO_DATA", "ERROR") else "  ")
        print(f"  [{i:>2}/{len(tickers)}] {ticker:<12} {name:<12} close={close}  RSI={rsi}  {label}")

        if sig == "YES":
            buy_rows.append(r)
        sf = r.get("sell_flags", {})
        sell_count = sum(1 for v in sf.values() if v)
        if sell_count >= 2:
            sell_rows.append(r)
        elif sell_count == 1 and sig != "YES":
            warn_rows.append(r)

    elapsed = time.time() - t0
    print(f"\n  Done in {elapsed:.0f}s — 🟢 {len(buy_rows)} buy  ⚠️ {len(warn_rows)} watch  🔴 {len(sell_rows)} sell\n")

    if dry_run:
        print("  [dry-run] Skipping GSheets write.")
        if buy_rows:
            print("\n  🟢 Buy signals:")
            for r in buy_rows:
                print(f"    {r['ticker']} {r.get('company_name','')} close={r.get('close')} RSI={r.get('rsi14')}")
        return

    print("[7] Saving to GSheets...")
    _save_scan_cache(sh, results, market)
    _save_scan_log(sh, results, market)

    if notify and (buy_rows or sell_rows):
        flag = "🇹🇼" if market == "tw" else "🇺🇸"
        mkt_name = "台股" if market == "tw" else "美股"
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M")
        lines = [
            f"\n{flag} {mkt_name} 本地掃描 {now_str}",
            f"📡 掃描 {len(tickers)} 支 | 🟢買進 {len(buy_rows)} | ⚠️警示 {len(warn_rows)} | 🔴出場 {len(sell_rows)}",
        ]
        if buy_rows:
            lines.append("\n🚀 買進訊號：")
            for r in buy_rows[:8]:
                conf  = " ✅確認" if r.get("confirmed_signal") else ""
                earn  = " ⚠️財報" if r.get("near_earnings") else ""
                lines.append(f"• {r['ticker']} {r.get('company_name','')}{conf}{earn}")
        if sell_rows:
            lines.append("\n🔴 出場訊號：")
            for r in sell_rows[:5]:
                flags = [k.replace("is_", "").replace("_", " ") for k, v in r.get("sell_flags", {}).items() if v]
                lines.append(f"• {r['ticker']} {r.get('company_name','')} — {', '.join(flags[:2])}")
        lines.append(f"\n🔗 https://gigi-frontend-mu.vercel.app")
        send_telegram("\n".join(lines), TG_BOT_TOKEN, TG_CHAT_ID)
        print("  ↑ Telegram sent")

    print("\nAll done.")


def main():
    ap = argparse.ArgumentParser(description="Local stock scanner — pushes results to GSheets")
    ap.add_argument("--market",   default="tw", choices=["tw", "us", "both"])
    ap.add_argument("--notify",   action="store_true", help="Send Telegram alert")
    ap.add_argument("--dry-run",  action="store_true", help="Print only, no GSheets write")
    args = ap.parse_args()

    markets = ["tw", "us"] if args.market == "both" else [args.market]
    for m in markets:
        run_scan(m, args.notify, args.dry_run)


if __name__ == "__main__":
    main()
