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

_BUNDLE_PATH = os.path.join(os.path.dirname(__file__), "tw_names.json")

# ── Constants ─────────────────────────────────────────────────────────────────
_SHEET_NAME     = "gigi-war-room-watchlist"
_SHEET_TABS     = {"tw": "gigi-war-room-watchlist", "us": "gigi-us-watchlist"}
_TICKER_COL     = "ticker"
_SHEETS_SCOPES  = [
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
    "rs_score", "weekly_trend", "rsi14", "bias",
]

# ── Caches ────────────────────────────────────────────────────────────────────
_INDEX_CACHE: dict = {}
_INST_CACHE:  dict = {}


def _load_tw_names() -> dict:
    result = {}
    try:
        with open(_BUNDLE_PATH, encoding="utf-8") as f:
            result = json.load(f)
        print(f"Loaded {len(result)} TW names from bundle.")
    except Exception as e:
        print(f"Warning: could not read {_BUNDLE_PATH}: {e}")

    sources = [
        ("https://openapi.twse.com.tw/v1/opendata/t187ap03_L",    "公司代號",             "公司簡稱"),
        ("https://www.tpex.org.tw/openapi/v1/mopsfin_t187ap03_O", "SecuritiesCompanyCode", "CompanyAbbreviation"),
    ]
    live: dict = {}
    for url, code_field, name_field in sources:
        try:
            r = requests.get(url, timeout=8, headers={"Accept": "application/json"})
            if r.ok:
                for item in r.json():
                    code = item.get(code_field, "").strip()
                    name = item.get(name_field, "").strip()
                    if code and name:
                        live[code] = name
        except Exception:
            pass
    if live:
        result.update(live)
        print(f"Refreshed {len(live)} live names.")
    return result


_TW_NAME_MAP = _load_tw_names()


def get_company_name(ticker: str) -> str:
    code = ticker.split(".")[0]
    return _TW_NAME_MAP.get(code, ticker)


def _get_index_df(market: str) -> Optional[pd.DataFrame]:
    ticker = "^TWII" if market == "tw" else "^GSPC"
    cached = _INDEX_CACHE.get(market)
    if cached and time.time() - cached[1] < 600:
        return cached[0]
    try:
        df = yf.Ticker(ticker).history(period="1y")
        if not df.empty:
            _INDEX_CACHE[market] = (df, time.time())
            return df
    except Exception:
        pass
    return None


def _load_inst_data() -> dict:
    now_ts = time.time()
    if _INST_CACHE.get("data") and now_ts - _INST_CACHE.get("ts", 0) < 3600:
        return _INST_CACHE["data"]

    for days_back in range(6):
        d = datetime.now() - timedelta(days=days_back)
        if d.weekday() >= 5:
            continue
        date_str = d.strftime("%Y%m%d")
        try:
            url = (f"https://www.twse.com.tw/rwd/zh/fund/T86"
                   f"?response=json&date={date_str}&selectType=ALL")
            r = requests.get(url, timeout=12, headers={
                "Accept": "application/json", "User-Agent": "Mozilla/5.0",
            })
            if not r.ok:
                continue
            payload = r.json()
            if payload.get("stat") != "OK" or not payload.get("data"):
                continue

            fields = payload.get("fields", [])
            def _find(keywords, exclude=""):
                for i, f in enumerate(fields):
                    if all(kw in f for kw in keywords) and (not exclude or exclude not in f):
                        return i
                return None

            fi_foreign = _find(["外陸資買賣超"], "自營") or 4
            fi_trust   = _find(["投信買賣超"])            or 10
            fi_dealer  = _find(["自營商買賣超", "合計"])  or 11

            def _n(s):
                try:
                    return int(str(s).replace(",", "").replace("+", "").strip() or "0")
                except Exception:
                    return 0

            result = {}
            for row in payload["data"]:
                try:
                    code = str(row[0]).strip()
                    result[code] = {
                        "foreign": _n(row[fi_foreign]),
                        "trust":   _n(row[fi_trust]),
                        "dealer":  _n(row[fi_dealer]),
                    }
                except Exception:
                    pass

            if result:
                _INST_CACHE["data"] = result
                _INST_CACHE["ts"]   = now_ts
                _INST_CACHE["date"] = date_str
                print(f"Institutional data loaded: {date_str}, {len(result)} stocks")
                return result
        except Exception as e:
            print(f"TWSE inst fetch failed ({date_str}): {e}")
    return {}


# ── Sheets helpers ─────────────────────────────────────────────────────────────

def _get_gc():
    raw = os.environ.get("GOOGLE_CREDENTIALS_JSON")
    if not raw:
        raise HTTPException(status_code=503, detail="GOOGLE_CREDENTIALS_JSON not configured")
    creds = Credentials.from_service_account_info(json.loads(raw), scopes=_SHEETS_SCOPES)
    return gspread.authorize(creds)


def _get_sheet_tab(market: str = "tw"):
    gc = _get_gc()
    sh = gc.open(_SHEET_NAME)
    tab_name = _SHEET_TABS.get(market, _SHEET_TABS["tw"])
    try:
        return sh.worksheet(tab_name)
    except Exception:
        ws = sh.add_worksheet(title=tab_name, rows=200, cols=1)
        ws.update("A1", [[_TICKER_COL]])
        return ws


def _get_or_create_tab(tab_name: str, headers: list):
    try:
        raw = os.environ.get("GOOGLE_CREDENTIALS_JSON")
        if not raw:
            return None
        creds = Credentials.from_service_account_info(json.loads(raw), scopes=_SHEETS_SCOPES)
        gc = gspread.authorize(creds)
        sh = gc.open(_SHEET_NAME)
        try:
            ws = sh.worksheet(tab_name)
        except Exception:
            ws = sh.add_worksheet(title=tab_name, rows=5000, cols=len(headers))
            ws.update("A1", [headers])
        return ws
    except Exception as e:
        print(f"_get_or_create_tab({tab_name}) error: {e}")
        return None


# ── Regression helpers ─────────────────────────────────────────────────────────

def _row_to_features(r: dict) -> list:
    """Convert scan result or scan_log record to regression feature vector (must match frontend)."""
    macd_map = {"golden": 2.0, "above": 1.0, "none": 0.0, "below": -1.0, "death": -2.0}

    def _bool_val(v, true_vals=("True", "true", "1", True, 1)):
        return 1.0 if v in true_vals else (-1.0 if v in ("False", "false", "0", False, 0, "None", None, "") and v not in (None, "") else 0.0)

    monthly = r.get("monthly_trend")
    weekly  = r.get("weekly_trend")

    return [
        macd_map.get(str(r.get("macd_cross") or "none"), 0.0),
        min(float(r.get("adx14") or 25), 50.0) / 50.0,
        1.0 if str(r.get("obv_trend")) == "rising" else (-1.0 if str(r.get("obv_trend")) == "falling" else 0.0),
        1.0 if monthly in (True, "True", "true", "1") else (-1.0 if monthly in (False, "False", "false", "0") else 0.0),
        1.0 if str(r.get("is_breakout20")) in ("True", "true", "1") or r.get("is_breakout20") is True else 0.0,
        1.0 if str(r.get("vol_expansion")) in ("True", "true", "1") or r.get("vol_expansion") is True else 0.0,
        max(-10.0, min(10.0, float(r.get("inst_foreign") or 0) / 1000.0)),
        max(-5.0,  min(5.0,  float(r.get("inst_trust")   or 0) / 1000.0)),
        max(-3.0,  min(3.0,  float(r.get("rs_score")     or 0))),
        1.0 if weekly in (True, "True", "true", "1") else (-1.0 if weekly in (False, "False", "false", "0") else 0.0),
        (float(r.get("rsi14") or 50) - 50.0) / 50.0,
        float(r.get("bias") or 0),
    ]


def _append_scan_log(results: list, market: str):
    if market != "tw":
        return
    try:
        ws = _get_or_create_tab(SCAN_LOG_TAB, SCAN_LOG_HEADERS)
        if not ws:
            return
        today = datetime.now().strftime("%Y-%m-%d")
        rows = []
        for r in results:
            if r.get("signal") in ("NO_DATA", "ERROR") or r.get("close") is None:
                continue
            rows.append([
                today, r["ticker"], r.get("close", ""),
                r.get("macd_cross", ""),
                r.get("adx14", ""),
                r.get("obv_trend", ""),
                str(r.get("monthly_trend", "")),
                str(r.get("is_breakout20", "")),
                str(r.get("vol_expansion", "")),
                r.get("inst_foreign", ""),
                r.get("inst_trust", ""),
                r.get("rs_score", ""),
                str(r.get("weekly_trend", "")),
                r.get("rsi14", ""),
                r.get("bias", ""),
            ])
        if rows:
            ws.append_rows(rows, value_input_option="RAW")
            print(f"Logged {len(rows)} rows to scan_log ({today})")
    except Exception as e:
        print(f"scan_log append error: {e}")


def _load_reg_coeffs() -> Optional[dict]:
    """Load regression coefficients from Sheets (returns None if not trained yet)."""
    try:
        ws = _get_or_create_tab(REG_COEFFS_TAB, ["feature", "value"])
        if not ws:
            return None
        records = ws.get_all_records()
        if not records:
            return None
        meta, coeffs = {}, {}
        for rec in records:
            feat = str(rec.get("feature", ""))
            val  = str(rec.get("value", ""))
            if feat.startswith("_"):
                meta[feat[1:]] = val
            elif feat:
                try:
                    coeffs[feat] = float(val)
                except Exception:
                    pass
        if not coeffs:
            return None
        return {
            "intercept":  float(meta.get("intercept", 0)),
            "r2":         float(meta.get("r2", 0)),
            "n_samples":  int(float(meta.get("n", 0))),
            "updated":    meta.get("updated", ""),
            "coefficients": [coeffs.get(name, 0.0) for name in FEATURE_NAMES],
        }
    except Exception:
        return None


def _apply_reg_coeffs(row: dict, reg: dict) -> Optional[float]:
    if not reg:
        return None
    try:
        features = _row_to_features(row)
        pred = reg["intercept"] + sum(f * c for f, c in zip(features, reg["coefficients"]))
        return round(pred * 100, 2)  # as percentage
    except Exception:
        return None


# ── Candlestick patterns ───────────────────────────────────────────────────────

def _detect_pattern(df: pd.DataFrame) -> str:
    if len(df) < 3:
        return ""
    o, c, h, l = df["Open"].values, df["Close"].values, df["High"].values, df["Low"].values
    body       = lambda i: abs(c[i] - o[i])
    upper_wick = lambda i: h[i] - max(c[i], o[i])
    lower_wick = lambda i: min(c[i], o[i]) - l[i]
    rng        = lambda i: h[i] - l[i]
    is_bull    = lambda i: c[i] > o[i]
    is_bear    = lambda i: c[i] < o[i]

    if is_bull(-3) and is_bull(-2) and is_bull(-1) and c[-2]>c[-3] and c[-1]>c[-2] and o[-2]>o[-3] and o[-1]>o[-2]:
        return "紅三兵"
    if is_bear(-3) and is_bear(-2) and is_bear(-1) and c[-2]<c[-3] and c[-1]<c[-2]:
        return "黑三兵"
    if is_bear(-2) and is_bull(-1) and o[-1]<=c[-2] and c[-1]>=o[-2]:
        return "多頭吞噬"
    if is_bull(-2) and is_bear(-1) and o[-1]>=c[-2] and c[-1]<=o[-2]:
        return "空頭吞噬"
    if rng(-1)>0 and body(-1)>0 and lower_wick(-1)>=2*body(-1) and upper_wick(-1)<=body(-1)*0.3:
        return "錘子線"
    if rng(-1)>0 and body(-1)>0 and upper_wick(-1)>=2*body(-1) and lower_wick(-1)<=body(-1)*0.3:
        return "射擊之星"
    if rng(-1)>0 and body(-1)<=rng(-1)*0.1:
        return "十字星"
    return ""


# ── Watchlist endpoints ────────────────────────────────────────────────────────

@app.get("/api/watchlist")
async def get_watchlist(market: str = "tw"):
    try:
        ws = _get_sheet_tab(market)
        records = ws.get_all_records()
        tickers = [r[_TICKER_COL] for r in records if r.get(_TICKER_COL, "").strip()]
        return {"tickers": tickers}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}")


class WatchlistUpdate(BaseModel):
    tickers: List[str]


@app.put("/api/watchlist")
async def put_watchlist(body: WatchlistUpdate, market: str = "tw"):
    try:
        ws = _get_sheet_tab(market)
        ws.clear()
        ws.update("A1", [[_TICKER_COL]] + [[t] for t in body.tickers])
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}")
    return {"status": "ok", "count": len(body.tickers)}


# ── Scan endpoint ──────────────────────────────────────────────────────────────

class ScanRequest(BaseModel):
    tickers: List[str]
    market: str = "tw"
    line_token: str = ""


def send_line_notify(message: str, token: str):
    if not token:
        return
    try:
        requests.post(
            "https://notify-api.line.me/api/notify",
            headers={"Authorization": f"Bearer {token}"},
            data={"message": message},
        )
    except Exception as e:
        print(f"LINE notify failed: {e}")


@app.post("/api/scan")
async def scan_stocks(request: ScanRequest):
    results = []
    triggered_alerts = []

    index_df = _get_index_df(request.market)
    index_20d_return = None
    if index_df is not None and len(index_df) >= 21:
        idx_c = index_df["Close"]
        index_20d_return = float((idx_c.iloc[-1] - idx_c.iloc[-21]) / idx_c.iloc[-21])

    market_regime_bull = None
    if index_df is not None and len(index_df) >= 20:
        idx_ma20 = index_df["Close"].rolling(20).mean().iloc[-1]
        market_regime_bull = bool(index_df["Close"].iloc[-1] > idx_ma20)

    inst_data = _load_inst_data() if request.market == "tw" else {}

    # Load regression coefficients (best-effort)
    reg_coeffs = None
    try:
        reg_coeffs = _load_reg_coeffs()
    except Exception:
        pass

    def _no_data_row(ticker):
        return {
            "ticker": ticker, "companyName": get_company_name(ticker),
            "close": None, "open": None, "high": None, "low": None,
            "ma20": None, "ma60": None, "ma120": None,
            "volume": None, "vol_ma20": None,
            "rsi14": None, "bias": None,
            "week52_high": None, "week52_low": None, "pct_from_52high": None,
            "rs_score": None, "weekly_trend": None,
            "max_drawdown_1y": None, "stop_loss": None, "target_price": None,
            "pattern": "",
            "macd_line": None, "macd_signal": None, "macd_hist": None, "macd_cross": None,
            "adx14": None, "di_plus": None, "di_minus": None,
            "obv_trend": None, "monthly_trend": None,
            "is_breakout20": None, "vol_expansion": None,
            "inst_foreign": None, "inst_trust": None,
            "market_regime_bull": market_regime_bull,
            "predicted_return": None,
            "conds": {}, "sell_flags": {}, "signal": "NO_DATA",
        }

    for ticker in request.tickers:
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
                row = _no_data_row(ticker)
                row["companyName"] = company_name
                results.append(row)
                continue

            # ── Moving averages ───────────────────────────────────────────────
            df["MA20"]  = df["Close"].rolling(20).mean()
            df["MA60"]  = df["Close"].rolling(60).mean()
            df["MA120"] = df["Close"].rolling(120).mean()
            df["VMA20"] = df["Volume"].rolling(20).mean()

            # ── RSI-14 (Wilder) ───────────────────────────────────────────────
            delta    = df["Close"].diff()
            gain     = delta.clip(lower=0)
            loss     = (-delta).clip(lower=0)
            avg_gain = gain.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
            avg_loss = loss.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
            safe_loss = avg_loss.copy(); safe_loss[safe_loss == 0] = 1e-10
            df["RSI14"] = 100.0 - (100.0 / (1.0 + avg_gain / safe_loss))

            # ── Bias ──────────────────────────────────────────────────────────
            df["Bias"] = (df["Close"] - df["MA20"]) / df["MA20"]

            # ── MACD (12/26/9) ────────────────────────────────────────────────
            exp12 = df["Close"].ewm(span=12, adjust=False).mean()
            exp26 = df["Close"].ewm(span=26, adjust=False).mean()
            df["MACD"]      = exp12 - exp26
            df["MACD_Sig"]  = df["MACD"].ewm(span=9, adjust=False).mean()
            df["MACD_Hist"] = df["MACD"] - df["MACD_Sig"]

            # ── ADX-14 ────────────────────────────────────────────────────────
            df["H-L"]  = df["High"] - df["Low"]
            df["H-PC"] = (df["High"] - df["Close"].shift(1)).abs()
            df["L-PC"] = (df["Low"]  - df["Close"].shift(1)).abs()
            df["TR"]   = df[["H-L", "H-PC", "L-PC"]].max(axis=1)
            hl_diff  = df["High"] - df["High"].shift(1)
            lh_diff  = df["Low"].shift(1) - df["Low"]
            df["DM+"] = np.where((hl_diff > lh_diff) & (hl_diff > 0), hl_diff, 0.0)
            df["DM-"] = np.where((lh_diff > hl_diff) & (lh_diff > 0), lh_diff, 0.0)
            atr    = df["TR"].ewm(alpha=1/14, min_periods=14, adjust=False).mean()
            di_pos = 100 * df["DM+"].ewm(alpha=1/14, min_periods=14, adjust=False).mean() / atr.replace(0, np.nan)
            di_neg = 100 * df["DM-"].ewm(alpha=1/14, min_periods=14, adjust=False).mean() / atr.replace(0, np.nan)
            dx_denom = (di_pos + di_neg).replace(0, np.nan)
            df["DX"]  = 100 * (di_pos - di_neg).abs() / dx_denom
            df["ADX"] = df["DX"].ewm(alpha=1/14, min_periods=14, adjust=False).mean()
            df["DI+"] = di_pos
            df["DI-"] = di_neg

            # ── OBV ──────────────────────────────────────────────────────────
            df["OBV"]    = (np.sign(df["Close"].diff()) * df["Volume"]).cumsum()
            df["OBV_MA"] = df["OBV"].rolling(20).mean()

            # ── 52-week high/low ──────────────────────────────────────────────
            week52_high = float(df["High"].max())
            week52_low  = float(df["Low"].min())

            # ── Max drawdown ──────────────────────────────────────────────────
            rolling_max     = df["Close"].cummax()
            max_drawdown_1y = float(((df["Close"] - rolling_max) / rolling_max).min())

            # ── Weekly trend ──────────────────────────────────────────────────
            weekly = df["Close"].resample("W").last().dropna()
            wma5   = weekly.rolling(5).mean()
            wma10  = weekly.rolling(10).mean()
            wma20  = weekly.rolling(20).mean()
            weekly_trend = None
            if len(weekly) >= 20 and not any(pd.isna([wma5.iloc[-1], wma10.iloc[-1], wma20.iloc[-1]])):
                weekly_trend = bool(wma5.iloc[-1] > wma10.iloc[-1] > wma20.iloc[-1])

            # ── Monthly trend ─────────────────────────────────────────────────
            monthly = df["Close"].resample("MS").last().dropna()
            mma5    = monthly.rolling(5).mean()
            mma10   = monthly.rolling(10).mean()
            monthly_trend = None
            if len(monthly) >= 10 and not any(pd.isna([mma5.iloc[-1], mma10.iloc[-1]])):
                monthly_trend = bool(mma5.iloc[-1] > mma10.iloc[-1])

            # ── Latest bar values ─────────────────────────────────────────────
            last     = df.iloc[-1]
            prev     = df.iloc[-2]
            prev_rsi = df["RSI14"].iloc[-2]

            c_close  = last["Close"]
            c_open   = last["Open"]
            c_high   = last["High"]
            c_low    = last["Low"]
            c_vol    = last["Volume"]
            ma20     = last["MA20"]
            ma60     = last["MA60"]
            ma120    = last["MA120"]
            vol_ma20 = last["VMA20"]
            rsi14    = last["RSI14"]
            bias     = last["Bias"]

            if any(pd.isna(v) for v in [ma20, ma60, ma120, vol_ma20, rsi14, bias, prev_rsi]):
                row = _no_data_row(ticker)
                row["companyName"] = company_name
                if not pd.isna(c_close):
                    row["close"] = round(float(c_close), 2)
                results.append(row)
                continue

            # ── MACD cross ───────────────────────────────────────────────────
            macd_now  = last["MACD"]
            macd_prev = prev["MACD"]
            sig_now   = last["MACD_Sig"]
            sig_prev  = prev["MACD_Sig"]
            macd_cross = "none"
            if not any(pd.isna([macd_now, macd_prev, sig_now, sig_prev])):
                if macd_prev < sig_prev and macd_now > sig_now:
                    macd_cross = "golden"
                elif macd_prev > sig_prev and macd_now < sig_now:
                    macd_cross = "death"
                elif macd_now > sig_now:
                    macd_cross = "above"
                else:
                    macd_cross = "below"

            # ── ADX ──────────────────────────────────────────────────────────
            adx14       = None if pd.isna(last["ADX"]) else round(float(last["ADX"]), 1)
            di_plus_val = None if pd.isna(last["DI+"]) else round(float(last["DI+"]), 1)
            di_minus_val= None if pd.isna(last["DI-"]) else round(float(last["DI-"]), 1)

            # ── OBV trend ────────────────────────────────────────────────────
            obv_trend = None
            if not pd.isna(last["OBV"]) and not pd.isna(last["OBV_MA"]):
                obv_trend = "rising" if last["OBV"] > last["OBV_MA"] else "falling"

            # ── Breakout & volume expansion ───────────────────────────────────
            is_breakout20 = False
            if len(df) >= 21:
                high20 = df["High"].iloc[-21:-1].max()
                is_breakout20 = bool(float(c_close) > float(high20))

            vol_expansion = False
            if len(df) >= 3:
                v1, v2, v3 = df["Volume"].iloc[-3], df["Volume"].iloc[-2], df["Volume"].iloc[-1]
                vol_expansion = bool(v3 > v2 > v1 and v3 > vol_ma20)

            # ── RS ────────────────────────────────────────────────────────────
            rs_score = None
            if len(df) >= 21:
                stock_20d = float((c_close - df["Close"].iloc[-21]) / df["Close"].iloc[-21])
                if index_20d_return is not None and index_20d_return != 0:
                    rs_score = round(stock_20d / abs(index_20d_return), 2)

            # ── Institutional data ────────────────────────────────────────────
            code = ticker.split(".")[0]
            inst = inst_data.get(code, {})
            inst_foreign = inst.get("foreign") if inst else None
            inst_trust   = inst.get("trust")   if inst else None

            # ── 6 buy conditions ──────────────────────────────────────────────
            mid_range = (c_high + c_low) / 2.0
            conds = {
                "price":  bool(c_close > ma20),
                "volume": bool(c_vol > 1.2 * vol_ma20),
                "trend":  bool(ma20 > ma60 and ma60 > ma120),
                "candle": bool(c_close > c_open and c_close > mid_range),
                "rsi":    bool(60.0 < rsi14 < 70.0 and rsi14 > prev_rsi),
                "bias":   bool(bias < 0.03),
            }
            sell_flags = {
                "is_trend_broken":       bool(c_close < ma20),
                "is_momentum_lost":      bool(rsi14 < 50.0),
                "is_heavy_distribution": bool(c_close < c_open and c_vol > vol_ma20),
            }
            is_buy = all(conds.values())

            pattern        = _detect_pattern(df.tail(5))
            stop_loss      = round(float(ma20) * 0.97, 2)
            target_price   = round(float(c_close) * 1.15, 2)
            pct_from_52high = round((float(c_close) - week52_high) / week52_high * 100, 1)

            # ── Build result row ──────────────────────────────────────────────
            result_row = {
                "ticker":      ticker,
                "companyName": company_name,
                "close":  round(float(c_close),  2),
                "open":   round(float(c_open),   2),
                "high":   round(float(c_high),   2),
                "low":    round(float(c_low),    2),
                "ma20":   round(float(ma20),     2),
                "ma60":   round(float(ma60),     2),
                "ma120":  round(float(ma120),    2),
                "volume":   int(c_vol),
                "vol_ma20": int(vol_ma20),
                "rsi14":  round(float(rsi14), 1),
                "bias":   round(float(bias) * 100, 2),
                "week52_high":      round(week52_high, 2),
                "week52_low":       round(week52_low,  2),
                "pct_from_52high":  pct_from_52high,
                "rs_score":         rs_score,
                "weekly_trend":     weekly_trend,
                "max_drawdown_1y":  round(max_drawdown_1y * 100, 1),
                "stop_loss":        stop_loss,
                "target_price":     target_price,
                "pattern":          pattern,
                "macd_line":    round(float(macd_now),  4) if not pd.isna(macd_now) else None,
                "macd_signal":  round(float(sig_now),   4) if not pd.isna(sig_now)  else None,
                "macd_hist":    round(float(last["MACD_Hist"]), 4) if not pd.isna(last["MACD_Hist"]) else None,
                "macd_cross":   macd_cross,
                "adx14":        adx14,
                "di_plus":      di_plus_val,
                "di_minus":     di_minus_val,
                "obv_trend":    obv_trend,
                "monthly_trend": monthly_trend,
                "is_breakout20": is_breakout20,
                "vol_expansion": vol_expansion,
                "inst_foreign":  inst_foreign,
                "inst_trust":    inst_trust,
                "market_regime_bull": market_regime_bull,
                "conds":      conds,
                "sell_flags": sell_flags,
                "signal":     "YES" if is_buy else "NO",
                "predicted_return": None,  # filled below
            }

            # Apply regression model if available
            result_row["predicted_return"] = _apply_reg_coeffs(result_row, reg_coeffs)

            results.append(result_row)

            if is_buy:
                triggered_alerts.append(
                    f"🚀 {ticker} 高品質買進訊號！"
                    f"Close:{round(float(c_close),2)} RSI:{round(float(rsi14),1)} MACD:{macd_cross}"
                )

        except Exception as e:
            print(f"Error scanning {ticker}: {e}")
            results.append({
                "ticker": ticker, "companyName": get_company_name(ticker),
                "close": None, "open": None, "high": None, "low": None,
                "ma20": None, "ma60": None, "ma120": None,
                "volume": None, "vol_ma20": None, "rsi14": None, "bias": None,
                "week52_high": None, "week52_low": None, "pct_from_52high": None,
                "rs_score": None, "weekly_trend": None,
                "max_drawdown_1y": None, "stop_loss": None, "target_price": None,
                "pattern": "",
                "macd_line": None, "macd_signal": None, "macd_hist": None, "macd_cross": None,
                "adx14": None, "di_plus": None, "di_minus": None,
                "obv_trend": None, "monthly_trend": None,
                "is_breakout20": None, "vol_expansion": None,
                "inst_foreign": None, "inst_trust": None,
                "market_regime_bull": market_regime_bull,
                "predicted_return": None,
                "conds": {}, "sell_flags": {}, "signal": "ERROR",
            })

    if triggered_alerts and request.line_token:
        send_line_notify("\n" + "\n".join(triggered_alerts), request.line_token)

    # Auto-log to scan_log (best-effort, non-blocking)
    try:
        _append_scan_log(results, request.market)
    except Exception as e:
        print(f"scan_log error (non-fatal): {e}")

    return {"status": "success", "data": results}


# ── Notify endpoint ────────────────────────────────────────────────────────────

class NotifyRequest(BaseModel):
    message: str
    line_token: str = ""


@app.post("/api/notify")
async def send_notify(req: NotifyRequest):
    if not req.line_token:
        return {"status": "skipped", "reason": "no token"}
    send_line_notify(req.message, req.line_token)
    return {"status": "ok"}


# ── Backtest endpoint ──────────────────────────────────────────────────────────

@app.post("/api/backtest")
async def backtest(request: ScanRequest):
    results = []
    for ticker in request.tickers:
        try:
            stock = yf.Ticker(ticker)
            df = stock.history(period="2y")
            if df.empty or len(df) < 200:
                results.append({"ticker": ticker, "error": "insufficient data"})
                continue

            df["MA20"]  = df["Close"].rolling(20).mean()
            df["MA60"]  = df["Close"].rolling(60).mean()
            df["MA120"] = df["Close"].rolling(120).mean()
            df["VMA20"] = df["Volume"].rolling(20).mean()
            delta    = df["Close"].diff()
            gain     = delta.clip(lower=0)
            loss     = (-delta).clip(lower=0)
            avg_gain = gain.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
            avg_loss = loss.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
            safe_loss = avg_loss.copy(); safe_loss[safe_loss == 0] = 1e-10
            df["RSI14"] = 100.0 - (100.0 / (1.0 + avg_gain / safe_loss))
            df["Bias"]  = (df["Close"] - df["MA20"]) / df["MA20"]
            exp12 = df["Close"].ewm(span=12, adjust=False).mean()
            exp26 = df["Close"].ewm(span=26, adjust=False).mean()
            df["MACD"]     = exp12 - exp26
            df["MACD_Sig"] = df["MACD"].ewm(span=9, adjust=False).mean()

            signals = []
            start_idx = max(136, len(df) - 252)
            for i in range(start_idx, len(df) - 10):
                row  = df.iloc[i]
                prev = df.iloc[i - 1]
                if any(pd.isna([row["MA20"], row["MA60"], row["MA120"], row["VMA20"], row["RSI14"], row["Bias"], prev["RSI14"]])):
                    continue
                mid = (row["High"] + row["Low"]) / 2.0
                is_buy = (
                    row["Close"] > row["MA20"] and
                    row["Volume"] > 1.2 * row["VMA20"] and
                    row["MA20"] > row["MA60"] and row["MA60"] > row["MA120"] and
                    row["Close"] > row["Open"] and row["Close"] > mid and
                    60.0 < row["RSI14"] < 70.0 and row["RSI14"] > prev["RSI14"] and
                    row["Bias"] < 0.03
                )
                macd_ok = (not pd.isna(row["MACD"]) and not pd.isna(row["MACD_Sig"])
                           and row["MACD"] > row["MACD_Sig"])

                if is_buy:
                    entry = float(row["Close"])
                    exit_price = float(df["Close"].iloc[i + 10])
                    for j in range(1, 11):
                        if float(df["Close"].iloc[i + j]) < float(df["MA20"].iloc[i + j]):
                            exit_price = float(df["Close"].iloc[i + j])
                            break
                    fwd_return = (exit_price - entry) / entry
                    signals.append({
                        "date": df.index[i].strftime("%Y-%m-%d"),
                        "entry": round(entry, 2),
                        "exit10d": round(exit_price, 2),
                        "return10d": round(fwd_return * 100, 2),
                        "win": fwd_return > 0,
                        "macd_ok": macd_ok,
                    })

            if signals:
                win_rate   = sum(1 for s in signals if s["win"]) / len(signals)
                avg_return = sum(s["return10d"] for s in signals) / len(signals)
                macd_sig   = [s for s in signals if s["macd_ok"]]
                macd_wr    = sum(1 for s in macd_sig if s["win"]) / len(macd_sig) if macd_sig else None
                results.append({
                    "ticker": ticker,
                    "companyName": get_company_name(ticker),
                    "total_signals": len(signals),
                    "win_rate": round(win_rate * 100, 1),
                    "avg_return_10d": round(avg_return, 2),
                    "macd_filtered_wr": round(macd_wr * 100, 1) if macd_wr is not None else None,
                    "signals": signals[-20:],
                })
            else:
                results.append({
                    "ticker": ticker, "companyName": get_company_name(ticker),
                    "total_signals": 0, "win_rate": None, "avg_return_10d": None,
                    "macd_filtered_wr": None, "signals": [],
                })
        except Exception as e:
            results.append({"ticker": ticker, "error": str(e)})

    return {"status": "success", "data": results}


# ── Regression endpoints ───────────────────────────────────────────────────────

@app.post("/api/regression/train")
async def regression_train(market: str = "tw"):
    if market != "tw":
        raise HTTPException(400, "Regression currently only supported for TW market")

    ws = _get_or_create_tab(SCAN_LOG_TAB, SCAN_LOG_HEADERS)
    if not ws:
        raise HTTPException(503, "Google Sheets not configured")

    records = ws.get_all_records()
    if not records:
        return {"status": "insufficient_data", "message": "No scan log yet. Scan stocks daily to build training data.", "eligible": 0}

    # Only use entries older than 10 calendar days
    cutoff = (datetime.now() - timedelta(days=10)).strftime("%Y-%m-%d")
    eligible = [r for r in records if str(r.get("scan_date", "")) <= cutoff and r.get("ticker")]

    if len(eligible) < 20:
        return {
            "status": "insufficient_data",
            "message": f"Need 20+ data points older than 10 days (have {len(eligible)}). Keep scanning daily!",
            "eligible": len(eligible),
        }

    # Deduplicate (date, ticker)
    seen, deduped = set(), []
    for r in eligible:
        key = (r["scan_date"], r["ticker"])
        if key not in seen:
            seen.add(key)
            deduped.append(r)

    # Fetch prices per unique ticker
    unique_tickers = list(set(r["ticker"] for r in deduped))
    price_cache = {}
    for ticker in unique_tickers:
        try:
            hist = yf.Ticker(ticker).history(period="1y")
            if not hist.empty:
                s = hist["Close"].copy()
                s.index = pd.to_datetime(s.index).normalize().tz_localize(None)
                price_cache[ticker] = s
        except Exception:
            pass

    # Build feature matrix X and target y
    X_rows, y_vals = [], []
    for r in deduped:
        ticker = r["ticker"]
        if ticker not in price_cache:
            continue
        try:
            prices    = price_cache[ticker]
            scan_date = pd.Timestamp(r["scan_date"]).tz_localize(None)
            future    = prices[prices.index >= scan_date]
            if len(future) < 11:
                continue
            entry  = float(future.iloc[0])
            exit_  = float(future.iloc[10])
            ret    = (exit_ - entry) / entry
            features = _row_to_features(r)
            X_rows.append(features)
            y_vals.append(ret)
        except Exception:
            pass

    n = len(X_rows)
    if n < 20:
        return {
            "status": "insufficient_data",
            "message": f"Only {n} valid samples after price matching (need 20+). Keep scanning!",
            "eligible": n,
        }

    X = np.array(X_rows, dtype=float)
    y = np.array(y_vals, dtype=float)

    # OLS regression
    X_b = np.column_stack([np.ones(n), X])
    coeffs, _, _, _ = np.linalg.lstsq(X_b, y, rcond=None)

    y_pred = X_b @ coeffs
    ss_res = float(np.sum((y - y_pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0

    intercept   = float(coeffs[0])
    feat_coeffs = [float(c) for c in coeffs[1:]]

    # Save to Google Sheets
    ws_reg = _get_or_create_tab(REG_COEFFS_TAB, ["feature", "value"])
    if ws_reg:
        ws_reg.clear()
        ws_reg.update("A1", [["feature", "value"]] + [
            ["_r2",        str(round(r2, 6))],
            ["_n",         str(n)],
            ["_updated",   datetime.now().strftime("%Y-%m-%d %H:%M")],
            ["_intercept", str(intercept)],
        ] + [[name, str(c)] for name, c in zip(FEATURE_NAMES, feat_coeffs)])

    return {
        "status": "ok",
        "r2": round(r2, 4),
        "n_samples": n,
        "intercept": round(intercept, 6),
        "feature_names": FEATURE_NAMES,
        "coefficients": {name: round(c, 6) for name, c in zip(FEATURE_NAMES, feat_coeffs)},
    }


@app.get("/api/regression/coeffs")
async def regression_coeffs_get(market: str = "tw"):
    reg = _load_reg_coeffs()
    if not reg:
        return {"status": "no_data"}
    return {
        "status": "ok",
        "intercept":    reg["intercept"],
        "r2":           reg["r2"],
        "n_samples":    reg["n_samples"],
        "updated":      reg["updated"],
        "feature_names": FEATURE_NAMES,
        "coefficients": reg["coefficients"],
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
