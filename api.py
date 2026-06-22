import json
import os
import pandas as pd
import yfinance as yf
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


def _load_tw_names() -> dict[str, str]:
    result: dict[str, str] = {}
    try:
        with open(_BUNDLE_PATH, encoding="utf-8") as f:
            result = json.load(f)
        print(f"Loaded {len(result)} TW company names from bundle.")
    except Exception as e:
        print(f"Warning: could not read {_BUNDLE_PATH}: {e}")

    sources = [
        ("https://openapi.twse.com.tw/v1/opendata/t187ap03_L",    "公司代號",             "公司簡稱"),
        ("https://www.tpex.org.tw/openapi/v1/mopsfin_t187ap03_O", "SecuritiesCompanyCode", "CompanyAbbreviation"),
    ]
    live: dict[str, str] = {}
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
        print(f"Refreshed with {len(live)} live entries from TWSE/TPEX.")

    return result


_TW_NAME_MAP = _load_tw_names()

# Cache for market index data (^TWII for TW, ^GSPC for US)
_INDEX_CACHE: dict[str, tuple] = {}  # market -> (df, timestamp)


def _get_index_df(market: str) -> pd.DataFrame | None:
    """Fetch market index 1y daily data with 10-min cache."""
    import time
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


def get_company_name(ticker: str) -> str:
    code = ticker.split(".")[0]
    return _TW_NAME_MAP.get(code, ticker)


def _detect_pattern(df: pd.DataFrame) -> str:
    """Detect the most recent notable candlestick pattern (last 3 bars)."""
    if len(df) < 3:
        return ""
    o = df["Open"].values
    c = df["Close"].values
    h = df["High"].values
    l = df["Low"].values

    # Last 3 candles: [-3], [-2], [-1]
    body = lambda i: abs(c[i] - o[i])
    upper_wick = lambda i: h[i] - max(c[i], o[i])
    lower_wick = lambda i: min(c[i], o[i]) - l[i]
    is_bull = lambda i: c[i] > o[i]
    is_bear = lambda i: c[i] < o[i]
    rng = lambda i: h[i] - l[i]

    # 紅三兵 Three White Soldiers
    if (is_bull(-3) and is_bull(-2) and is_bull(-1) and
            c[-2] > c[-3] and c[-1] > c[-2] and
            o[-2] > o[-3] and o[-1] > o[-2]):
        return "紅三兵"

    # 黑三兵 Three Black Crows
    if (is_bear(-3) and is_bear(-2) and is_bear(-1) and
            c[-2] < c[-3] and c[-1] < c[-2]):
        return "黑三兵"

    # 多頭吞噬 Bullish Engulfing
    if (is_bear(-2) and is_bull(-1) and
            o[-1] <= c[-2] and c[-1] >= o[-2]):
        return "多頭吞噬"

    # 空頭吞噬 Bearish Engulfing
    if (is_bull(-2) and is_bear(-1) and
            o[-1] >= c[-2] and c[-1] <= o[-2]):
        return "空頭吞噬"

    # 錘子線 Hammer (lower wick >= 2x body, small upper wick, bullish context)
    if (rng(-1) > 0 and body(-1) > 0 and
            lower_wick(-1) >= 2 * body(-1) and
            upper_wick(-1) <= body(-1) * 0.3):
        return "錘子線"

    # 射擊之星 Shooting Star
    if (rng(-1) > 0 and body(-1) > 0 and
            upper_wick(-1) >= 2 * body(-1) and
            lower_wick(-1) <= body(-1) * 0.3):
        return "射擊之星"

    # 十字星 Doji
    if rng(-1) > 0 and body(-1) <= rng(-1) * 0.1:
        return "十字星"

    return ""


_SHEETS_SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive.readonly",
]
_SHEET_NAME = "gigi-war-room-watchlist"
_TICKER_COL = "ticker"

_SHEET_TABS = {
    "tw": "gigi-war-room-watchlist",
    "us": "gigi-us-watchlist",
}


def _get_sheet_tab(market: str = "tw"):
    raw = os.environ.get("GOOGLE_CREDENTIALS_JSON")
    if not raw:
        raise HTTPException(status_code=503, detail="GOOGLE_CREDENTIALS_JSON not configured")
    creds_dict = json.loads(raw)
    creds = Credentials.from_service_account_info(creds_dict, scopes=_SHEETS_SCOPES)
    gc = gspread.authorize(creds)
    sh = gc.open(_SHEET_NAME)
    tab_name = _SHEET_TABS.get(market, _SHEET_TABS["tw"])
    try:
        return sh.worksheet(tab_name)
    except Exception:
        ws = sh.add_worksheet(title=tab_name, rows=200, cols=1)
        ws.update("A1", [[_TICKER_COL]])
        return ws


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
        print(f"Failed to send LINE notification: {e}")


@app.post("/api/scan")
async def scan_stocks(request: ScanRequest):
    results = []
    triggered_alerts = []

    # Fetch market index for relative strength calculation
    index_df = _get_index_df(request.market)
    index_20d_return = None
    if index_df is not None and len(index_df) >= 21:
        idx_close = index_df["Close"]
        index_20d_return = float((idx_close.iloc[-1] - idx_close.iloc[-21]) / idx_close.iloc[-21])

    _NO_DATA_ROW = lambda ticker: {
        "ticker": ticker, "companyName": get_company_name(ticker),
        "close": None, "open": None, "high": None, "low": None,
        "ma20": None, "ma60": None, "ma120": None,
        "volume": None, "vol_ma20": None,
        "rsi14": None, "bias": None,
        "week52_high": None, "week52_low": None, "pct_from_52high": None,
        "rs_score": None, "weekly_trend": None,
        "max_drawdown_1y": None, "stop_loss": None, "target_price": None,
        "pattern": "",
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
                row = _NO_DATA_ROW(ticker)
                row["companyName"] = company_name
                results.append(row)
                continue

            # ── Moving averages & volume MA ───────────────────────────────────
            df["MA20"]  = df["Close"].rolling(20).mean()
            df["MA60"]  = df["Close"].rolling(60).mean()
            df["MA120"] = df["Close"].rolling(120).mean()
            df["VMA20"] = df["Volume"].rolling(20).mean()

            # ── Wilder RSI-14 ────────────────────────────────────────────────
            delta    = df["Close"].diff()
            gain     = delta.clip(lower=0)
            loss     = (-delta).clip(lower=0)
            avg_gain = gain.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
            avg_loss = loss.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
            safe_loss = avg_loss.copy()
            safe_loss[safe_loss == 0] = 1e-10
            df["RSI14"] = 100.0 - (100.0 / (1.0 + avg_gain / safe_loss))

            # ── Bias ─────────────────────────────────────────────────────────
            df["Bias"] = (df["Close"] - df["MA20"]) / df["MA20"]

            # ── 52-week high/low ──────────────────────────────────────────────
            week52_high = float(df["High"].max())
            week52_low  = float(df["Low"].min())

            # ── Max drawdown (1y) ────────────────────────────────────────────
            rolling_max = df["Close"].cummax()
            drawdown    = (df["Close"] - rolling_max) / rolling_max
            max_drawdown_1y = float(drawdown.min())  # negative number, e.g. -0.35

            # ── Weekly trend (resample to weekly) ────────────────────────────
            weekly = df["Close"].resample("W").last().dropna()
            weekly_ma5  = weekly.rolling(5).mean()
            weekly_ma10 = weekly.rolling(10).mean()
            weekly_ma20 = weekly.rolling(20).mean()
            if len(weekly) >= 20 and not any(pd.isna([weekly_ma5.iloc[-1], weekly_ma10.iloc[-1], weekly_ma20.iloc[-1]])):
                weekly_trend = bool(weekly_ma5.iloc[-1] > weekly_ma10.iloc[-1] > weekly_ma20.iloc[-1])
            else:
                weekly_trend = None

            # ── Extract latest + previous bar ─────────────────────────────────
            last     = df.iloc[-1]
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
                row = _NO_DATA_ROW(ticker)
                if not pd.isna(c_close):
                    row["close"]  = round(float(c_close), 2)
                if not pd.isna(c_vol):
                    row["volume"] = int(c_vol)
                results.append(row)
                continue

            # ── Relative strength (20-day return vs index) ───────────────────
            stock_20d_return = float((c_close - df["Close"].iloc[-21]) / df["Close"].iloc[-21]) if len(df) >= 21 else None
            if stock_20d_return is not None and index_20d_return is not None and index_20d_return != 0:
                rs_score = round(stock_20d_return / abs(index_20d_return), 2)
            else:
                rs_score = None

            # ── Six conditions ────────────────────────────────────────────────
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
                "is_trend_broken":        bool(c_close < ma20),
                "is_momentum_lost":       bool(rsi14 < 50.0),
                "is_heavy_distribution":  bool(c_close < c_open and c_vol > vol_ma20),
            }
            is_buy = all(conds.values())

            # ── Candlestick pattern ───────────────────────────────────────────
            pattern = _detect_pattern(df.tail(5))

            # ── Stop loss & target ────────────────────────────────────────────
            stop_loss    = round(float(ma20) * 0.97, 2)
            target_price = round(float(c_close) * 1.15, 2)
            pct_from_52high = round((float(c_close) - week52_high) / week52_high * 100, 1)

            results.append({
                "ticker":      ticker,
                "companyName": company_name,
                "close":    round(float(c_close),  2),
                "open":     round(float(c_open),   2),
                "high":     round(float(c_high),   2),
                "low":      round(float(c_low),    2),
                "ma20":     round(float(ma20),     2),
                "ma60":     round(float(ma60),     2),
                "ma120":    round(float(ma120),    2),
                "volume":   int(c_vol),
                "vol_ma20": int(vol_ma20),
                "rsi14":    round(float(rsi14),    1),
                "bias":     round(float(bias) * 100, 2),
                # v3.0 new fields
                "week52_high":    round(week52_high, 2),
                "week52_low":     round(week52_low,  2),
                "pct_from_52high": pct_from_52high,
                "rs_score":       rs_score,
                "weekly_trend":   weekly_trend,
                "max_drawdown_1y": round(max_drawdown_1y * 100, 1),
                "stop_loss":      stop_loss,
                "target_price":   target_price,
                "pattern":        pattern,
                "conds":      conds,
                "sell_flags": sell_flags,
                "signal":   "YES" if is_buy else "NO",
            })

            if is_buy:
                triggered_alerts.append(
                    f"🚀 {ticker} 高品質買進訊號！"
                    f"Close:{round(float(c_close),2)} RSI:{round(float(rsi14),1)} Bias:{round(float(bias)*100,2)}%"
                )

        except Exception as e:
            print(f"Error scanning {ticker}: {e}")
            results.append({
                "ticker": ticker, "companyName": get_company_name(ticker),
                "close": None, "open": None, "high": None, "low": None,
                "ma20": None, "ma60": None, "ma120": None,
                "volume": None, "vol_ma20": None,
                "rsi14": None, "bias": None,
                "week52_high": None, "week52_low": None, "pct_from_52high": None,
                "rs_score": None, "weekly_trend": None,
                "max_drawdown_1y": None, "stop_loss": None, "target_price": None,
                "pattern": "",
                "conds": {}, "sell_flags": {}, "signal": "ERROR",
            })

    if triggered_alerts and request.line_token:
        send_line_notify("\n" + "\n".join(triggered_alerts), request.line_token)

    return {"status": "success", "data": results}


class NotifyRequest(BaseModel):
    message: str
    line_token: str = ""


@app.post("/api/notify")
async def send_notify(req: NotifyRequest):
    if not req.line_token:
        return {"status": "skipped", "reason": "no token"}
    send_line_notify(req.message, req.line_token)
    return {"status": "ok"}


@app.post("/api/backtest")
async def backtest(request: ScanRequest):
    """
    Backtest: for each ticker, scan every day in the past year,
    record buy signals, then check 10-day forward return.
    Returns win rate and avg return per ticker.
    """
    results = []
    for ticker in request.tickers:
        try:
            stock = yf.Ticker(ticker)
            df = stock.history(period="2y")  # 2y for enough lookback
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

            signals = []
            # Only scan the last 1 year (skip first 120 bars for warmup)
            start_idx = max(136, len(df) - 252)
            for i in range(start_idx, len(df) - 10):
                row = df.iloc[i]
                prev_rsi = df["RSI14"].iloc[i - 1]
                if any(pd.isna([row["MA20"], row["MA60"], row["MA120"], row["VMA20"], row["RSI14"], row["Bias"], prev_rsi])):
                    continue
                mid = (row["High"] + row["Low"]) / 2.0
                is_buy = (
                    row["Close"] > row["MA20"] and
                    row["Volume"] > 1.2 * row["VMA20"] and
                    row["MA20"] > row["MA60"] and row["MA60"] > row["MA120"] and
                    row["Close"] > row["Open"] and row["Close"] > mid and
                    60.0 < row["RSI14"] < 70.0 and row["RSI14"] > prev_rsi and
                    row["Bias"] < 0.03
                )
                if is_buy:
                    entry = float(row["Close"])
                    exit_price = float(df["Close"].iloc[i + 10])
                    fwd_return = (exit_price - entry) / entry
                    signals.append({
                        "date": df.index[i].strftime("%Y-%m-%d"),
                        "entry": round(entry, 2),
                        "exit10d": round(exit_price, 2),
                        "return10d": round(fwd_return * 100, 2),
                        "win": fwd_return > 0,
                    })

            if signals:
                win_rate = sum(1 for s in signals if s["win"]) / len(signals)
                avg_return = sum(s["return10d"] for s in signals) / len(signals)
                results.append({
                    "ticker": ticker,
                    "companyName": get_company_name(ticker),
                    "total_signals": len(signals),
                    "win_rate": round(win_rate * 100, 1),
                    "avg_return_10d": round(avg_return, 2),
                    "signals": signals[-20:],  # last 20 signals
                })
            else:
                results.append({
                    "ticker": ticker,
                    "companyName": get_company_name(ticker),
                    "total_signals": 0,
                    "win_rate": None,
                    "avg_return_10d": None,
                    "signals": [],
                })
        except Exception as e:
            results.append({"ticker": ticker, "error": str(e)})

    return {"status": "success", "data": results}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
