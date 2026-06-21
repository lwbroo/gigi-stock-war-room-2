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
    """Load Traditional Chinese company names from bundled JSON, refreshed from TWSE/TPEX if reachable."""
    result: dict[str, str] = {}

    # 1. Load bundled file (always works, no network needed)
    try:
        with open(_BUNDLE_PATH, encoding="utf-8") as f:
            result = json.load(f)
        print(f"Loaded {len(result)} TW company names from bundle.")
    except Exception as e:
        print(f"Warning: could not read {_BUNDLE_PATH}: {e}")

    # 2. Try live refresh from TWSE + TPEX (fails silently on restricted networks)
    sources = [
        "https://openapi.twse.com.tw/v1/opendata/t187ap03_L",
        "https://www.tpex.org.tw/openapi/v1/mopsfin_t187ap03_O",
    ]
    live: dict[str, str] = {}
    for url in sources:
        try:
            r = requests.get(url, timeout=8, headers={"Accept": "application/json"})
            if r.ok:
                for item in r.json():
                    code = item.get("公司代號", "").strip()
                    name = item.get("公司簡稱", "").strip()
                    if code and name:
                        live[code] = name
        except Exception:
            pass
    if live:
        result.update(live)
        print(f"Refreshed with {len(live)} live entries from TWSE/TPEX.")

    return result


_TW_NAME_MAP = _load_tw_names()


def get_company_name(ticker: str) -> str:
    """Return Traditional Chinese name for a TW ticker, or the ticker itself."""
    code = ticker.split(".")[0]
    return _TW_NAME_MAP.get(code, ticker)


_SHEETS_SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive.readonly",
]
_SHEET_NAME = "gigi-war-room-watchlist"
_TICKER_COL = "ticker"

# Each market maps to its own worksheet tab; the US tab is auto-created on first use.
_SHEET_TABS = {
    "tw": "gigi-war-room-watchlist",
    "us": "gigi-us-watchlist",
}


def _get_sheet_tab(market: str = "tw"):
    """Return the worksheet for the given market, auto-creating the tab if needed."""
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
    """Return the ticker list for the given market from Google Sheets."""
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
    """Replace the ticker list for the given market in Google Sheets."""
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

    _NO_DATA_ROW = lambda ticker: {
        "ticker": ticker, "companyName": get_company_name(ticker),
        "close": None, "open": None, "high": None, "low": None,
        "ma20": None, "ma60": None, "ma120": None,
        "volume": None, "vol_ma20": None,
        "rsi14": None, "bias": None,
        "conds": {}, "sell_flags": {}, "signal": "NO_DATA",
    }

    for ticker in request.tickers:
        try:
            stock = yf.Ticker(ticker)
            # 1 year → ~252 trading days; need 120 (MA120) + 14 (RSI warmup) + 2 (shift)
            df = stock.history(period="1y")

            # Resolve company name: map first, yfinance info as fallback
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

            # ── Wilder RSI-14 (EWM with alpha = 1/14) ────────────────────────
            delta    = df["Close"].diff()
            gain     = delta.clip(lower=0)
            loss     = (-delta).clip(lower=0)
            avg_gain = gain.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
            avg_loss = loss.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
            safe_loss = avg_loss.copy()
            safe_loss[safe_loss == 0] = 1e-10          # avoid /0; near-100 RSI is correct
            df["RSI14"] = 100.0 - (100.0 / (1.0 + avg_gain / safe_loss))

            # ── Bias: how far Close is above/below MA20 (decimal) ────────────
            df["Bias"] = (df["Close"] - df["MA20"]) / df["MA20"]

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

            # ── Strict NaN guard — any missing indicator → NO_DATA ────────────
            if any(pd.isna(v) for v in [ma20, ma60, ma120, vol_ma20, rsi14, bias, prev_rsi]):
                row = _NO_DATA_ROW(ticker)
                if not pd.isna(c_close):
                    row["close"]  = round(float(c_close), 2)
                if not pd.isna(c_vol):
                    row["volume"] = int(c_vol)
                results.append(row)
                continue

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
                "bias":     round(float(bias) * 100, 2),  # stored as % for display
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
                "ticker": ticker, "companyName": get_company_name(ticker),  # best-effort; company_name may not be set
                "close": None, "open": None, "high": None, "low": None,
                "ma20": None, "ma60": None, "ma120": None,
                "volume": None, "vol_ma20": None,
                "rsi14": None, "bias": None,
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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
