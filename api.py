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

    for ticker in request.tickers:
        try:
            stock = yf.Ticker(ticker)
            df = stock.history(period="60d")
            if df.empty or len(df) < 20:
                results.append({
                    "ticker": ticker,
                    "companyName": get_company_name(ticker),
                    "close": None,
                    "ma20": None,
                    "volume": None,
                    "vol_ma20": None,
                    "signal": "NO_DATA",
                })
                continue

            latest_close = df["Close"].iloc[-1]
            latest_vol = df["Volume"].iloc[-1]

            df["20MA"] = df["Close"].rolling(window=20).mean()
            df["20VMA"] = df["Volume"].rolling(window=20).mean()

            ma20 = df["20MA"].iloc[-1]
            vma20 = df["20VMA"].iloc[-1]

            cond1 = latest_close > ma20
            cond2 = latest_vol > vma20
            is_buy = cond1 and cond2

            results.append({
                "ticker": ticker,
                "companyName": get_company_name(ticker),
                "close": round(float(latest_close), 2),
                "ma20": round(float(ma20), 2),
                "volume": int(latest_vol),
                "vol_ma20": int(vma20),
                "signal": "YES" if is_buy else "NO",
            })

            if is_buy:
                triggered_alerts.append(
                    f"🚀 {ticker} 觸發買進訊號！股價:{round(float(latest_close), 2)} > 20MA, 成交量爆發！"
                )

        except Exception as e:
            print(f"Error scanning {ticker}: {e}")
            results.append({
                "ticker": ticker,
                "companyName": get_company_name(ticker),
                "close": None,
                "ma20": None,
                "volume": None,
                "vol_ma20": None,
                "signal": "ERROR",
            })

    if triggered_alerts and request.line_token:
        send_line_notify("\n" + "\n".join(triggered_alerts), request.line_token)

    return {"status": "success", "data": results}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
