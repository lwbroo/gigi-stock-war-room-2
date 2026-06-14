import pandas as pd
import yfinance as yf
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import requests

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _load_tw_names() -> dict[str, str]:
    """Fetch Traditional Chinese company names from TWSE and TPEX at startup."""
    result: dict[str, str] = {}
    sources = [
        "https://openapi.twse.com.tw/v1/opendata/t187ap03_L",   # TWSE listed
        "https://www.tpex.org.tw/openapi/v1/mopsfin_t187ap03_O", # OTC / TPEX listed
    ]
    for url in sources:
        try:
            r = requests.get(url, timeout=10, headers={"Accept": "application/json"})
            if r.ok:
                for item in r.json():
                    code = item.get("公司代號", "").strip()
                    name = item.get("公司簡稱", "").strip()
                    if code and name:
                        result[code] = name
        except Exception as e:
            print(f"Warning: could not load names from {url}: {e}")
    print(f"Loaded {len(result)} TW company names.")
    return result


_TW_NAME_MAP = _load_tw_names()


def get_company_name(ticker: str) -> str:
    """Return Traditional Chinese name for a TW ticker, or the ticker itself."""
    code = ticker.split(".")[0]
    return _TW_NAME_MAP.get(code, ticker)


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

    if triggered_alerts and request.line_token:
        send_line_notify("\n" + "\n".join(triggered_alerts), request.line_token)

    return {"status": "success", "data": results}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
