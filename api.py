import os
import pandas as pd
import yfinance as yf
from fastapi import FastAPI, HTTPException
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

class ScanRequest(BaseModel):
    tickers: List[str]
    line_token: str = ""

def send_line_notify(message: str, token: str):
    if not token:
        return
    url = "https://notify-api.line.me/api/notify"
    headers = {"Authorization": f"Bearer {token}"}
    data = {"message": message}
    try:
        requests.post(url, headers=headers, data=data)
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
                
            latest_close = df['Close'].iloc[-1]
            latest_vol = df['Volume'].iloc[-1]
            
            df['20MA'] = df['Close'].rolling(window=20).mean()
            df['20VMA'] = df['Volume'].rolling(window=20).mean()
            
            ma20 = df['20MA'].iloc[-1]
            vma20 = df['20VMA'].iloc[-1]
            
            cond1 = latest_close > ma20
            cond2 = latest_vol > vma20
            is_buy = cond1 and cond2
            
            results.append({
                "ticker": ticker,
                "price": round(latest_close, 2),
                "ma20": round(ma20, 2),
                "volume": int(latest_vol),
                "vma20": int(vma20),
                "signal": "YES" if is_buy else "NO"
            })
            
            if is_buy:
                triggered_alerts.append(f"🚀 {ticker} 觸發買進訊號！股價:{round(latest_close,2)} > 20MA, 成交量爆發！")
                
        except Exception as e:
            print(f"Error scanning {ticker}: {e}")
            
    if triggered_alerts and request.line_token:
        alert_msg = "\n" + "\n".join(triggered_alerts)
        send_line_notify(alert_msg, request.line_token)
        
    return {"status": "success", "data": results}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
