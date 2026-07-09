#!/usr/bin/env python3
"""
market_risk_local.py — v11.0 大盤大跌/大漲 偵測 + 風險情境警示

兩種訊號，誠實分開標示，不誇大：
  1. 「確認訊號」— 當天大盤實際漲跌幅 >= 閾值，收盤後立即推播。
     可靠、簡單，但是「當天已發生」的確認，不是預測。
  2. 「風險情境」— 法人資金流／VIX 相對自己近期均值的統計離群值，
     推播用詞刻意保守（"風險略升"），因為研究過 2026-07-07 台股重挫的
     實際資料後發現：外資賣超與當天跌幅大多是同一天發生，沒有乾淨的
     「連續兩天賣超→隔天崩盤」領先關係（07-03 當週最大單日賣超那天
     指數幾乎打平；真正下跌的 07-07 賣超金額反而較小）。所以這只當作
     「風險升高的統計脈絡」，不是「明天會崩盤」的預測。

用法:
  python3 market_risk_local.py --market tw
  python3 market_risk_local.py --market us
  python3 market_risk_local.py --market tw --dry-run
"""
import argparse
import datetime as _dt
import os
import statistics

import requests
import yfinance as yf
from dotenv import load_dotenv

import simulate as sim  # reuse _get_or_create_tab

load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

TG_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TG_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")
FINMIND_TOKEN = os.environ.get("FINMIND_TOKEN", "")
FINMIND_BASE = "https://api.finmindtrade.com/api/v4/data"

MOVE_THRESHOLD_PCT = 2.0        # 確認訊號：大盤單日漲跌幅門檻
FLOW_LOOKBACK_DAYS = 20         # 風險情境：法人流量/VIX 計算均值+標準差用的天數
FLOW_Z_THRESHOLD = 1.0          # 風險情境：偏離均值幾個標準差算「風險略升」

_MARKET_FLOW_TAB = "market_flow"
_MARKET_FLOW_HDR = ["date", "market", "index_close", "index_pct_chg",
                     "flow_or_vix", "flow_or_vix_z", "note"]


def send_telegram(msg: str):
    if not TG_BOT_TOKEN or not TG_CHAT_ID:
        print("  [Telegram 未設定，僅印出]")
        print(msg)
        return
    try:
        requests.post(
            f"https://api.telegram.org/bot{TG_BOT_TOKEN}/sendMessage",
            json={"chat_id": TG_CHAT_ID, "text": msg, "parse_mode": "HTML"},
            timeout=10,
        )
        print("  ↑ Telegram 已推播")
    except Exception as e:
        print(f"  Telegram 推播失敗: {e}")


def _upsert_flow_row(ws, row: list):
    date_key, market_key = row[0], row[1]
    all_rows = ws.get_all_values()
    for i, r in enumerate(all_rows[1:], 2):
        if len(r) >= 2 and r[0] == date_key and r[1] == market_key:
            ws.update(f"A{i}:G{i}", [row])
            return
    ws.append_rows([row], value_input_option="RAW")


def _history(ws, market: str, exclude_date: str, n: int) -> list:
    rows = [r for r in ws.get_all_values()[1:]
            if len(r) >= 5 and r[1] == market and r[0] != exclude_date]
    rows.sort(key=lambda r: r[0])
    return rows[-n:]


# ── TW ───────────────────────────────────────────────────────────────────────

def _tw_index_close(as_of: str):
    """Return (actual_date, close, prev_close) for the most recent TAIEX close on/before as_of."""
    r = requests.get(FINMIND_BASE, params={
        "dataset": "TaiwanStockPrice", "data_id": "TAIEX",
        "start_date": (_dt.date.fromisoformat(as_of) - _dt.timedelta(days=15)).isoformat(),
        "end_date": as_of, "token": FINMIND_TOKEN,
    }, timeout=20)
    data = r.json().get("data", [])
    if len(data) < 2:
        return None
    data.sort(key=lambda x: x["date"])
    return data[-1]["date"], float(data[-1]["close"]), float(data[-2]["close"])


def _tw_foreign_flow(date_str: str):
    ds = date_str.replace("-", "")
    r = requests.get(
        f"https://www.twse.com.tw/rwd/zh/fund/BFI82U?response=json&dayDate={ds}&type=day",
        timeout=12, headers={"User-Agent": "Mozilla/5.0"},
    )
    data = r.json()
    if data.get("stat") != "OK":
        return None
    for row in data.get("data", []):
        if row[0].strip() == "外資及陸資(不含外資自營商)":
            return int(row[3].replace(",", "")) / 1e8  # 億元
    return None


def check_tw(dry_run: bool):
    today = _dt.date.today().isoformat()
    result = _tw_index_close(today)
    if not result:
        print("找不到近期 TWII 收盤資料，跳過")
        return
    date_str, close, prev_close = result
    pct_chg = round((close - prev_close) / prev_close * 100, 2)
    flow = _tw_foreign_flow(date_str)
    print(f"[TW {date_str}] TAIEX={close:.0f} ({pct_chg:+.2f}%)  外資淨額={flow:+.0f}億" if flow is not None
          else f"[TW {date_str}] TAIEX={close:.0f} ({pct_chg:+.2f}%)  外資資料無法取得")

    ws = sim._get_or_create_tab(_MARKET_FLOW_TAB, _MARKET_FLOW_HDR)
    hist = _history(ws, "tw", date_str, FLOW_LOOKBACK_DAYS)

    # ── 1. 確認訊號：今日已發生的大跌/大漲 ──
    if abs(pct_chg) >= MOVE_THRESHOLD_PCT:
        direction = "大跌" if pct_chg < 0 else "大漲"
        msg = (f"⚠️ <b>台股今日{direction} {pct_chg:+.2f}%</b>\n"
               f"TAIEX 收盤 {close:,.0f}\n"
               + (f"外資淨{'賣超' if flow < 0 else '買超'} {abs(flow):.0f}億\n" if flow is not None else "")
               + f"📅 {date_str}\n請留意明日開盤走勢與持倉風險。")
        print("  🚨 確認訊號觸發")
        if not dry_run:
            send_telegram(msg)
    else:
        print(f"  今日漲跌 {pct_chg:+.2f}% 未達 ±{MOVE_THRESHOLD_PCT}% 門檻")

    # ── 2. 風險情境：外資近3日賣超 相對自己近期均值的離群程度 ──
    if flow is not None and len(hist) >= 10:
        recent_flows = [float(r[4]) for r in hist if r[4]]
        cum3d_hist = [sum(recent_flows[max(0, i - 2):i + 1]) for i in range(2, len(recent_flows))]
        if len(cum3d_hist) >= 5:
            mean3d = statistics.mean(cum3d_hist)
            std3d = statistics.stdev(cum3d_hist) if len(cum3d_hist) > 1 else 1.0
            last3 = [flow] + recent_flows[-2:]
            cum3d_now = sum(last3)
            z = (cum3d_now - mean3d) / std3d if std3d > 0 else 0.0
            print(f"  近3日外資合計 {cum3d_now:+.0f}億  (近{len(cum3d_hist)}期均值 {mean3d:+.0f}億, z={z:.2f})")
            if z <= -FLOW_Z_THRESHOLD and cum3d_now < 0:
                msg = (f"📊 <b>外資近3日賣超轉強</b>\n"
                       f"合計 {cum3d_now:+.0f}億（高於近期平均賣壓約 {abs(z):.1f} 個標準差）\n"
                       f"⚠️ 這是風險情境提示，不是崩盤預測 — 僅供留意，非確認訊號。\n"
                       f"📅 {date_str}")
                print("  📊 風險情境觸發")
                if not dry_run:
                    send_telegram(msg)
    else:
        print(f"  歷史資料不足（{len(hist)}/10），暫不計算風險情境")

    if flow is not None and not dry_run:
        note = "confirmed_move" if abs(pct_chg) >= MOVE_THRESHOLD_PCT else ""
        _upsert_flow_row(ws, [date_str, "tw", close, pct_chg, flow, "", note])


# ── US ───────────────────────────────────────────────────────────────────────

def check_us(dry_run: bool):
    spx = yf.Ticker("^GSPC").history(period="10d")
    vix = yf.Ticker("^VIX").history(period=f"{FLOW_LOOKBACK_DAYS + 5}d")
    if spx.empty or vix.empty:
        print("找不到近期 SPX/VIX 資料，跳過")
        return

    date_str = spx.index[-1].date().isoformat()
    close = float(spx["Close"].iloc[-1])
    prev_close = float(spx["Close"].iloc[-2])
    pct_chg = round((close - prev_close) / prev_close * 100, 2)
    vix_now = float(vix["Close"].iloc[-1])

    print(f"[US {date_str}] SPX={close:.0f} ({pct_chg:+.2f}%)  VIX={vix_now:.1f}")

    if abs(pct_chg) >= MOVE_THRESHOLD_PCT:
        direction = "大跌" if pct_chg < 0 else "大漲"
        msg = (f"⚠️ <b>美股今日{direction} {pct_chg:+.2f}%</b>\n"
               f"S&P500 收盤 {close:,.0f}\nVIX {vix_now:.1f}\n"
               f"📅 {date_str}\n請留意明日開盤走勢與持倉風險。")
        print("  🚨 確認訊號觸發")
        if not dry_run:
            send_telegram(msg)
    else:
        print(f"  今日漲跌 {pct_chg:+.2f}% 未達 ±{MOVE_THRESHOLD_PCT}% 門檻")

    vix_hist = vix["Close"].iloc[:-1].tail(FLOW_LOOKBACK_DAYS).tolist()
    if len(vix_hist) >= 10:
        mean_vix = statistics.mean(vix_hist)
        std_vix = statistics.stdev(vix_hist)
        z = (vix_now - mean_vix) / std_vix if std_vix > 0 else 0.0
        print(f"  VIX={vix_now:.1f}  近{len(vix_hist)}日均值={mean_vix:.1f}  z={z:.2f}")
        if z >= FLOW_Z_THRESHOLD:
            msg = (f"📊 <b>VIX 急升</b>\n"
                   f"目前 {vix_now:.1f}（高於近期均值約 {z:.1f} 個標準差）\n"
                   f"⚠️ 這是風險情境提示，不是崩盤預測 — 僅供留意，非確認訊號。\n"
                   f"📅 {date_str}")
            print("  📊 風險情境觸發")
            if not dry_run:
                send_telegram(msg)
    else:
        print(f"  歷史資料不足（{len(vix_hist)}/10），暫不計算風險情境")

    if not dry_run:
        ws = sim._get_or_create_tab(_MARKET_FLOW_TAB, _MARKET_FLOW_HDR)
        note = "confirmed_move" if abs(pct_chg) >= MOVE_THRESHOLD_PCT else ""
        _upsert_flow_row(ws, [date_str, "us", close, pct_chg, vix_now, "", note])


def main():
    ap = argparse.ArgumentParser(description="大盤大跌/大漲偵測 + 風險情境警示")
    ap.add_argument("--market", required=True, choices=["tw", "us"])
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if args.market == "tw":
        check_tw(args.dry_run)
    else:
        check_us(args.dry_run)


if __name__ == "__main__":
    main()
