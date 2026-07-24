#!/usr/bin/env python3
"""
market_warning_notify.py — 每日盤前大盤燈號 + 進場評等 Telegram 推播 (v11.4)

每天 07:40（Mon-Fri）由 crontab 執行：
  40  7 * * 1-5 cd /path/to/repo && python3 market_warning_notify.py >> /tmp/gigi_warning.log 2>&1

流程：呼叫雲端 /api/market-warning（五年邏輯斯回歸模型，單一真相來源）
      + /api/night-futures（期交所夜盤）→ 組成盤前簡訊 → Telegram。

資料日期說明：燈號預測的是「下一個台股交易日」。07:40 執行時用的是
昨晚（美國時間昨日）收盤資料，正是今天台股的盤前預警。
"""
import os
import sys
import requests
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

TG_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TG_CHAT_ID   = os.environ.get("TELEGRAM_CHAT_ID", "")
API_BASE     = os.environ.get("GIGI_API_BASE", "https://gigi-stock-war-room-2.onrender.com")

LIGHT_EMOJI = {"紅燈": "🔴", "黃燈": "🟡", "綠燈": "🟢"}
GRADE_EMOJI = {"X": "⛔", "A": "🚀", "B": "✅", "C": "⚖️"}


def send_telegram(msg: str):
    if not TG_BOT_TOKEN or not TG_CHAT_ID:
        print("Telegram 憑證未設定，僅輸出到 stdout")
        print(msg)
        return
    try:
        r = requests.post(
            f"https://api.telegram.org/bot{TG_BOT_TOKEN}/sendMessage",
            json={"chat_id": TG_CHAT_ID, "text": msg, "parse_mode": "HTML"},
            timeout=10,
        )
        print(f"Telegram: HTTP {r.status_code}")
    except Exception as e:
        print(f"Telegram: {e}")


def main():
    warn = requests.get(f"{API_BASE}/api/market-warning", timeout=30).json()
    if warn.get("error") or not warn.get("light"):
        send_telegram(f"⚠️ 大盤燈號取得失敗：{warn.get('error', '未知錯誤')}")
        sys.exit(1)

    f = warn.get("features", {})
    light = warn["light"]
    grade = warn.get("entry_grade", "C")
    prob_pct = round((warn.get("prob") or 0) * 100, 1)

    lines = [
        f"{LIGHT_EMOJI.get(light, '⚪')} <b>台股盤前燈號：{light}</b>　{GRADE_EMOJI.get(grade, '')} <b>{warn.get('entry_grade_label', '')}</b>",
        f"大跌機率 <b>{prob_pct}%</b>（基準 8.3%）· 資料日 {warn.get('asof', '—')}",
        "",
        f"📋 {warn.get('today_action', warn.get('advice', ''))}",
        "",
        f"📊 依據：昨晚美股 {f.get('sp_ret')}% · VIX {f.get('vix')}（Δ{f.get('vix_chg')}）"
        f" · 台股5日動能 {f.get('tw_mom5')}% · 距20日高點 {f.get('tw_dd20')}%",
    ]

    # 夜盤資訊（期交所日報，可能有一日落差，附資料日期）
    try:
        nf = requests.get(f"{API_BASE}/api/night-futures", timeout=20).json()
        tx = nf.get("tx")
        if tx:
            chg = tx.get("night_change") or 0
            arrow = "▲" if chg >= 0 else "▼"
            basis = tx.get("basis")
            basis_txt = f"正價差 +{basis}" if (basis or 0) >= 0 else f"逆價差 {basis}"
            lines.append(
                f"🌙 夜盤（{tx.get('session_date')}）：{int(tx.get('night_last', 0)):,} "
                f"{arrow}{abs(chg)}（{tx.get('night_pct')}）· {basis_txt}"
            )
    except Exception:
        pass

    lines.append("")
    lines.append("<i>五年回測：紅燈日 81% 次日下跌；A 級日 92.5% 勝率。僅供參考，非投資建議。</i>")
    send_telegram("\n".join(lines))


if __name__ == "__main__":
    main()
