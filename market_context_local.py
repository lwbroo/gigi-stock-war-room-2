#!/usr/bin/env python3
"""
market_context_local.py — v11.0 本機 VIX + Grok 市場情緒更新器

取代雲端 /api/vix 即時抓取 + /api/sentiment 排程呼叫 Grok。
本機每天跑一次，寫入 market_context tab；雲端只讀這個 cache。

需要 GROK_API_KEY（.env）才能做情緒分析；沒有的話只更新 VIX，情緒欄位留空，
雲端 /api/sentiment 會 fallback 回自己即時呼叫 Grok（如果 Render 上有設 key）。

用法:
  python3 market_context_local.py
  python3 market_context_local.py --dry-run
"""
import argparse
import datetime as _dt
import json
import os

import requests

import simulate as sim  # reuse _get_gc / _get_or_create_tab

GROK_API_KEY = os.environ.get("GROK_API_KEY", "")
_TW_TZ = _dt.timezone(_dt.timedelta(hours=8))

_MARKET_CONTEXT_TAB = "market_context"
_MARKET_CONTEXT_HDR = ["date", "vix", "vix_status", "vix_score",
                        "sentiment_score", "key_reason", "target_sectors", "fetched_at"]


def _get_vix() -> float:
    r = requests.get(
        "https://query1.finance.yahoo.com/v8/finance/chart/%5EVIX",
        headers={"User-Agent": "Mozilla/5.0"}, timeout=10,
    )
    data = r.json()
    return float(data["chart"]["result"][0]["meta"]["regularMarketPrice"])


def _vix_status(level: float):
    if level > 30:   return "極度恐慌", -20
    elif level > 25: return "恐慌", -10
    elif level > 20: return "警戒", -3
    elif level > 15: return "中性", 0
    else:            return "樂觀", 5


def _fetch_news_text() -> str:
    items, seen = [], set()
    for sym in ["SPY", "QQQ", "NVDA", "AAPL", "MSFT"]:
        try:
            r = requests.get(
                f"https://query1.finance.yahoo.com/v1/finance/search?q={sym}&newsCount=4",
                headers={"User-Agent": "Mozilla/5.0"}, timeout=8,
            )
            for n in r.json().get("news", [])[:4]:
                title = n.get("title", "").strip()
                if not title or title in seen:
                    continue
                seen.add(title)
                items.append(f"標題: {title}")
        except Exception:
            pass
    if not items:
        return f"今天是 {_dt.date.today().isoformat()}，請根據你的最新知識對當前全球金融市場情緒做出評估。"
    return "\n".join(items[:10])


def _analyze_with_grok(news_text: str) -> dict:
    system_prompt = (
        "你是一位頂尖的量化交易員。請評估以下最新財經新聞，為今日市場情緒打分。"
        "嚴格回傳標準 JSON，包含三個欄位："
        "1. 'sentiment_score': 浮點數，範圍 -1.0（極度悲觀）到 +1.0（極度樂觀）。"
        "2. 'key_reason': 繁體中文，一句話說明核心原因。"
        "3. 'target_sectors': 受影響最大的產業板塊陣列，如 ['半導體', 'AI']。"
        "不要包含任何 markdown 語法，直接輸出純 JSON 字串。"
    )
    resp = requests.post(
        "https://api.x.ai/v1/chat/completions",
        headers={"Authorization": f"Bearer {GROK_API_KEY}", "Content-Type": "application/json"},
        json={"model": "grok-3-mini-beta", "temperature": 0.0, "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"今日新聞：\n{news_text}"},
        ]},
        timeout=30,
    )
    raw = resp.json()["choices"][0]["message"]["content"]
    try:
        return json.loads(raw)
    except Exception:
        return {"sentiment_score": 0.0, "key_reason": "解析失敗，預設中立", "target_sectors": []}


def _upsert_today(ws, row: list):
    today = row[0]
    all_rows = ws.get_all_values()
    for i, r in enumerate(all_rows[1:], 2):
        if r and r[0] == today:
            ws.update(f"A{i}:H{i}", [row])
            return "updated"
    ws.append_rows([row], value_input_option="RAW")
    return "appended"


def main():
    ap = argparse.ArgumentParser(description="Refresh VIX + Grok sentiment, cache to GSheets")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    today = _dt.datetime.now(_TW_TZ).date().isoformat()
    fetched_at = _dt.datetime.now(_TW_TZ).isoformat()

    print("[1] Fetching VIX...")
    try:
        vix = _get_vix()
        vix_status, vix_score = _vix_status(vix)
        print(f"    VIX={vix} ({vix_status}, {vix_score:+d})")
    except Exception as e:
        print(f"    VIX fetch failed: {e}")
        vix, vix_status, vix_score = None, "未知", 0

    sentiment_score, key_reason, target_sectors = "", "", ""
    if GROK_API_KEY:
        print("[2] Fetching news + Grok sentiment...")
        try:
            news = _fetch_news_text()
            result = _analyze_with_grok(news)
            sentiment_score = result.get("sentiment_score", 0.0)
            key_reason = result.get("key_reason", "")
            target_sectors = json.dumps(result.get("target_sectors", []), ensure_ascii=False)
            print(f"    sentiment={sentiment_score}  reason={key_reason}")
        except Exception as e:
            print(f"    Grok sentiment failed: {e}")
    else:
        print("[2] GROK_API_KEY not set locally — skipping sentiment (cloud /api/sentiment will fallback)")

    row = [today, vix if vix is not None else "", vix_status, vix_score,
           sentiment_score, key_reason, target_sectors, fetched_at]

    if args.dry_run:
        print("[dry-run] Would write:", row)
        return

    ws = sim._get_or_create_tab(_MARKET_CONTEXT_TAB, _MARKET_CONTEXT_HDR)
    action = _upsert_today(ws, row)
    print(f"[3] market_context row {action} for {today}")


if __name__ == "__main__":
    main()
