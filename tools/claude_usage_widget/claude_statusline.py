#!/usr/bin/env python3
"""
claude_statusline.py — Claude Code 狀態列 script

這支腳本接在 Claude Code 的 statusLine hook 上執行，每次都會收到官方即時的
rate_limits（5小時 / 7天官方額度百分比 + 重置時間），印出一行狀態列文字，
同時把同一份資料寫到 ~/.gigi_claude_rate_limits.json，讓
claude_usage_server.py 網頁面板可以顯示「真正」的官方額度，而不只是本機
估算的花費。

設定方式：把以下加進 ~/.claude/settings.json（沒有 statusLine 欄位就整段加）：

{
  "statusLine": {
    "type": "command",
    "command": "python3 /Users/kurtchiang/Desktop/gigi-stock-war-room/tools/claude_usage_widget/claude_statusline.py"
  }
}

rate_limits 只有 Claude.ai Pro/Max 訂閱者、且該 session 至少呼叫過一次 API
後才會出現，欄位缺席時腳本會優雅地略過該段文字。
"""
import json
import sys
from datetime import datetime
from pathlib import Path

RATE_LIMIT_FILE = Path.home() / ".gigi_claude_rate_limits.json"


def main() -> None:
    try:
        data = json.load(sys.stdin)
    except (json.JSONDecodeError, OSError):
        print("[Claude Code]")
        return

    model = (data.get("model") or {}).get("display_name", "Claude")
    rate = data.get("rate_limits") or {}
    five_h = rate.get("five_hour") or {}
    week = rate.get("seven_day") or {}

    five_pct = five_h.get("used_percentage")
    week_pct = week.get("used_percentage")

    record = {
        "updated_at": datetime.now().astimezone().isoformat(),
        "model": model,
        "five_hour": {
            "used_percentage": five_pct,
            "resets_at": five_h.get("resets_at"),
        },
        "seven_day": {
            "used_percentage": week_pct,
            "resets_at": week.get("resets_at"),
        },
    }
    try:
        RATE_LIMIT_FILE.write_text(json.dumps(record), encoding="utf-8")
    except OSError:
        pass

    parts = []
    if five_pct is not None:
        parts.append(f"5h {five_pct:.0f}%")
    if week_pct is not None:
        parts.append(f"7d {week_pct:.0f}%")

    print(f"[{model}] | " + " · ".join(parts) if parts else f"[{model}]")


if __name__ == "__main__":
    main()
