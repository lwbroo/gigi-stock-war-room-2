#!/usr/bin/env python3
"""
claude_usage_server.py — Claude Code 用量小面板
使用方式: python3 tools/claude_usage_widget/claude_usage_server.py
然後開啟瀏覽器: http://localhost:8899

資料來源：解析本機 ~/.claude/projects/**/*.jsonl（Claude Code 的對話紀錄），
用官方每 MTok 價格反推估算花費。這不是 Anthropic 官方的「本週剩餘額度」API
（目前沒有公開），只是本機 token 用量的估算，用來大概掌握花費趨勢。
週預算 / 5 小時預算是自己設定的參考值，不是真正的帳戶額度上限。
"""
import json
import os
import sys
import threading
import webbrowser
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import uvicorn

PORT = 8899
CONFIG_PATH = Path.home() / ".gigi_claude_usage_config.json"
PROJECTS_DIR = Path.home() / ".claude" / "projects"
RATE_LIMIT_FILE = Path.home() / ".gigi_claude_rate_limits.json"

# 每 MTok 價格 (USD)，來源：platform.claude.com/docs/en/pricing（2026-06 快照）
# cache write 假設 5 分鐘 TTL（1.25x input），cache read 為 0.1x input
PRICING: dict[str, dict[str, float]] = {
    "claude-opus-4-8": {"input": 5.00, "output": 25.00},
    "claude-opus-4-7": {"input": 5.00, "output": 25.00},
    "claude-opus-4-6": {"input": 5.00, "output": 25.00},
    "claude-opus-4-5": {"input": 5.00, "output": 25.00},
    "claude-sonnet-5": {"input": 3.00, "output": 15.00},
    "claude-sonnet-4-6": {"input": 3.00, "output": 15.00},
    "claude-sonnet-4-5": {"input": 3.00, "output": 15.00},
    "claude-haiku-4-5": {"input": 1.00, "output": 5.00},
    "claude-fable-5": {"input": 10.00, "output": 50.00},
    "claude-mythos-5": {"input": 10.00, "output": 50.00},
}
DEFAULT_PRICING = {"input": 3.00, "output": 15.00}  # 找不到型號時的保守估算
CACHE_WRITE_MULT = 1.25
CACHE_READ_MULT = 0.1

DEFAULT_CONFIG = {"weekly_budget_usd": 50.0, "session5h_budget_usd": 10.0}

app = FastAPI(title="Claude Usage Widget")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class ConfigUpdate(BaseModel):
    weekly_budget_usd: float
    session5h_budget_usd: float


def _load_config() -> dict:
    if CONFIG_PATH.exists():
        try:
            data = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
            return {**DEFAULT_CONFIG, **data}
        except (json.JSONDecodeError, OSError):
            pass
    return dict(DEFAULT_CONFIG)


def _save_config(cfg: dict) -> None:
    CONFIG_PATH.write_text(json.dumps(cfg, indent=2), encoding="utf-8")


def _load_rate_limits() -> dict | None:
    """讀取 claude_statusline.py 寫入的官方 rate_limits 快照（若有設定）。"""
    if not RATE_LIMIT_FILE.exists():
        return None
    try:
        return json.loads(RATE_LIMIT_FILE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def _price_for_model(model: str) -> dict[str, float]:
    for key in sorted(PRICING, key=len, reverse=True):
        if model.startswith(key):
            return PRICING[key]
    return DEFAULT_PRICING


def _cost_for_entry(e: dict) -> float:
    p = _price_for_model(e["model"])
    return (
        e["input_tokens"] / 1_000_000 * p["input"]
        + e["output_tokens"] / 1_000_000 * p["output"]
        + e["cache_creation_input_tokens"] / 1_000_000 * p["input"] * CACHE_WRITE_MULT
        + e["cache_read_input_tokens"] / 1_000_000 * p["input"] * CACHE_READ_MULT
    )


def _iter_usage_entries():
    if not PROJECTS_DIR.exists():
        return
    for jsonl_file in PROJECTS_DIR.rglob("*.jsonl"):
        try:
            with open(jsonl_file, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    msg = obj.get("message") or {}
                    usage = msg.get("usage")
                    ts = obj.get("timestamp")
                    if not usage or not ts:
                        continue
                    try:
                        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                    except ValueError:
                        continue
                    if dt.tzinfo is None:
                        dt = dt.replace(tzinfo=timezone.utc)
                    yield {
                        "dt": dt.astimezone(),
                        "model": msg.get("model", "unknown"),
                        "session_id": obj.get("sessionId") or jsonl_file.stem,
                        "input_tokens": usage.get("input_tokens", 0) or 0,
                        "output_tokens": usage.get("output_tokens", 0) or 0,
                        "cache_creation_input_tokens": usage.get("cache_creation_input_tokens", 0) or 0,
                        "cache_read_input_tokens": usage.get("cache_read_input_tokens", 0) or 0,
                    }
        except OSError:
            continue


def _tokens_total(e: dict) -> int:
    return (
        e["input_tokens"]
        + e["output_tokens"]
        + e["cache_creation_input_tokens"]
        + e["cache_read_input_tokens"]
    )


def _build_usage_report() -> dict:
    now = datetime.now().astimezone()
    today = now.date()
    week_start_date = today - timedelta(days=6)
    five_hours_ago = now - timedelta(hours=5)

    day_totals: dict[str, dict[str, float]] = defaultdict(lambda: {"tokens": 0, "cost": 0.0})
    day_sessions: dict[str, set] = defaultdict(set)
    model_week: dict[str, dict[str, float]] = defaultdict(lambda: {"tokens": 0, "cost": 0.0})
    week_sessions: set = set()
    today_sessions: set = set()
    window5h_tokens = 0
    window5h_cost = 0.0

    for e in _iter_usage_entries():
        d = e["dt"].date()
        if d < week_start_date:
            continue
        cost = _cost_for_entry(e)
        tokens = _tokens_total(e)
        key = d.isoformat()
        day_totals[key]["tokens"] += tokens
        day_totals[key]["cost"] += cost
        day_sessions[key].add(e["session_id"])
        week_sessions.add(e["session_id"])
        model_week[e["model"]]["tokens"] += tokens
        model_week[e["model"]]["cost"] += cost
        if d == today:
            today_sessions.add(e["session_id"])
        if e["dt"] >= five_hours_ago:
            window5h_tokens += tokens
            window5h_cost += cost

    days = []
    for i in range(6, -1, -1):
        d = today - timedelta(days=i)
        key = d.isoformat()
        entry = day_totals.get(key, {"tokens": 0, "cost": 0.0})
        days.append({
            "date": key,
            "label": d.strftime("%m/%d"),
            "tokens": int(entry["tokens"]),
            "cost": round(entry["cost"], 4),
            "sessions": len(day_sessions.get(key, set())),
        })

    week_tokens = sum(d["tokens"] for d in days)
    week_cost = round(sum(d["cost"] for d in days), 4)
    today_entry = next((d for d in days if d["date"] == today.isoformat()), {"tokens": 0, "cost": 0.0})

    models = [
        {"model": m, "tokens": int(v["tokens"]), "cost": round(v["cost"], 4)}
        for m, v in sorted(model_week.items(), key=lambda kv: kv[1]["cost"], reverse=True)
    ]

    cfg = _load_config()

    return {
        "generated_at": now.isoformat(),
        "today": {
            "date": today.isoformat(),
            "tokens": today_entry["tokens"],
            "cost": today_entry["cost"],
            "sessions": len(today_sessions),
        },
        "week": {
            "start_date": week_start_date.isoformat(),
            "tokens": week_tokens,
            "cost": week_cost,
            "sessions": len(week_sessions),
            "budget_usd": cfg["weekly_budget_usd"],
            "budget_pct": round(week_cost / cfg["weekly_budget_usd"] * 100, 1) if cfg["weekly_budget_usd"] > 0 else 0,
        },
        "window_5h": {
            "tokens": window5h_tokens,
            "cost": round(window5h_cost, 4),
            "budget_usd": cfg["session5h_budget_usd"],
            "budget_pct": round(window5h_cost / cfg["session5h_budget_usd"] * 100, 1) if cfg["session5h_budget_usd"] > 0 else 0,
        },
        "days": days,
        "models": models,
        "config": cfg,
        "projects_dir_found": PROJECTS_DIR.exists(),
        "official_rate_limits": _load_rate_limits(),
    }


@app.get("/", response_class=HTMLResponse)
async def index() -> str:
    return _HTML


@app.get("/api/usage")
async def get_usage() -> dict:
    return _build_usage_report()


@app.get("/api/config")
async def get_config() -> dict:
    return _load_config()


@app.post("/api/config")
async def update_config(cfg: ConfigUpdate) -> dict:
    data = {"weekly_budget_usd": cfg.weekly_budget_usd, "session5h_budget_usd": cfg.session5h_budget_usd}
    _save_config(data)
    return data


def _open_browser():
    import time
    time.sleep(1.2)
    webbrowser.open(f"http://localhost:{PORT}")


_HTML = """<!DOCTYPE html>
<html lang="zh-TW">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Claude Code Usage</title>
<style>
  *{box-sizing:border-box;margin:0;padding:0}
  body{background:#0d1117;color:#e6edf3;font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;padding:20px 24px;min-height:100vh;max-width:640px}
  h1{font-size:17px;font-weight:700;margin-bottom:2px;display:flex;align-items:center;gap:8px}
  .sub{font-size:11px;color:#7d8590;margin-bottom:18px}
  .disclaimer{font-size:11px;color:#9e6a03;background:#2b2110;border:1px solid #4d3a10;border-radius:6px;padding:8px 10px;margin-bottom:18px;line-height:1.5}
  .grid{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-bottom:18px}
  .tile{background:#161b22;border:1px solid #21262d;border-radius:8px;padding:12px 14px}
  .tile-label{font-size:11px;color:#7d8590;text-transform:uppercase;letter-spacing:.5px;margin-bottom:6px}
  .tile-value{font-size:22px;font-weight:700}
  .tile-sub{font-size:11px;color:#7d8590;margin-top:4px}
  .bar-bg{background:#21262d;border-radius:4px;height:6px;margin-top:8px;overflow:hidden}
  .bar-fill{height:100%;border-radius:4px;transition:width .3s}
  .ok{background:#3fb950}.warn{background:#e3b341}.over{background:#f85149}
  .section{margin-bottom:18px}
  .section-label{font-size:11px;text-transform:uppercase;letter-spacing:.8px;color:#7d8590;margin-bottom:10px;font-weight:600}
  .chart{display:flex;align-items:flex-end;gap:6px;height:90px;background:#161b22;border:1px solid #21262d;border-radius:8px;padding:10px}
  .chart-col{flex:1;display:flex;flex-direction:column;align-items:center;justify-content:flex-end;height:100%}
  .chart-bar{width:100%;background:#1f6feb;border-radius:3px 3px 0 0;min-height:2px}
  .chart-label{font-size:9px;color:#7d8590;margin-top:4px}
  table{width:100%;border-collapse:collapse;font-size:12px}
  th,td{text-align:left;padding:6px 8px;border-bottom:1px solid #21262d}
  th{color:#7d8590;font-weight:600;font-size:11px;text-transform:uppercase}
  .settings{background:#161b22;border:1px solid #21262d;border-radius:8px;padding:12px 14px;display:flex;gap:10px;align-items:flex-end;flex-wrap:wrap}
  .field{display:flex;flex-direction:column;gap:4px}
  .field label{font-size:11px;color:#7d8590}
  .field input{background:#0d1117;border:1px solid #30363d;border-radius:6px;color:#e6edf3;padding:6px 8px;width:100px;font-size:13px}
  .btn{padding:6px 14px;border:none;border-radius:6px;cursor:pointer;font-size:12px;font-weight:600;background:#1f6feb;color:#fff}
  .btn:hover{opacity:.85}
  .empty-note{font-size:12px;color:#f85149;margin-bottom:14px}
  .official-tile{background:#0d2818;border:1px solid #1f6feb33;border-radius:8px;padding:12px 14px;margin-bottom:8px}
  .official-badge{display:inline-block;font-size:10px;font-weight:700;color:#3fb950;background:#0d2818;border:1px solid #235c3a;border-radius:4px;padding:1px 6px;margin-bottom:8px;letter-spacing:.4px}
  .official-row{display:flex;justify-content:space-between;align-items:baseline;margin-bottom:4px}
  .official-row .name{font-size:12px;color:#9aa5b1}
  .official-row .pct{font-size:15px;font-weight:700}
  .official-reset{font-size:11px;color:#7d8590;margin-top:2px}
  .setup-hint{background:#161b22;border:1px dashed #30363d;border-radius:8px;padding:12px 14px;margin-bottom:18px;font-size:12px;color:#9aa5b1;line-height:1.6}
  .setup-hint code{background:#0d1117;border:1px solid #30363d;border-radius:4px;padding:1px 5px;font-size:11px}
</style>
</head>
<body>

<h1>📊 Claude Code Usage</h1>
<div class="sub" id="updated">載入中...</div>

<div class="section" id="official-section" style="display:none">
  <div class="official-badge">✅ 官方即時額度</div>
  <div class="official-tile">
    <div class="official-row"><span class="name">5 小時額度</span><span class="pct" id="off-5h-pct">–</span></div>
    <div class="bar-bg"><div class="bar-fill" id="off-5h-bar"></div></div>
    <div class="official-reset" id="off-5h-reset"></div>
  </div>
  <div class="official-tile">
    <div class="official-row"><span class="name">7 天（本週）額度</span><span class="pct" id="off-7d-pct">–</span></div>
    <div class="bar-bg"><div class="bar-fill" id="off-7d-bar"></div></div>
    <div class="official-reset" id="off-7d-reset"></div>
  </div>
</div>
<div class="setup-hint" id="setup-hint">
  💡 目前看不到官方即時額度。設定 statusline 就能顯示真正的 5小時/7天額度百分比：
  在 <code>~/.claude/settings.json</code> 加上
  <code>"statusLine": {"type": "command", "command": "python3 .../tools/claude_usage_widget/claude_statusline.py"}</code>，
  之後在 Claude Code 裡跑一輪對話，這個頁面重新整理就會出現。
</div>

<div class="disclaimer">
  ⚠️ 下面這些是本機解析 ~/.claude/projects 對話紀錄、用官方牌價反推的<b>估算</b>花費，
  跟上面的官方額度不同，只是用來大概掌握花費趨勢。
  「週預算」「5小時預算」是你自己設定的參考值，不是帳戶真正的上限。
</div>
<div id="empty-note" class="empty-note" style="display:none">
  找不到 ~/.claude/projects 目錄，可能這台機器上還沒有 Claude Code 對話紀錄。
</div>

<div class="grid">
  <div class="tile">
    <div class="tile-label">今天</div>
    <div class="tile-value" id="today-cost">$0.00</div>
    <div class="tile-sub" id="today-sub">0 tokens · 0 sessions</div>
  </div>
  <div class="tile">
    <div class="tile-label">最近 5 小時</div>
    <div class="tile-value" id="w5h-cost">$0.00</div>
    <div class="tile-sub" id="w5h-sub">0 tokens</div>
    <div class="bar-bg"><div class="bar-fill" id="w5h-bar"></div></div>
  </div>
</div>

<div class="tile" style="margin-bottom:18px">
  <div class="tile-label">本週（近 7 天）</div>
  <div class="tile-value" id="week-cost">$0.00</div>
  <div class="tile-sub" id="week-sub">0 tokens · 0 sessions</div>
  <div class="bar-bg"><div class="bar-fill" id="week-bar"></div></div>
</div>

<div class="section">
  <div class="section-label">📅 每日花費</div>
  <div class="chart" id="chart"></div>
</div>

<div class="section">
  <div class="section-label">🧩 本週各模型花費</div>
  <table>
    <thead><tr><th>模型</th><th>Tokens</th><th>估算花費</th></tr></thead>
    <tbody id="model-table"></tbody>
  </table>
</div>

<div class="section">
  <div class="section-label">⚙️ 預算設定（僅本機參考用）</div>
  <div class="settings">
    <div class="field">
      <label>週預算 (USD)</label>
      <input type="number" id="cfg-weekly" step="1" min="0">
    </div>
    <div class="field">
      <label>5小時預算 (USD)</label>
      <input type="number" id="cfg-5h" step="0.5" min="0">
    </div>
    <button class="btn" onclick="saveConfig()">儲存</button>
  </div>
</div>

<script>
function barClass(pct) {
  if (pct >= 100) return 'over';
  if (pct >= 75) return 'warn';
  return 'ok';
}

function fmtUSD(n) { return '$' + n.toFixed(2); }

function fmtResetCountdown(epochSeconds) {
  if (!epochSeconds) return '';
  const ms = epochSeconds * 1000 - Date.now();
  if (ms <= 0) return '即將重置';
  const hours = Math.floor(ms / 3600000);
  const mins = Math.floor((ms % 3600000) / 60000);
  const when = new Date(epochSeconds * 1000).toLocaleString('zh-TW', {month: 'numeric', day: 'numeric', hour: '2-digit', minute: '2-digit'});
  if (hours >= 24) {
    const days = Math.floor(hours / 24);
    return `${when} 重置（還有約 ${days} 天）`;
  }
  return `${when} 重置（還有約 ${hours}h ${mins}m）`;
}

function renderOfficialWindow(prefix, windowData) {
  const pct = windowData && windowData.used_percentage;
  if (pct === null || pct === undefined) {
    document.getElementById(prefix + '-pct').textContent = '–';
    document.getElementById(prefix + '-bar').style.width = '0%';
    document.getElementById(prefix + '-reset').textContent = '';
    return;
  }
  document.getElementById(prefix + '-pct').textContent = pct.toFixed(0) + '%';
  document.getElementById(prefix + '-bar').style.width = Math.min(pct, 100) + '%';
  document.getElementById(prefix + '-bar').className = 'bar-fill ' + barClass(pct);
  document.getElementById(prefix + '-reset').textContent = fmtResetCountdown(windowData.resets_at);
}

async function refresh() {
  const res = await fetch('/api/usage');
  const d = await res.json();

  document.getElementById('updated').textContent =
    '更新於 ' + new Date(d.generated_at).toLocaleTimeString('zh-TW');
  document.getElementById('empty-note').style.display = d.projects_dir_found ? 'none' : 'block';

  const official = d.official_rate_limits;
  if (official) {
    document.getElementById('official-section').style.display = 'block';
    document.getElementById('setup-hint').style.display = 'none';
    renderOfficialWindow('off-5h', official.five_hour);
    renderOfficialWindow('off-7d', official.seven_day);
  } else {
    document.getElementById('official-section').style.display = 'none';
    document.getElementById('setup-hint').style.display = 'block';
  }

  document.getElementById('today-cost').textContent = fmtUSD(d.today.cost);
  document.getElementById('today-sub').textContent =
    d.today.tokens.toLocaleString() + ' tokens · ' + d.today.sessions + ' sessions';

  document.getElementById('w5h-cost').textContent = fmtUSD(d.window_5h.cost);
  document.getElementById('w5h-sub').textContent = d.window_5h.tokens.toLocaleString() + ' tokens';
  const w5hPct = Math.min(d.window_5h.budget_pct, 100);
  document.getElementById('w5h-bar').style.width = w5hPct + '%';
  document.getElementById('w5h-bar').className = 'bar-fill ' + barClass(d.window_5h.budget_pct);

  document.getElementById('week-cost').textContent = fmtUSD(d.week.cost);
  document.getElementById('week-sub').textContent =
    d.week.tokens.toLocaleString() + ' tokens · ' + d.week.sessions + ' sessions';
  const weekPct = Math.min(d.week.budget_pct, 100);
  document.getElementById('week-bar').style.width = weekPct + '%';
  document.getElementById('week-bar').className = 'bar-fill ' + barClass(d.week.budget_pct);

  const maxCost = Math.max(...d.days.map(x => x.cost), 0.01);
  const chart = document.getElementById('chart');
  chart.innerHTML = '';
  d.days.forEach(day => {
    const col = document.createElement('div');
    col.className = 'chart-col';
    const bar = document.createElement('div');
    bar.className = 'chart-bar';
    bar.style.height = Math.max(2, (day.cost / maxCost) * 100) + '%';
    bar.title = day.date + ': ' + fmtUSD(day.cost);
    const label = document.createElement('div');
    label.className = 'chart-label';
    label.textContent = day.label;
    col.appendChild(bar);
    col.appendChild(label);
    chart.appendChild(col);
  });

  const tbody = document.getElementById('model-table');
  tbody.innerHTML = '';
  if (d.models.length === 0) {
    tbody.innerHTML = '<tr><td colspan="3" style="color:#7d8590">本週尚無資料</td></tr>';
  }
  d.models.forEach(m => {
    const tr = document.createElement('tr');
    tr.innerHTML = '<td>' + m.model + '</td><td>' + m.tokens.toLocaleString() +
      '</td><td>' + fmtUSD(m.cost) + '</td>';
    tbody.appendChild(tr);
  });

  document.getElementById('cfg-weekly').value = d.config.weekly_budget_usd;
  document.getElementById('cfg-5h').value = d.config.session5h_budget_usd;
}

async function saveConfig() {
  const weekly_budget_usd = parseFloat(document.getElementById('cfg-weekly').value) || 0;
  const session5h_budget_usd = parseFloat(document.getElementById('cfg-5h').value) || 0;
  await fetch('/api/config', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({weekly_budget_usd, session5h_budget_usd}),
  });
  refresh();
}

refresh();
setInterval(refresh, 30000);
</script>
</body>
</html>"""


if __name__ == "__main__":
    threading.Thread(target=_open_browser, daemon=True).start()
    print("📊 Claude Code Usage Widget")
    print(f"   http://localhost:{PORT}")
    print("   Ctrl-C 停止\n")
    uvicorn.run(app, host="127.0.0.1", port=PORT, log_level="warning")
