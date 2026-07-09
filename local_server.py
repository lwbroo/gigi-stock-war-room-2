#!/usr/bin/env python3
"""
local_server.py — Gigi Stock War Room 本機控制台
使用方式: python3 local_server.py
然後開啟瀏覽器: http://localhost:8888
"""
import asyncio
import os
import sys
import webbrowser
import threading

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
import uvicorn

PYTHON  = sys.executable
WORKDIR = os.path.dirname(os.path.abspath(__file__))
PORT    = 8888

app = FastAPI(title="Gigi Local Control")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# Each action: (script_args, display_label)
ACTIONS: dict[str, tuple[list[str], str]] = {
    "scan_tw":     (["scan_local.py", "--market", "tw",   "--notify"],              "台股掃描"),
    "scan_us":     (["scan_local.py", "--market", "us",   "--notify"],              "美股掃描"),
    "scan_both":   (["scan_local.py", "--market", "both", "--notify"],              "全部掃描"),
    "scan_tw_dry": (["scan_local.py", "--market", "tw",   "--dry-run"],             "台股 Dry Run (不推送)"),
    "opt_tw":      (["simulate.py", "--mode", "optimize", "--market", "tw", "--years", "5"], "台股優化參數"),
    "opt_us":      (["simulate.py", "--mode", "optimize", "--market", "us", "--years", "5"], "美股優化參數"),
    "paper_tw":    (["simulate.py", "--mode", "paper",    "--market", "tw", "--years", "5"], "台股 Paper 回測"),
    "paper_us":    (["simulate.py", "--mode", "paper",    "--market", "us", "--years", "5"], "美股 Paper 回測"),
    "both_tw":     (["simulate.py", "--mode", "both",     "--market", "tw", "--years", "5"], "台股 優化 + 回測"),
    "both_us":     (["simulate.py", "--mode", "both",     "--market", "us", "--years", "5"], "美股 優化 + 回測"),
    "auto_tw":     (["auto_optimize.py", "--market", "tw", "--target", "72", "--rounds", "8", "--years", "5"], "台股 自動優化 (目標72%)"),
    "auto_us":     (["auto_optimize.py", "--market", "us", "--target", "70", "--rounds", "8", "--years", "5"], "美股 自動優化 (目標70%)"),
    "auto_both":   (["auto_optimize.py", "--market", "both", "--target", "72", "--rounds", "8", "--years", "5"], "全市場 自動優化"),
    "regression":  (["regression_train_local.py"],                                    "OLS 回歸訓練 (僅台股)"),
    "risk_tw":     (["market_risk_local.py", "--market", "tw"],                        "台股 大盤風險檢查"),
    "risk_us":     (["market_risk_local.py", "--market", "us"],                        "美股 大盤風險檢查"),
}

_state: dict = {"proc": None}  # holds the running asyncio subprocess


@app.get("/", response_class=HTMLResponse)
async def index() -> str:
    return _HTML


@app.get("/run/{action}")
async def run_action(action: str) -> StreamingResponse:
    if action not in ACTIONS:
        async def _err():
            yield "data: ❌ 未知操作\n\n"
            yield "data: [DONE]\n\n"
        return StreamingResponse(_err(), media_type="text/event-stream")

    args, label = ACTIONS[action]
    cmd = [PYTHON] + args

    async def stream():
        from datetime import datetime
        start = datetime.now().strftime("%H:%M:%S")
        yield f"data: ▶ [{start}] 開始執行: {label}\n\n"
        yield f"data: {'─' * 60}\n\n"
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                cwd=WORKDIR,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )
            _state["proc"] = proc

            async for raw in proc.stdout:
                line = raw.decode("utf-8", errors="replace").rstrip()
                if line:
                    yield f"data: {line}\n\n"

            await proc.wait()
            end = datetime.now().strftime("%H:%M:%S")
            yield f"data: {'─' * 60}\n\n"
            status = "✅ 完成" if proc.returncode == 0 else f"⚠️ 結束 (exit {proc.returncode})"
            yield f"data: {status}  [{end}]\n\n"
        except Exception as exc:
            yield f"data: ❌ 錯誤: {exc}\n\n"
        finally:
            _state["proc"] = None
            yield "data: [DONE]\n\n"

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={"X-Accel-Buffering": "no", "Cache-Control": "no-cache"},
    )


@app.get("/kill")
async def kill_job() -> dict:
    proc = _state.get("proc")
    if proc and proc.returncode is None:
        proc.kill()
        return {"killed": True}
    return {"killed": False}


@app.get("/status")
async def get_status() -> dict:
    proc = _state.get("proc")
    running = proc is not None and proc.returncode is None
    return {"running": running}


def _open_browser():
    import time
    time.sleep(1.2)
    webbrowser.open(f"http://localhost:{PORT}")


# ─── Inline HTML ──────────────────────────────────────────────────────────────

_HTML = """<!DOCTYPE html>
<html lang="zh-TW">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Gigi Local Control</title>
<style>
  *{box-sizing:border-box;margin:0;padding:0}
  body{background:#0d1117;color:#e6edf3;font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;padding:24px 28px;min-height:100vh}
  h1{font-size:18px;font-weight:700;margin-bottom:4px;display:flex;align-items:center;gap:8px}
  .sub{font-size:12px;color:#7d8590;margin-bottom:28px}
  .section{margin-bottom:22px}
  .section-label{font-size:11px;text-transform:uppercase;letter-spacing:.8px;color:#7d8590;margin-bottom:10px;font-weight:600}
  .btn-row{display:flex;flex-wrap:wrap;gap:8px}
  .btn{padding:8px 18px;border:none;border-radius:6px;cursor:pointer;font-size:13px;font-weight:600;letter-spacing:.2px;transition:opacity .15s,transform .08s}
  .btn:hover:not(:disabled){opacity:.82;transform:translateY(-1px)}
  .btn:active:not(:disabled){transform:translateY(0)}
  .btn:disabled{opacity:.35;cursor:not-allowed}
  .g{background:#1a7f37;color:#fff}
  .b{background:#1f6feb;color:#fff}
  .p{background:#7038a0;color:#fff}
  .y{background:#9e6a03;color:#fff}
  .r{background:#b91c1c;color:#fff}
  hr{border:none;border-top:1px solid #21262d;margin:20px 0}
  .log-header{display:flex;align-items:center;justify-content:space-between;margin-bottom:8px}
  .log-title{font-size:13px;font-weight:600;display:flex;align-items:center;gap:8px}
  .dot{width:8px;height:8px;border-radius:50%;background:#3fb950;flex-shrink:0;transition:background .3s}
  .dot.on{background:#f85149;animation:pulse 1s ease-in-out infinite}
  @keyframes pulse{0%,100%{opacity:1}50%{opacity:.35}}
  #log{background:#010409;border:1px solid #21262d;border-radius:8px;padding:14px 16px;height:420px;overflow-y:auto;font-family:"JetBrains Mono","Menlo","Monaco",monospace;font-size:12px;line-height:1.6;white-space:pre-wrap;word-break:break-all}
  .buy{color:#3fb950}.sell{color:#f85149}.warn{color:#e3b341}.done{color:#58a6ff;font-weight:700}.sep{color:#30363d}
  .clear-btn{font-size:11px;padding:3px 10px;border-radius:4px}
  .kill-btn{font-size:11px;padding:3px 10px;border-radius:4px}
</style>
</head>
<body>

<h1>🎛️ Gigi Local Control Panel</h1>
<div class="sub">http://localhost:8888 — 直接在瀏覽器控制本機腳本</div>

<div class="section">
  <div class="section-label">📡 掃描 Scan</div>
  <div class="btn-row">
    <button class="btn g" onclick="run('scan_tw')">🇹🇼 台股掃描</button>
    <button class="btn g" onclick="run('scan_us')">🇺🇸 美股掃描</button>
    <button class="btn g" onclick="run('scan_both')">🌏 全部掃描</button>
    <button class="btn y" onclick="run('scan_tw_dry')">🔍 台股 Dry Run</button>
  </div>
</div>

<div class="section">
  <div class="section-label">⚙️ 優化 &amp; 回測 — 台股 Taiwan</div>
  <div class="btn-row">
    <button class="btn b" onclick="run('opt_tw')">⚙️ 台股優化</button>
    <button class="btn p" onclick="run('paper_tw')">📊 台股回測</button>
    <button class="btn b" onclick="run('both_tw')">🚀 台股 優化+回測</button>
  </div>
</div>

<div class="section">
  <div class="section-label">⚙️ 優化 &amp; 回測 — 美股 US</div>
  <div class="btn-row">
    <button class="btn b" onclick="run('opt_us')">⚙️ 美股優化</button>
    <button class="btn p" onclick="run('paper_us')">📊 美股回測</button>
    <button class="btn b" onclick="run('both_us')">🚀 美股 優化+回測</button>
  </div>
</div>

<div class="section">
  <div class="section-label">🤖 自動優化 — 持續迭代直到達標</div>
  <div class="btn-row">
    <button class="btn r" onclick="run('auto_tw')">🇹🇼 台股自動優化</button>
    <button class="btn r" onclick="run('auto_us')">🇺🇸 美股自動優化</button>
    <button class="btn r" onclick="run('auto_both')">🌏 全市場自動優化</button>
  </div>
</div>

<div class="section">
  <div class="section-label">📐 OLS 回歸 — 僅台股</div>
  <div class="btn-row">
    <button class="btn y" onclick="run('regression')">📐 訓練回歸模型</button>
  </div>
</div>

<div class="section">
  <div class="section-label">🚨 大盤風險偵測</div>
  <div class="btn-row">
    <button class="btn y" onclick="run('risk_tw')">🇹🇼 台股大盤風險</button>
    <button class="btn y" onclick="run('risk_us')">🇺🇸 美股大盤風險</button>
  </div>
</div>

<hr>

<div class="log-header">
  <div class="log-title">
    <span class="dot" id="dot"></span>
    <span id="status">閒置</span>
  </div>
  <div style="display:flex;gap:8px">
    <button class="btn r kill-btn" onclick="killJob()" id="kill-btn" disabled>⛔ 中止</button>
    <button class="btn y clear-btn" onclick="clearLog()">🗑 清除</button>
  </div>
</div>
<div id="log">等待執行...</div>

<script>
let es = null;

function run(action) {
  if (es) { es.close(); es = null; }
  const log = document.getElementById('log');
  log.innerHTML = '';
  setRunning(true);

  es = new EventSource('/run/' + action);
  es.onmessage = (e) => {
    if (e.data === '[DONE]') {
      es.close(); es = null;
      setRunning(false);
      return;
    }
    appendLine(e.data);
  };
  es.onerror = () => {
    setRunning(false);
    if (es) { es.close(); es = null; }
  };
}

function appendLine(text) {
  const log = document.getElementById('log');
  const div = document.createElement('div');
  div.textContent = text;
  const t = text.toLowerCase();
  if (t.includes('✅') || t.includes('完成') || t.includes('done')) div.className = 'done';
  else if (t.includes('buy') || t.includes('🟢') || t.includes('watch')) div.className = 'buy';
  else if (t.includes('sell') || t.includes('🔴')) div.className = 'sell';
  else if (t.includes('⚠️') || t.includes('warn') || t.includes('error') || t.includes('❌')) div.className = 'warn';
  else if (text.startsWith('─')) div.className = 'sep';
  log.appendChild(div);
  log.scrollTop = log.scrollHeight;
}

function setRunning(on) {
  document.getElementById('dot').className = 'dot' + (on ? ' on' : '');
  document.getElementById('status').textContent = on ? '執行中...' : '閒置';
  document.getElementById('kill-btn').disabled = !on;
  document.querySelectorAll('.btn.g,.btn.b,.btn.p,.btn.y').forEach(b => b.disabled = on);
}

function killJob() {
  fetch('/kill').then(r => r.json()).then(d => {
    appendLine(d.killed ? '⛔ 已中止程序' : '⚠️ 沒有正在執行的工作');
    setRunning(false);
  });
}

function clearLog() {
  document.getElementById('log').innerHTML = '';
}
</script>
</body>
</html>"""


if __name__ == "__main__":
    threading.Thread(target=_open_browser, daemon=True).start()
    print(f"🎛️  Gigi Local Control Panel")
    print(f"   http://localhost:{PORT}")
    print(f"   Ctrl-C 停止\n")
    uvicorn.run(app, host="127.0.0.1", port=PORT, log_level="warning")
