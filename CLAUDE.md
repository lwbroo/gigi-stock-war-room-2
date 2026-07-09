# Gigi Stock War Room — Claude Code Context

> **現行版本: v11.0** (2026-07-06)

## 專案架構 — v11.0：全面本機運算

```
本機 Mac（真正的大腦，所有重運算都在這裡）
├── scan_local.py            — yfinance/FinMind 抓資料 + 技術指標 + BUY訊號偵測
│                               → 寫入 scan_cache / scan_log / signal_outcomes
├── update_outcomes_local.py — 回填 signal_outcomes 的 5d/10d 實際報酬
├── market_context_local.py  — VIX + Grok AI 市場情緒 → market_context tab
├── regression_train_local.py — OLS 回歸（12指標→10日報酬，僅台股，用 scan_log 訓練）
├── market_risk_local.py     — 大盤大跌/大漲偵測（見下方「大盤風險偵測」章節）
├── simulate.py               — Grid Search 參數優化 + XGBoost 訓練 + Paper Trading（手動用）
├── auto_optimize.py          — 自動迴圈調參直到達到目標勝率（含 paper trading 驗證，週排程用這個）
├── local_server.py           — http://localhost:8888 網頁控制台，手動觸發以上任一工具
└── crontab（`crontab -l` 查看，11 項排程，Mon-Fri + Sunday）

雲端 Render（api.py，很薄，只做這些事）
├── 讀 GSheets cache 給前端看（/api/scan/cache, /api/paper-report, /api/model/stats...）
├── 讀 market_context cache 給 /api/vix、/api/sentiment（本機沒跑過就即時 fallback）
├── /api/sentiment/refresh — 手動即時重新分析（前端按鈕用，之後也會寫回 cache）
├── /api/chat — Claude AI 助手，讀寫 GitHub + 觸發 Vercel 部署（需要能從任何地方維護，故留在雲端）
└── 其餘全部 stub：/api/scan、/api/scheduled/*、/api/backtest*、/api/model/train-xgb、
    /api/model/walk-forward、/api/regression/train、/api/forecast/{code}
    → 一律回傳 {"status":"run_locally", ...}

前端: frontend/index.html → 單檔 React 18 (CDN + Babel) → Vercel
```

- **後端 repo**: https://github.com/lwbroo/gigi-stock-war-room-2.git
- **前端 URL**: https://gigi-frontend-mu.vercel.app
- **後端 URL**: https://gigi-stock-war-room-2.onrender.com

## ⚠️ v11.0 修復的關鍵bug（都是「移到本機」refactor留下的坑）

1. **`signal_outcomes` 從未被寫入** — `scan_local.py` 移植時漏掉了原本 cloud 版
   `_log_buy_signals()` 的邏輯，導致整個 ML 自我學習迴圈（walk-forward + XGBoost）
   從 refactor 後就一直在讀空資料。已在 `scan_local.py` 補上 `_log_buy_signals()`。
2. **`/api/outcomes/update` 呼叫不存在的函式** `_price_n_days_later` — 被外層
   try/except 吞掉，每天靜默失敗。已改用 `update_outcomes_local.py`（本機執行，
   重用 `simulate.py` 的 `_fetch_ohlcv`）。
3. **Render 部署阻斷bug** — `api.py` 裡已 stub 掉的舊回測程式碼，函式簽名仍標注
   `pd.DataFrame`/`np.ndarray`，但 `import pandas as pd`/`import numpy as np` 早被
   移除，導致 `import api` 直接 `NameError` 崩潰。這代表 2026-06-27 之後的每次
   `git push` 可能都沒有真正部署成功，Render 只是一直серving舊的健康 build。
   已加 `from __future__ import annotations` 修復（讓型別註記變成 lazy，不在
   import 時求值）。**這是這次修復最重要的一項**，之後的 push 才會真正生效。
4. **本機 `.env` 缺 `TELEGRAM_BOT_TOKEN` / `TELEGRAM_CHAT_ID`** — 本機排程掃描
   （`scan_local.py --notify`）一直沒有真的推播出去，靜默無錯誤。已補上。

## 部署方式

```bash
# 後端 + 前端（都是 git push 觸發自動 deploy）
cd /Users/kurtchiang/Desktop/gigi-stock-war-room
git add -A && git commit -m "描述" && git push

# 前端手動部署（備用，最穩定）
cd frontend && npx vercel --prod --yes
```

> Push 後務必確認 Render 真的重新部署成功（見上面第3點的教訓）——
> 可以打 `/api/scan` 看回傳內容是否符合預期，或看 Render dashboard 的 deploy log。

## 本機 crontab（`crontab -l` 查看/`crontab -e` 編輯）

```
08:00 Mon-Fri  scan_local.py --market tw   --notify
15:00 Mon-Fri  scan_local.py --market both --notify
21:00 Mon-Fri  scan_local.py --market us   --notify
07:30 Mon-Fri  market_context_local.py            (VIX + Grok情緒)
15:30 Mon-Fri  update_outcomes_local.py --market tw
06:30 Tue-Sat  update_outcomes_local.py --market us
14:30 Mon-Fri  market_risk_local.py --market tw    (大盤大跌/大漲偵測，見下方章節)
05:45 Tue-Sat  market_risk_local.py --market us
15:30 Mon-Fri  update_outcomes_local.py --market tw
06:30 Tue-Sat  update_outcomes_local.py --market us
10:00 Sunday   auto_optimize.py --market tw --target 72 --rounds 8 --years 5
11:30 Sunday   auto_optimize.py --market us --target 70 --rounds 8 --years 5
13:00 Sunday   regression_train_local.py          (OLS 回歸，僅台股)
```

auto_optimize.py 每輪都包含 grid search + XGBoost 重訓 + paper trading 驗證，
比單純 `simulate.py --mode both` 多了「沒達標就繼續調整、最多8輪」的邏輯，
兩者都會寫 `paper_results`/`model_params`。

日誌：`/tmp/gigi_scan.log`、`/tmp/gigi_context.log`、`/tmp/gigi_marketrisk.log`、
`/tmp/gigi_outcomes.log`、`/tmp/gigi_autooptimize.log`、`/tmp/gigi_regression.log`

## 大盤風險偵測（v11.0，2026-07-08 新增）

動機：2026-07-07 台股單日重挫 -2.31%，事後想「應該要有訊號」。研究了實際的
外資買賣超資料後發現：外資賣超與當天跌幅大多是**同一天**發生，07-03 當週最大
單日賣超（-778億）那天指數幾乎打平，真正下跌的 07-07 賣超金額反而較小
（-547億）— 沒有乾淨的「連續賣超→隔天崩盤」領先關係。所以這個功能刻意分成
兩種誠實分開標示的訊號，不假裝能預測崩盤：

1. **確認訊號**（可靠）— 大盤當日漲跌幅 ≥ ±2%，收盤結算後立即 Telegram 推播。
   不是預測，是「今天已經發生了」的即時通知 — 但因為是收盤後馬上推播，
   還是能讓你在下一個交易日開始前就知道。
2. **風險情境**（謹慎）— TW 用外資近3日累計買賣超、US 用 VIX，各自對照自己
   近20個交易日的均值+標準差，超出 1 個標準差才推播，用詞刻意保守
   （"風險略升"、"僅供留意，非確認訊號"），避免製造錯誤的確定感。

閾值都在 `market_risk_local.py` 檔案開頭常數區（`MOVE_THRESHOLD_PCT`、
`FLOW_Z_THRESHOLD`），可以自行調整。`market_flow` tab 存歷史資料，風險情境
需要累積 ≥10 個交易日才會開始計算（剛部署的頭兩週會先跳過這部分）。

## 環境變數

**Render Dashboard**（雲端，不在 code 裡）：
`FINMIND_TOKEN`、`TELEGRAM_BOT_TOKEN`、`TELEGRAM_CHAT_ID`、`ANTHROPIC_API_KEY`、
`GITHUB_TOKEN`、`VERCEL_DEPLOY_HOOK`、`GOOGLE_CREDENTIALS_JSON`、`GROK_API_KEY`（sentiment手動refresh用）

**本機 `.env`**（`/Users/kurtchiang/Desktop/gigi-stock-war-room/.env`）：
`GOOGLE_CREDENTIALS_JSON`、`FINMIND_TOKEN`、`TELEGRAM_BOT_TOKEN`、`TELEGRAM_CHAT_ID`、
`GROK_API_KEY`（目前留空 — 填上才會由本機排程做 sentiment 分析，否則雲端手動
refresh 仍會 fallback 即時呼叫）

## GSheets tabs

```
gigi-war-room-watchlist / gigi-us-watchlist  — 觀察清單
scan_cache      — 最新一次掃描結果（前端讀這個）
scan_log        — 歷史掃描 log（僅 TW）
signal_outcomes — 每次BUY訊號 + 15天後的 5d/10d 實際報酬（餵給 walk-forward + XGBoost）
model_params    — 各市場目前最佳參數（walk-forward 更新）
model_store     — XGBoost model（base64 + gzip pickle）
sim_results     — Grid Search 優化歷史
paper_results   — Paper Trading 回測結果
market_context  — v11.0 新增：VIX + Grok情緒 每日 cache
regression_coeffs — OLS 回歸係數（僅台股，regression_train_local.py 週更新）
market_flow     — v11.0 新增：大盤漲跌幅 + 外資流量/VIX 歷史（market_risk_local.py）
universe_tw     — 台股 top-150 市值宇宙（update_universe.py 維護）
```

## 重要技術細節

- Python 3.9 on Render — 用 `Optional[X]` 不用 `X | None`（但 `api.py` 已加
  `from __future__ import annotations`，新程式碼两種寫法都可以）
- `api.py` 裡有大量 v10.0 之前的舊回測程式碼還沒真的刪除（`_compute_bt_indicators`、
  `_bt_is_buy`、`_fetch_ohlcv`、`_analyze_signals`... 等），全部是 dead code，只有
  已stub的 `/api/backtest*` 端點語法上引用它們。之後有空可以整段刪掉瘦身。
- `@babel/standalone@7.23.10` with `data-presets="react,env"` (classic JSX)
- `PatternBadge`/`TradingViewModal`/`AutoLearnPanel` 元件必須宣告在 `App()` 外
- `sectorHeat` useMemo 必須在 `isRowBuy/Sell/Warn` 之後（TDZ fix）
- `.TW` 後綴：呼叫 `_fetch_ohlcv`/`simulate.py` 系列函式前，TW ticker 要先
  `code.split(".")[0]`，否則會變成 `2330.TW.TW`
- Render Build Command: `bash build.sh`（安裝 Playwright Chromium，供 Grok 新聞
  fallback 使用；現在情緒抓取已主要移到本機，這個依賴之後可以評估要不要拿掉）

## Roadmap

- v11.0 ✅ 全面本機運算 + 修復多個關鍵bug（本檔案上方）
- v11.1 🔜 模擬交易（虛擬持倉自動執行 + 每日損益追蹤 + Telegram 報告）
- v12.0 🔜 真實下單（Fugle 台股 + Alpaca 美股）
