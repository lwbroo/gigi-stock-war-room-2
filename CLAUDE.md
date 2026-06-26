# Gigi Stock War Room — Claude Code Context

> **現行版本: v7.0** (2026-06-27)

## 專案架構

- **後端**: `api.py` → FastAPI，部署在 Render (free tier, ~30s cold start)
- **前端**: `frontend/index.html` → 單檔 React 18 (CDN + Babel)，部署在 Vercel
- **資料庫**: Google Sheets via gspread（watchlist + scan log）
- **後端 repo**: https://github.com/lwbroo/gigi-stock-war-room-2.git
- **前端 URL**: https://gigi-frontend-mu.vercel.app
- **後端 URL**: https://gigi-stock-war-room-2.onrender.com

## 部署方式

```bash
# 後端（Render 自動 deploy 當 git push）
git add api.py requirements.txt build.sh
git commit -m "描述"
git push

# 前端（GitHub Actions 自動 deploy，或手動）
git add frontend/index.html
git commit -m "描述"
git push

# 前端手動部署（備用，最穩定）
cd /path/to/gigi-stock-war-room/frontend && npx vercel --prod --yes
```

> ⚠️ GitHub Actions 只有當 commit 包含 `frontend/` 路徑的檔案時才觸發前端 deploy。
> 不確定時，直接用 `npx vercel --prod --yes`。

## 環境變數（Render Dashboard 上設定，不在 code 裡）

- `GROK_API_KEY` — xAI Grok API key（v7.0 市場情緒分析）
- `FINMIND_TOKEN` — FinMind API token（台股歷史資料）
- `LINE_NOTIFY_TOKEN` — LINE 通知（可選）
- Google Sheets 認證透過 gspread service account JSON

## v7.0 新功能

### Grok AI 市場情緒

```python
GET  /api/sentiment         # 取得當日情緒（快取，不重複打 API）
POST /api/sentiment/refresh # 強制重新分析
```

- 情緒分數 -1.0（極悲觀）～ +1.0（極樂觀）
- 融入 `computeOverall()` 評分：≥+0.5→+10, ≥+0.2→+5, ≤-0.2→-8, ≤-0.5→-15
- 新聞來源：Playwright（主）→ yfinance SPY/QQQ（備）→ date prompt（最後）
- Grok model: `grok-3-mini-beta`

### UI 版面（v7.0）

```
🇹🇼 Taiwan | 🇺🇸 USA          ← 頂層市場切換（影響所有 tab）
[Grok Sentiment Bar]          ← 情緒分數 + 理由 + 板塊
📡 Scanner | 💼 Portfolio | 🧪 Backtest | 🧠 Model | 📊 History | ⚙️ Settings
```

## 市場參數（已由 Grid Search 優化）

| 參數 | 台股 TW | 美股 US |
|------|---------|---------|
| RSI | 52–60 | 60–65 |
| Bias | 4–8% | 4–8% |
| ADX | 18–35 | 18–30 |
| MACD_H | ≥60th pct | ≥60th pct |
| 回測勝率 | 94.1% | 78% |
| Sharpe | 4.43 | 2.09 |
| 持有天數 | 10 日 | 10 日 |

美股 RSI 60-65：動能已確立才入場；台股 RSI 52-60：早期動能區入場。

## 掃描器 Buy Signal 條件

```python
Close > MA20, Volume > VMA20
RSI14 in (52,60] TW / (60,65] US, and RSI14 > prev_RSI
4% <= Bias <= 8%
MACD > MACD_Signal
MACD_H > MACD_H.rolling(50).quantile(0.6)
18 <= ADX14 <= 35 (TW) / 30 (US)
OBV > OBV_MA20
monthly_trend: MA20 > MA60 > MA120
NOT is_extended: 5日漲幅 < 8%
```

## API Endpoints

```
GET  /api/watchlist?market=tw|us
PUT  /api/watchlist
POST /api/scan
POST /api/backtest/full       → 完整回測 + condition analysis
POST /api/backtest/gridsearch → 參數 Grid Search（9216 組合）
GET  /api/forecast/{code}     → 8步 EPS 預測
GET  /api/regression/coeffs
POST /api/regression/train
GET  /api/sentiment
POST /api/sentiment/refresh
```

## 重要技術細節

- Python 3.9 on Render：用 `Optional[X]` 不用 `X | None`
- `@babel/standalone@7.23.10` with `data-presets="react,env"` (classic JSX)
- `PatternBadge` component 必須在 `App()` 外宣告，否則 blank page
- `sectorHeat` useMemo 必須在 `isRowBuy/Sell/Warn` 之後（TDZ fix）
- Render Build Command: `bash build.sh`（安裝 Playwright Chromium）

## localStorage keys

```
gigiTickersTW / gigiTickersUS  — watchlist
gigiScanResults                — 上次掃描
gigiScanHistory                — 最近10次掃描
gigiPortfolio                  — 持倉記錄
gigiThresholds                 — 自訂門檻
gigiAlerts                     — 價格警示
gigiLineToken                  — LINE token
gigiWhatsNewDismissed          — "What's new" banner 已讀
```

## 重要函數（api.py）

- `_fetch_ohlcv(code, start, end, market)` — FinMind(TW) / yfinance(US)
- `_bt_is_buy(row, params)` — 回測買進條件（parameterized）
- `_get_market_params(market)` — 回傳 TW_BEST_PARAMS 或 US_DEFAULT_PARAMS
- `_compute_bt_indicators(df)` — MA/RSI/MACD/ADX/OBV 向量化計算
- `_backtest_ticker(code, start, end, hold_days, params)` — 單股回測
- `_get_finmind_fundamentals(code, shares_actual)` — 財務資料 + 8步EPS
- `_fetch_news_yfinance()` — 美股新聞 fallback（SPY/QQQ/NVDA/AAPL/MSFT）
- `_analyze_with_grok(news_text)` — Grok AI 情緒分析
- `_get_or_refresh_sentiment(force)` — 情緒快取管理

## 已完成功能

- [x] MA20 + 15指標掃描器（台股/美股）
- [x] FinMind 真實歷史回測（`/api/backtest/full`）
- [x] Grid Search 參數優化（9216 組合）
- [x] 8步 EPS 預測模型
- [x] OLS 自適應回歸模型
- [x] v7.0 Grok AI 市場情緒
- [x] 台股/美股頂層切換
- [x] Premium Dark Glass UI

## 待完成 / 可延伸方向

- [ ] 全量美股 59支 × 10日 回測確認
- [ ] Grid Search cross-validation（避免 overfit）
- [ ] Grok 情緒歷史追蹤（每日記錄到 Google Sheets）
- [ ] Playwright 在 Render 上的穩定性測試
