# Gigi Stock War Room — Claude Code Context

## 專案架構

- **後端**: `api.py` → FastAPI，部署在 Render (free tier, ~30s cold start)
- **前端**: `frontend/index.html` → 單檔 React 18 (CDN + Babel)，部署在 Vercel
- **資料庫**: Google Sheets via gspread（watchlist + scan log）
- **後端 repo**: https://github.com/lwbroo/gigi-stock-war-room-2.git
- **前端 URL**: https://gigi-frontend-mu.vercel.app
- **後端 URL**: https://gigi-stock-war-room-2.onrender.com

## 部署方式

```bash
# 後端 + 前端（推上 GitHub，Render + GitHub Actions 自動部署）
git add api.py frontend/index.html
git commit -m "描述"
git push

# 前端手動部署（備用）
cd frontend && npx vercel --prod --yes
```

## 資料來源

- **FinMind API**: `https://api.finmindtrade.com/api/v4/data`
  - datasets: `TaiwanStockPrice`, `TaiwanStockMonthRevenue`, `TaiwanStockFinancialStatements`, `TaiwanStockDividend`
- **yfinance**: 美股 + 台股基本資料（sharesOutstanding 常為 0，有 fallback）
- **TWSE T86**: 外資/投信 institutional 資料

## 現行版本：v6.5

### 掃描器 Buy Signal 條件（已由 Grid Search 優化）

```python
Close > MA20
Volume > VMA20
52 < RSI14 <= 60  and RSI14 > prev_RSI   # Grid Search BEST
4% <= Bias <= 8%                          # Grid Search BEST：高偏離訊號更強
MACD > MACD_Signal
MACD_H > MACD_H.rolling(50).quantile(0.6)  # ≥60th percentile
18 <= ADX14 <= 35                         # Grid Search BEST：放寬上限
OBV > OBV_MA20
monthly_trend (MA20 > MA60 > MA120)
NOT is_extended (5日漲幅 < 8%)
```

### Backtest 優化歷程

| 版本 | 訊號數 | 勝率 | Avg Return | Sharpe |
|------|-------|------|-----------|--------|
| 原始 RSI 40-75 | 145 | ~57% | ~+2% | ~0.8 |
| +MACD_H ≥50th% | 89 | 55% | +2.9% | 1.09 |
| +RSI 50-60 | 47 | 57% | +4.4% | 1.30 |
| +ADX ≤30 | 38 | 63% | +5.7% | 1.63 |
| Grid Search v1 | ~39 | 77% | +10.4% | 3.04 |
| Grid Search v2 BEST | 21 | **86%** | +12.1% | **3.73** |

Grid Search v2（+Bias 參數）最優：RSI 50-60 / ADX 18-35 / MACD_H ≥60% / Bias ≥4%

## API Endpoints

```
GET  /api/watchlist?market=tw|us
PUT  /api/watchlist
POST /api/scan
POST /api/backtest/full       → 完整回測 + condition analysis
POST /api/backtest/gridsearch → 參數 Grid Search（640 組合）
GET  /api/forecast/{code}     → 8步 EPS 預測
GET  /api/regression/coeffs
POST /api/regression/train
```

## 重要函數

- `_bt_is_buy(row)` — 回測買進條件
- `_compute_bt_indicators(df)` — 向量化指標計算（MA/RSI/MACD/ADX/OBV）
- `_backtest_ticker(code, start, end, hold_days)` — 單股回測
- `_analyze_signals(all_signals)` — Direction C：指標 bucket 分析
- `_collect_wide_signals(code, ...)` — Direction A Grid Search 用的寬鬆訊號收集
- `_get_finmind_fundamentals(code, shares_actual)` — FinMind 財務資料 + 8步EPS
- `_forecast_eps_8step(...)` — 預測EPS、股息

## Frontend 關鍵狀態

```javascript
// localStorage keys
gigiTickersTW, gigiTickersUS   // watchlist
gigiScanResults                // 上次掃描結果
gigiThresholds                 // 自訂指標門檻
gigiPortfolio                  // 持股記錄

// 預設指標門檻（對應後端 Grid Search BEST）
DEFAULT_THRESH = { volMult:1.2, rsiLow:52, rsiHigh:60, biasPct:3, minGreenBuy:4 }
```

## 環境變數（Render 上設定）

- `FINMIND_TOKEN` — FinMind API token
- `LINE_NOTIFY_TOKEN` — LINE 通知（可選）
- Google Sheets 認證透過 gspread service account JSON

## 已完成功能

- [x] MA20 + 多指標掃描器（台股/美股）
- [x] FinMind 真實歷史回測（`/api/backtest/full`）
- [x] Direction C：指標 Bucket 分析（condition_analysis）
- [x] Direction A：Grid Search 參數優化（640 組合）
- [x] 8步 EPS 預測模型（FinMind 月營收 + 季報）
- [x] v6.5 UI：凍結欄位 + 固定表頭 + 斑馬條紋 + Hover 高亮

## 待完成 / 可延伸方向

- [ ] OLS 回歸學習指標權重（Direction B）
- [ ] 問題股票自動過濾（欣興/臺慶科/創意 在此策略持續虧損）
- [ ] 跨時間段 cross-validation（驗證 Grid Search 參數不 overfit）
- [ ] 美股 backtest（目前只支援台股 FinMind）
