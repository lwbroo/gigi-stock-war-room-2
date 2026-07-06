#!/usr/bin/env python3
"""
auto_optimize.py — 持續自我優化直到勝率達到目標

每輪同時優化入場參數（RSI/ADX/BIAS/MACD）與出場參數（RSI_exit/hold_days），
找到最佳組合後縮小搜尋範圍，不斷迭代直到達標或輪數用完。

資料只下載一次，後續所有組合在記憶體中運算 → 每輪約 30-120 秒。

用法:
  python3 auto_optimize.py                         # TW, 目標 80%, 8 輪, 5 年
  python3 auto_optimize.py --market us --target 75
  python3 auto_optimize.py --market both --rounds 10 --target 72
"""

import sys, os, time, argparse
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import simulate as sim

# ── Constants ─────────────────────────────────────────────────────────────────
MIN_TRADES_TW   = 12    # TW 最少交易次數（股票多，容易達到）
MIN_TRADES_US   = 6     # US 最少交易次數（watchlist 較小）
TOP_CANDIDATES  = 5     # 每輪取前 N 個入場參數組合進行 paper sim
DL_WORKERS      = 8     # 下載平行執行緒
MAX_HOLD_BUFFER = 25    # 每個入場點預抓的未來天數

RSI_EXIT_OPTS  = [70, 73, 76, 78, 81, 84, 87]
HOLD_DAYS_OPTS = [7, 8, 10, 12, 15]


# ══════════════════════════════════════════════════════════════════════════════
#  Step 1: 下載 & 快取 OHLCV
# ══════════════════════════════════════════════════════════════════════════════

def _download_cache(universe: List[str], fetch_start: str, fetch_end: str, market: str) -> Dict:
    cache: Dict[str, pd.DataFrame] = {}

    def _one(code: str):
        # _fetch_ohlcv appends .TW internally — strip suffix if already present
        clean = code.split(".")[0] if market == "tw" else code
        df = sim._fetch_ohlcv(clean, fetch_start, fetch_end, market=market)
        if df is None or len(df) < 80:
            return clean, None
        return clean, sim._compute_bt_indicators(df)

    print(f"  下載 {len(universe)} 支股票 ({fetch_start} → {fetch_end})...")
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=DL_WORKERS) as ex:
        futs = {ex.submit(_one, c): c for c in universe}
        done = 0
        for fut in as_completed(futs):
            code, df = fut.result()
            if df is not None:
                cache[code] = df
            done += 1
            if done % 30 == 0:
                print(f"    {done}/{len(universe)} ...")
    print(f"  ✅ 取得 {len(cache)}/{len(universe)} 支  ({time.time()-t0:.0f}s)")
    return cache


# ══════════════════════════════════════════════════════════════════════════════
#  Step 2: 快速網格搜尋（入場參數，固定持倉天數）
# ══════════════════════════════════════════════════════════════════════════════

def _collect_signals(cache: Dict, start_dt, end_dt, hold_days: int) -> List[dict]:
    """收集通過基礎條件的入場點（不過濾入場參數），計算固定持倉天數報酬。"""
    sigs = []
    for code, df in cache.items():
        in_range = df[(df["date"] >= start_dt) & (df["date"] <= end_dt)]
        for idx, row in in_range.iterrows():
            try:
                for col in ["MA20","VMA20","RSI14","Bias","MACD","MACD_Sig","ADX14","MACD_H","OBV","OBV_MA20"]:
                    if pd.isna(row.get(col, float("nan"))): raise ValueError
                if not (row["Close"] > row["MA20"] and row["Volume"] > row["VMA20"] and
                        row["MACD"] > row["MACD_Sig"] and row["OBV"] > row["OBV_MA20"] and
                        bool(row.get("monthly_trend")) and not bool(row.get("is_extended"))):
                    continue
            except Exception:
                continue
            past_mh = df["MACD_H"].iloc[max(0, idx-50):idx].dropna()
            mh_pct = float((past_mh < float(row["MACD_H"])).mean() * 100) if len(past_mh) >= 5 else 50.0
            future = df[df.index > idx].head(hold_days)
            if len(future) < hold_days: continue
            entry = float(row["Close"])
            if entry <= 0: continue
            ret = (float(future.iloc[-1]["Close"]) - entry) / entry * 100
            sigs.append({
                "rsi14":       float(row["RSI14"]),
                "adx14":       float(row["ADX14"]),
                "bias":        float(row["Bias"]),
                "macd_h_pct":  mh_pct,
                "return_pct":  ret,
            })
    return sigs


def _grid_search(signals: List[dict], grid: List[dict], hold_days: int, min_n: int = 5) -> List[dict]:
    """對所有入場參數組合評分（固定持倉天數 WR）。"""
    results = []
    for p in grid:
        sub = [s for s in signals
               if p["rsi_lo"] <= s["rsi14"] <= p["rsi_hi"]
               and p["adx_lo"] <= s["adx14"] <= p["adx_hi"]
               and s["macd_h_pct"] >= p["macd_h_pct_min"]
               and p["bias_lo"] <= s["bias"] <= p["bias_hi"]]
        if len(sub) < min_n: continue
        rets = [s["return_pct"] for s in sub]
        wins = sum(1 for r in rets if r > 0)
        wr = wins / len(sub)
        if wr < 0.55: continue
        avg = float(np.mean(rets))
        std = float(np.std(rets)) if len(rets) > 1 else 1.0
        sharpe = round(avg / std * (252 / hold_days) ** 0.5, 2) if std > 0 else 0.0
        score = sharpe * (wr / 0.65) * min(1.0, len(sub) / 15)
        results.append({"params": p, "n": len(sub), "wr": wr, "sharpe": sharpe, "score": score})
    results.sort(key=lambda x: x["score"], reverse=True)
    return results


# ══════════════════════════════════════════════════════════════════════════════
#  Step 3: 找入場點（快取到記憶體，後續不再讀 df）
# ══════════════════════════════════════════════════════════════════════════════

def _find_entries(cache: Dict, entry_params: dict, mkt_trend, start_dt, end_dt) -> List[dict]:
    """找出所有符合入場條件的點，預存未來 MAX_HOLD_BUFFER 天的 index。"""
    entries = []
    for code, df in cache.items():
        in_range = df[(df["date"] >= start_dt) & (df["date"] <= end_dt)]
        for idx, row in in_range.iterrows():
            if not sim._bt_is_buy(row, entry_params): continue
            if mkt_trend is not None:
                ed = pd.to_datetime(row["date"]).normalize()
                if not bool(mkt_trend.get(ed, True)): continue
            past_mh = df["MACD_H"].iloc[max(0, idx-50):idx].dropna()
            mh_pct = float((past_mh < float(row["MACD_H"])).mean() * 100) if len(past_mh) >= 5 else 50.0
            if mh_pct < entry_params.get("macd_h_pct_min", 0): continue
            px = float(row["Close"])
            if px <= 0: continue
            atr = float(row["ATR14"]) if not pd.isna(row.get("ATR14", float("nan"))) else px * 0.03
            future_idx = [i for i in df.index if i > idx][:MAX_HOLD_BUFFER]
            if len(future_idx) < 3: continue
            entries.append({
                "code":       code,
                "entry_date": row["date"].strftime("%Y-%m-%d"),
                "entry_px":   px,
                "atr_stop":   px - atr * 1.5,
                "future_idx": future_idx,
                "df":         df,   # reference only — no copy
            })
    return entries


# ══════════════════════════════════════════════════════════════════════════════
#  Step 4: 套用出場規則（極快，純記憶體運算）
# ══════════════════════════════════════════════════════════════════════════════

def _apply_exits(entries: List[dict], rsi_exit: int, hold_days: int) -> List[dict]:
    """對預找的入場點套用不同出場規則，回傳交易清單。"""
    trades = []
    for e in entries:
        df = e["df"]
        fidx = e["future_idx"][:hold_days]
        if not fidx: continue
        exit_i = fidx[min(hold_days - 1, len(fidx) - 1)]
        reason = "持滿天數"
        prev_macd_above = True
        below_ma10 = 0
        for j, fi in enumerate(fidx):
            frow = df.loc[fi]
            day = j + 1
            macd_above = frow["MACD"] > frow["MACD_Sig"]
            if day >= 3:
                if frow["RSI14"] > rsi_exit:
                    exit_i = fi; reason = f"RSI>{rsi_exit}超買"; break
                ma10 = frow.get("MA10", float("nan"))
                if not pd.isna(ma10) and frow["Close"] < ma10:
                    below_ma10 += 1
                    if below_ma10 >= 2:
                        exit_i = fi; reason = "跌破MA10×2日"; break
                else:
                    below_ma10 = 0
                if prev_macd_above and not macd_above:
                    exit_i = fi; reason = "MACD死叉"; break
                if frow["Close"] < e["atr_stop"]:
                    exit_i = fi; reason = "ATR停損"; break
            prev_macd_above = macd_above
        exit_px = float(df.loc[exit_i]["Close"])
        held = fidx.index(exit_i) + 1
        ret = (exit_px - e["entry_px"]) / e["entry_px"] * 100
        trades.append({
            "ticker":      e["code"],
            "entry_date":  e["entry_date"],
            "exit_date":   df.loc[exit_i]["date"].strftime("%Y-%m-%d"),
            "entry_price": round(e["entry_px"], 2),
            "exit_price":  round(exit_px, 2),
            "return_pct":  round(ret, 2),
            "held_days":   held,
            "exit_reason": reason,
            "won":         ret > 0,
        })
    return trades


MIN_DISTINCT_MONTHS = 4  # guard against overfitting to one narrow time window

def _score(trades: List[dict], hold_days: int) -> Tuple[float, float, int, int]:
    if not trades: return 0.0, 0.0, 0, 0
    rets    = [t["return_pct"] for t in trades]
    wr      = sum(1 for t in trades if t["won"]) / len(trades)
    avg     = float(np.mean(rets))
    std     = float(np.std(rets)) if len(rets) > 1 else 1.0
    sh      = round(avg / std * (252 / hold_days) ** 0.5, 2) if std > 0 else 0.0
    n_months = len({t["entry_date"][:7] for t in trades})
    return wr, sh, len(trades), n_months


# ══════════════════════════════════════════════════════════════════════════════
#  入場參數網格（支援逐輪縮小搜尋範圍）
# ══════════════════════════════════════════════════════════════════════════════

def _entry_grid(best: Optional[dict], rnd: int, market: str) -> List[dict]:
    is_tw = market == "tw"
    if best is None or rnd == 1:
        rsi_los  = [54,56,58,60,62]      if is_tw else [45,50,55,58,60,62]
        rsi_his  = [60,62,64,66,68]      if is_tw else [60,63,66,68,72,75]
        adx_los  = [18,20,22,24,26]      if is_tw else [12,14,16,18,20,22]
        adx_his  = [28,32,36,40]         if is_tw else [26,30,34,38,42]
        mh_pcts  = [60,66,70,75,80]      if is_tw else [40,50,60,66,70]
        bias_los = [2,3,4,5]             if is_tw else [1,2,3,4,5]
        bias_his = [6,7,8,9]             if is_tw else [6,7,8,9,10,12]
    else:
        step = max(1, 5 - rnd)
        def rng(v, s, lo, hi):
            return sorted(set(max(lo, min(hi, v + i * s)) for i in range(-3, 4)))
        rsi_los  = rng(best["rsi_lo"],           step, 46, 68)
        rsi_his  = rng(best["rsi_hi"],           step, 54, 76)
        adx_los  = rng(best["adx_lo"],           step, 12, 30)
        adx_his  = rng(best["adx_hi"],           step, 24, 50)
        mh_pcts  = rng(best["macd_h_pct_min"],   5,    50, 90)
        bias_los = rng(best["bias_lo"],           1,     1,  8)
        bias_his = rng(best["bias_hi"],           1,     4, 12)
    return [
        {"rsi_lo":rsl,"rsi_hi":rsh,"adx_lo":adl,"adx_hi":adh,
         "macd_h_pct_min":mh,"bias_lo":blo,"bias_hi":bhi}
        for rsl in rsi_los for rsh in rsi_his if rsl < rsh
        for adl in adx_los for adh in adx_his if adl < adh
        for mh  in mh_pcts
        for blo in bias_los for bhi in bias_his if blo < bhi
    ]


# ══════════════════════════════════════════════════════════════════════════════
#  主迴圈
# ══════════════════════════════════════════════════════════════════════════════

def auto_optimize(market: str, target_wr: float, max_rounds: int, years: int):
    print(f"\n{'═'*64}")
    print(f"  🤖 自動優化迴圈 [{market.upper()}]")
    print(f"  目標勝率={target_wr}%  最大輪數={max_rounds}  歷史={years}年")
    print(f"{'═'*64}")

    today      = datetime.today()
    end_date   = today.strftime("%Y-%m-%d")
    start_date = (today - timedelta(days=int(years * 365.25))).strftime("%Y-%m-%d")
    fetch_s    = (today - timedelta(days=int(years * 365.25) + 420)).strftime("%Y-%m-%d")
    fetch_e    = (today + timedelta(days=30)).strftime("%Y-%m-%d")
    start_dt   = pd.to_datetime(start_date)
    end_dt     = pd.to_datetime(end_date)

    # ── 下載資料（只做一次）────────────────────────────────────────────────
    print(f"\n[1] 下載資料")
    raw_universe = sim._get_universe(market)
    # Normalise: strip any .TW / .US suffix so _fetch_ohlcv doesn't double-append
    universe = list(dict.fromkeys(c.split(".")[0] if market == "tw" else c for c in raw_universe))
    cache    = _download_cache(universe, fetch_s, fetch_e, market)

    # ── 大盤趨勢過濾 ─────────────────────────────────────────────────────
    print(f"\n[2] 大盤趨勢過濾")
    mkt_trend = sim._build_market_trend(market, start_date, end_date)
    if mkt_trend is not None:
        bull = int(mkt_trend.sum()); total = len(mkt_trend)
        print(f"  大盤多頭天數: {bull}/{total} ({bull/total:.0%})")

    best_result: Optional[dict] = None  # 全程最佳
    best_entry:  Optional[dict] = None  # 入場參數（供下輪縮小範圍）
    best_exit = {"rsi_exit": 78, "hold_days": 10}
    min_trades = MIN_TRADES_TW if market == "tw" else MIN_TRADES_US
    gs_min_n   = 8            if market == "tw" else 4   # grid search 最小信號數

    for rnd in range(1, max_rounds + 1):
        t_rnd = time.time()
        cur = f"{best_result['wr']:.1%}" if best_result else "—"
        print(f"\n{'─'*64}")
        print(f"  ROUND {rnd}/{max_rounds}   目前最佳={cur}")
        print(f"{'─'*64}")

        # A. 入場網格
        e_grid = _entry_grid(best_entry, rnd, market)
        print(f"\n[A] 入場網格: {len(e_grid)} 組")

        # B. 收集快速信號（固定持倉天數，用於網格搜尋排序）
        hd_ref  = best_exit["hold_days"]
        sigs    = _collect_signals(cache, start_dt, end_dt, hd_ref)
        print(f"[B] 基礎條件信號: {len(sigs)} 筆")
        if not sigs:
            print("    ⚠️ 無信號，跳過本輪")
            continue

        # C. 網格搜尋（快速排序入場參數）
        gs_results = _grid_search(sigs, e_grid, hd_ref, min_n=gs_min_n)
        if not gs_results:
            print("[C] ❌ 網格搜尋無結果")
            continue
        print(f"[C] 前 {min(5,len(gs_results))} 名（固定持倉 WR）:")
        for r in gs_results[:5]:
            p = r["params"]
            print(f"    GS_WR={r['wr']:.1%} sh={r['sharpe']:.2f} n={r['n']:3d}  RSI {p['rsi_lo']}-{p['rsi_hi']} ADX {p['adx_lo']}-{p['adx_hi']} B {p['bias_lo']}-{p['bias_hi']} MH {p['macd_h_pct_min']}")

        # D. Paper Sim（動態出場）× 出場參數組合
        top_eps = [r["params"] for r in gs_results[:TOP_CANDIDATES]]
        n_exit  = len(RSI_EXIT_OPTS) * len(HOLD_DAYS_OPTS)
        total   = len(top_eps) * n_exit
        print(f"\n[D] Paper Sim: {len(top_eps)} 入場 × {n_exit} 出場 = {total} 組合...")

        rnd_best = None
        checked  = 0

        for ep in top_eps:
            # 每個入場參數組合：找入場點（一次）
            entries = _find_entries(cache, ep, mkt_trend, start_dt, end_dt)
            if not entries:
                checked += n_exit
                continue

            for re in RSI_EXIT_OPTS:
                for hd in HOLD_DAYS_OPTS:
                    trades  = _apply_exits(entries, re, hd)
                    checked += 1
                    if len(trades) < min_trades:
                        continue
                    wr, sh, n, n_months = _score(trades, hd)
                    # dampen results concentrated in very few calendar months —
                    # a narrow filter that only ever fires during one rally is
                    # overfitting, not a real edge, however high its raw WR/Sharpe
                    score = sh * (wr / 0.65) * min(1.0, n / 20) * min(1.0, n_months / MIN_DISTINCT_MONTHS)
                    if rnd_best is None or score > rnd_best["score"]:
                        rnd_best = {
                            "params": {**ep, "rsi_exit": re, "hold_days": hd},
                            "trades": trades, "wr": wr, "sharpe": sh,
                            "n": n, "n_months": n_months, "score": score,
                        }
                    if checked % 30 == 0:
                        bwrstr = f"{rnd_best['wr']:.1%}" if rnd_best else "—"
                        print(f"    {checked}/{total}  本輪最佳: WR={bwrstr}", end="\r")

        print()  # 清除 \r 行

        if rnd_best is None:
            n_found = max((len(_apply_exits(_find_entries(cache, gs_results[0]["params"], mkt_trend, start_dt, end_dt), 78, 10)),) if gs_results else (0,))
            print(f"    ❌ 無有效組合（最多 {n_found} 筆，門檻 {min_trades} 筆）")
            if gs_results and min_trades > 4:
                min_trades = max(4, min_trades - 2)   # 動態放寬門檻再試
                print(f"    ↓ 放寬交易次數門檻至 {min_trades}，下輪再試")
            continue

        rb = rnd_best
        p  = rb["params"]
        elapsed = time.time() - t_rnd
        print(f"\n  ✅ 第 {rnd} 輪最佳  ({elapsed:.0f}s)")
        print(f"     WR={rb['wr']:.1%}  n={rb['n']}  跨{rb['n_months']}個月  Sharpe={rb['sharpe']:.2f}")
        print(f"     入場: RSI {p['rsi_lo']}-{p['rsi_hi']} / ADX {p['adx_lo']}-{p['adx_hi']} / Bias {p['bias_lo']}-{p['bias_hi']} / MACD_H {p['macd_h_pct_min']}%")
        print(f"     出場: RSI_exit={p['rsi_exit']}  hold_days={p['hold_days']}")

        diverse_enough = rb["n_months"] >= MIN_DISTINCT_MONTHS

        # 更新全程最佳（只在時間分散夠的結果之間比較，避免把過擬合的單月結果留下來）
        if diverse_enough and (best_result is None or not best_result.get("n_months", 0) >= MIN_DISTINCT_MONTHS
                                or rb["wr"] > best_result["wr"]):
            best_result = rb
            best_entry  = {k: v for k, v in p.items() if k not in ("rsi_exit","hold_days")}
            best_exit   = {"rsi_exit": p["rsi_exit"], "hold_days": p["hold_days"]}
        elif best_result is None:
            # 還沒有任何分散夠的結果 — 先暫存這輪，總比空手好，但下面不會用它來提早停止
            best_result = rb
            best_entry  = {k: v for k, v in p.items() if k not in ("rsi_exit","hold_days")}
            best_exit   = {"rsi_exit": p["rsi_exit"], "hold_days": p["hold_days"]}

        if rb["wr"] >= target_wr / 100 and diverse_enough:
            print(f"\n  🎉 達到目標勝率 {target_wr}%（且交易分散在 {rb['n_months']} 個月，非單一時段過擬合）！停止搜尋。")
            break
        elif rb["wr"] >= target_wr / 100 and not diverse_enough:
            print(f"\n  ⚠️  勝率達標但交易全部集中在 {rb['n_months']} 個月（<{MIN_DISTINCT_MONTHS}），")
            print(f"      疑似對單一行情過擬合，不採用，繼續搜尋...")
        else:
            gap = target_wr - rb["wr"] * 100
            print(f"     距離目標差 {gap:.1f}%，繼續縮小搜尋範圍...")

    # ── 儲存最佳結果 ────────────────────────────────────────────────────────
    print(f"\n{'═'*64}")
    print(f"  💾 儲存最佳結果到 GSheets")
    print(f"{'═'*64}")

    if not best_result:
        print("  ❌ 無有效結果，請嘗試增加股票數量或延長歷史年數")
        return

    br = best_result
    ep = {k: v for k, v in br["params"].items() if k not in ("rsi_exit","hold_days")}
    hd = br["params"]["hold_days"]

    # 完整 paper report（列印 + 回傳計算值，僅供人工查看，不代表會採用）
    result = sim.run_paper_report(br["trades"], market, br["params"], start_date, end_date, hd)

    if br.get("n_months", 0) < MIN_DISTINCT_MONTHS:
        print(f"\n  🛑 {max_rounds} 輪內找不到分散夠（≥{MIN_DISTINCT_MONTHS}個月）的結果 —")
        print(f"      全程最佳（WR={br['wr']:.1%}, n={br['n']}, 跨{br['n_months']}個月）疑似對單一")
        print(f"      時段過擬合，不寫入 GSheets、不更新 live params，維持現有設定不變。")
        print(f"      建議：延長 --years、增加股票數量，或接受較低的 --target 再試。")
        return

    if result:
        passed, annual_ret, cum_ret, sharpe, max_consec = result
        sim._save_paper_result(market, br["trades"], br["params"],
                               start_date, end_date, passed,
                               annual_ret, cum_ret, sharpe, max_consec)

    sim._save_live_params(market, ep, br["wr"] * 100, br["sharpe"])
    sim._notify_cloud_reload(market)
    print(f"\n  ✅ 參數已更新 GSheets 並通知雲端重載")

    if br["wr"] < target_wr / 100:
        print(f"\n  ⚠️  {max_rounds} 輪後最高達 {br['wr']:.1%}，未達目標 {target_wr}%")
        print(f"      台股市場歷史 70-75% WR 已屬頂尖，建議:")
        print(f"      1. 將目標調低至 70-72%  (--target 72)")
        print(f"      2. 增加股票掃描數量  (update_universe.py)")
        print(f"      3. 延長歷史年數  (--years 7)")


# ══════════════════════════════════════════════════════════════════════════════
#  Entry point
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="自動優化直到勝率達標")
    parser.add_argument("--market",  default="tw",   choices=["tw","us","both"],
                        help="市場 (default: tw)")
    parser.add_argument("--target",  default=80.0,   type=float,
                        help="目標勝率 %% (default: 80)")
    parser.add_argument("--rounds",  default=8,      type=int,
                        help="最大迭代輪數 (default: 8)")
    parser.add_argument("--years",   default=5,      type=int,
                        help="歷史回測年數 (default: 5)")
    args = parser.parse_args()

    markets = ["tw", "us"] if args.market == "both" else [args.market]
    for m in markets:
        auto_optimize(m, args.target, args.rounds, args.years)
