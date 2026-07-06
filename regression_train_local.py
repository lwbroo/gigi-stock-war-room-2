#!/usr/bin/env python3
"""
regression_train_local.py — v11.0 本機 OLS 回歸訓練（取代雲端 /api/regression/train，
該端點現在只是個 stub，從沒被移植到本機過）。

用 scan_log（僅台股）過去 10 天以上的掃描紀錄，對 12 個技術指標做 OLS 回歸，
預測 10 天後的實際報酬率。係數寫回 regression_coeffs，供前端顯示。

用法:
  python3 regression_train_local.py
  python3 regression_train_local.py --dry-run
"""
import argparse
import datetime as _dt

import numpy as np
import pandas as pd

import simulate as sim  # reuse _get_or_create_tab / _fetch_ohlcv

SCAN_LOG_TAB = "scan_log"
REG_COEFFS_TAB = "regression_coeffs"
FEATURE_NAMES = [
    "macd_num", "adx_norm", "obv_num", "monthly_num",
    "breakout20", "vol_exp", "inst_fgn_k", "inst_tst_k",
    "rs_clip", "weekly_num", "rsi_norm", "bias_pct",
]


def _row_to_features(r: dict) -> list:
    macd_map = {"golden": 2.0, "above": 1.0, "none": 0.0, "below": -1.0, "death": -2.0}
    monthly = r.get("monthly_trend")
    weekly = r.get("weekly_trend")
    return [
        macd_map.get(str(r.get("macd_cross") or "none"), 0.0),
        min(float(r.get("adx14") or 25), 50.0) / 50.0,
        1.0 if str(r.get("obv_trend")) == "rising" else (-1.0 if str(r.get("obv_trend")) == "falling" else 0.0),
        1.0 if monthly in (True, "True", "true", "1") else (-1.0 if monthly in (False, "False", "false", "0") else 0.0),
        1.0 if str(r.get("is_breakout20")) in ("True", "true", "1") else 0.0,
        1.0 if str(r.get("vol_expansion")) in ("True", "true", "1") else 0.0,
        max(-10.0, min(10.0, float(r.get("inst_foreign") or 0) / 1000.0)),
        max(-5.0, min(5.0, float(r.get("inst_trust") or 0) / 1000.0)),
        max(-3.0, min(3.0, float(r.get("rs_score") or 0))),
        1.0 if weekly in (True, "True", "true", "1") else (-1.0 if weekly in (False, "False", "false", "0") else 0.0),
        (float(r.get("rsi14") or 50) - 50.0) / 50.0,
        float(r.get("bias") or 0),
    ]


def main():
    ap = argparse.ArgumentParser(description="Train OLS regression on scan_log (TW only)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    ws = sim._get_or_create_tab(SCAN_LOG_TAB, ["scan_date", "ticker", "close",
                                                "macd_cross", "adx14", "obv_trend", "monthly_trend",
                                                "is_breakout20", "vol_expansion", "inst_foreign", "inst_trust",
                                                "rs_score", "weekly_trend", "rsi14", "bias", "is_buy"])
    records = ws.get_all_records()
    if not records:
        print("scan_log 是空的，跳過（需要每天掃描累積資料）")
        return

    cutoff = (_dt.datetime.now() - _dt.timedelta(days=10)).strftime("%Y-%m-%d")
    eligible = [r for r in records if str(r.get("scan_date", "")) <= cutoff and r.get("ticker")]
    if len(eligible) < 20:
        print(f"資料不足：需要 20+ 筆10天以上的紀錄（目前 {len(eligible)} 筆）。持續每天掃描即可累積。")
        return

    seen, deduped = set(), []
    for r in eligible:
        key = (r["scan_date"], r["ticker"])
        if key not in seen:
            seen.add(key)
            deduped.append(r)

    print(f"[1] {len(deduped)} 筆去重後的紀錄，抓取股價...")
    unique_tickers = sorted(set(r["ticker"] for r in deduped))
    price_cache = {}
    for ticker in unique_tickers:
        clean = ticker.split(".")[0]
        rows_for_t = [r["scan_date"] for r in deduped if r["ticker"] == ticker]
        fetch_start = min(rows_for_t)
        df = sim._fetch_ohlcv(clean, fetch_start, _dt.date.today().isoformat(), market="tw")
        if df is not None and not df.empty:
            price_cache[ticker] = df

    print("[2] 建立特徵矩陣...")
    X_rows, y_vals = [], []
    for r in deduped:
        ticker = r["ticker"]
        if ticker not in price_cache:
            continue
        try:
            df = price_cache[ticker]
            scan_date = pd.Timestamp(r["scan_date"])
            dates = df["date"]
            idx = df.index[dates >= scan_date]
            if len(idx) == 0:
                continue
            base = idx[0]
            if base + 10 >= len(df):
                continue
            entry = float(df.loc[base, "Close"])
            exit_ = float(df.loc[base + 10, "Close"])
            ret = (exit_ - entry) / entry
            X_rows.append(_row_to_features(r))
            y_vals.append(ret)
        except Exception:
            continue

    n = len(X_rows)
    if n < 20:
        print(f"配對股價後只剩 {n} 筆有效樣本（需要 20+）。持續每天掃描即可累積。")
        return

    X = np.array(X_rows, dtype=float)
    y = np.array(y_vals, dtype=float)
    X_b = np.column_stack([np.ones(n), X])
    coeffs, _, _, _ = np.linalg.lstsq(X_b, y, rcond=None)
    y_pred = X_b @ coeffs
    ss_res = float(np.sum((y - y_pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0
    intercept = float(coeffs[0])
    feat_coeffs = [float(c) for c in coeffs[1:]]

    print(f"[3] OLS 完成: n={n}  R²={r2:.4f}")
    for name, c in zip(FEATURE_NAMES, feat_coeffs):
        print(f"    {name:14s} {c:+.4f}")

    if args.dry_run:
        print("[dry-run] 不寫入 GSheets")
        return

    ws_reg = sim._get_or_create_tab(REG_COEFFS_TAB, ["feature", "value"])
    ws_reg.clear()
    ws_reg.update("A1", [["feature", "value"]] + [
        ["_r2", str(round(r2, 6))],
        ["_n", str(n)],
        ["_updated", _dt.datetime.now().strftime("%Y-%m-%d %H:%M")],
        ["_intercept", str(intercept)],
    ] + [[name, str(c)] for name, c in zip(FEATURE_NAMES, feat_coeffs)])
    print("[4] 已寫入 regression_coeffs")


if __name__ == "__main__":
    main()
