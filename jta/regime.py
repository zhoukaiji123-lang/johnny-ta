"""大盘状态判定：决定三套做多计划能不能开新仓。

规则在看回测结果之前定死，不调参：
  ADX(14) < 20                        → range（趋势强度不足）
  close > SMA200 且 SMA50 > SMA200    → up
  close < SMA200                      → down
  其余                                 → range

验证（25 个标的，2025-09 至 2026-09，距离匹配的随机入场对照）：
只在 up 时开仓，期望 +0.44R（n=411），被剔除的交易 +0.00R（n=419），
差 +0.44R、z=2.97，样本内与样本外两段方向一致。选点本身仍与随机入场
无差异——这个开关改善的是"什么时候做"，不是"在哪里做"。
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .indicators.adx import adx

ADX_PERIOD = 14
ADX_TREND_MIN = 20.0
SMA_FAST = 50
SMA_SLOW = 200

STATE_LABELS = {"up": "向上", "range": "震荡", "down": "向下", "unknown": "未知"}


def benchmark_regime(df: pd.DataFrame | None) -> dict[str, Any]:
    """df: 基准日线（小写列），最后一根为最近已收盘 bar。"""
    if df is None or len(df) < SMA_SLOW:
        return {
            "state": "unknown",
            "label": STATE_LABELS["unknown"],
            "reason": (
                "没有基准指数" if df is None
                else f"基准日线仅 {len(df)} 根，不足 {SMA_SLOW} 根，无法判定"
            ),
            "close": None, "sma50": None, "sma200": None, "adx": None,
        }

    close = df["close"]
    c = float(close.iloc[-1])
    s50 = float(close.rolling(SMA_FAST).mean().iloc[-1])
    s200 = float(close.rolling(SMA_SLOW).mean().iloc[-1])
    a = float(adx(df, ADX_PERIOD).iloc[-1])

    if not np.isfinite(a) or a < ADX_TREND_MIN:
        state = "range"
        reason = f"ADX {a:.1f} < {ADX_TREND_MIN:.0f}，趋势强度不足"
    elif c > s200 and s50 > s200:
        state = "up"
        reason = f"收盘 > SMA200 且 SMA50 > SMA200，ADX {a:.1f}"
    elif c < s200:
        state = "down"
        reason = f"收盘 {c:.2f} < SMA200 {s200:.2f}"
    else:
        state = "range"
        reason = f"SMA50 {s50:.2f} 未站上 SMA200 {s200:.2f}"

    return {
        "state": state,
        "label": STATE_LABELS[state],
        "reason": reason,
        "close": round(c, 4),
        "sma50": round(s50, 4),
        "sma200": round(s200, 4),
        "adx": round(a, 2),
    }
