"""大盘状态判定：决定三套做多计划能不能开新仓。

规则在看回测结果之前定死，不调参：
  ADX(14) < 20                        → range（趋势强度不足）
  close > SMA200 且 SMA50 > SMA200    → up
  close < SMA200                      → down
  其余                                 → range

验证（25 个标的，2025-09 至 2026-09，距离匹配的随机入场对照）：
最初的结果是 up 时期望 +0.44R、被剔除的交易 +0.00R，z=2.97。2026-10 修掉回测的口径问题
（计划日与突破确认日当天的 4H 参与成交、4H 与日线复权口径不一致、4H 量能不分时段、
跳空成交贴近止损时 R 被放大）后重跑：真实计划 up 时 +0.07R（n=273），被剔除的 +0.08R（n=300），
差 -0.01R、z=-0.07；距离匹配随机入场上同样过滤 -0.09R（z=-0.50）；牛市段 -0.06R、回调段 +0.02R。
区分度完全消失。默认仍开启，但它现在是风控规则而不是被验证过的择时信号：
样本里「向下」只有 10 笔，开关真正防范的下跌市没有被检验过。详见 README「口径修正后重跑」。
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
