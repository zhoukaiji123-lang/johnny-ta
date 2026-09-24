"""Wilder ADX。

只用于大盘状态判定（regime.py），衡量趋势强度而不是方向。
与 atr.py 一样按 Wilder 原始定义实现：首值取前 period 个值的简单平均，
其后 RMA 递推，不直接用 ewm——起点差异会一路带到当前值。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .atr import DEFAULT_PERIOD, true_range


def _rma(values: np.ndarray, period: int, start: int) -> np.ndarray:
    """从 start 起取 period 个值的均值做首值，之后 Wilder 递推。"""
    out = np.full(len(values), np.nan)
    first = start + period - 1
    if first >= len(values):
        return out
    out[first] = values[start:first + 1].mean()
    for i in range(first + 1, len(values)):
        out[i] = (out[i - 1] * (period - 1) + values[i]) / period
    return out


def adx(df: pd.DataFrame, period: int = DEFAULT_PERIOD) -> pd.Series:
    """需要至少 2 × period 根 bar 才有首个值，之前为 NaN。"""
    up = df["high"].diff().to_numpy()
    down = -df["low"].diff().to_numpy()
    plus_dm = np.where((up > down) & (up > 0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)
    tr = true_range(df).to_numpy()

    # 首根没有前一根，DM 无定义，平滑从第二根开始
    tr_s = _rma(tr, period, 1)
    plus_s = _rma(plus_dm, period, 1)
    minus_s = _rma(minus_dm, period, 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        plus_di = 100 * plus_s / tr_s
        minus_di = 100 * minus_s / tr_s
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)

    out = np.full(len(df), np.nan)
    valid = np.flatnonzero(np.isfinite(dx))
    if valid.size:
        out = _rma(np.nan_to_num(dx), period, int(valid[0]))
    return pd.Series(out, index=df.index, name=f"adx{period}")
