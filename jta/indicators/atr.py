"""真实波幅与 Wilder ATR。

ATR 在本项目只有三个用途：执行容差、止损缓冲、仓位反推。
它**不能**用来吞并承担不同触发作用的相邻关键位——这是原 skill 明确的纪律，
也是最容易被"两个点在同一个 ATR 内所以合并"这类推理破坏的地方。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

DEFAULT_PERIOD = 14


def true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    # 首根没有前收，退化为当根振幅
    tr.iloc[0] = df["high"].iloc[0] - df["low"].iloc[0]
    return tr


def atr(df: pd.DataFrame, period: int = DEFAULT_PERIOD) -> pd.Series:
    """Wilder 平滑：首值取前 period 根 TR 的简单平均，其后 RMA 递推。

    与 ewm(alpha=1/period) 的差别只在起点，但起点差异会一路带到当前值，
    因此这里按 Wilder 原始定义实现，而不是直接用 ewm。
    """
    tr = true_range(df)
    n = len(tr)
    out = np.full(n, np.nan)
    if n < period:
        return pd.Series(out, index=df.index, name=f"atr{period}")
    out[period - 1] = tr.iloc[:period].mean()
    tr_v = tr.to_numpy()
    for i in range(period, n):
        out[i] = (out[i - 1] * (period - 1) + tr_v[i]) / period
    return pd.Series(out, index=df.index, name=f"atr{period}")


def atr_pct(df: pd.DataFrame, period: int = DEFAULT_PERIOD) -> pd.Series:
    return atr(df, period) / df["close"]
