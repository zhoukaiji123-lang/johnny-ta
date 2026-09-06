"""周期重采样。

yfinance 没有原生 4H。美股 4H bar 的切法必须显式定义，否则同一个"4 小时收盘"
在不同工具里指向不同价格，后面所有 Fib / EMA / 确认口径都会错位。

本模块采用与 TradingView 美股 4H 一致的规则：以每个交易日的常规盘开盘
（通常 09:30 ET）为起点，按 4 小时切分，因此常规交易日得到 2 根 bar：
    bar1  09:30–13:30  （4 小时）
    bar2  13:30–16:00  （2.5 小时，非等宽，属于交易所日历的固有结果）
半日市（提前 13:00 收盘）当天只有 1 根 bar。

bar 时间戳一律使用 bar 的**开始**时间，与 yfinance 的日内约定一致。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

BAR_ALIGNMENT_4H = (
    "RTH-anchored: 每交易日自常规盘开盘起按 4 小时切分 "
    "(09:30-13:30 / 13:30-16:00 ET)，时间戳为 bar 开始时间"
)

AGG = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}


def to_4h(df60: pd.DataFrame, boundary_hours: int = 4) -> pd.DataFrame:
    """把 RTH 60m bar 聚合成 4H bar。

    输入必须是常规交易时段（prepost=False）的 60m 数据，索引为交易所本地时区。
    """
    if df60.empty:
        return df60.copy()
    if df60.index.tz is None:
        raise ValueError("60m 数据索引必须带时区")

    df60 = df60.sort_index()
    if df60.index.unit != "ns":
        df60 = df60.copy()
        df60.index = df60.index.as_unit("ns")
    idx = df60.index
    tz = idx.tz
    ns_per_hour = 3_600_000_000_000

    # 全程用 epoch 纳秒整数运算。groupby/transform 对 tz-aware Timestamp 会丢时区，
    # 而 transform 的结果可能降级为 float64——epoch 纳秒需要 61 位有效位，
    # float64 只有 53 位，会静默毁掉时间戳。
    ns = idx.asi8.astype("int64")
    day_key = idx.normalize().asi8.astype("int64")

    # 索引已升序，故每个交易日的首个 bar 即该日开盘 bar
    _, first_pos = np.unique(day_key, return_index=True)
    group_sizes = np.diff(np.append(first_pos, len(ns)))
    session_open_ns = np.repeat(ns[first_pos], group_sizes)

    slot = (ns - session_open_ns) // (boundary_hours * ns_per_hour)
    bar_start_ns = session_open_ns + slot * boundary_hours * ns_per_hour
    bar_start = pd.DatetimeIndex(
        pd.to_datetime(bar_start_ns, unit="ns", utc=True)
    ).tz_convert(tz)

    out = df60.groupby(bar_start).agg(AGG)
    out.index = pd.DatetimeIndex(out.index, name=df60.index.name or "Datetime")
    return out.sort_index()


def bars_per_day(df: pd.DataFrame) -> pd.Series:
    """每个交易日的 bar 数，用于体检异常缺口。"""
    if df.empty:
        return pd.Series(dtype=int)
    return df.groupby(df.index.normalize()).size()


def audit_4h(df4h: pd.DataFrame) -> list[str]:
    """返回可读的异常提示，不抛错——数据质量问题必须显式暴露给下游。"""
    warnings: list[str] = []
    if df4h.empty:
        return ["4H 序列为空"]
    counts = bars_per_day(df4h)
    odd = counts[(counts < 1) | (counts > 2)]
    if len(odd):
        sample = ", ".join(f"{d.date()}={n}" for d, n in odd.head(5).items())
        warnings.append(f"{len(odd)} 个交易日的 4H bar 数异常（期望 1-2 根）：{sample}")
    half = counts[counts == 1]
    if len(half):
        warnings.append(f"{len(half)} 个交易日只有 1 根 4H bar（半日市或数据缺失）")
    return warnings
