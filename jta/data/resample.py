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


def align_to_daily(intraday: pd.DataFrame, daily: pd.DataFrame) -> pd.DataFrame:
    """把 4H 价格缩放到日线的复权口径。

    yfinance 日线 auto_adjust 会把历史分红折进价格，60m（4H 由它聚合）只做拆股调整，
    缓存又是分批抓的、各批复权基准不同：JNJ 2024 年的 4H 比日线高约 8%，QQQ 约 1.4%。
    计划价与关键位按日线算、成交与 4H 候选位按 4H 算，两套口径一混，共振打分和成交都被污染。
    这里以日线为准，按每天"日线收盘 / 当天最后一根 4H 收盘"缩放当天的 4H。

    只有 1 根 4H 的日子（半日市，或 Yahoo 缺了下午的 60m）不参与估计、沿用前一个
    完整交易日的因子：那根 bar 的收盘不是当天收盘，硬拉到日线收盘会把整根 bar 平移，
    2026-01-30 缺下午数据时 MU 上午那根会被压低 4.6%，凭空造出一根假下影线。
    因子只用当天及之前的数据（仅序列开头没有完整交易日时向后借），回放中不引入前视。
    """
    if intraday.empty or daily.empty:
        return intraday
    day = intraday.index.normalize()
    grp = intraday["close"].groupby(day)
    last = grp.last()
    dclose = daily["close"].copy()
    dclose.index = dclose.index.normalize()
    raw = (dclose.reindex(last.index) / last).replace([np.inf, -np.inf], np.nan)
    complete = grp.size() >= 2
    factor = raw.where(complete) if complete.any() else raw
    factor = factor.ffill().bfill().fillna(1.0)
    f = factor.reindex(day).to_numpy(dtype=float)
    out = intraday.copy()
    for col in ("open", "high", "low", "close"):
        out[col] = out[col].to_numpy(dtype=float) * f
    return out
