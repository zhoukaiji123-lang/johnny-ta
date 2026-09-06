"""价格行为与量能反应。

共振八项里的第 6 项。Johnny 方法反复强调"触线不等于买入，必须有价格确认"，
因此这里检测的是**某个价位附近是否真的发生过反应**，而不是价位本身。
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import TYPE_CHECKING, Any, Sequence

if TYPE_CHECKING:
    from .swing import SwingPoint

import numpy as np
import pandas as pd

#: 下影/上影至少要占整根 bar 振幅的比例
WICK_RATIO = 0.5

#: 影线相对实体的最小倍数
WICK_BODY_MULT = 2.0

#: 放量门槛（相对前 20 根均量）
VOLUME_SPIKE_MULT = 1.5
VOLUME_LOOKBACK = 20


@dataclass(frozen=True)
class Reaction:
    ts: str
    signals: list[str]
    price_tested: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def bar_signals(df: pd.DataFrame) -> pd.DataFrame:
    o, h, l, c = (df[k].to_numpy(dtype=float) for k in ("open", "high", "low", "close"))
    v = df["volume"].to_numpy(dtype=float)
    rng = np.maximum(h - l, 1e-12)
    body = np.abs(c - o)
    upper = h - np.maximum(o, c)
    lower = np.minimum(o, c) - l

    long_lower = (lower >= WICK_RATIO * rng) & (lower >= WICK_BODY_MULT * np.maximum(body, 1e-12))
    long_upper = (upper >= WICK_RATIO * rng) & (upper >= WICK_BODY_MULT * np.maximum(body, 1e-12))

    prev_o, prev_c = np.roll(o, 1), np.roll(c, 1)
    bull_engulf = (c > o) & (prev_c < prev_o) & (c >= prev_o) & (o <= prev_c)
    bear_engulf = (c < o) & (prev_c > prev_o) & (c <= prev_o) & (o >= prev_c)
    bull_engulf[0] = bear_engulf[0] = False

    avg_v = pd.Series(v).rolling(VOLUME_LOOKBACK, min_periods=5).mean().to_numpy()
    with np.errstate(invalid="ignore"):
        vol_spike = v >= VOLUME_SPIKE_MULT * avg_v
        vol_dry = v <= 0.6 * avg_v
    vol_spike = np.nan_to_num(vol_spike, nan=0).astype(bool)
    vol_dry = np.nan_to_num(vol_dry, nan=0).astype(bool)

    return pd.DataFrame(
        {
            "long_lower_wick": long_lower,
            "long_upper_wick": long_upper,
            "bullish_engulfing": bull_engulf,
            "bearish_engulfing": bear_engulf,
            "volume_spike": vol_spike,
            "volume_dry_up": vol_dry,
        },
        index=df.index,
    )


SUPPORT_SIGNALS = ("long_lower_wick", "bullish_engulfing", "volume_dry_up", "volume_spike")
RESISTANCE_SIGNALS = ("long_upper_wick", "bearish_engulfing", "volume_spike")


def reactions_near(
    df: pd.DataFrame,
    price: float,
    side: str,
    *,
    tolerance: float,
    lookback: int = 120,
    signals: pd.DataFrame | None = None,
) -> list[Reaction]:
    """某价位附近是否出现过带确认信号的真实触碰。

    支撑侧看 low 是否触及并出现止跌信号；压力侧看 high 是否触及并出现拒绝信号。
    """
    if tolerance <= 0 or df.empty:
        return []
    sig = signals if signals is not None else bar_signals(df)
    window = df.tail(lookback)
    sig = sig.loc[window.index]

    wanted = SUPPORT_SIGNALS if side == "support" else RESISTANCE_SIGNALS
    probe = window["low"] if side == "support" else window["high"]
    touched = (probe - price).abs() <= tolerance

    out: list[Reaction] = []
    for ts in window.index[touched]:
        fired = [s for s in wanted if bool(sig.at[ts, s])]
        if fired:
            out.append(Reaction(ts.isoformat(), fired, round(float(probe.loc[ts]), 4)))
    return out


def double_bottom_near(
    price: float,
    swings: "Sequence[SwingPoint]",
    *,
    tolerance: float,
    min_rally: float,
) -> bool:
    """价位附近是否形成过双底（形态学定义）。

    必须是**相邻的两个摆动低点**落在同一水平，且中间的摆动高点高出足够幅度。
    早先按"最近 N 根里任意两次触及"实现会在高波动标的上恒为真——
    MU 的日 ATR 约 68，120 根里任何价位都能找到两次触碰和一次大反弹。
    复用已冻结的摆动点序列，避免再引入一个可调自由度。
    """
    if tolerance <= 0 or min_rally <= 0:
        return False
    for i in range(len(swings) - 2):
        a, mid, b = swings[i], swings[i + 1], swings[i + 2]
        if a.kind != "low" or mid.kind != "high" or b.kind != "low":
            continue
        if abs(a.price - price) > tolerance or abs(b.price - price) > tolerance:
            continue
        if mid.price - max(a.price, b.price) >= min_rally:
            return True
    return False


def double_top_near(
    price: float,
    swings: "Sequence[SwingPoint]",
    *,
    tolerance: float,
    min_drop: float,
) -> bool:
    """双顶，与双底对称。"""
    if tolerance <= 0 or min_drop <= 0:
        return False
    for i in range(len(swings) - 2):
        a, mid, b = swings[i], swings[i + 1], swings[i + 2]
        if a.kind != "high" or mid.kind != "low" or b.kind != "high":
            continue
        if abs(a.price - price) > tolerance or abs(b.price - price) > tolerance:
            continue
        if min(a.price, b.price) - mid.price >= min_drop:
            return True
    return False
