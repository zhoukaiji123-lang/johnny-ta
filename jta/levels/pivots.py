"""水平结构：枢轴、前一交易日高低收、缺口、整数关口。

Johnny 方法把"反复切换支撑/压力的水平位"排在 Fib 之前——先看裸 K 与成交量，
再叠指标。因此这里的枢轴强度只由**真实触碰次数**决定，不掺任何指标。
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Literal, Sequence

import numpy as np
import pandas as pd

from ..indicators.atr import atr as atr_series
from ..indicators.swing import SwingPoint

#: 枢轴聚类半径（ATR 倍数）。同一水平的多次触碰会散布在一个小范围内
DEFAULT_CLUSTER_ATR = 0.25

#: 计入枢轴所需的最少触碰次数。1 次不构成"反复反应"
MIN_TOUCHES = 2

#: 缺口的最小幅度（ATR 倍数），滤掉一跳的伪缺口
MIN_GAP_ATR = 0.3


@dataclass(frozen=True)
class Level:
    price: float
    source: str
    side: Literal["support", "resistance", "neutral"]
    timeframe: str
    detail: dict[str, Any]
    confirmed_at: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _side(price: float, current: float) -> str:
    return "support" if price < current else "resistance"


def horizontal_pivots(
    swings: Sequence[SwingPoint],
    df: pd.DataFrame,
    current_price: float,
    *,
    timeframe: str,
    cluster_atr: float = DEFAULT_CLUSTER_ATR,
    min_touches: int = MIN_TOUCHES,
) -> list[Level]:
    """把摆动点按价格聚类成水平枢轴，触碰次数即强度。"""
    if not swings:
        return []
    atr_v = atr_series(df).to_numpy()
    if not np.isfinite(atr_v).any():
        return []

    def radius_at(point: SwingPoint) -> float:
        """聚类半径取该摆动点**当时**的 ATR，而不是最新 ATR。

        用最新 ATR 做跨越大价格区间的聚类会灾难性失真：MU 现价 910 的 ATR 约 68，
        套到股价 100 附近的历史摆动上，会把几十个互不相干的低点糊成一个"枢轴"。
        """
        i = min(point.bar_index, len(atr_v) - 1)
        a = atr_v[i]
        if not np.isfinite(a) or a <= 0:
            finite = atr_v[np.isfinite(atr_v)]
            if finite.size == 0:
                return 0.0
            # 退化时按同期 ATR 比例换算到该点价位，仍不使用最新绝对值
            a = float(np.median(finite)) * point.price / float(df["close"].iloc[-1])
        return cluster_atr * float(a)

    ordered = sorted(swings, key=lambda s: s.price)
    clusters: list[list[SwingPoint]] = [[ordered[0]]]
    for p in ordered[1:]:
        anchor = clusters[-1][-1]
        if p.price - anchor.price <= radius_at(anchor):
            clusters[-1].append(p)
        else:
            clusters.append([p])

    out: list[Level] = []
    for c in clusters:
        if len(c) < min_touches:
            continue
        prices = [p.price for p in c]
        rep = float(np.median(prices))
        last = max(c, key=lambda p: p.ts)
        out.append(
            Level(
                price=rep,
                source="horizontal_pivot",
                side=_side(rep, current_price),
                timeframe=timeframe,
                confirmed_at=max(p.confirmed_at for p in c).isoformat(),
                detail={
                    "touches": len(c),
                    "price_range": [round(min(prices), 4), round(max(prices), 4)],
                    "first_touch": min(p.ts for p in c).isoformat(),
                    "last_touch": last.ts.isoformat(),
                    "kinds": sorted({p.kind for p in c}),
                    "flipped": len({p.kind for p in c}) > 1,
                },
            )
        )
    return out


def prior_session_levels(
    df: pd.DataFrame, current_price: float, *, timeframe: str
) -> list[Level]:
    """前一交易日高/低/收。

    按自然日分组取上一个完整交易日的 H/L/C，而不是简单地取 iloc[-2]。
    日线数据下两者等价（一天一根 bar）；但 4H 一天有两根 bar，iloc[-2]
    在下午那根 bar 上其实是"今天上午"，不是前一交易日——会产出一个跟现价
    只差几美分的伪"结构证据"，被合并算法当成独立支撑/压力，把关键位拽到
    贴着现价的位置。按日期分组能在任意 bar 粒度下都取到真正的前一交易日。
    """
    if len(df) < 2:
        return []
    dates = df.index.normalize()
    last_date = dates[-1]
    prior_mask = dates < last_date
    if not prior_mask.any():
        return []
    prior_date = dates[prior_mask][-1]
    day_df = df.loc[dates == prior_date]
    ts = day_df.index[-1].isoformat()
    values = {
        "prev_high": float(day_df["high"].max()),
        "prev_low": float(day_df["low"].min()),
        "prev_close": float(day_df["close"].iloc[-1]),
    }
    out = []
    for name, price in values.items():
        out.append(
            Level(
                price=price,
                source=name,
                side=_side(price, current_price),
                timeframe=timeframe,
                confirmed_at=df.index[-1].isoformat(),
                detail={"bar_ts": ts, "session_date": str(prior_date.date())},
            )
        )
    return out


def gaps(
    df: pd.DataFrame,
    current_price: float,
    *,
    timeframe: str,
    lookback: int = 120,
    min_gap_atr: float = MIN_GAP_ATR,
) -> list[Level]:
    """未回补缺口的边缘。已被完全回补的缺口不再作为候选位。"""
    if len(df) < 3:
        return []
    a = atr_series(df)
    window = df.tail(lookback)
    out: list[Level] = []
    highs, lows = df["high"], df["low"]

    for i in range(1, len(window)):
        ts = window.index[i]
        pos = df.index.get_loc(ts)
        prev_high, prev_low = float(highs.iloc[pos - 1]), float(lows.iloc[pos - 1])
        cur_high, cur_low = float(highs.iloc[pos]), float(lows.iloc[pos])
        scale = float(a.iloc[pos]) if np.isfinite(a.iloc[pos]) else 0.0
        if scale <= 0:
            continue

        if cur_low > prev_high and (cur_low - prev_high) >= min_gap_atr * scale:
            edge, kind = prev_high, "gap_up"
            filled = bool((lows.iloc[pos + 1 :] <= prev_high).any())
        elif cur_high < prev_low and (prev_low - cur_high) >= min_gap_atr * scale:
            edge, kind = prev_low, "gap_down"
            filled = bool((highs.iloc[pos + 1 :] >= prev_low).any())
        else:
            continue
        if filled:
            continue
        out.append(
            Level(
                price=edge,
                source="gap_edge",
                side=_side(edge, current_price),
                timeframe=timeframe,
                confirmed_at=ts.isoformat(),
                detail={
                    "gap_kind": kind,
                    "gap_ts": ts.isoformat(),
                    "gap_size": round(abs(cur_low - prev_high) if kind == "gap_up" else abs(prev_low - cur_high), 4),
                    "unfilled": True,
                },
            )
        )
    return out


def round_numbers(
    current_price: float, atr_value: float, *, timeframe: str, reach_atr: float = 3.0
) -> list[Level]:
    """现价附近的整数关口。

    整数本身不是证据——它在共振评分里与水平平台共用同一项，且合计只算一分。
    """
    if current_price <= 0 or not np.isfinite(atr_value) or atr_value <= 0:
        return []
    magnitude = 10 ** int(np.floor(np.log10(current_price)))
    step = magnitude / 2
    reach = reach_atr * atr_value
    lo = np.floor((current_price - reach) / step) * step
    hi = np.ceil((current_price + reach) / step) * step
    out: list[Level] = []
    price = lo
    while price <= hi + 1e-9:
        if price > 0 and abs(price - current_price) <= reach:
            out.append(
                Level(
                    price=float(price),
                    source="round_number",
                    side=_side(float(price), current_price),
                    timeframe=timeframe,
                    detail={"step": step, "note": "心理关口，不单独构成证据"},
                )
            )
        price += step
    return out
