"""趋势线与下降通道。

趋势线是动态值：同一条线每根 bar 给出不同价格。因此本模块的输出必须带
`as_of_bar` 和"约/参考位"标注，绝不能被下游当成固定水平价。

拟合规则冻结为：只用算法产出的摆动点作锚点，枚举点对连线，
以"是否被显著穿透"和"触碰次数"排序。不允许为了贴合某个目标价手挑锚点。
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Literal, Sequence

import numpy as np
import pandas as pd

from ..indicators.atr import atr as atr_series
from ..indicators.swing import SwingPoint

#: 判定"触碰"与"穿透"的容差（ATR 倍数）。趋势线永远是区域，不是一像素的线
TOUCH_ATR = 0.25

#: 参与枚举的最近摆动点数量
MAX_ANCHOR_POINTS = 10

#: 一条线至少要有的触碰数（含两个锚点）
MIN_TOUCHES = 2


@dataclass(frozen=True)
class TrendLine:
    kind: Literal["up", "down"]
    slope: float
    x0: int
    y0: float
    anchors: list[dict[str, Any]]
    touches: int
    current_value: float
    as_of_bar: str
    broken_at: str | None
    span_bars: int

    def value_at(self, x: int) -> float:
        return self.y0 + self.slope * (x - self.x0)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["display"] = f"约 {self.current_value:,.2f}"
        d["dynamic"] = True
        d["note"] = "动态趋势线，每根 bar 数值不同；须标注数据时间，不得当作固定水平位"
        return d


def _fit(
    df: pd.DataFrame,
    a: SwingPoint,
    b: SwingPoint,
    kind: Literal["up", "down"],
    atr_v: np.ndarray,
    touch_atr: float,
) -> TrendLine | None:
    if b.bar_index <= a.bar_index:
        return None
    slope = (b.price - a.price) / (b.bar_index - a.bar_index)
    if kind == "up" and slope <= 0:
        return None
    if kind == "down" and slope >= 0:
        return None

    series = df["low"] if kind == "up" else df["high"]
    vals = series.to_numpy(dtype=float)
    n = len(vals)
    xs = np.arange(a.bar_index, n)
    line = a.price + slope * (xs - a.bar_index)
    tol = np.where(np.isfinite(atr_v[a.bar_index :]), atr_v[a.bar_index :], 0.0) * touch_atr

    seg = vals[a.bar_index :]
    if kind == "up":
        pierced = seg < (line - tol)
    else:
        pierced = seg > (line + tol)

    broken_at = None
    idx = np.flatnonzero(pierced)
    # 只有发生在第二个锚点之后的穿透才算破位；之前的穿透说明这条线画错了
    after = idx[idx + a.bar_index > b.bar_index]
    before = idx[idx + a.bar_index <= b.bar_index]
    if before.size:
        return None
    if after.size:
        broken_at = df.index[a.bar_index + int(after[0])].isoformat()

    touches = int(np.count_nonzero(np.abs(seg - line) <= tol))
    if touches < MIN_TOUCHES:
        return None

    return TrendLine(
        kind=kind,
        slope=float(slope),
        x0=a.bar_index,
        y0=float(a.price),
        anchors=[
            {"ts": a.ts.isoformat(), "price": round(a.price, 4)},
            {"ts": b.ts.isoformat(), "price": round(b.price, 4)},
        ],
        touches=touches,
        current_value=float(a.price + slope * (n - 1 - a.bar_index)),
        as_of_bar=df.index[-1].isoformat(),
        broken_at=broken_at,
        span_bars=int(n - 1 - a.bar_index),
    )


def fit_trendlines(
    df: pd.DataFrame,
    swings: Sequence[SwingPoint],
    kind: Literal["up", "down"],
    *,
    touch_atr: float = TOUCH_ATR,
    max_points: int = MAX_ANCHOR_POINTS,
    top_n: int = 2,
) -> list[TrendLine]:
    """枚举摆动点对，返回最优的若干条趋势线。

    上升趋势线连低点，下降趋势线连高点。排序优先未破位、其次触碰多、再次跨度长。
    """
    want = "low" if kind == "up" else "high"
    pts = [p for p in swings if p.kind == want][-max_points:]
    if len(pts) < 2:
        return []
    atr_v = atr_series(df).to_numpy()

    lines: list[TrendLine] = []
    for i, a in enumerate(pts):
        for b in pts[i + 1 :]:
            line = _fit(df, a, b, kind, atr_v, touch_atr)
            if line is not None:
                lines.append(line)

    lines.sort(key=lambda t: (t.broken_at is None, t.touches, t.span_bars), reverse=True)
    return lines[:top_n]


def parallel_channel(
    df: pd.DataFrame, line: TrendLine, swings: Sequence[SwingPoint]
) -> dict[str, Any] | None:
    """把趋势线平移到反向摆动的极值，构成通道。上轨是压力，下轨是支撑。"""
    want = "high" if line.kind == "up" else "low"
    pts = [p for p in swings if p.kind == want and p.bar_index >= line.x0]
    if not pts:
        return None
    offsets = [p.price - line.value_at(p.bar_index) for p in pts]
    pick = max(offsets) if line.kind == "up" else min(offsets)
    anchor = pts[int(np.argmax(offsets) if line.kind == "up" else np.argmin(offsets))]
    n = len(df)
    return {
        "offset": float(pick),
        "anchor": {"ts": anchor.ts.isoformat(), "price": round(anchor.price, 4)},
        "current_value": float(line.value_at(n - 1) + pick),
        "as_of_bar": df.index[-1].isoformat(),
        "role": "channel_upper" if line.kind == "up" else "channel_lower",
        "dynamic": True,
    }
