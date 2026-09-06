"""四套斐波那契。

设计上的硬约束：**所有 Fib 锚点只能是 SwingPoint**，不接受裸价格。
这样锚点必然来自 swing.py 的算法输出，带 confirmed_at，可被前向测试约束。
原 skill 允许分析者手选台阶底，这正是"嵌套锚点 Fib 能凑出任意价位"的来源。

四套的分工（严格区分，不得混用锚点）：
  primary_retracement   主升波段回撤 → 支撑    H - r(H-L)
  primary_rebound       主跌波段反弹 → 压力    L + r(H-L)
  local_navigation      最近完成的短波段全比例 → 现价附近逐级导航
  nested_anchors        固定同一极值，对多个突破基座取同一比例 → 逐级防线
  extension             早期推动段扩展 → 过度延伸后的目标
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Any, Literal, Sequence

from ..indicators.swing import SwingPoint

RETRACEMENT_RATIOS = (0.236, 0.382, 0.5, 0.618, 0.786)
NAVIGATION_RATIOS = (0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0)
EXTENSION_RATIOS = (1.618, 2.0, 2.618)

#: 嵌套锚点最多允许几个台阶。放开数量等于放开自由度
MAX_NESTED_ANCHORS = 4

#: 嵌套锚点默认只用黄金分割；要用别的比例必须显式传入并说明理由
NESTED_DEFAULT_RATIOS = (0.618,)

FibKind = Literal[
    "primary_retracement",
    "primary_rebound",
    "local_navigation",
    "nested_anchor",
    "extension",
]


@dataclass(frozen=True)
class FibLevel:
    price: float
    ratio: float
    kind: FibKind
    side: Literal["support", "resistance", "neutral"]
    timeframe: str
    anchor_low: dict[str, Any]
    anchor_high: dict[str, Any]
    confirmed_at: str | None = None
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Swing:
    """一段用于计算 Fib 的完整波段。"""

    low: SwingPoint
    high: SwingPoint
    direction: Literal["up", "down"]
    timeframe: str

    @property
    def span(self) -> float:
        return self.high.price - self.low.price

    @property
    def confirmed_at(self):
        return max(self.low.confirmed_at, self.high.confirmed_at)

    def describe(self) -> dict[str, Any]:
        return {
            "direction": self.direction,
            "timeframe": self.timeframe,
            "low": {"ts": self.low.ts.isoformat(), "price": round(self.low.price, 4)},
            "high": {"ts": self.high.ts.isoformat(), "price": round(self.high.price, 4)},
            "span": round(self.span, 4),
            "confirmed_at": self.confirmed_at.isoformat(),
        }


def _anchor(p: SwingPoint) -> dict[str, Any]:
    return {"ts": p.ts.isoformat(), "price": round(p.price, 4), "kind": p.kind}


# ------------------------------------------------------------------ 波段选择


def select_dominant_swing(
    points: Sequence[SwingPoint],
    direction: Literal["up", "down"],
    *,
    timeframe: str,
    lookback: int = 12,
) -> Swing | None:
    """选主导波段：在最近 lookback 个摆动点内取**幅度最大**的一段。

    规则冻结为"幅度最大"，而不是"看起来改变了结构"。前者可复现、可回测；
    后者是原 skill 留给分析者的自由度，也是事后贴合的入口。
    """
    pts = list(points)[-lookback:]
    best: Swing | None = None
    for i, a in enumerate(pts):
        for b in pts[i + 1 :]:
            if direction == "up" and a.kind == "low" and b.kind == "high":
                cand = Swing(a, b, "up", timeframe)
            elif direction == "down" and a.kind == "high" and b.kind == "low":
                cand = Swing(b, a, "down", timeframe)
            else:
                continue
            if cand.span <= 0:
                continue
            if best is None or cand.span > best.span:
                best = cand
    return best


def select_recent_swing(
    points: Sequence[SwingPoint], *, timeframe: str
) -> Swing | None:
    """最近一段完成的摆动，用于局部导航 Fib。方向由两点先后顺序决定。"""
    if len(points) < 2:
        return None
    a, b = points[-2], points[-1]
    if a.kind == b.kind:
        return None
    if a.kind == "low":
        return Swing(a, b, "up", timeframe)
    return Swing(b, a, "down", timeframe)


# ------------------------------------------------------------------ 各套 Fib


def primary_retracement(
    swing: Swing, ratios: Sequence[float] = RETRACEMENT_RATIOS
) -> list[FibLevel]:
    """上涨后的回撤支撑：H - r × (H-L)。"""
    h, l = swing.high.price, swing.low.price
    return [
        FibLevel(
            price=h - r * (h - l),
            ratio=r,
            kind="primary_retracement",
            side="support",
            timeframe=swing.timeframe,
            anchor_low=_anchor(swing.low),
            anchor_high=_anchor(swing.high),
            confirmed_at=swing.confirmed_at.isoformat(),
            note=f"{r:.3f} 回撤（{r * 100:.1f}%，非 {r}%）",
        )
        for r in ratios
    ]


def primary_rebound(
    swing: Swing, ratios: Sequence[float] = RETRACEMENT_RATIOS
) -> list[FibLevel]:
    """下跌后的反弹压力：L + r × (H-L)。"""
    h, l = swing.high.price, swing.low.price
    return [
        FibLevel(
            price=l + r * (h - l),
            ratio=r,
            kind="primary_rebound",
            side="resistance",
            timeframe=swing.timeframe,
            anchor_low=_anchor(swing.low),
            anchor_high=_anchor(swing.high),
            confirmed_at=swing.confirmed_at.isoformat(),
            note=f"{r:.3f} 反弹位",
        )
        for r in ratios
    ]


def local_navigation(
    swing: Swing, current_price: float, ratios: Sequence[float] = NAVIGATION_RATIOS
) -> list[FibLevel]:
    """局部导航 Fib：同一对锚点、全比例，只负责排列现价上下的下一档。

    落在现价附近的比例标为 neutral（中轴），不强行归入支撑或压力侧。
    """
    h, l = swing.high.price, swing.low.price
    band = abs(h - l) * 0.01
    out: list[FibLevel] = []
    for r in ratios:
        price = h - r * (h - l) if swing.direction == "up" else l + r * (h - l)
        if abs(price - current_price) <= band:
            side = "neutral"
        else:
            side = "support" if price < current_price else "resistance"
        out.append(
            FibLevel(
                price=price,
                ratio=r,
                kind="local_navigation",
                side=side,
                timeframe=swing.timeframe,
                anchor_low=_anchor(swing.low),
                anchor_high=_anchor(swing.high),
                confirmed_at=swing.confirmed_at.isoformat(),
                note="局部导航，仅表达路径，不单独构成入场",
            )
        )
    return out


def nested_anchors(
    apex: SwingPoint,
    steps: Sequence[SwingPoint],
    *,
    timeframe: str,
    ratios: Sequence[float] = NESTED_DEFAULT_RATIOS,
    max_anchors: int = MAX_NESTED_ANCHORS,
) -> list[FibLevel]:
    """嵌套锚点 Fib：固定极值 apex，对多个台阶取同一比例。

    台阶必须满足全部准入条件，不满足就返回空列表——宁可没有阶梯，
    也不允许用任意摆动点凑出一串"支撑"。
    """
    if apex.kind == "high":
        eligible = [
            s for s in steps if s.kind == "low" and s.is_base and s.ts < apex.ts
        ]
    else:
        eligible = [
            s for s in steps if s.kind == "high" and s.is_base and s.ts < apex.ts
        ]
    if not eligible:
        return []

    eligible = sorted(eligible, key=lambda s: s.ts, reverse=True)[:max_anchors]
    side = "support" if apex.kind == "high" else "resistance"
    out: list[FibLevel] = []
    for step in eligible:
        for r in ratios:
            if apex.kind == "high":
                price = apex.price - r * (apex.price - step.price)
                lo, hi = step, apex
            else:
                price = apex.price + r * (step.price - apex.price)
                lo, hi = apex, step
            confirmed = max(
                apex.confirmed_at, step.base_confirmed_at or step.confirmed_at
            )
            out.append(
                FibLevel(
                    price=price,
                    ratio=r,
                    kind="nested_anchor",
                    side=side,
                    timeframe=timeframe,
                    anchor_low=_anchor(lo),
                    anchor_high=_anchor(hi),
                    confirmed_at=confirmed.isoformat(),
                    note="台阶已通过突破基座校验（放量 + 反弹幅度）",
                )
            )
    return sorted(out, key=lambda f: f.price, reverse=True)


def extension(
    swing: Swing, ratios: Sequence[float] = EXTENSION_RATIOS
) -> list[FibLevel]:
    """早期推动段扩展。锚点必须与主波段分开披露，不得为制造共振而复用。"""
    h, l = swing.high.price, swing.low.price
    out: list[FibLevel] = []
    for r in ratios:
        price = l + r * (h - l) if swing.direction == "up" else h - r * (h - l)
        out.append(
            FibLevel(
                price=price,
                ratio=r,
                kind="extension",
                side="resistance" if swing.direction == "up" else "support",
                timeframe=swing.timeframe,
                anchor_low=_anchor(swing.low),
                anchor_high=_anchor(swing.high),
                confirmed_at=swing.confirmed_at.isoformat(),
                note="扩展目标，需与回撤位或真实枢轴重合才算共振",
            )
        )
    return out


def visible_levels(levels: Sequence[FibLevel], as_of) -> list[FibLevel]:
    """按锚点确认时间过滤，防止用当时尚不可见的摆动画 Fib。"""
    if as_of is None:
        return list(levels)
    import pandas as pd

    ts = pd.Timestamp(as_of)
    return [
        f
        for f in levels
        if f.confirmed_at is None or pd.Timestamp(f.confirmed_at) <= ts
    ]
