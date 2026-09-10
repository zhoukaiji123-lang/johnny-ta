"""候选位构建、去重、贴合与角色标注。

这里实现原 skill 里最拗口也最关键的一条纪律：
**"距离近"不是合并理由，但"多个来源落在同一价位"要算共振。**

工程化的办法是把两个半径分开：
  DEDUP_ATR      极小（0.05 ATR）——只有数值几乎相同才合并成一个展示点；
  RESONANCE_ATR  较大（0.25 ATR）——统计一个点周围有哪些来源支持它。
用一个半径同时做这两件事，必然要么吞掉独立点位，要么算不出共振。
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Literal, Sequence

import numpy as np

Side = Literal["support", "resistance", "neutral"]

#: 数值去重半径。只用于合并"同一个数字"，不用于合并"相近的两个结构"
DEDUP_ATR = 0.05

#: 共振检测半径。多套证据落在这个范围内才算互相印证
RESONANCE_ATR = 0.25

#: Fib 原值贴合到真实枢轴的最大距离
SNAP_ATR = 0.25

#: 超过这个距离的候选不进入答案（现价 ± N × ATR）
MAX_DISTANCE_ATR = 6.0

SUPPORT_ROLES = ("immediate_defense", "executable_pullback", "core_defense")
RESISTANCE_ROLES = ("immediate_resistance", "breakout_confirmation", "higher_target")

ROLE_LABELS = {
    "immediate_defense": "即时持仓防守枢轴",
    "executable_pullback": "可执行回踩支撑",
    "core_defense": "更深核心防守",
    "immediate_resistance": "即时阻力/减仓位",
    "breakout_confirmation": "突破确认位",
    "higher_target": "更高止盈目标",
}

#: 哪些来源属于"动态"——每根 bar 数值都会变，必须标注约/参考位
DYNAMIC_SOURCES = {"ema", "vegas", "trendline", "channel"}


@dataclass
class Candidate:
    price: float
    side: Side
    sources: list[dict[str, Any]] = field(default_factory=list)
    display: float | None = None
    role: str | None = None
    snap: dict[str, Any] | None = None
    resonance: dict[str, Any] = field(default_factory=dict)
    distance_atr: float = 0.0

    @property
    def dynamic(self) -> bool:
        return any(s.get("family") in DYNAMIC_SOURCES for s in self.sources)

    @property
    def source_families(self) -> set[str]:
        return {s.get("family", "") for s in self.sources}

    @property
    def confirmed_at(self) -> str | None:
        stamps = [s.get("confirmed_at") for s in self.sources if s.get("confirmed_at")]
        return max(stamps) if stamps else None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["dynamic"] = self.dynamic
        d["confirmed_at"] = self.confirmed_at
        d["role_label"] = ROLE_LABELS.get(self.role or "", None)
        shown = self.display if self.display is not None else self.price
        text = f"{shown:,.10g}" if shown == int(shown) else f"{shown:,}"
        d["display_text"] = f"约 {text}" if self.dynamic else text
        d["raw_price"] = round(self.price, 4)
        return d


def make_source(
    family: str,
    label: str,
    *,
    timeframe: str,
    confirmed_at: str | None = None,
    detail: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "family": family,
        "label": label,
        "timeframe": timeframe,
        "confirmed_at": confirmed_at,
        "detail": detail or {},
    }


#: 展示粒度目标：约 0.1 个 ATR
DISPLAY_ATR_FRACTION = 0.1

#: 美股最小跳动，展示步长不得比它更细
MIN_DISPLAY_STEP = 0.01


def display_step(atr_value: float) -> float:
    """展示步长：把价格取整到约 0.1 ATR 的 1-2-5 阶梯上。

    锚点选择在合理范围内摇摆时，同一条 Fib 的结果会漂移接近一个 ATR
    （SNDK 的对照里，换一组台阶后三档支撑分别差 0.11–0.34 ATR）。
    既然固有精度就是 ATR 量级，把关键位报到个位数就是伪精度：
    1185.7 看起来比"约 1190"精确，实际上并不是。原始计算值仍保留在 price 字段。
    """
    if not np.isfinite(atr_value) or atr_value <= 0:
        return MIN_DISPLAY_STEP
    target = DISPLAY_ATR_FRACTION * atr_value
    exp = int(np.floor(np.log10(target)))
    base = 10.0**exp
    ladder = [base, 2 * base, 5 * base, 10 * base]
    step = min(ladder, key=lambda c: abs(c - target))
    return max(step, MIN_DISPLAY_STEP)


def quantize(price: float, step: float) -> float:
    if step <= 0:
        return price
    # 步长本身可能是 0.05 这类值，二次取整避免浮点尾巴
    decimals = max(0, -int(np.floor(np.log10(step))) + 2)
    return round(round(price / step) * step, decimals)


def build_candidates(
    raw: Sequence[tuple[float, dict[str, Any]]],
    current_price: float,
    atr_value: float,
    *,
    max_distance_atr: float = MAX_DISTANCE_ATR,
    dedup_atr: float = DEDUP_ATR,
) -> list[Candidate]:
    """把 (价格, 来源) 对合并成候选点。

    合并只发生在数值几乎相同、且位于现价同一侧的来源之间。

    合并判据是"新点是否落在簇的**锚点**（该簇第一个成员）半径内"，
    不是跟不断漂移的簇内均值比较。后者会链式传导：A-B 差 0.4、B-C 差 0.4，
    各自都在半径内，但 A-C 可能已经差出去一个 ATR——三个本不该合并的独立
    来源就这样被拼成一个跨度远超 dedup_atr 的候选点。锚定在第一个成员上，
    能保证任何一簇的总跨度都不超过 radius。
    """
    if atr_value <= 0 or not np.isfinite(atr_value):
        return []
    reach = max_distance_atr * atr_value
    items = [
        (p, s)
        for p, s in raw
        if np.isfinite(p) and p > 0 and abs(p - current_price) <= reach
    ]
    if not items:
        return []

    items.sort(key=lambda t: t[0])
    radius = dedup_atr * atr_value
    out: list[Candidate] = []
    anchors: list[float] = []
    members: list[list[float]] = []
    for price, src in items:
        side: Side = "support" if price < current_price else "resistance"
        if out and out[-1].side == side and price - anchors[-1] <= radius:
            cur = out[-1]
            cur.sources.append(src)
            members[-1].append(float(price))
            cur.price = float(np.mean(members[-1]))
            continue
        out.append(Candidate(price=float(price), side=side, sources=[src]))
        anchors.append(float(price))
        members.append([float(price)])

    step = display_step(atr_value)
    for c in out:
        c.distance_atr = round(abs(c.price - current_price) / atr_value, 3)
        c.display = quantize(c.price, step)
    return out


def annotate_resonance(
    candidates: Sequence[Candidate], atr_value: float, *, radius_atr: float = RESONANCE_ATR
) -> None:
    """统计每个候选点周围（而不是точно同价）有哪些来源族在印证它。

    这一步只记录事实，不做加权求和——八项证据彼此高度相关，
    等权相加得出的分数没有统计含义。
    """
    radius = radius_atr * atr_value
    for c in candidates:
        families: set[str] = set(c.source_families)
        neighbours: list[dict[str, Any]] = []
        for other in candidates:
            if other is c or abs(other.price - c.price) > radius:
                continue
            families |= other.source_families
            neighbours.append(
                {"price": round(other.price, 4), "families": sorted(other.source_families)}
            )
        c.resonance = {
            "radius_atr": radius_atr,
            "families": sorted(families),
            "neighbours": neighbours,
        }


def note_pivot_confluence(
    candidates: Sequence[Candidate], atr_value: float, *, snap_atr: float = SNAP_ATR
) -> None:
    """记录 Fib 原值与真实枢轴的重合关系。

    原 skill 允许把 Fib 原值"贴合"到附近枢轴上展示。放到确定性计算里这是有害的：
    贴合会让两个独立候选显示成同一个数字，读者无法分辨逐级路径。
    因此这里只记录重合事实——展示价仍由该候选自己的计算值量化而来，
    重合信息进证据栏。
    """
    tol = snap_atr * atr_value
    pivots = [c for c in candidates if "pivot" in c.source_families]
    for c in candidates:
        if "pivot" in c.source_families or not c.source_families & {"fib", "extension"}:
            continue
        near = [p for p in pivots if abs(p.price - c.price) <= tol and p.side == c.side]
        if not near:
            continue
        target = min(near, key=lambda p: abs(p.price - c.price))
        c.snap = {
            "raw_price": round(c.price, 4),
            "confluent_pivot": round(target.price, 4),
            "distance_atr": round(abs(target.price - c.price) / atr_value, 3),
            "reason": "Fib 原值与有真实触碰记录的水平枢轴重合；展示价仍取自身计算值",
        }


def assign_roles(candidates: Sequence[Candidate], current_price: float) -> None:
    """按距现价的远近分配结构角色。

    角色决定该点用于持仓管理还是新开仓；两个角色不同的点即使价格接近也不合并。
    """
    supports = sorted(
        [c for c in candidates if c.side == "support"], key=lambda c: -c.price
    )
    resistances = sorted(
        [c for c in candidates if c.side == "resistance"], key=lambda c: c.price
    )
    for i, c in enumerate(supports):
        c.role = SUPPORT_ROLES[min(i, len(SUPPORT_ROLES) - 1)]
    for i, c in enumerate(resistances):
        c.role = RESISTANCE_ROLES[min(i, len(RESISTANCE_ROLES) - 1)]
