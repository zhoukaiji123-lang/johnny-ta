"""摆动点检测。

这是整个项目里最要紧的一块：Johnny 方法中的"嵌套锚点 Fib"允许固定一个高点 H，
对多个历史台阶底 Lᵢ 取同一比例。只要 Lᵢ 可以由人手选，这个工具就能凑出任意价位——
原 skill 的证据文件自己承认 MU 的 770/756/715/660 是从答案逆向重建出来的锚点。

因此这里把锚点的产生方式冻结成算法：
  1. 分形检测（左右各 k 根严格极值）；
  2. 强制 high/low 交替；
  3. 按 ATR 倍数过滤幅度不足的摆动。
每个摆动点都带 confirmed_at（比摆动本身晚 k 根），下游做前向测试时只能使用
confirmed_at <= as_of 的点，杜绝用"当时还看不见的低点"事后画 Fib。
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Literal

import numpy as np
import pandas as pd

from .atr import atr as atr_series

Kind = Literal["high", "low"]

#: 分形左右确认 bar 数。日线偏大以过滤噪声，日内周期可调小
DEFAULT_K = {"1wk": 3, "1d": 3, "4h": 2, "60m": 2, "30m": 2, "15m": 2}

#: 相邻反向摆动的最小幅度（ATR 倍数）。0 表示不过滤
DEFAULT_MIN_ATR_MULT = 1.0

#: 突破基座认定：低点之后的反弹幅度门槛与放量门槛
BASE_MIN_RALLY_ATR = 1.5
BASE_VOLUME_MULT = 1.2
BASE_VOLUME_LOOKBACK = 20
BASE_VOLUME_WINDOW = 3


@dataclass(frozen=True)
class SwingPoint:
    ts: pd.Timestamp
    kind: Kind
    price: float
    confirmed_at: pd.Timestamp
    bar_index: int
    is_base: bool = False
    base_confirmed_at: pd.Timestamp | None = None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        for key in ("ts", "confirmed_at", "base_confirmed_at"):
            v = d[key]
            d[key] = v.isoformat() if isinstance(v, pd.Timestamp) else v
        return d


def _fractals(df: pd.DataFrame, k: int) -> list[SwingPoint]:
    high = df["high"].to_numpy()
    low = df["low"].to_numpy()
    idx = df.index
    n = len(df)
    pts: list[SwingPoint] = []
    for i in range(k, n - k):
        left_h, right_h = high[i - k : i], high[i + 1 : i + k + 1]
        if high[i] > left_h.max() and high[i] > right_h.max():
            pts.append(SwingPoint(idx[i], "high", float(high[i]), idx[i + k], i))
        left_l, right_l = low[i - k : i], low[i + 1 : i + k + 1]
        if low[i] < left_l.min() and low[i] < right_l.min():
            pts.append(SwingPoint(idx[i], "low", float(low[i]), idx[i + k], i))
    pts.sort(key=lambda p: (p.bar_index, p.kind))
    return pts


def _alternate(points: list[SwingPoint]) -> list[SwingPoint]:
    """强制 high/low 交替；连续同向时保留更极端的一个。"""
    out: list[SwingPoint] = []
    for p in points:
        if not out:
            out.append(p)
            continue
        last = out[-1]
        if p.kind != last.kind:
            out.append(p)
            continue
        keep_new = p.price > last.price if p.kind == "high" else p.price < last.price
        if keep_new:
            out[-1] = p
    return out


def _filter_amplitude(
    points: list[SwingPoint], atr_v: np.ndarray, mult: float
) -> list[SwingPoint]:
    """删除幅度不足的摆动。

    以后一个点处的 ATR 为尺度；ATR 缺失时该摆动一律保留，宁可多留也不静默丢弃。
    """
    if mult <= 0 or len(points) < 3:
        return points
    keep = [True] * len(points)
    for i in range(1, len(points)):
        a, b = points[i - 1], points[i]
        scale = atr_v[b.bar_index]
        if not np.isfinite(scale) or scale <= 0:
            continue
        if abs(b.price - a.price) < mult * scale:
            # 删掉这一对里较不重要的那个：保留更极端的极值
            weaker = i if _is_weaker(b, a) else i - 1
            keep[weaker] = False
    return [p for p, k in zip(points, keep) if k]


def _is_weaker(b: SwingPoint, a: SwingPoint) -> bool:
    """b 相对 a 是否是更弱的摆动（同向比极值，反向比后来者优先保留前者）。"""
    if b.kind == a.kind:
        return b.price <= a.price if b.kind == "high" else b.price >= a.price
    return True


def _mark_bases(
    df: pd.DataFrame, points: list[SwingPoint], atr_v: np.ndarray
) -> list[SwingPoint]:
    """标记哪些 swing low 构成"突破基座"。

    只有基座才有资格充当嵌套锚点 Fib 的台阶底。判定需要低点之后的数据，
    因此单独记录 base_confirmed_at——它必然晚于 confirmed_at。
    """
    vol = df["volume"].to_numpy(dtype=float)
    avg_vol = (
        pd.Series(vol).rolling(BASE_VOLUME_LOOKBACK, min_periods=5).mean().to_numpy()
    )
    out: list[SwingPoint] = []
    for i, p in enumerate(points):
        if p.kind != "low":
            out.append(p)
            continue
        nxt = next((q for q in points[i + 1 :] if q.kind == "high"), None)
        scale = atr_v[p.bar_index]
        is_base = False
        base_at = None
        if nxt is not None and np.isfinite(scale) and scale > 0:
            rally_ok = (nxt.price - p.price) >= BASE_MIN_RALLY_ATR * scale
            w0, w1 = p.bar_index, min(p.bar_index + BASE_VOLUME_WINDOW, len(vol))
            base_avg = avg_vol[p.bar_index]
            vol_ok = (
                np.isfinite(base_avg)
                and base_avg > 0
                and vol[w0:w1].max() >= BASE_VOLUME_MULT * base_avg
            )
            is_base = bool(rally_ok and vol_ok)
            if is_base:
                base_at = nxt.confirmed_at
        out.append(
            SwingPoint(
                p.ts, p.kind, p.price, p.confirmed_at, p.bar_index, is_base, base_at
            )
        )
    return out


def detect_swings(
    df: pd.DataFrame,
    *,
    k: int = 3,
    min_atr_mult: float = DEFAULT_MIN_ATR_MULT,
    atr_period: int = 14,
) -> list[SwingPoint]:
    if len(df) < 2 * k + 1:
        return []
    atr_v = atr_series(df, atr_period).to_numpy()
    points = _alternate(_fractals(df, k))
    if min_atr_mult > 0:
        while True:
            filtered = _alternate(_filter_amplitude(points, atr_v, min_atr_mult))
            if len(filtered) == len(points):
                break
            points = filtered
    return _mark_bases(df, points, atr_v)


def swings_frame(points: list[SwingPoint]) -> pd.DataFrame:
    if not points:
        return pd.DataFrame(
            columns=["kind", "price", "confirmed_at", "bar_index", "is_base"]
        )
    return pd.DataFrame([p.to_dict() for p in points]).set_index("ts")


def visible_at(points: list[SwingPoint], as_of: pd.Timestamp | None) -> list[SwingPoint]:
    """只返回在 as_of 时点已经确认的摆动点。前向测试的唯一合法入口。"""
    if as_of is None:
        return points
    ts = pd.Timestamp(as_of)
    return [p for p in points if p.confirmed_at <= ts]
