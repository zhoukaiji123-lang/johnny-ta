"""指数移动平均。

口径：alpha = 2/(n+1)，递归式 EMA_i = alpha*P_i + (1-alpha)*EMA_{i-1}，
与 TradingView Pine `ta.ema` 一致（pandas ewm(adjust=False)）。

递归 EMA 对起点有记忆，warmup 不足时数值会明显偏离图表。因此本模块强制
输出每条 EMA 的 warmup 状态，由上层决定是否允许把它当作关键位依据。
"""

from __future__ import annotations

import pandas as pd

#: Johnny 方法用到的两组 EMA：短周期判断反弹/反转，Vegas 隧道判断长期牛熊
SHORT_SPANS = (8, 13, 21)
VEGAS_SPANS = (144, 169)
ALL_SPANS = SHORT_SPANS + VEGAS_SPANS

#: 递归 EMA 达到与"无限历史"结果的可忽略偏差所需的最小 bar 数倍率
WARMUP_MULTIPLE = 5


def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def ema_set(close: pd.Series, spans: tuple[int, ...] = ALL_SPANS) -> pd.DataFrame:
    return pd.DataFrame({f"ema{s}": ema(close, s) for s in spans}, index=close.index)


def warmup_status(n_bars: int, spans: tuple[int, ...] = ALL_SPANS) -> dict[str, dict]:
    """每条 EMA 的可信度。bar 数不足 5×span 时数值仍会输出，但必须标注不可信。"""
    out: dict[str, dict] = {}
    for s in spans:
        need = s * WARMUP_MULTIPLE
        out[f"ema{s}"] = {
            "span": s,
            "bars_available": n_bars,
            "bars_needed": need,
            "reliable": n_bars >= need,
        }
    return out


def ema_stack(row: pd.Series, spans: tuple[int, ...] = SHORT_SPANS) -> str:
    """短周期 EMA 排列：bull / bear / mixed。用于状态判定，不单独构成信号。"""
    vals = [row.get(f"ema{s}") for s in spans]
    if any(v is None or pd.isna(v) for v in vals):
        return "unknown"
    if all(vals[i] > vals[i + 1] for i in range(len(vals) - 1)):
        return "bull"
    if all(vals[i] < vals[i + 1] for i in range(len(vals) - 1)):
        return "bear"
    return "mixed"


def vegas_zone(row: pd.Series) -> dict[str, float] | None:
    """Vegas 隧道上下沿。EMA144 通常是第一防守，EMA169 是更深防守。"""
    a, b = row.get("ema144"), row.get("ema169")
    if a is None or b is None or pd.isna(a) or pd.isna(b):
        return None
    return {"upper": float(max(a, b)), "lower": float(min(a, b))}
