"""八项共振检查。

**这里刻意不做加权求和。**原 skill 给八项各记 1 分、5–8 分算强区，
但这八项彼此高度相关（Fib 与整数关口、EMA 簇与趋势线、TD9 与价格行为），
等权相加得到的数字没有统计含义，阈值也是拍的。

因此本模块只回答"哪几类证据确实存在、证据是什么"，把命中数原样报出并附警告。
真实权重要等前向记录攒够样本后再定，而不是现在假设。
"""

from __future__ import annotations

from typing import Any, Sequence

import pandas as pd

from .indicators.price_action import (
    double_bottom_near,
    double_top_near,
    reactions_near,
)
from .levels.candidates import Candidate, RESONANCE_ATR

FACTORS: dict[str, str] = {
    "primary_fib": "主波段回撤/反弹 Fib",
    "extension_fib": "次级推动段扩展 Fib",
    "horizontal": "历史平台/前高前低/缺口/整数关口（同类合计一项）",
    "trendline": "趋势线或通道边界",
    "ema_cluster": "EMA 簇或 Vegas 隧道",
    "price_action": "长影线/双底/反包/量能等真实反应",
    "td9": "TD 九转衰竭",
    "index": "基准指数同向确认",
}

FAMILY_TO_FACTOR = {
    "fib": "primary_fib",
    "extension": "extension_fib",
    "pivot": "horizontal",
    "prev_session": "horizontal",
    "gap": "horizontal",
    "round_number": "horizontal",
    "trendline": "trendline",
    "channel": "trendline",
    "ema": "ema_cluster",
    "vegas": "ema_cluster",
}

SCORE_CAVEAT = (
    "命中数只是证据类型的计数，不是加权评分：八项彼此相关，"
    "求和没有统计含义。强/中/弱仅为可读性标签，不得直接当作胜率或仓位依据。"
)


def _band(hits: int) -> str:
    if hits >= 5:
        return "strong"
    if hits >= 3:
        return "moderate"
    return "weak"


def evaluate(
    candidate: Candidate,
    *,
    df: pd.DataFrame,
    swings: Sequence,
    atr_value: float,
    td_signal: dict | None,
    index_state: dict | None,
    resonance_atr: float = RESONANCE_ATR,
) -> dict[str, Any]:
    tol = resonance_atr * atr_value
    factors: dict[str, dict[str, Any]] = {
        name: {"label": label, "hit": False, "evidence": []}
        for name, label in FACTORS.items()
    }

    # 1-5：来自候选点自身与共振半径内的来源族
    for family in candidate.resonance.get("families", candidate.source_families):
        factor = FAMILY_TO_FACTOR.get(family)
        if factor:
            factors[factor]["hit"] = True
            factors[factor]["evidence"].append(family)

    # 6：该价位附近是否真的发生过带确认信号的反应
    side = candidate.side if candidate.side != "neutral" else "support"
    reactions = reactions_near(df, candidate.price, side, tolerance=tol)
    pattern = (
        double_bottom_near(candidate.price, swings, tolerance=tol, min_rally=1.5 * atr_value)
        if side == "support"
        else double_top_near(candidate.price, swings, tolerance=tol, min_drop=1.5 * atr_value)
    )
    if reactions or pattern:
        factors["price_action"]["hit"] = True
        factors["price_action"]["evidence"] = [r.to_dict() for r in reactions[-3:]]
        if pattern:
            factors["price_action"]["evidence"].append(
                {"pattern": "double_bottom" if side == "support" else "double_top"}
            )

    # 7：TD9 方向必须与该点的用途一致
    if td_signal:
        wants_buy = side == "support"
        if (td_signal["kind"] == "buy_setup_9") == wants_buy:
            factors["td9"]["hit"] = True
            factors["td9"]["evidence"].append(td_signal)

    # 8：基准指数
    if index_state and index_state.get("aligned"):
        factors["index"]["hit"] = True
        factors["index"]["evidence"].append(index_state)

    hits = sum(1 for f in factors.values() if f["hit"])
    return {
        "factors": factors,
        "hits": hits,
        "band": _band(hits),
        "caveat": SCORE_CAVEAT,
    }


def index_alignment(
    stock_side: str, index_summary: dict | None
) -> dict[str, Any] | None:
    """判断基准指数是否与个股当前判断同向。

    指数走强只在评估支撑时算正面证据，评估压力时反而降低突破受阻的概率——
    因此方向必须显式匹配，不能笼统地"指数好=加分"。
    """
    if not index_summary:
        return None
    stack = index_summary.get("ema_stack")
    aligned = (stock_side == "support" and stack == "bull") or (
        stock_side == "resistance" and stack == "bear"
    )
    return {
        "symbol": index_summary.get("symbol"),
        "ema_stack": stack,
        "aligned": bool(aligned),
        "note": "指数共振只调整置信度，不替代个股失效位",
    }
