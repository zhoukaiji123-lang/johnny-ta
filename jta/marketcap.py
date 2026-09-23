"""4A 可选市值里程碑映射（上游 methodology.md 第 4A 节）。

市值本身只是股价的线性换算，不能单独触发交易——这里只做两件事：
把整数总市值（及 0.382 派生中间位）换算成等价股价，再看这个等价价是否跟
已有的关键位（Fib/枢轴/EMA/趋势线……）挨得够近。挨得近才在那条关键位上
挂一条"市值 XT 附近"的备注；挨不近就只在 market_cap 区块里如实报出来，
不伪造共振。不新增共振评分项——市值 Fib 与价格 Fib 落在同一点不算两份
独立证据，这是上游原文自己的立场，跟本仓库"8 项证据都已回测过，不能
拍脑袋加第 9 项"的立场完全一致。

只覆盖美股单一股权类别的普通股：没有流通股数据（ETF/指数）、非美元计价、
或历史回放（as_of 早于今天）都直接判 available=False，不硬凑——跟
events.py 处理财报日程是同一个原则：数据源只给当前快照，没有历史股数，
拿现在的股数去凑历史价位就是伪造信息。ADR、多类别股权换算暂不支持，
是已知局限，不是被忽略。
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

#: 判定"周线收盘等于锚点"的容差——数据源本身的报价误差，不代表股数或汇率精度
BOUNDARY_TOLERANCE_PCT = 0.001


def _bracket_anchors(market_cap: float) -> tuple[float, float]:
    """市值量级附近，按 1-2-5 阶梯找到上下两个整数锚点。"""
    magnitude = 10.0 ** np.floor(np.log10(market_cap))
    ladder = sorted(
        {magnitude * 0.1 * m for m in (1, 2, 5)}
        | {magnitude * m for m in (1, 2, 5)}
        | {magnitude * 10 * m for m in (1, 2, 5)}
    )
    below = max(v for v in ladder if v <= market_cap)
    above = min(v for v in ladder if v > market_cap)
    return below, above


def _completed_weekly_closes(df: pd.DataFrame) -> pd.Series:
    """按自然周（周五收盘）聚合，只保留已经真正走完的周。

    df 已经是结构计算用的日线（今天未收盘的 bar 已被上游 split_incomplete
    剔除），所以只需要看最后一根 bar 是不是周五——不是就说明这一周还没走完，
    丢掉最后一个还在累积的桶。
    """
    weekly = df["close"].resample("W-FRI").last()
    if df.index[-1].weekday() != 4:
        weekly = weekly.iloc[:-1]
    return weekly


def _touched(bars: pd.DataFrame, target_price: float) -> bool:
    """这段 bar 的价格区间有没有真的碰到过 target_price。

    不能拆成"high>=target 或 low<=target"分别判断——只要 target 在这段
    bar 的整体价格范围之外的任一侧，其中一半条件必然恒真（远低于这段 bar
    的目标价，low 永远 <= target），会把"根本没到过"误判成"触碰过"。
    """
    return bool(((bars["low"] <= target_price) & (bars["high"] >= target_price)).any())


def _classify_weekly_state(
    df: pd.DataFrame, target_price: float, tolerance: float
) -> dict[str, Any]:
    completed = _completed_weekly_closes(df)
    if completed.empty:
        return {"state": "数据不足", "note": "还没有一整个完整交易周的数据"}

    last_week_end = completed.index[-1]
    last_close = float(completed.iloc[-1])
    prev_close = float(completed.iloc[-2]) if len(completed) >= 2 else None

    if abs(last_close - target_price) <= tolerance:
        state = "边界不确定"
        note = "上周收盘与锚点的差距在报价误差范围内，不宣称站稳"
    else:
        accepted_now = last_close >= target_price
        accepted_before = prev_close is not None and prev_close >= target_price
        if accepted_now:
            state = "周线接受"
            note = "本周首次收盘站上该锚点" if not accepted_before else "已连续站稳"
        elif accepted_before:
            state = "确认后失守"
            note = "上周还站在锚点上方，最近一个完整周收盘跌破"
        else:
            week_start = last_week_end - pd.Timedelta(days=6)
            last_week_df = df[(df.index > week_start) & (df.index <= last_week_end)]
            attempted = _touched(last_week_df, target_price)
            state = "初始周线假突破" if attempted else "未触及"
            note = "上周盘中触碰过但收盘未能站稳" if attempted else "上周价格没有到过这一带"

    current_week_df = df[df.index > last_week_end]
    in_progress = not current_week_df.empty and _touched(current_week_df, target_price)

    return {
        "state": state,
        "note": note,
        "in_progress_attempt": in_progress and state != "周线接受",
        "last_weekly_close": round(last_close, 4),
    }


def fetch_market_cap_context(
    symbol: str,
    df: pd.DataFrame,
    *,
    as_of: datetime | None = None,
) -> dict[str, Any]:
    """取市值/股数快照，算出整数锚点、0.382 中间位与各自的周线确认状态。"""
    tz = str(df.index.tz)
    today = pd.Timestamp.now(tz=tz).normalize()
    if as_of is not None and pd.Timestamp(as_of).tz_convert(tz).normalize() < today:
        return {
            "available": False,
            "reason": "市值/股数无法按历史时点回放（数据源只提供当前快照，没有历史股数）",
        }

    try:
        import logging

        import yfinance as yf

        logging.getLogger("yfinance").setLevel(logging.CRITICAL)
        info = yf.Ticker(symbol).get_info()
    except Exception as exc:  # noqa: BLE001
        return {"available": False, "reason": f"股数/市值接口暂不可用: {exc}"}

    quote_type = (info.get("quoteType") or "").upper()
    if quote_type and quote_type != "EQUITY":
        # 上游原文明确写了 ETF/杠杆ETF/指数默认跳过：份额数不是公司市值，
        # 市值心理锚点这套逻辑不适用于被动跟踪的基金
        return {
            "available": False,
            "reason": f"标的类型为 {quote_type}，非普通股不适用市值里程碑（ETF/指数默认跳过）",
        }

    shares = info.get("sharesOutstanding")
    currency = info.get("currency")
    if not shares or shares <= 0:
        return {"available": False, "reason": "该标的没有流通股数据（数据源缺失）"}
    if currency and currency != "USD":
        return {
            "available": False,
            "reason": f"计价货币为 {currency}，暂不支持非美元市值换算（已知局限）",
        }

    price = float(df["close"].iloc[-1])
    if price <= 0:
        return {"available": False, "reason": "现价异常，无法换算市值"}
    market_cap = price * shares

    below, above = _bracket_anchors(market_cap)
    mid = below + 0.382 * (above - below)

    anchors = []
    for label, mc in (("低位整数锚点", below), ("0.382 中间位", mid), ("高位整数锚点", above)):
        equiv = mc / shares
        anchors.append(
            {
                "label": label,
                "market_cap": round(mc, 2),
                "equivalent_price": round(equiv, 4),
                "weekly": _classify_weekly_state(
                    df, equiv, tolerance=equiv * BOUNDARY_TOLERANCE_PCT
                ),
            }
        )

    return {
        "available": True,
        "shares_outstanding": int(shares),
        "shares_source_note": "数据源当前快照，非历史某时点的股数",
        "currency": currency or "USD",
        "current_market_cap": round(market_cap, 2),
        "anchors": anchors,
        "note": (
            "市值路径是辅助解释，不能单独触发交易；"
            "只有换算价与已有关键位挨得够近时才会在关键位表里留痕"
        ),
    }
