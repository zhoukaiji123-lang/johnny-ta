"""编排层：从行情到候选位与共振证据。

本层只产出**可复现的事实**：状态判定、候选位、证据、确认与失效规则。
它不写结论叙事，也不给"应该买/应该卖"——那是 skill 叙事层的事，
而叙事层不得再自行计算任何数字。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import pandas as pd

from .data.provider import MAX_DATA_AGE_SESSIONS, OHLCV
from .events import fetch_events
from .data.yf_provider import YFinanceProvider
from .indicators.atr import atr as atr_series
from .indicators.ema import ALL_SPANS, ema_set, ema_stack, vegas_zone, warmup_status
from .indicators.price_action import bar_signals
from .indicators.swing import DEFAULT_K, detect_swings, visible_at
from .indicators.td import COUNTDOWN_IMPLEMENTED, latest_td_signal, td_setup
from .levels import fib as fibmod
from .levels.candidates import (
    RESISTANCE_ROLES,
    display_step,
    SUPPORT_ROLES,
    Candidate,
    annotate_resonance,
    assign_roles,
    build_candidates,
    make_source,
    note_pivot_confluence,
)
from .levels.pivots import gaps, horizontal_pivots, prior_session_levels, round_numbers
from .plans import build_plans, holder_playbook, size_position
from .levels.trendline import fit_trendlines, parallel_channel
from .scoring import evaluate, index_alignment

#: 每侧最多输出的关键位数量。证据不足时允许更少，禁止用弱点补位
MAX_LEVELS_PER_SIDE = 3

#: 进入答案所需的最低证据类型数
MIN_HITS = 3

#: 相邻入选关键位的最小间距（ATR 倍数）。
#: 这不是"距离近就合并候选"——候选全部保留在内部；但输出的 S1/S2/S3 必须表达
#: 逐级路径，三个挤在 0.3 ATR 内的点无法回答"失守后看哪一档"。
MIN_SEPARATION_ATR = 0.5

DAILY_LOOKBACK = 500
INTRADAY_LOOKBACK = 400


@dataclass
class TimeframeContext:
    interval: str
    series: OHLCV
    df: pd.DataFrame
    atr: float
    swings: list
    emas: pd.DataFrame
    td: pd.DataFrame
    full_bars: int = 0


def _prepare(series: OHLCV, lookback: int, as_of) -> TimeframeContext:
    """截断只针对结构识别，不针对递推指标。

    EMA 与 ATR 都是有记忆的递推式：在 tail(500) 上起算，等于把 EMA169 的
    warmup 砍到 500 根，数值永远不可信——而这与实际有多少历史无关，
    是自己把历史丢了。递推指标一律用完整序列算完再取尾段。
    摆动点与 TD 计数只关心近期结构，用截断后的数据即可。
    """
    full = series.df
    df = full.tail(lookback)
    k = DEFAULT_K.get(series.meta.interval, 3)
    swings = visible_at(detect_swings(df, k=k), as_of)
    return TimeframeContext(
        interval=series.meta.interval,
        series=series,
        df=df,
        atr=float(atr_series(full).iloc[-1]),
        swings=swings,
        emas=ema_set(full["close"]).tail(lookback),
        td=td_setup(df),
        full_bars=len(full),
    )


# ------------------------------------------------------------------ 状态判定


def _structure(swings: Sequence) -> str:
    highs = [s for s in swings if s.kind == "high"][-2:]
    lows = [s for s in swings if s.kind == "low"][-2:]
    if len(highs) < 2 or len(lows) < 2:
        return "unknown"
    hh, hl = highs[1].price > highs[0].price, lows[1].price > lows[0].price
    if hh and hl:
        return "higher_highs_higher_lows"
    if not hh and not hl:
        return "lower_highs_lower_lows"
    return "mixed"


def market_state(daily: TimeframeContext, intraday: TimeframeContext) -> dict[str, Any]:
    price = float(daily.df["close"].iloc[-1])
    last = daily.emas.iloc[-1]
    vz = vegas_zone(last)
    stack = ema_stack(last)
    structure = _structure(daily.swings)

    warm = warmup_status(daily.full_bars, ALL_SPANS)
    vegas_ok = warm["ema144"]["reliable"] and warm["ema169"]["reliable"]

    if not vegas_ok:
        # 递归 EMA 对起点有记忆：50 根数据算出的 EMA169 只是把起始价拖了一段，
        # 拿它判断长期牛熊纯属自欺。宁可说"历史不够"，也不给一个看似确定的方位。
        long_term = "insufficient_history"
    elif vz is None:
        long_term = "unknown"
    elif price > vz["upper"]:
        long_term = "above_vegas"
    elif price < vz["lower"]:
        long_term = "below_vegas"
    else:
        long_term = "inside_vegas"

    if structure == "lower_highs_lower_lows" and stack == "bear":
        phase = "downtrend"
    elif structure == "lower_highs_lower_lows" and stack != "bear":
        phase = "stabilizing"
    elif structure == "higher_highs_higher_lows" and stack == "bull":
        phase = "trend_improving"
    elif structure == "higher_highs_higher_lows":
        phase = "reversal_candidate"
    else:
        phase = "range_or_mixed"

    return {
        "phase": phase,
        "phase_note": (
            "4 小时走强但日线未突破关键压力时只算反弹，不算反转——"
            "低周期不得推翻高周期状态"
        ),
        "long_term_vs_vegas": long_term,
        "vegas": vz if vegas_ok else None,
        "vegas_reliable": vegas_ok,
        "daily_ema_stack": stack,
        "intraday_ema_stack": ema_stack(intraday.emas.iloc[-1]),
        "daily_structure": structure,
        "intraday_structure": _structure(intraday.swings),
        "ema_warmup": warm,
        "unreliable_emas": [k for k, v in warm.items() if not v["reliable"]],
    }


# ------------------------------------------------------------------ 候选来源


def _fib_sources(ctx: TimeframeContext, price: float, as_of) -> list[tuple[float, dict]]:
    out: list[tuple[float, dict]] = []
    tf = ctx.interval

    up = fibmod.select_dominant_swing(ctx.swings, "up", timeframe=tf)
    down = fibmod.select_dominant_swing(ctx.swings, "down", timeframe=tf)
    recent = fibmod.select_recent_swing(ctx.swings, timeframe=tf)

    def add(levels, family):
        for f in fibmod.visible_levels(levels, as_of):
            out.append(
                (
                    f.price,
                    make_source(
                        family,
                        f"{f.kind} {f.ratio:.3f}",
                        timeframe=tf,
                        confirmed_at=f.confirmed_at,
                        detail={
                            "ratio": f.ratio,
                            "kind": f.kind,
                            "anchor_low": f.anchor_low,
                            "anchor_high": f.anchor_high,
                            "note": f.note,
                        },
                    ),
                )
            )

    if up:
        add(fibmod.primary_retracement(up), "fib")
        add(fibmod.extension(up), "extension")
    if down:
        add(fibmod.primary_rebound(down), "fib")
    if recent:
        add(fibmod.local_navigation(recent, price), "fib")

    apex = max(
        (s for s in ctx.swings if s.kind == "high"), key=lambda s: s.price, default=None
    )
    if apex is not None:
        add(fibmod.nested_anchors(apex, ctx.swings, timeframe=tf), "fib")
    return out


def _structural_sources(ctx: TimeframeContext, price: float) -> list[tuple[float, dict]]:
    tf = ctx.interval
    out: list[tuple[float, dict]] = []

    for lv in horizontal_pivots(ctx.swings, ctx.df, price, timeframe=tf):
        out.append((lv.price, make_source("pivot", "水平枢轴", timeframe=tf,
                                          confirmed_at=lv.confirmed_at, detail=lv.detail)))
    for lv in prior_session_levels(ctx.df, price, timeframe=tf):
        out.append((lv.price, make_source("prev_session", lv.source, timeframe=tf,
                                          confirmed_at=lv.confirmed_at, detail=lv.detail)))
    for lv in gaps(ctx.df, price, timeframe=tf):
        out.append((lv.price, make_source("gap", "未回补缺口边缘", timeframe=tf,
                                          confirmed_at=lv.confirmed_at, detail=lv.detail)))
    for lv in round_numbers(price, ctx.atr, timeframe=tf):
        out.append((lv.price, make_source("round_number", "整数关口", timeframe=tf,
                                          detail=lv.detail)))

    last = ctx.emas.iloc[-1]
    for span in ALL_SPANS:
        val = last.get(f"ema{span}")
        if val is None or not np.isfinite(val):
            continue
        family = "vegas" if span >= 144 else "ema"
        out.append(
            (
                float(val),
                make_source(
                    family,
                    f"EMA{span}",
                    timeframe=tf,
                    confirmed_at=ctx.df.index[-1].isoformat(),
                    detail={"span": span, "snapshot": round(float(val), 4),
                            "as_of_bar": ctx.df.index[-1].isoformat()},
                ),
            )
        )

    for kind in ("up", "down"):
        for line in fit_trendlines(ctx.df, ctx.swings, kind):
            out.append(
                (
                    line.current_value,
                    make_source("trendline", f"{kind}趋势线", timeframe=tf,
                                confirmed_at=line.as_of_bar, detail=line.to_dict()),
                )
            )
            ch = parallel_channel(ctx.df, line, ctx.swings)
            if ch:
                out.append((ch["current_value"],
                            make_source("channel", ch["role"], timeframe=tf,
                                        confirmed_at=ch["as_of_bar"], detail=ch)))
    return out


# ------------------------------------------------------------------ 筛选


def _timeframe_weight(c: Candidate) -> int:
    order = {"1wk": 3, "1d": 2, "4h": 1, "60m": 0}
    return max(order.get(s.get("timeframe", ""), 0) for s in c.sources)


def select_key_levels(
    scored: list[tuple[Candidate, dict]],
    side: str,
    atr_value: float,
    current_price: float,
) -> tuple[list[tuple[Candidate, dict]], dict[str, Any]]:
    """每侧筛出 2–3 个作用独立的关键位。

    证据不足时输出更少并说明缺口，绝不用弱点凑数——这是原 skill 的硬纪律，
    也是这个函数唯一不该被"优化"的地方。
    """
    pool = [(c, s) for c, s in scored if c.side == side]
    eligible = [(c, s) for c, s in pool if s["hits"] >= MIN_HITS]
    eligible.sort(
        key=lambda cs: (cs[1]["hits"], _timeframe_weight(cs[0]), -cs[0].distance_atr),
        reverse=True,
    )

    def shown(c: Candidate) -> float:
        return c.display if c.display is not None else c.price

    picked: list[tuple[Candidate, dict]] = []
    crowded_out: list[dict[str, Any]] = []
    separation = MIN_SEPARATION_ATR * atr_value
    for c, s in eligible:
        if len(picked) >= MAX_LEVELS_PER_SIDE:
            break
        too_close = next(
            (p for p, _ in picked if abs(shown(p) - shown(c)) < separation), None
        )
        if too_close is not None:
            crowded_out.append(
                {
                    "price": shown(c),
                    "hits": s["hits"],
                    "reason": f"与已入选的 {shown(too_close):,.2f} 相距不足 "
                    f"{MIN_SEPARATION_ATR} ATR，无法表达独立的下一档路径",
                }
            )
            continue
        picked.append((c, s))

    # 编号按**展示值**排序：贴合会让展示值与原始计算值分离，
    # 若按原始值编号，读者会看到 R2 的数字比 R3 还大
    picked.sort(key=lambda cs: abs(shown(cs[0]) - current_price))
    prefix = "S" if side == "support" else "R"
    roles = SUPPORT_ROLES if side == "support" else RESISTANCE_ROLES
    for i, (c, _) in enumerate(picked, start=1):
        c.label = f"{prefix}{i}"  # type: ignore[attr-defined]
        # 角色按最终入选顺序分配：它描述的是这一档在路径中的作用，
        # 若按全部候选的距离预先分配，所有较深的候选都会共享同一个角色
        c.role = roles[min(i - 1, len(roles) - 1)]

    gap_note = None
    if len(picked) < 2:
        reason = (
            f"仅 {len(eligible)} 个候选达到最低证据标准（>= {MIN_HITS} 类证据）"
            if len(eligible) < 2
            else f"{len(crowded_out)} 个达标候选与已入选点的间距不足 "
            f"{MIN_SEPARATION_ATR} ATR，无法构成独立的下一档"
        )
        gap_note = f"{side} 侧只输出 {len(picked)} 个关键位：{reason}。按纪律不用弱点补足数量。"
    return picked, {
        "candidates_considered": len(pool),
        "eligible": len(eligible),
        "selected": len(picked),
        "crowded_out": crowded_out,
        "gap_note": gap_note,
    }


def confirmation_rules(candidate: Candidate) -> dict[str, Any]:
    """确认与失效规则。口径必须入场前定死，事后不得更改。"""
    if candidate.side == "support":
        return {
            "confirmation": "4 小时收盘不再有效跌破，并出现止跌信号（长下影/缩量/反包/更高低点）之一",
            "invalidation": "日线实体收于该位下方且次日无法收复",
            "note": "单次触及不构成买入；跌破后不沿途补仓，等待下一支撑重新确认",
        }
    return {
        "confirmation": "日线收盘站上该位，随后回踩不破",
        "invalidation": "冲高后日线收回该位下方",
        "note": "压力下方不重仓追涨；突破仅在回踩确认后才提高仓位",
    }


# ------------------------------------------------------------------ 主入口


def analyze(
    symbol: str,
    *,
    as_of: pd.Timestamp | None = None,
    benchmark: str | None = None,
    provider: Any | None = None,
    include_events: bool = True,
    holding: dict[str, Any] | None = None,
    account: float | None = None,
    risk_pct: float = 0.01,
) -> dict[str, Any]:
    provider = provider or YFinanceProvider()
    daily_series = provider.fetch(symbol, "1d", as_of=as_of)
    intraday_series = provider.fetch(symbol, "4h", as_of=as_of)

    daily = _prepare(daily_series, DAILY_LOOKBACK, as_of)
    intraday = _prepare(intraday_series, INTRADAY_LOOKBACK, as_of)

    # 现价用未完成 bar 的最新成交价（如果盘中），但结构计算一律只用已收盘的 bar：
    # 位置判断问的是"现在在哪"，突破/失守问的是"收在哪"，两者口径必须分开
    live = daily_series.meta.live_bar
    price = float(live["close"]) if live else float(daily.df["close"].iloc[-1])

    raw: list[tuple[float, dict]] = []
    for ctx in (daily, intraday):
        raw += _fib_sources(ctx, price, as_of)
        raw += _structural_sources(ctx, price)

    candidates = build_candidates(raw, price, daily.atr)
    annotate_resonance(candidates, daily.atr)
    note_pivot_confluence(candidates, daily.atr)
    assign_roles(candidates, price)

    index_summary = None
    if benchmark:
        b = provider.fetch(benchmark, "1d", as_of=as_of)
        bdf = b.df.tail(DAILY_LOOKBACK)
        index_summary = {
            "symbol": benchmark,
            "price": round(float(bdf["close"].iloc[-1]), 4),
            "ema_stack": ema_stack(ema_set(bdf["close"]).iloc[-1]),
            "as_of_bar": bdf.index[-1].isoformat(),
            # 基准驱动着方向降级判断；它自己的数据可信度必须一起传出，
            # 否则一份过期的基准会悄悄改变个股结论
            "stale": bool(b.meta.stale),
            "too_old": bool(b.meta.too_old),
            "age_sessions": b.meta.age_sessions,
            "fetched_at": b.meta.fetched_at.isoformat(),
        }

    td_sig = latest_td_signal(daily.td, within=3)
    scored: list[tuple[Candidate, dict]] = []
    for c in candidates:
        scored.append(
            (
                c,
                evaluate(
                    c,
                    df=daily.df,
                    swings=daily.swings,
                    atr_value=daily.atr,
                    td_signal=td_sig,
                    index_state=index_alignment(c.side, index_summary),
                ),
            )
        )

    supports, sup_meta = select_key_levels(scored, "support", daily.atr, price)
    resistances, res_meta = select_key_levels(scored, "resistance", daily.atr, price)

    def render(items):
        out = []
        for c, s in items:
            d = c.to_dict()
            d["label"] = getattr(c, "label", None)
            d["evidence_families"] = c.resonance.get("families", sorted(c.source_families))
            d["scoring"] = s
            d.update(confirmation_rules(c))
            out.append(d)
        return out

    near_s = supports[0][0].distance_atr if supports else float("inf")
    near_r = resistances[0][0].distance_atr if resistances else float("inf")
    if min(near_s, near_r) > 0.5:
        zone = "between"  # 支撑与压力中间，按纪律默认不交易
    elif near_s <= near_r:
        zone = "at_support"
    else:
        zone = "at_resistance"

    events: dict[str, Any] = {"available": False, "reason": "未请求", "event_mode": False}
    if include_events:
        events = fetch_events(symbol, daily.df, atr_series(daily.df), as_of=as_of)

    state = market_state(daily, intraday)
    support_rows, resistance_rows = render(supports), render(resistances)
    # 三套计划都是做多，因此一律以"指数多头是否成立"判断方向是否有利，
    # 不按各自关键位所在的一侧来判断
    index_bullish = (
        None if index_summary is None else index_summary.get("ema_stack") == "bull"
    )
    plans = build_plans(
        support_rows,
        resistance_rows,
        atr=daily.atr,
        zone=zone,
        event_mode=bool(events.get("event_mode")),
        event_reason=events.get("event_mode_reason"),
        index_bullish=index_bullish,
        index_symbol=(index_summary or {}).get("symbol"),
    )
    if account:
        for p in plans:
            if p["entry"] is not None and p["stop"] is not None:
                sizing = size_position(account, risk_pct, p["entry"], p["stop"])
                scale = p.get("position_scale", 1.0)
                if scale < 1 and "quantity" in sizing:
                    sizing["quantity"] = int(sizing["quantity"] * scale)
                    sizing["notional"] = round(sizing["quantity"] * p["entry"], 2)
                    sizing["risk_amount"] = round(sizing["risk_amount"] * scale, 2)
                    sizing["scaled_by"] = scale
                if account and sizing.get("notional", 0) > account:
                    sizing["warning"] = (
                        f"名义金额 {sizing['notional']:,.0f} 超过账户净值 "
                        f"{account:,.0f}，需要保证金或按现金上限缩减股数"
                    )
                p["sizing"] = sizing

    return {
        "schema_version": "1.1",
        "symbol": symbol,
        "current_price": round(price, 4),
        "price_is_live": live is not None,
        "structure_as_of": daily.df.index[-1].isoformat(),
        "live_bar": live,
        "data": {
            "daily": daily_series.meta.to_dict(),
            "intraday": intraday_series.meta.to_dict(),
            "atr_daily": round(daily.atr, 4),
            "atr_intraday": round(intraday.atr, 4),
            "display_step": display_step(daily.atr),
            "display_step_note": (
                "关键位按约 0.1 ATR 取整展示；原始计算值见每行 raw_price。"
                "换一组合理锚点后同一条 Fib 可漂移接近一个 ATR，"
                "报到个位数是伪精度"
            ),
        },
        "data_health": {
            "stale": bool(
                daily_series.meta.stale
                or intraday_series.meta.stale
                or (index_summary or {}).get("stale")
            ),
            "too_old": bool(
                daily_series.meta.too_old
                or intraday_series.meta.too_old
                or (index_summary or {}).get("too_old")
            ),
            "age_sessions": daily_series.meta.age_sessions,
            "max_age_sessions": MAX_DATA_AGE_SESSIONS,
            "sources": {
                "daily": {"stale": daily_series.meta.stale,
                          "age_sessions": daily_series.meta.age_sessions},
                "intraday": {"stale": intraday_series.meta.stale,
                             "age_sessions": intraday_series.meta.age_sessions},
                "benchmark": {
                    "stale": (index_summary or {}).get("stale"),
                    "age_sessions": (index_summary or {}).get("age_sessions"),
                } if index_summary else None,
            },
        },
        "market_state": state,
        "benchmark": index_summary,
        "td_signal": td_sig,
        "events": events,
        "supports": support_rows,
        "resistances": resistance_rows,
        "selection": {"support": sup_meta, "resistance": res_meta},
        "position_zone": zone,
        "plans": plans,
        "holder_playbook": holder_playbook(
            support_rows, resistance_rows, holding=holding, current_price=price
        ),
        "position_sizing": {
            "formula": "数量 = (账户净值 × 单笔风险比例) ÷ |入场价 - 止损价|",
            "risk_pct_guidance": {"swing": [0.005, 0.01], "leveraged_etf": [0.0025, 0.005]},
            "note": "先定失效位再算仓位；不得为了放大仓位而放宽止损",
            "account": account,
            "risk_pct": risk_pct,
        },
        "known_gaps": [
            *(
                [
                    f"{'、'.join(state['unreliable_emas'])} 的 warmup 不足"
                    f"（仅 {daily.full_bars} 根日线，上市时间短或历史缺失），"
                    "含这些来源的证据不可信"
                ]
                if state["unreliable_emas"] else []
            ),
            "TD Countdown 13 未实现，只有 Setup 9",
            "基本面否决层未实现，需人工检查盈利周期与资本开支",
            "市值里程碑模块按要求未移植",
            *([events["reason"]] if not events.get("available") and events.get("reason") else []),
            *daily_series.meta.warnings,
            *intraday_series.meta.warnings,
        ],
        "validation_disclaimer": (
            "前向验证（7 标的 / 2025-08–2026-07 / 1023 条已判定观察）显示：控制距现价"
            "远近后，这些关键位的守住率与同侧同距离的随机价位没有可检测差异，"
            "证据命中数与守住率也没有单调关系。本图的价值在于口径统一、"
            "失效位明确与仓位可反推，不在于预测胜率。"
        ),
    }
