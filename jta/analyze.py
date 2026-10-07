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

from .data.fallback_provider import build_provider
from .data.provider import MAX_DATA_AGE_SESSIONS, OHLCV
from .data.resample import align_to_daily
from .events import fetch_events
from .indicators.atr import atr as atr_series
from .indicators.ema import ALL_SPANS, ema_set, ema_stack, vegas_zone, warmup_status
from .indicators.price_action import bar_signals
from .indicators.swing import DEFAULT_K, detect_swings, visible_at
from .indicators.td import COUNTDOWN_IMPLEMENTED, latest_td_signal, td_setup
from .levels import fib as fibmod
from .levels.candidates import (
    RESISTANCE_ROLES,
    STRUCTURE_PIVOT_ROLE,
    display_step,
    SUPPORT_ROLES,
    Candidate,
    annotate_resonance,
    assign_roles,
    build_candidates,
    make_source,
    note_market_cap_confluence,
    note_pivot_confluence,
)
from .levels.pivots import gaps, horizontal_pivots, prior_session_levels, round_ladder, round_numbers
from .marketcap import fetch_market_cap_context
from .brooks import EMA_SPAN as BROOKS_EMA_SPAN, brooks_analysis
from .indicators.ema import ema as ema_series
from .plans import build_plans, holder_playbook, sides, size_position
from .regime import benchmark_regime
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

#: 关键位筛选口径。
#:
#: - "legacy"：达标候选按命中数从高到低取前 3，再按距离编号。命中数统计的是 0.25 ATR
#:   半径内有几类证据，主波段回撤区四套 Fib 扎堆、天然偏高，刚涨上来的价位附近天然偏低——
#:   强势上涨后 S1 会被系统性推到深处（LITE 2026-10-05：现价 1085，S1 落在 1.74 ATR 外的 970，
#:   近处的前日低点 1080、1000 整数 + 枢轴 + Fib 都轮不到）。等于拿一个未被回测证实的计数做了加权。
#: - "proximity"：命中数 >= MIN_HITS 只作噪声下限、不排序。S1、S2 由近及远取"有真实反应"的
#:   达标候选（自身来源含枢轴 / 前日高低 / 缺口，或价格行为项命中），相邻至少 MIN_SEPARATION_ATR；
#:   S3 取 S2 外侧命中数最高的一档作结构覆盖（上游 v2.0.6：近端有效结构不因深层存在而被跳过，
#:   深层结构也不被近端挤掉）。S2 不在窗口里按命中数挑：窗口一宽就又够到深处的 Fib 密集区。
#:
#: 默认 proximity（2026-10-05 起）。25 标的 / 2025-09-01 至 2026-09-23 回测，A/B/C 合计：
#: legacy +0.036R（591 笔）比距离匹配随机对照低 0.225R（z=-1.9）；proximity +0.160R（629 笔），
#: 与对照持平（-0.023R，z=-0.21）。proximity 比 legacy 高 0.125R（z=1.23），前后两段、A/B/C/L
#: 各自方向一致但都不显著。读法是"去掉了按命中数排序带来的负向选择"，不是"选点有了超额价值"。
#:
#: - "structure"：在 proximity 之上补上游 7.1 节的"结构支点"。日线最近一个已确认的摆动低点
#:   （现价下方）/ 摆动高点（现价上方）作为 swing 证据进入候选；它若未入选、在 S2 外侧至少
#:   MIN_SEPARATION_ATR、且达到 MIN_HITS，就占第三档，被替换的原第三档记进 crowded_out。
#:   S1、S2 的选法与 proximity 相同；swing 证据也会计入 0.25 ATR 内邻近候选的共振，
#:   个别情况下可能让邻近候选多一类证据。水平枢轴要求 >= 2 次触碰，单个 HL 进不了候选，
#:   这是与上游手工判读分歧最多的地方（2026-10-07 QQQ/SOXX/LITE 的 HL 均缺席）。
#:   只取日线：4H 的小摆动会把第三档拉到现价附近的噪声上（QQQ 选到 736–737 而不是 731.63）。
SELECTION_MODE = "proximity"
SELECTION_MODES = ("legacy", "proximity", "structure")

#: 算作"真实价格反应"的来源族：历史上价格确实在这里转过向，而不是只由公式算出来。
#: swing 只在 structure 口径下产生，不影响另外两种口径
REACTION_FAMILIES = {"pivot", "prev_session", "gap", "swing"}

DAILY_LOOKBACK = 500
INTRADAY_LOOKBACK = 400


#: 4H 价格口径说明，随 JSON 一起输出
INTRADAY_SCALE_NOTE = (
    "4H 已按每天「日线收盘 / 当天最后一根 4H 收盘」缩放到日线复权口径；"
    "只有 1 根 4H 的日子沿用前一个完整交易日的因子"
)


def fetch_timeframes(provider: Any, symbol: str, as_of=None) -> tuple[OHLCV, OHLCV]:
    """取日线与 4H，并把 4H 缩放到日线的复权口径。

    日线把历史分红折进价格，60m（4H 由它聚合）只做拆股调整：回看 400 根 4H 的最早一段，
    分红股（PG/XOM/KO/JNJ 等）会比日线高约 2%，4H 候选位（局部 Fib、4H 摆动、4H EMA、
    4H 趋势线）和日线候选位就不在同一尺度上，共振打分被污染。主备数据源混用时
    （日线来自 yfinance、4H 来自 twelvedata）口径差异也在这里一并抹平。
    analyze 与图表必须走同一个入口，否则图上的 K 线和关键位会错位。
    """
    daily = provider.fetch(symbol, "1d", as_of=as_of)
    intraday = provider.fetch(symbol, "4h", as_of=as_of)
    return daily, OHLCV(df=align_to_daily(intraday.df, daily.df), meta=intraday.meta)


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


#: 相邻同类摆动点相差不足这个 ATR 倍数时记为等高 / 等低，不用小数差异宣布创新高低（上游 7.1）
EQUAL_SWING_ATR = 0.1

SWING_LABELS = {
    "HH": "更高高点", "LH": "更低高点", "EH": "等高",
    "HL": "更高低点", "LL": "更低低点", "EL": "等低",
}


def swing_sequence(swings: Sequence, atr_value: float, n: int = 6) -> list[dict[str, Any]]:
    """最近 n 个已确认摆动点，各自与前一个同类点比较，标 HH/HL/LH/LL（上游 v2.0.4 第 7.1 节）。

    只陈述关系，不推断趋势或形态——那是叙事层的判断。swings 已经过 visible_at，
    只含截至 as_of 已确认的点，第一个同类点没有比较对象，label 为 None。
    """
    tol = EQUAL_SWING_ATR * atr_value
    prev: dict[str, float] = {}
    out: list[dict[str, Any]] = []
    for s in swings:
        p = prev.get(s.kind)
        label = None
        if p is not None:
            if abs(s.price - p) <= tol:
                label = "EH" if s.kind == "high" else "EL"
            elif s.kind == "high":
                label = "HH" if s.price > p else "LH"
            else:
                label = "HL" if s.price > p else "LL"
        prev[s.kind] = s.price
        out.append({
            "ts": s.ts.isoformat(),
            "kind": s.kind,
            "price": round(float(s.price), 4),
            "label": label,
            "label_text": SWING_LABELS.get(label or ""),
            "confirmed_at": s.confirmed_at.isoformat(),
        })
    return out[-n:]


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
        "swing_sequence": {
            "daily": swing_sequence(daily.swings, daily.atr),
            "intraday": swing_sequence(intraday.swings, intraday.atr),
            "equal_tolerance_atr": EQUAL_SWING_ATR,
            "note": "只陈述相邻同类摆动点的高低关系，不代表趋势已确认，也不识别形态",
        },
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


def _swing_sources(ctx: TimeframeContext, price: float) -> list[tuple[float, dict]]:
    """日线结构支点：现价下方最近确认的摆动低点、上方最近确认的摆动高点（上游 7.1 节）。

    ctx.swings 已经过 visible_at，只含截至 as_of 已确认的点，回放不会看到未来的拐点。
    最近一个低点若已被跌破（在现价上方），取更早的、仍在现价下方的那个——
    已失守的支点不再是支撑，它的角色转换由水平枢轴等其他来源去表达。
    """
    seq = swing_sequence(ctx.swings, ctx.atr, n=len(ctx.swings))
    out: list[tuple[float, dict]] = []
    for kind, below in (("low", True), ("high", False)):
        pt = next((p for p in reversed(seq)
                   if p["kind"] == kind and (p["price"] < price) == below), None)
        if pt is None:
            continue
        rel = pt["label"]
        out.append((
            pt["price"],
            make_source(
                "swing",
                f"日线结构{'低' if kind == 'low' else '高'}点" + (f"（{rel}）" if rel else ""),
                timeframe=ctx.interval,
                confirmed_at=pt["confirmed_at"],
                detail={"ts": pt["ts"], "price": pt["price"], "relation": rel,
                        "relation_text": pt["label_text"]},
            ),
        ))
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
    mode: str | None = None,
) -> tuple[list[tuple[Candidate, dict]], dict[str, Any]]:
    """每侧筛出 2–3 个作用独立的关键位。

    证据不足时输出更少并说明缺口，绝不用弱点凑数——这是原 skill 的硬纪律，
    也是这个函数唯一不该被"优化"的地方。
    """
    mode = mode or SELECTION_MODE
    if mode not in SELECTION_MODES:
        raise ValueError(f"未知的筛选口径 {mode!r}，可选 {SELECTION_MODES}")
    pool = [(c, s) for c, s in scored if c.side == side]
    eligible = [(c, s) for c, s in pool if s["hits"] >= MIN_HITS]

    def shown(c: Candidate) -> float:
        return c.display if c.display is not None else c.price

    def strength(cs: tuple[Candidate, dict]) -> tuple:
        return (cs[1]["hits"], _timeframe_weight(cs[0]), -cs[0].distance_atr)

    picked: list[tuple[Candidate, dict]] = []
    crowded_out: list[dict[str, Any]] = []
    separation = MIN_SEPARATION_ATR * atr_value
    pivot_meta: dict[str, Any] | None = None

    def crowd(c: Candidate, s: dict) -> bool:
        too_close = next(
            (p for p, _ in picked if abs(shown(p) - shown(c)) < separation), None
        )
        if too_close is None:
            return False
        crowded_out.append(
            {
                "price": shown(c),
                "hits": s["hits"],
                "reason": f"与已入选的 {shown(too_close):,.2f} 相距不足 "
                f"{MIN_SEPARATION_ATR} ATR，无法表达独立的下一档路径",
            }
        )
        return True

    if mode == "legacy":
        for c, s in sorted(eligible, key=strength, reverse=True):
            if len(picked) >= MAX_LEVELS_PER_SIDE:
                break
            if not crowd(c, s):
                picked.append((c, s))
    elif eligible:
        by_dist = sorted(eligible, key=lambda cs: cs[0].distance_atr)

        def reaction(cs: tuple[Candidate, dict]) -> bool:
            pa = (cs[1].get("factors") or {}).get("price_action") or {}
            return bool(cs[0].source_families & REACTION_FAMILIES) or bool(pa.get("hit"))

        def nearest_beyond(d: float) -> tuple[Candidate, dict] | None:
            pool_ = [cs for cs in by_dist if cs[0].distance_atr - d >= MIN_SEPARATION_ATR]
            # 没有带真实反应的就退回最近的达标候选，不留空
            return next((cs for cs in pool_ if reaction(cs)), pool_[0] if pool_ else None)

        s1 = next((cs for cs in by_dist if reaction(cs)), by_dist[0])
        picked.append(s1)
        s2 = nearest_beyond(s1[0].distance_atr)
        if s2 is not None:
            picked.append(s2)
            deeper = [cs for cs in by_dist
                      if cs[0].distance_atr - s2[0].distance_atr >= MIN_SEPARATION_ATR]
            if deeper:
                picked.append(max(deeper, key=strength))
        if mode == "structure":
            pivot_meta = _place_structure_pivot(picked, eligible, pool, crowded_out)
        # 只列出被挤掉的最强几个，不把几十个候选全部倒出来
        chosen = {id(c) for c, _ in picked}
        for c, s in sorted(eligible, key=strength, reverse=True):
            if len(crowded_out) >= 5:
                break
            if id(c) not in chosen:
                crowd(c, s)

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
        if pivot_meta and pivot_meta.get("placed") and "swing" in c.source_families:
            c.role = STRUCTURE_PIVOT_ROLE[side]

    gap_note = None
    if len(picked) < 2:
        reason = (
            f"仅 {len(eligible)} 个候选达到最低证据标准（>= {MIN_HITS} 类证据）"
            if len(eligible) < 2
            else f"{len(crowded_out)} 个达标候选与已入选点的间距不足 "
            f"{MIN_SEPARATION_ATR} ATR，无法构成独立的下一档"
        )
        gap_note = f"{side} 侧只输出 {len(picked)} 个关键位：{reason}。按纪律不用弱点补足数量。"
    meta = {
        "mode": mode,
        "candidates_considered": len(pool),
        "eligible": len(eligible),
        "selected": len(picked),
        "crowded_out": crowded_out,
        "gap_note": gap_note,
    }
    if mode == "structure":
        meta["structure_pivot"] = pivot_meta
    return picked, meta


def _place_structure_pivot(
    picked: list[tuple[Candidate, dict]],
    eligible: list[tuple[Candidate, dict]],
    pool: list[tuple[Candidate, dict]],
    crowded_out: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """structure 口径：让日线结构支点占第三档（原地修改 picked）。

    只动第三档，S1、S2 沿用 proximity 的选法。支点已入选、离已入选点不足
    MIN_SEPARATION_ATR、或证据不足 MIN_HITS 时都不动，原因写进返回值。
    """
    holders = sorted((cs for cs in pool if "swing" in cs[0].source_families),
                     key=lambda cs: cs[0].distance_atr)
    if not holders:
        return None
    c, s = holders[0]
    src = next(x for x in c.sources if x.get("family") == "swing")
    info: dict[str, Any] = {
        "price": round(float(src["detail"]["price"]), 4),
        "relation": src["detail"].get("relation"),
        "ts": src["detail"].get("ts"),
        "hits": s["hits"],
        "placed": False,
    }
    if any(p is c for p, _ in picked):
        info["note"] = "结构支点已经入选"
        return info
    if all(p is not c for p, _ in eligible):
        info["note"] = f"证据不足 {MIN_HITS} 类，不进入关键位"
        return info
    near = next((p for p, _ in picked
                 if abs(p.distance_atr - c.distance_atr) < MIN_SEPARATION_ATR), None)
    if near is not None:
        info["note"] = f"与已入选的 {near.display:,.2f} 相距不足 {MIN_SEPARATION_ATR} ATR，由该档代表"
        return info
    if len(picked) >= 2 and c.distance_atr < picked[1][0].distance_atr:
        # proximity 的 S2 已是 S1 外侧最近的真实反应位，支点不可能落在两者之间而未入选；
        # 防御性保留，不改动 S1、S2
        info["note"] = "支点位于 S1 与 S2 之间，保持 proximity 结果"
        return info
    if len(picked) >= MAX_LEVELS_PER_SIDE:
        old_c, old_s = picked.pop(MAX_LEVELS_PER_SIDE - 1)
        crowded_out.append({
            "price": old_c.display if old_c.display is not None else old_c.price,
            "hits": old_s["hits"],
            "reason": "第三档让位给日线结构支点（上游 7.1 节：维持趋势的结构低点 / 高点）",
        })
    picked.append((c, s))
    info["placed"] = True
    return info


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


# ------------------------------------------------------------------ 整数关口梯子

#: 一侧没有关键位时，整数关口往外取多远（ATR 倍数）
ROUND_REACH_ATR = 2.0

ROUND_NOTE = (
    "整数关口是心理参考位，只穿插在关键位之间展示，不参与关键位筛选、证据计数与三套计划。"
    "回放检验（26 个标的、2025-03 至 2026-09）里，整数位的守住率 46.0%，"
    "同侧同距离的非整数价位 44.7%（z=0.72），没有可检测的差异。"
)


def _shown(level: dict[str, Any]) -> float:
    return level["display"] if level.get("display") is not None else level["raw_price"]


def _covers(level: dict[str, Any], price: float) -> bool:
    """整数位是否落在关键位的展示区间里（单值展示时要求相等）。"""
    lo = level.get("range_low") if level.get("range_low") is not None else _shown(level)
    hi = level.get("range_high") if level.get("range_high") is not None else _shown(level)
    return lo - 1e-9 <= price <= hi + 1e-9


def _fmt_price(v: float) -> str:
    return f"{v:,.10g}" if v == int(v) else f"{v:,}"


def build_ladder(
    supports: list[dict[str, Any]],
    resistances: list[dict[str, Any]],
    price: float,
    atr: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """把整数关口穿插进 S1–S3 / R1–R3 之间，返回 (round_levels, ladder)。

    只在最外侧关键位之间取整数位（一侧没有关键位时取 ROUND_REACH_ATR）。
    整数位落在某个关键位的展示区间里时，标注在那个关键位上（round_number 字段），不重复列出；
    只是挨得近（比如 1800 与 R2「约 1780–1790」）仍单独列出——那正是读者想看到的数字。
    ladder 按价格从高到低，含关键位、整数关口与现价，前端照这个顺序画。
    """
    keys = [("resistance", r) for r in resistances] + [("support", s) for s in supports]
    # 上下界取最外侧关键位的区间沿，落在区间里的整数位才能并到那一档上
    low = min((s.get("range_low") or _shown(s) for s in supports),
              default=price - ROUND_REACH_ATR * atr)
    high = max((r.get("range_high") or _shown(r) for r in resistances),
               default=price + ROUND_REACH_ATR * atr)
    rounds: list[dict[str, Any]] = []
    for rd in round_ladder(price, atr, min(low, price), max(high, price)):
        host = next((lv for _, lv in keys if _covers(lv, rd["price"])), None)
        if host is not None:
            host["round_number"] = {"price": rd["price"], "tier": rd["tier"]}
            continue
        rounds.append({
            **rd,
            "side": "support" if rd["price"] < price else "resistance",
            "distance_atr": round(abs(rd["price"] - price) / atr, 3),
            "text": _fmt_price(rd["price"]),
        })

    # 排序用原始计算值：展示值按 0.1 ATR 取整，S1 原值 1716.3 会显示成 1720、排到现价 1719.99 上面
    ladder = (
        [{"kind": kind, "label": lv["label"], "price": _shown(lv), "text": lv["display_text"],
          "_order": lv["raw_price"]} for kind, lv in keys]
        + [{"kind": "round", "label": rd["tier"], "price": rd["price"], "text": rd["text"],
            "_order": rd["price"]} for rd in rounds]
        + [{"kind": "now", "label": "现价", "price": round(price, 4),
            "text": _fmt_price(round(price, 2)), "_order": price}]
    )
    ladder.sort(key=lambda it: -it["_order"])
    for it in ladder:
        del it["_order"]
    # 每个整数位标明落在哪两档之间，叙事层可以直接说"R1 与 R2 之间的 1800"
    for i, it in enumerate(ladder):
        if it["kind"] != "round":
            continue
        above = next((x["label"] for x in reversed(ladder[:i]) if x["kind"] != "round"), None)
        below = next((x["label"] for x in ladder[i + 1:] if x["kind"] != "round"), None)
        between = " 与 ".join(x for x in (above, below) if x)
        it["between"] = between
        next(rd for rd in rounds if rd["price"] == it["price"])["between"] = between
    return rounds, ladder


# ------------------------------------------------------------------ 主入口


def analyze(
    symbol: str,
    *,
    as_of: pd.Timestamp | None = None,
    benchmark: str | None = None,
    provider: Any | None = None,
    include_events: bool = True,
    include_market_cap: bool = True,
    holding: dict[str, Any] | None = None,
    account: float | None = None,
    risk_pct: float = 0.01,
    use_live: bool = False,
    selection: str | None = None,
) -> dict[str, Any]:
    provider = provider or build_provider("auto")
    daily_series, intraday_series = fetch_timeframes(provider, symbol, as_of)

    daily = _prepare(daily_series, DAILY_LOOKBACK, as_of)
    intraday = _prepare(intraday_series, INTRADAY_LOOKBACK, as_of)

    # 默认用最近一个已收盘交易日的收盘价定当天的点位：同一天不管几点跑，
    # 关键位、距离、位置与计划都一样，计划是开盘前定死的，不随盘中价漂移。
    # use_live=True 时现价改用未完成 bar 的最新成交价（盘中看"现在在哪"），
    # 但结构计算仍只用已收盘的 bar——突破/失守问的是"收在哪"
    live = daily_series.meta.live_bar if use_live else None
    price = float(live["close"]) if live else float(daily.df["close"].iloc[-1])

    raw: list[tuple[float, dict]] = []
    for ctx in (daily, intraday):
        raw += _fib_sources(ctx, price, as_of)
        raw += _structural_sources(ctx, price)
    # 结构支点只在 structure 口径下进入候选，保证 legacy / proximity 的输出逐字不变
    if (selection or SELECTION_MODE) == "structure":
        raw += _swing_sources(daily, price)

    candidates = build_candidates(raw, price, daily.atr)
    annotate_resonance(candidates, daily.atr)
    note_pivot_confluence(candidates, daily.atr)

    market_cap: dict[str, Any] = {"available": False, "reason": "未请求"}
    if include_market_cap:
        market_cap = fetch_market_cap_context(symbol, daily.df, as_of=as_of)
    note_market_cap_confluence(candidates, daily.atr, market_cap)

    assign_roles(candidates, price)

    index_summary = None
    if benchmark:
        b = provider.fetch(benchmark, "1d", as_of=as_of)
        bdf = b.df.tail(DAILY_LOOKBACK)
        # 基准指数走的是普通日线抓取，没有盘中 live 拼接，iloc[-1]/[-2] 就是
        # 最近两根已收盘的 bar，算当日涨跌幅不涉及"现价是否盘中"的歧义
        change_pct = (
            round((float(bdf["close"].iloc[-1]) / float(bdf["close"].iloc[-2]) - 1) * 100, 2)
            if len(bdf) >= 2 and float(bdf["close"].iloc[-2])
            else None
        )
        index_summary = {
            "symbol": benchmark,
            "price": round(float(bdf["close"].iloc[-1]), 4),
            "change_pct": change_pct,
            # 递推指标用完整序列算完再取末值，与 _prepare 同一口径
            "ema_stack": ema_stack(ema_set(b.df["close"]).iloc[-1]),
            "as_of_bar": bdf.index[-1].isoformat(),
            # 基准驱动着方向降级判断；它自己的数据可信度必须一起传出，
            # 否则一份过期的基准会悄悄改变个股结论
            "stale": bool(b.meta.stale),
            "too_old": bool(b.meta.too_old),
            "age_sessions": b.meta.age_sessions,
            "fetched_at": b.meta.fetched_at.isoformat(),
            "regime": benchmark_regime(bdf),
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

    supports, sup_meta = select_key_levels(scored, "support", daily.atr, price, selection)
    resistances, res_meta = select_key_levels(scored, "resistance", daily.atr, price, selection)

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
    round_rows, ladder = build_ladder(support_rows, resistance_rows, price, daily.atr)
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
        current_price=price,
        event_mode=bool(events.get("event_mode")),
        event_reason=events.get("event_mode_reason"),
        index_bullish=index_bullish,
        index_symbol=(index_summary or {}).get("symbol"),
        regime=(index_summary or {}).get("regime") or benchmark_regime(None),
    )
    # Brooks 层：EMA20 用完整历史递推后再取尾段，与其他递推指标同一口径
    brooks = brooks_analysis(
        daily.df,
        ema_series(daily_series.df["close"], BROOKS_EMA_SPAN).tail(DAILY_LOOKBACK),
        atr=daily.atr,
        event_mode=bool(events.get("event_mode")),
        event_reason=events.get("event_mode_reason"),
    )
    brooks_plan = brooks.get("plan") if brooks.get("available") else None

    if account:
        for p in plans + ([brooks_plan] if brooks_plan else []):
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
        "schema_version": "1.6",
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
            "intraday_scale": INTRADAY_SCALE_NOTE,
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
        "market_cap": market_cap,
        "supports": support_rows,
        "resistances": resistance_rows,
        "selection": {"support": sup_meta, "resistance": res_meta},
        "round_levels": round_rows,
        "round_levels_note": ROUND_NOTE,
        "ladder": ladder,
        "position_zone": zone,
        "plans": plans,
        "sides": sides(plans),
        "brooks": brooks,
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
            *([events["reason"]] if not events.get("available") and events.get("reason") else []),
            *(
                [market_cap["reason"]]
                if not market_cap.get("available") and market_cap.get("reason")
                else []
            ),
            *daily_series.meta.warnings,
            *intraday_series.meta.warnings,
        ],
        "validation_disclaimer": (
            "前向验证（7 标的 / 2025-08–2026-07 / 973 条已判定观察）显示：控制距现价"
            "远近后，这些关键位的守住率与同侧同距离的随机价位没有可检测差异，"
            "证据命中数与守住率也没有单调关系；计划级回测里照计划入场也没有跑赢"
            "同样远近的随机入场。本图的价值在于口径统一、"
            "失效位明确与仓位可反推，不在于预测胜率。"
        ),
    }
