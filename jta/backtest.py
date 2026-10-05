"""计划级回测。

P5 验证的是"关键位守不守得住"，这里验证的是"照计划做的期望值"——
两者不是一回事：守住率 50% 完全可以配出正期望，只要盈亏比够。

模拟粒度是 4H 而不是日线：方法论本身要求 4 小时确认，而且 4H 能区分
同一天内止损与止盈的先后，日线粒度只能靠"保守假设"糊过去。
代价是 Yahoo 的 60m 历史上限约 730 天，因此回测窗口最多两年。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd

from .data.resample import align_to_daily
from .indicators.atr import atr as atr_series
from .indicators.price_action import bar_signals

Outcome = Literal[
    "not_triggered", "cancelled", "stopped", "t1_then_stop", "t1_then_timeout",
    "target", "timeout",
]

#: 支撑侧入场需要的止跌信号；压力突破侧回踩确认用同一组
ENTRY_SIGNALS = ("long_lower_wick", "bullish_engulfing", "volume_dry_up")


@dataclass(frozen=True)
class TradeRules:
    """回测规则。每一条都是可争论的选择，因此全部显式冻结并随结果一起报出。"""

    touch_tol_atr: float = 0.1      # 触及入场位的容差
    confirm_window: int = 3          # 触及后多少根 4H 内必须出现确认信号
    ttl_bars: int = 40               # 计划有效期（4H bar，约 20 个交易日）
    fill_window: int = 4             # 确认后多少根 4H 内限价单仍有效
    max_holding_bars: int = 80       # 最长持仓（4H bar，约 40 个交易日）
    t1_fraction: float = 0.5         # 触及 T1 减仓比例
    breakeven_after_t1: bool = True  # T1 后把剩余仓位的止损抬到入场价
    trail_atr: float = 2.0           # T1 后的跟踪止损宽度（最高价回撤 N×ATR）
    same_bar_policy: Literal["stop", "target"] = "stop"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TradeResult:
    symbol: str
    plan_key: str
    plan_date: str
    executable: bool
    index_bullish: bool | None
    entry_plan: float
    stop_plan: float
    t1: float | None
    t2: float | None
    rr_plan: float | None
    outcome: Outcome = "not_triggered"
    entry_date: str | None = None
    entry_fill: float | None = None
    exit_date: str | None = None
    exit_fill: float | None = None
    r_multiple: float | None = None
    bars_held: int | None = None
    mae_r: float | None = None
    mfe_r: float | None = None
    gapped: bool = False
    regime: str | None = None
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _r_unit(entry: float, stop: float) -> float:
    return abs(entry - stop)


def _first_touch(
    lows: np.ndarray, highs: np.ndarray, level: float, tol: float, side: str
) -> int | None:
    hit = (lows <= level + tol) if side == "long_support" else (highs >= level - tol)
    idx = np.flatnonzero(hit)
    return int(idx[0]) if idx.size else None


def simulate(
    plan: dict[str, Any],
    intraday: pd.DataFrame,
    daily: pd.DataFrame,
    plan_ts: pd.Timestamp,
    *,
    symbol: str,
    index_bullish: bool | None,
    rules: TradeRules,
    signals: pd.DataFrame | None = None,
    regime: str | None = None,
) -> TradeResult:
    """模拟一套计划从等待到离场的完整过程。

    plan_ts 必须是计划日收盘之后的时刻（replay_plans 给的 as_of）：
    4H 只取其后的 bar；ATR 取到计划日为止；突破类的日线确认从下一个交易日起算。
    """
    entry, stop = plan.get("entry"), plan.get("stop")
    res = TradeResult(
        symbol=symbol, plan_key=plan["key"], plan_date=plan_ts.isoformat(),
        executable=bool(plan.get("executable")), index_bullish=index_bullish,
        entry_plan=entry, stop_plan=stop, t1=plan.get("t1"), t2=plan.get("t2"),
        rr_plan=plan.get("rr"), regime=regime,
    )
    if entry is None or stop is None:
        res.note = "计划缺少入场或止损"
        return res
    if plan.get("t1") is None:
        # 没有第一目标就没有离场依据，只能靠超时平仓——那不是交易计划，
        # 而且在趋势行情里会产出 +19R 这种完全由持仓期长度决定的假结果
        res.note = "计划缺少第一目标，不可交易"
        return res

    fwd = intraday[intraday.index > plan_ts]
    if fwd.empty:
        res.note = "计划日之后没有可用 4H 数据"
        return res
    # 突破类先等日线确认（最多 ttl/2 个交易日 ≈ ttl 根 4H）再进入 ttl 窗口找回踩，
    # 窗口要留够，否则持仓期被截断、提前记成超时
    fwd = fwd.iloc[: 2 * rules.ttl_bars + rules.confirm_window + rules.fill_window
                   + rules.max_holding_bars + 2]
    sig = (signals if signals is not None else bar_signals(intraday)).loc[fwd.index]

    atr = float(atr_series(daily.loc[:plan_ts]).iloc[-1])
    if not np.isfinite(atr) or atr <= 0:
        res.note = "ATR 不可用"
        return res
    tol = rules.touch_tol_atr * atr

    o = fwd["open"].to_numpy(float)
    h = fwd["high"].to_numpy(float)
    l = fwd["low"].to_numpy(float)
    c = fwd["close"].to_numpy(float)

    # ---------------------------------------------------------------- 触发
    breakout = plan["key"] == "breakout"
    if breakout:
        # 突破类：先要日线收盘站上该位，再等回踩。方法论要求的是日线确认，
        # 用 4H 收盘代替会把大量假突破算成有效突破。
        d_fwd = daily[(daily.index > plan_ts)].head(rules.ttl_bars // 2 + 1)
        above = d_fwd[d_fwd["close"] > entry]
        if above.empty:
            res.note = "有效期内日线未收盘站上该位"
            return res
        # 日线收盘确认要到确认日 16:00 才成立，回踩只能从下一个交易日的 4H 找起；
        # 从确认日 00:00 算起会让当天两根 4H 用上当天收盘，和计划日前视是同一类错误
        confirm_close = above.index[0].normalize() + pd.Timedelta(hours=16)
        after = np.flatnonzero(fwd.index >= confirm_close)
        if after.size == 0:
            res.note = "突破确认后没有可用 4H 数据"
            return res
        search_from = int(after[0])
    else:
        search_from = 0

    window = slice(search_from, min(search_from + rules.ttl_bars, len(fwd)))
    side = "long_support"
    touch_rel = _first_touch(l[window], h[window], entry, tol, side)
    if touch_rel is None:
        res.note = "有效期内未触及入场位"
        return res
    touch = search_from + touch_rel

    # ---------------------------------------------------------------- 确认
    #
    # 支撑计划不存在"触发前跌破止损"：止损在入场位之下，价格要跌破它必然
    # 先触及入场位。真正对应 cancel_if 的情形是——触及后没等到确认，
    # 价格反而收盘跌穿了止损位，这时计划作废而不是继续挂着等。
    conf_end = min(touch + rules.confirm_window, len(fwd))
    conf_idx = None
    for i in range(touch, conf_end):
        if c[i] < stop:
            res.outcome = "cancelled"
            res.note = "触及后未确认即收盘跌破止损位，计划作废"
            return res
        if c[i] < entry - tol:            # 有效跌破入场位，确认失败
            break
        if any(bool(sig.iloc[i][s]) for s in ENTRY_SIGNALS):
            conf_idx = i
            break
    if conf_idx is None:
        res.note = "触及后未出现止跌确认信号"
        return res

    # 成交模拟：挂限价单在入场位，而不是"确认根的下一根按开盘价买入"。
    # 后者在跳空行情里会产生 10% 以上的假滑点——MU 2025-10 的一笔计划入场位 166，
    # 按开盘价成交是 184.54，等于凭空多付 11%，整份统计都会被这种成交污染。
    limit = entry + tol
    fill_idx = fill = None
    for i in range(conf_idx + 1, min(conf_idx + 1 + rules.fill_window, len(fwd))):
        if o[i] <= limit:                 # 开盘已在限价内，按开盘成交
            fill_idx, fill = i, float(o[i])
            break
        if l[i] <= limit:                 # 盘中回落到限价，按限价成交
            fill_idx, fill = i, float(limit)
            break
        if c[i] < stop:                   # 等待期间已跌破止损，计划作废
            res.outcome = "cancelled"
            res.note = "等待成交期间跌破止损位"
            return res
    if fill_idx is None:
        res.note = "确认后价格未回到限价内，未成交"
        return res
    if fill <= stop:
        res.note = "成交价已在止损位之下"
        return res
    res.entry_date = fwd.index[fill_idx].isoformat()
    res.entry_fill = round(fill, 4)

    return _hold(res, fwd, fill_idx, fill, stop, plan.get("t1"), plan.get("t2"), atr, rules,
                 r_unit=_r_unit(entry, stop))


def _hold(
    res: TradeResult,
    fwd: pd.DataFrame,
    fill_idx: int,
    fill: float,
    stop: float,
    t1: float | None,
    t2: float | None,
    atr: float,
    rules: TradeRules,
    *,
    r_unit: float,
) -> TradeResult:
    """成交之后的持仓管理。三套计划、Brooks D 与随机对照共用这一段，
    保证几组之间只有入场不同、离场规则完全一致。

    r_unit 是**计划**风险 |计划入场 − 止损|：仓位按它反推（size_position），
    1R 就是计划里准备亏的那笔钱。用"成交价 − 止损"当 1R，跳空低开成交在止损附近时
    1R 会缩到几乎为零，随后再一跳空就记成几十 R 的亏损（对照组出现过 −55.9R），
    而真实账户按计划仓位只亏了一两个 R。"""
    o = fwd["open"].to_numpy(float)
    h = fwd["high"].to_numpy(float)
    l = fwd["low"].to_numpy(float)
    c = fwd["close"].to_numpy(float)
    # ---------------------------------------------------------------- 持仓
    remaining = 1.0
    realised = 0.0
    cur_stop = stop
    took_t1 = False
    peak = fill
    mae = mfe = 0.0

    end = min(fill_idx + rules.max_holding_bars, len(fwd))
    for i in range(fill_idx, end):
        bar_high, bar_low, bar_open = h[i], l[i], o[i]
        mae = min(mae, (bar_low - fill) / r_unit)
        mfe = max(mfe, (bar_high - fill) / r_unit)

        hit_stop = bar_low <= cur_stop
        hit_t1 = (t1 is not None) and (not took_t1) and bar_high >= t1
        hit_t2 = (t2 is not None) and took_t1 and bar_high >= t2

        if hit_stop and (hit_t1 or hit_t2) and rules.same_bar_policy == "stop":
            hit_t1 = hit_t2 = False

        if hit_stop:
            # 跳空越过止损时按开盘价成交——假设按止损价成交会系统性低估损失
            exit_px = min(cur_stop, bar_open) if bar_open < cur_stop else cur_stop
            res.gapped = bool(bar_open < cur_stop)
            realised += remaining * (exit_px - fill) / r_unit
            res.exit_date = fwd.index[i].isoformat()
            res.exit_fill = round(float(exit_px), 4)
            res.outcome = "t1_then_stop" if took_t1 else "stopped"
            remaining = 0.0
            break

        if hit_t1:
            realised += rules.t1_fraction * (t1 - fill) / r_unit
            remaining -= rules.t1_fraction
            took_t1 = True
            if rules.breakeven_after_t1:
                cur_stop = max(cur_stop, fill)

        # T1 之后启用跟踪止损。只抬到成本价、然后一路持有到超时，
        # 会让结果由 max_holding_bars 决定而不是由策略决定：
        # 趋势行情里能刷出 +19R，那是持仓期长度的产物，不是计划的产物。
        if took_t1 and rules.trail_atr > 0:
            peak = max(peak, float(bar_high))
            cur_stop = max(cur_stop, peak - rules.trail_atr * atr)

        if hit_t2:
            realised += remaining * (t2 - fill) / r_unit
            res.exit_date = fwd.index[i].isoformat()
            res.exit_fill = round(float(t2), 4)
            res.outcome = "target"
            remaining = 0.0
            break

    if remaining > 0:
        last = min(end, len(fwd)) - 1
        exit_px = float(c[last])
        realised += remaining * (exit_px - fill) / r_unit
        res.exit_date = fwd.index[last].isoformat()
        res.exit_fill = round(exit_px, 4)
        res.outcome = "t1_then_timeout" if took_t1 else "timeout"
        last_i = last
    else:
        last_i = fwd.index.get_loc(pd.Timestamp(res.exit_date))

    res.r_multiple = round(realised, 4)
    res.bars_held = int(last_i - fill_idx + 1)
    res.mae_r = round(mae, 3)
    res.mfe_r = round(mfe, 3)
    return res


# ------------------------------------------------------------------ 回放与汇总

#: 同一套计划在多少 ATR 内视为同一次机会（避免一个计划被连续 20 天重复计数）
PLAN_DEDUP_ATR = 0.25


def replay_plans(
    symbol: str,
    provider: Any,
    *,
    start: str,
    end: str,
    benchmark: str | None = None,
    dedup_atr: float = PLAN_DEDUP_ATR,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """逐日回放，一次收集当天生成的三套计划与 Brooks 计划 D。

    D 不去重：它的挂单只在下一个交易日有效，每天按当天的分析重新挂，
    重复计数由"同一标的同时只持有一笔"处理（见 run_brooks_backtest）。

    去重按 (计划类型, 入场价) 做：一个支撑位会连续很多天出现在计划里，
    不去重的话同一次机会会被计成几十笔，统计立刻失真。

    两类计划的 ts 都是 as_of（计划日收盘之后）：计划用到了当天收盘，
    当天 09:30 / 13:30 两根 4H 不能参与触及、确认与成交。
    """
    from .analyze import analyze

    daily = provider.daily(symbol)
    days = daily.loc[(daily.index >= start) & (daily.index <= end)].index

    out: list[dict[str, Any]] = []
    brooks: list[dict[str, Any]] = []
    seen: list[tuple[str, float, pd.Timestamp]] = []
    for day in days:
        as_of = day + pd.Timedelta(hours=23)
        try:
            r = analyze(
                symbol, as_of=as_of, benchmark=benchmark,
                provider=provider, include_events=False, include_market_cap=False,
            )
        except Exception:
            continue
        atr = r["data"]["atr_daily"]
        if not atr or not np.isfinite(atr):
            continue
        idx_bull = (
            None if r.get("benchmark") is None
            else r["benchmark"]["ema_stack"] == "bull"
        )
        regime = ((r.get("benchmark") or {}).get("regime") or {}).get("state", "unknown")
        bk = r.get("brooks") or {}
        if bk.get("available") and (bk.get("plan") or {}).get("entry") is not None:
            brooks.append({"ts": as_of, "plan": bk["plan"], "atr": atr, "regime": regime,
                           "brooks_state": bk.get("state")})
        seen = [s for s in seen if (day - s[2]).days <= 60]
        for plan in r["plans"]:
            if plan.get("entry") is None:
                continue
            key, entry = plan["key"], float(plan["entry"])
            if any(k == key and abs(e - entry) <= dedup_atr * atr for k, e, _ in seen):
                continue
            seen.append((key, entry, day))
            out.append({"ts": as_of, "plan": plan, "index_bullish": idx_bull, "atr": atr,
                        "regime": regime})
    return out, brooks


def collect_plans(
    symbol: str,
    provider: Any,
    *,
    start: str,
    end: str,
    benchmark: str | None = None,
    dedup_atr: float = PLAN_DEDUP_ATR,
) -> list[dict[str, Any]]:
    """逐日回放，收集当天生成的三套计划（去重）。"""
    return replay_plans(symbol, provider, start=start, end=end, benchmark=benchmark,
                        dedup_atr=dedup_atr)[0]


def _subset(provider: Any, keep: set[str]) -> Any:
    """进程池任务只带自己用得到的行情，避免把全部标的序列化进每个子进程。"""
    if hasattr(provider, "_series"):
        return type(provider)({k: v for k, v in provider._series.items() if k[0] in keep})
    return provider


def _abc_symbol(job: tuple) -> list[dict[str, Any]]:
    """一个标的的 A/B/C 回测（进程池任务，必须是模块级函数）。

    control_seeds 非空时，每条计划再按距离匹配复制若干份（distance_matched），
    与真实计划用同一套触发、确认、成交与离场规则——两组唯一的差别是"在哪里等"。
    """
    import zlib

    from . import plans as plans_mod

    symbol, bm, provider, start, end, rules, executable_only, seeds, min_dist = job
    saved = plans_mod.MIN_ENTRY_DISTANCE_ATR
    if min_dist is not None:
        plans_mod.MIN_ENTRY_DISTANCE_ATR = min_dist
    try:
        daily = provider.daily(symbol)
        # ReplayProvider 已对齐过；这里再对齐一次是幂等的，换别的 provider 也不会混口径
        intraday = align_to_daily(provider.fetch(symbol, "4h").df, daily)
        sig = bar_signals(intraday)
        items = collect_plans(symbol, provider, start=start, end=end, benchmark=bm)
    finally:
        plans_mod.MIN_ENTRY_DISTANCE_ATR = saved
    rows: list[dict[str, Any]] = []
    for n, item in enumerate(items):
        plan = item["plan"]
        if executable_only and not plan.get("executable"):
            continue
        streams = [("real", plan)]
        for seed in seeds:
            rng = np.random.RandomState(
                (seed * 100_003 + zlib.crc32(f"{symbol}|{n}".encode())) % 2**31)
            streams.append((f"ctrl_{seed}", distance_matched(plan, item["atr"], rng)))
        for name, p in streams:
            d = simulate(
                p, intraday, daily, item["ts"], symbol=symbol,
                index_bullish=item["index_bullish"], rules=rules, signals=sig,
                regime=item["regime"],
            ).to_dict()
            d["stream"] = name
            rows.append(d)
    return rows


def run_backtest(
    pairs: list[tuple[str, str | None]],
    provider: Any,
    *,
    start: str,
    end: str,
    rules: TradeRules | None = None,
    executable_only: bool = False,
    control_seeds: tuple[int, ...] = (),
    min_entry_distance_atr: float | None = None,
    jobs: int = 1,
) -> pd.DataFrame:
    """A/B/C 逐笔结果。stream：real / ctrl_<seed>（距离匹配随机对照）。

    min_entry_distance_atr 覆盖 plans.MIN_ENTRY_DISTANCE_ATR，用来复现"距离过滤开/关"的对比。
    """
    rules = rules or TradeRules()
    tasks = [
        (symbol, bm, _subset(provider, {symbol, bm} - {None}), start, end, rules,
         executable_only, tuple(control_seeds), min_entry_distance_atr)
        for symbol, bm in pairs
    ]
    rows: list[dict[str, Any]] = []
    if jobs > 1:
        from concurrent.futures import ProcessPoolExecutor

        with ProcessPoolExecutor(max_workers=jobs) as ex:
            for part in ex.map(_abc_symbol, tasks):
                rows += part
    else:
        for t in tasks:
            rows += _abc_symbol(t)
    return pd.DataFrame(rows)


TRIGGERED = {"stopped", "t1_then_stop", "t1_then_timeout", "target", "timeout"}


def _stats(df: pd.DataFrame) -> dict[str, Any]:
    total = len(df)
    trig = df[df["outcome"].isin(TRIGGERED)]
    n = len(trig)
    if n == 0:
        return {"plans": total, "triggered": 0, "trigger_rate": 0.0, "expectancy_r": None}
    r = trig["r_multiple"].astype(float)
    wins, losses = r[r > 0], r[r <= 0]
    return {
        "plans": total,
        "triggered": n,
        "trigger_rate": round(n / total, 3),
        "win_rate": round(len(wins) / n, 3),
        "expectancy_r": round(float(r.mean()), 3),
        "stderr_r": round(float(r.std(ddof=1) / np.sqrt(n)), 3) if n > 1 else None,
        "median_r": round(float(r.median()), 3),
        "avg_win_r": round(float(wins.mean()), 3) if len(wins) else None,
        "avg_loss_r": round(float(losses.mean()), 3) if len(losses) else None,
        "total_r": round(float(r.sum()), 2),
        "worst_r": round(float(r.min()), 3),
        "gap_rate": round(float(trig["gapped"].mean()), 3),
        "outcomes": trig["outcome"].value_counts().to_dict(),
    }


#: 被大盘开关拦截的状态（unknown 不拦截，只提示）
GATE_BLOCKED = ("range", "down")


def _diff(a: pd.DataFrame, b: pd.DataFrame, seeds_a: int = 1, seeds_b: int = 1) -> dict[str, Any]:
    """两组期望之差与 z。对照组多个种子来自同一批计划、彼此相关，标准误按单个种子的样本量算。"""
    ra = a[a["outcome"].isin(TRIGGERED)]["r_multiple"].astype(float)
    rb = b[b["outcome"].isin(TRIGGERED)]["r_multiple"].astype(float)
    out: dict[str, Any] = {"a": round(float(ra.mean()), 3) if len(ra) else None, "n_a": len(ra),
                           "b": round(float(rb.mean()), 3) if len(rb) else None, "n_b": len(rb),
                           "diff_r": None, "z": None}
    if len(ra) > 1 and len(rb) > 1:
        d = float(ra.mean() - rb.mean())
        se = np.sqrt(ra.var(ddof=1) / max(len(ra) / seeds_a, 1)
                     + rb.var(ddof=1) / max(len(rb) / seeds_b, 1))
        out["diff_r"] = round(d, 3)
        out["z"] = round(d / se, 2) if se > 0 else None
    return out


def _gate(df: pd.DataFrame, seeds: int = 1) -> dict[str, Any]:
    """大盘开关：向上（可开仓）对被拦截（震荡 + 向下）。"""
    if "regime" not in df.columns:
        return {}
    up, blocked = df[df["regime"] == "up"], df[df["regime"].isin(GATE_BLOCKED)]
    return {"up_minus_blocked": _diff(up, blocked, seeds, seeds),
            "unfiltered": _stats(df).get("expectancy_r"), "filtered": _stats(up).get("expectancy_r")}


def summarise_trades(df: pd.DataFrame, *, split: str | None = None) -> dict[str, Any]:
    """split：按计划日把窗口切成前后两段（例如 2026-06-25），分别报真实 vs 对照与开关效果。"""
    if df.empty:
        return {"error": "无记录"}
    ctrl = pd.DataFrame()
    if "stream" in df.columns:
        ctrl = df[df["stream"].str.startswith("ctrl_")]
        df = df[df["stream"] == "real"]
    n_seeds = ctrl["stream"].nunique() if len(ctrl) else 0
    exe = df[df["executable"]]
    out = {
        "overall": _stats(df),
        "executable_only": _stats(exe) if len(exe) else None,
        "by_plan": {k: _stats(g) for k, g in df.groupby("plan_key")},
        "by_index": {
            ("bullish" if k else "not_bullish"): _stats(g)
            for k, g in df.dropna(subset=["index_bullish"]).groupby("index_bullish")
        },
        # 回测默认模拟全部计划（含被大盘状态拦截的），这里看开关的效果
        "by_regime": (
            {k: _stats(g) for k, g in df.groupby("regime")} if "regime" in df else {}
        ),
        "regime_gate": _gate(df),
        "caveats": [
            "期望值以 R 倍数计（1R = 计划入场到止损的距离），未计滑点、佣金与融资成本",
            "同一根 4H 内同时触及止损与目标时按规则取一侧，双向都跑一遍才知道区间",
            "样本高度相关（同行业标的），有效样本量远小于记录条数",
            "窗口末尾的计划持仓期不足，按最后一根 4H 收盘强制平仓（记为 timeout）",
        ],
    }
    if n_seeds:
        out["control"] = {
            "seeds": n_seeds,
            "overall": _compare(df, ctrl, n_seeds),
            "by_plan": {k: _compare(g, ctrl[ctrl["plan_key"] == k], n_seeds)
                        for k, g in df.groupby("plan_key")},
            "regime_gate": _gate(ctrl, n_seeds),
        }
    if split:
        cut = pd.Timestamp(split)

        def first(d: pd.DataFrame) -> pd.Series:
            return pd.to_datetime(d["plan_date"], utc=True).dt.tz_convert(None).dt.normalize() <= cut

        halves = {}
        for name, pick in (("first", True), ("second", False)):
            r = df[first(df) == pick]
            h: dict[str, Any] = {"real": _stats(r), "regime_gate": _gate(r)}
            if n_seeds:
                c = ctrl[first(ctrl) == pick]
                h["control"] = _compare(r, c, n_seeds)
                h["control_regime_gate"] = _gate(c, n_seeds)
            halves[name] = h
        out["halves"] = {"split": split, **halves}
    return out


# ------------------------------------------------------------------ Brooks 计划 D

#: 并入"有可执行计划"分组的判定标准。跑之前写定，看到结果后不改（见 README）
BROOKS_MIN_TRADES = 100
BROOKS_MIN_Z = 2.0

#: 距离匹配随机对照：入场价随机挪开 ±0.5–1.0 ATR，止损 / T1 / T2 同步平移，
#: 订单类型与成交、离场规则全部不变——两组唯一的差别是"在哪里等"
CONTROL_OFFSET_ATR = (0.5, 1.0)
CONTROL_SEEDS = (0, 1, 2, 3, 4)


def simulate_brooks(
    plan: dict[str, Any],
    intraday: pd.DataFrame,
    daily: pd.DataFrame,
    plan_ts: pd.Timestamp,
    *,
    symbol: str,
    rules: TradeRules,
    regime: str | None = None,
) -> TradeResult:
    """D 的成交：挂单只在计划日之后的下一个交易日有效。

    stop：4H 高点越过入场价成交，跳空高开按开盘价；limit：低点回落到入场价成交，
    跳空低开按开盘价；market：次日第一根开盘价。成交后与三套计划共用 _hold。
    plan_ts 必须是计划日收盘之后的时刻——计划用到了当天收盘，当天盘中的 K 线不能参与成交。
    """
    entry, stop, t1 = plan.get("entry"), plan.get("stop"), plan.get("t1")
    res = TradeResult(
        symbol=symbol, plan_key=plan.get("setup_code") or "brooks",
        plan_date=plan_ts.isoformat(), executable=bool(plan.get("executable")),
        index_bullish=None, entry_plan=entry, stop_plan=stop, t1=t1, t2=plan.get("t2"),
        rr_plan=plan.get("rr"), regime=regime,
    )
    kind = plan.get("order_type")
    if entry is None or stop is None or t1 is None or kind not in ("stop", "limit", "market"):
        res.note = "计划缺少入场、止损、目标或订单类型"
        return res
    fwd = intraday[intraday.index > plan_ts]
    if fwd.empty:
        res.note = "计划日之后没有可用 4H 数据"
        return res
    fwd = fwd.iloc[: rules.max_holding_bars + 4]
    day0 = fwd.index[0].date()
    session = sum(1 for t in fwd.index[:4] if t.date() == day0)

    atr = float(atr_series(daily.loc[:plan_ts]).iloc[-1])
    if not np.isfinite(atr) or atr <= 0:
        res.note = "ATR 不可用"
        return res

    o = fwd["open"].to_numpy(float)
    h = fwd["high"].to_numpy(float)
    l = fwd["low"].to_numpy(float)
    fill_idx = fill = None
    for i in range(session):
        if kind == "market":
            fill_idx, fill = i, float(o[i])
            break
        if kind == "stop" and h[i] >= entry:
            fill_idx, fill = i, float(max(o[i], entry))
            break
        if kind == "limit" and l[i] <= entry:
            fill_idx, fill = i, float(min(o[i], entry))
            break
    if fill_idx is None:
        res.note = "下一个交易日未成交"
        return res
    if fill <= stop:
        res.note = "成交价已在止损位之下，撤单"
        return res
    res.entry_date = fwd.index[fill_idx].isoformat()
    res.entry_fill = round(fill, 4)
    return _hold(res, fwd, fill_idx, fill, stop, t1, plan.get("t2"), atr, rules,
                 r_unit=_r_unit(entry, stop))


def distance_matched(plan: dict[str, Any], atr: float, rng: np.random.RandomState) -> dict[str, Any]:
    """距离匹配的随机对照：整组价位平移同一个随机偏移。"""
    off = float(rng.choice([-1.0, 1.0]) * rng.uniform(*CONTROL_OFFSET_ATR) * atr)
    q = dict(plan)
    for k in ("entry", "stop", "t1", "t2"):
        if q.get(k) is not None:
            q[k] = round(float(q[k]) + off, 4)
    q["control_offset"] = round(off, 4)
    return q


def _brooks_symbol(job: tuple) -> list[dict[str, Any]]:
    """一个标的的 D 回测（进程池任务，必须是模块级函数）。"""
    import zlib

    symbol, bm, provider, start, end, rules, seeds = job
    daily = provider.daily(symbol)
    intraday = align_to_daily(provider.fetch(symbol, "4h").df, daily)
    _, items = replay_plans(symbol, provider, start=start, end=end, benchmark=bm)

    streams: dict[str, list[tuple[dict, dict]]] = {"real_all": [], "real_exe": []}
    for it in items:
        streams["real_all"].append((it, it["plan"]))
        if it["plan"].get("executable"):
            streams["real_exe"].append((it, it["plan"]))
    for seed in seeds:
        rng = np.random.RandomState(seed * 100_003 + zlib.crc32(symbol.encode()) % 100_000)
        streams[f"ctrl_{seed}"] = [
            (it, distance_matched(p, it["atr"], rng)) for it, p in streams["real_exe"]
        ]

    rows: list[dict[str, Any]] = []
    for name, plans in streams.items():
        busy_until = None     # 同一标的同时只持有一笔：持仓期间的新计划不计
        for it, plan in plans:
            if busy_until is not None and it["ts"] < busy_until:
                continue
            res = simulate_brooks(plan, intraday, daily, it["ts"], symbol=symbol,
                                  rules=rules, regime=it["regime"])
            d = res.to_dict()
            d.update(stream=name, order_type=plan.get("order_type"),
                     setup_code=plan.get("setup_code"), brooks_state=it.get("brooks_state"))
            rows.append(d)
            if res.outcome in TRIGGERED and res.exit_date:
                busy_until = pd.Timestamp(res.exit_date)
    return rows


def run_brooks_backtest(
    pairs: list[tuple[str, str | None]],
    provider: Any,
    *,
    start: str,
    end: str,
    rules: TradeRules | None = None,
    seeds: tuple[int, ...] = CONTROL_SEEDS,
    jobs: int = 1,
) -> pd.DataFrame:
    """D 与距离匹配随机对照的逐笔结果。stream：real_exe / real_all / ctrl_<seed>。"""
    rules = rules or TradeRules()
    tasks = [(symbol, bm, _subset(provider, {symbol, bm} - {None}), start, end, rules, tuple(seeds))
             for symbol, bm in pairs]
    rows: list[dict[str, Any]] = []
    if jobs > 1:
        from concurrent.futures import ProcessPoolExecutor

        with ProcessPoolExecutor(max_workers=jobs) as ex:
            for part in ex.map(_brooks_symbol, tasks):
                rows += part
    else:
        for t in tasks:
            rows += _brooks_symbol(t)
    return pd.DataFrame(rows)


def _compare(real: pd.DataFrame, ctrl: pd.DataFrame, n_seeds: int) -> dict[str, Any]:
    """真实 vs 对照。对照组 n 个种子来自同一批计划、彼此相关，
    标准误按单个种子的样本量算（保守），不按合并后的笔数。"""
    rs, cs = _stats(real), _stats(ctrl)
    out: dict[str, Any] = {"real": rs, "control": cs, "diff_r": None, "z_diff": None, "z_real": None}
    rt = real[real["outcome"].isin(TRIGGERED)]["r_multiple"].astype(float)
    ct = ctrl[ctrl["outcome"].isin(TRIGGERED)]["r_multiple"].astype(float)
    if len(rt) and len(ct):
        out["diff_r"] = round(float(rt.mean() - ct.mean()), 3)
    if len(rt) > 1:
        se_r = rt.std(ddof=1) / np.sqrt(len(rt))
        out["z_real"] = round(float(rt.mean() / se_r), 2) if se_r > 0 else None
        if len(ct) > 1:
            se_c = ct.std(ddof=1) / np.sqrt(max(len(ct) / max(n_seeds, 1), 1))
            se = np.sqrt(se_r**2 + se_c**2)
            out["z_diff"] = round(out["diff_r"] / se, 2) if se > 0 else None
    return out


def summarise_brooks(df: pd.DataFrame, *, start: str, end: str) -> dict[str, Any]:
    if df.empty:
        return {"error": "无记录"}
    real = df[df["stream"] == "real_exe"]
    ctrl = df[df["stream"].str.startswith("ctrl_")]
    n_seeds = ctrl["stream"].nunique()
    mid = pd.Timestamp(start) + (pd.Timestamp(end) - pd.Timestamp(start)) / 2

    def first_half(d: pd.DataFrame) -> pd.Series:
        return pd.to_datetime(d["plan_date"], utc=True).dt.tz_convert(None) < mid

    main = _compare(real, ctrl, n_seeds)
    halves = {
        "first": _compare(real[first_half(real)], ctrl[first_half(ctrl)], n_seeds),
        "second": _compare(real[~first_half(real)], ctrl[~first_half(ctrl)], n_seeds),
    }
    by_setup = {k: _compare(g, ctrl[ctrl["setup_code"] == k], n_seeds)
                for k, g in real.groupby("setup_code")}
    by_regime = {k: _compare(g, ctrl[ctrl["regime"] == k], n_seeds)
                 for k, g in real.groupby("regime")}

    n = main["real"].get("triggered", 0)
    c1 = n >= BROOKS_MIN_TRADES and (main["z_real"] or 0) >= BROOKS_MIN_Z \
        and (main["real"].get("expectancy_r") or 0) > 0
    c2 = (main["z_diff"] or 0) >= BROOKS_MIN_Z
    c3 = all((h["diff_r"] or 0) > 0 for h in halves.values())
    up = (by_regime.get("up") or {}).get("real", {}).get("expectancy_r")
    rest = [v["real"].get("expectancy_r") for k, v in by_regime.items() if k != "up"]
    gate_needed = bool(up is not None and up > 0 and rest and all((x or 0) <= 0 for x in rest))
    return {
        "window": {"start": start, "end": end, "split": mid.date().isoformat()},
        "criteria": {
            "1_real_positive": {"pass": bool(c1), "rule": f"可执行 D 触发 >= {BROOKS_MIN_TRADES} 笔、期望 > 0、z >= {BROOKS_MIN_Z}"},
            "2_beats_control": {"pass": bool(c2), "rule": f"D − 距离匹配随机对照 > 0，z >= {BROOKS_MIN_Z}"},
            "3_both_halves": {"pass": bool(c3), "rule": "前后两半窗口里差值都 > 0"},
            "4_regime_gate_needed": {"value": gate_needed, "rule": "只在大盘向上时为正 → 并入时须接 REGIME_GATE"},
        },
        "merge_recommended": bool(c1 and c2 and c3),
        "main": main,
        "all_plans": _stats(df[df["stream"] == "real_all"]),
        "halves": halves,
        "by_setup": by_setup,
        "by_regime": by_regime,
        "by_order_type": {k: _stats(g) for k, g in real.groupby("order_type")},
        "caveats": [
            "期望值以 R 倍数计，未计滑点、佣金与融资成本",
            "同一根 4H 内同时触及止损与目标按止损算；止损单成交那根若开盘就在止损下方，按开盘价出场（偏保守）",
            "对照组标准误按单个种子样本量计（保守）",
            "样本高度相关（同行业标的），有效样本量远小于记录条数",
            "Brooks 概率是课程经验值，这里检验的是代码化后的规则在个股日线上的表现",
        ],
    }

