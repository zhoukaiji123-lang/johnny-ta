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
) -> TradeResult:
    """模拟一套计划从等待到离场的完整过程。"""
    entry, stop = plan.get("entry"), plan.get("stop")
    res = TradeResult(
        symbol=symbol, plan_key=plan["key"], plan_date=plan_ts.isoformat(),
        executable=bool(plan.get("executable")), index_bullish=index_bullish,
        entry_plan=entry, stop_plan=stop, t1=plan.get("t1"), t2=plan.get("t2"),
        rr_plan=plan.get("rr"),
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
    fwd = fwd.iloc[: rules.ttl_bars + rules.max_holding_bars + rules.confirm_window]
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
        confirm_ts = above.index[0]
        after = np.flatnonzero(fwd.index > confirm_ts)
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
    res.entry_date = fwd.index[fill_idx].isoformat()
    res.entry_fill = round(fill, 4)
    r_unit = _r_unit(fill, stop)
    if r_unit <= 0:
        res.note = "成交价已在止损位之下"
        return res

    # ---------------------------------------------------------------- 持仓
    t1, t2 = plan.get("t1"), plan.get("t2")
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


def collect_plans(
    symbol: str,
    provider: Any,
    *,
    start: str,
    end: str,
    benchmark: str | None = None,
    dedup_atr: float = PLAN_DEDUP_ATR,
) -> list[dict[str, Any]]:
    """逐日回放，收集当天生成的三套计划。

    去重按 (计划类型, 入场价) 做：一个支撑位会连续很多天出现在计划里，
    不去重的话同一次机会会被计成几十笔，统计立刻失真。
    """
    from .analyze import analyze

    daily = provider.daily(symbol)
    days = daily.loc[(daily.index >= start) & (daily.index <= end)].index

    out: list[dict[str, Any]] = []
    seen: list[tuple[str, float, pd.Timestamp]] = []
    for day in days:
        as_of = day + pd.Timedelta(hours=23)
        try:
            r = analyze(
                symbol, as_of=as_of, benchmark=benchmark,
                provider=provider, include_events=False,
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
        seen = [s for s in seen if (day - s[2]).days <= 60]
        for plan in r["plans"]:
            if plan.get("entry") is None:
                continue
            key, entry = plan["key"], float(plan["entry"])
            if any(k == key and abs(e - entry) <= dedup_atr * atr for k, e, _ in seen):
                continue
            seen.append((key, entry, day))
            out.append({"ts": day, "plan": plan, "index_bullish": idx_bull, "atr": atr})
    return out


def run_backtest(
    pairs: list[tuple[str, str | None]],
    provider: Any,
    *,
    start: str,
    end: str,
    rules: TradeRules | None = None,
    executable_only: bool = False,
) -> pd.DataFrame:
    rules = rules or TradeRules()
    rows: list[dict[str, Any]] = []
    for symbol, bm in pairs:
        daily = provider.daily(symbol)
        intraday = provider.fetch(symbol, "4h").df
        sig = bar_signals(intraday)
        for item in collect_plans(symbol, provider, start=start, end=end, benchmark=bm):
            plan = item["plan"]
            if executable_only and not plan.get("executable"):
                continue
            res = simulate(
                plan, intraday, daily, item["ts"], symbol=symbol,
                index_bullish=item["index_bullish"], rules=rules, signals=sig,
            )
            rows.append(res.to_dict())
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


def summarise_trades(df: pd.DataFrame) -> dict[str, Any]:
    if df.empty:
        return {"error": "无记录"}
    exe = df[df["executable"]]
    out = {
        "overall": _stats(df),
        "executable_only": _stats(exe) if len(exe) else None,
        "by_plan": {k: _stats(g) for k, g in df.groupby("plan_key")},
        "by_index": {
            ("bullish" if k else "not_bullish"): _stats(g)
            for k, g in df.dropna(subset=["index_bullish"]).groupby("index_bullish")
        },
        "caveats": [
            "期望值以 R 倍数计，未计滑点、佣金与融资成本",
            "同一根 4H 内同时触及止损与目标时按规则取一侧，双向都跑一遍才知道区间",
            "样本高度相关（同行业标的），有效样本量远小于记录条数",
        ],
    }
    return out
