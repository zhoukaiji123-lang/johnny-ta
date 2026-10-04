"""Al Brooks 价格行为层（日线）。

规则来源：Al Brooks《How to Trade Price Action》课程（Brooks Trading Course），
经 brooks-price-action skill 整理。注释里的 chNN 指课程章节。这里只把能量化的部分
落成确定性代码：Always In 方向、市场周期阶段、K 线计数、MTR 清单、高潮/楔形提示、
测量移动目标，以及一套按交易者方程式判定的第 4 套计划（D）。

口径（必须一起读）：
- 课程例子以 Emini / 外汇日内图为主，搬到个股日线上属于推演。阈值一律取原文数字，
  不按本仓库数据调参——调了就成了对样本的拟合，不再是 Brooks。
- 概率是课程给的经验值（ch30、ch34、ch43、ch45 等），不是本仓库的回测结果。
- 只做多，与 A/B/C 一致。AIS 时 D 只在 MTR 条件齐备时给出做多计划，其余时候只描述。
- 不受 REGIME_GATE 约束（Brooks 用自己的 Always In 判断背景），也不改变看板分组：
  在回测验证之前，它只作为独立参考。

实现上把空头方向的 K 线取负（high↔-low），同一套"顺势"代码同时处理两个方向，
避免多空两份镜像代码各自漂移。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

#: 美股最小报价单位。Brooks 的"信号 K 线上方 1 tick"
TICK = 0.01

#: Brooks 用的均线（ch14、ch22）
EMA_SPAN = 20

#: 平均 K 线幅度的回看根数
AVG_RANGE_BARS = 20

#: 趋势 K 线：实体至少占振幅一半，收盘在顺势一端的 1/3 内（ch08、ch10）。
#: 日线股票按"含缺口"口径：实体 = 前收到收盘，振幅 = 真实波幅。ch11："缺口本身就是
#: BO 趋势 K 线"，日线股票上缺口很常见——只看开收实体，跳空低开再收阳的暴跌日
#: 会被当成阳线，整段跳空下跌都认不出空头 BO
TREND_BODY_RATIO = 0.5
CLOSE_ZONE = 1 / 3

#: 单根就能构成 BO 的"巨大趋势 K 线"：振幅 >= 2 倍平均（ch13）
HUGE_BAR_MULT = 2.0

#: 单根巨大 K 线 BO 等 FT 的最长根数（ch15）
FT_WINDOW = 10

#: BO 要越过的前高/前低回看根数
BO_REF_BARS = 10

#: Always In 只看最近这么多根里的 BO
AI_LOOKBACK = 120

#: PB 持续 >= 20 根 → TR / Endless PB（ch09、ch18）
TR_PB_BARS = 20

#: 数到 H4 就停：市场已是 TR 或反转（ch09）
MAX_COUNT = 4

#: 紧密通道：PB 多为 1–3 根、小于平均 K 线 2 倍、回撤 <= 1/3–1/2（ch17、ch43）
TIGHT_PB_BARS = 3
TIGHT_PB_RANGE_MULT = 2.0
TIGHT_RETRACE = 0.5

#: 回调幅度 >= 2 倍平均 K 线才算主要回调、才刷新"前一腿起点"（主要 HL）。
#: 紧密通道里的回调按定义小于 2 倍平均 K 线（ch17），Brooks 也要求止损不放在
#: minor HL 下方（ch33）。小回调仍计入通道松紧统计，但不定义腿——否则一根 K 线的
#: 小回调就会被当成 HL，下一次小回调立刻被判成"更低低点"
MAJOR_PB_MULT = 2.0

#: "强 BO"阶段：BO 结束后最多这么多根仍没出现回调（推演：cheatsheet"回调迟迟不来"）
BREAKOUT_FRESH_BARS = 5

#: 只用最近这么多根里的 PB 判断通道松紧，太久远的结构不代表当前阶段
CHANNEL_WINDOW = 60

#: TTR：整个区间高度 < 3 倍平均 K 线（ch17、ch47）
TTR_HEIGHT_MULT = 3.0

#: 趋势 20 根以上后出现全趋势最大的 K 线 → 更像衰竭（ch29、ch42）
CLIMAX_TREND_BARS = 20
CLIMAX_RECENT_BARS = 3

#: 价格在均线同侧 20 根以上 → 首次回踩均线（A8，ch14）
GAP_BARS = 20

#: MTR 的反向腿：>= 5 根强 K 线，或 >= 10 根普通 K 线（ch39）；测试回撤 >= 1/3（ch22）
MTR_STRONG_BARS = 5
MTR_WEAK_BARS = 10
MTR_TEST_RETRACE = 1 / 3

#: 交易者方程式：概率 → 最低回报倍数（cheatsheet §2：40%/50% 配 >= 2 倍，60% 配 >= 1 倍）
MIN_RR = {0.6: 1.0, 0.5: 2.0, 0.4: 2.0}

#: 正确止损超过平均 K 线 2 倍 → 仓位降到 1/2，约 3 倍 → 1/3（ch33、ch51）
WIDE_STOP_MULT = 2.0
VERY_WIDE_STOP_MULT = 3.0

STATE_LABELS = {
    "breakout": "强 BO",
    "tight_channel": "紧密通道",
    "broad_channel": "宽幅通道",
    "trading_range": "交易区间 TR",
    "tight_trading_range": "紧密交易区间 TTR",
}
AI_LABELS = {1: "Always In Long", -1: "Always In Short", 0: "Always In 不明"}
AI_SHORT = {1: "AIL", -1: "AIS", 0: "不明"}


# ------------------------------------------------------------------ 基础


def _oriented(df: pd.DataFrame, sign: int):
    o, h, l, c = (df[k].to_numpy(dtype=float) for k in ("open", "high", "low", "close"))
    if sign > 0:
        return o, h, l, c
    return -o, -l, -h, -c


def _avg_range(df: pd.DataFrame) -> np.ndarray:
    """每根 bar 之前 20 根的平均真实波幅（不含自身，否则大 K 线会抬高自己的门槛）。"""
    pc = df["close"].shift(1).fillna(df["open"])
    rng = (np.maximum(df["high"], pc) - np.minimum(df["low"], pc)).astype(float)
    avg = rng.rolling(AVG_RANGE_BARS, min_periods=5).mean().shift(1)
    return avg.bfill().fillna(rng).to_numpy()


def _trend_bars(o, h, l, c, avg):
    """含缺口的趋势 K 线（定向价，顺势为正）：返回 (趋势 K 线, 大, 巨大)。"""
    pc = np.r_[o[0], c[:-1]]
    hh, ll = np.maximum(h, pc), np.minimum(l, pc)
    rng = np.maximum(hh - ll, 1e-12)
    trend = ((c - pc) >= TREND_BODY_RATIO * rng) & ((hh - c) <= CLOSE_ZONE * rng)
    big = trend & (rng >= avg)
    return trend, big, trend & (rng >= HUGE_BAR_MULT * avg)


def _strong_bars(o, h, l, c, avg):
    """强趋势 K 线（大）与巨大趋势 K 线，供 MTR 反弹腿计数用。"""
    _, big, huge = _trend_bars(o, h, l, c, avg)
    return big, huge


def _signal_quality(o, h, l, c, i: int) -> str:
    """信号 K 线质量（ch08）：顺势实体且收在顺势 1/3 为强，逆势实体收在逆势 1/3 为弱。"""
    rng = max(h[i] - l[i], 1e-12)
    if c[i] > o[i] and (h[i] - c[i]) <= CLOSE_ZONE * rng:
        return "strong"
    if c[i] < o[i] and (c[i] - l[i]) <= CLOSE_ZONE * rng:
        return "weak"
    return "fair"


SIGNAL_LABELS = {"strong": "强", "fair": "一般", "weak": "弱"}


@dataclass
class Breakout:
    sign: int
    start: int
    end: int
    origin: float      # 定向价：BO 起点（多头为 BO 段最低价）
    top: float         # 定向价：BO 段极值
    confirmed: bool
    confirm_index: int | None


def _breakouts(df: pd.DataFrame, sign: int, avg: np.ndarray, start: int) -> list[Breakout]:
    """强 BO：连续同向趋势 K 线，收盘越过前 10 根极值，且满足 ch13 三种形成方式之一：
    ① 一根巨大趋势 K 线（>= 2 倍平均）收在极值；② 两根大趋势 K 线各自收在极值；
    ③ 3–5 根较小的同向趋势 K 线。

    ②③ 自带 FT；①需要 FT——之后 10 根内出现一根同向趋势 K 线、收盘越过 BO 收盘，
    且期间没有收回 BO 起点（ch15：FT 有时晚到，10 根内不来就转为 BO Mode）。
    还在等 FT 的记为未确认。
    """
    o, h, l, c = _oriented(df, sign)
    trend, big, huge = _trend_bars(o, h, l, c, avg)
    pc = np.r_[o[0], c[:-1]]
    n = len(c)
    out: list[Breakout] = []
    i = max(start, BO_REF_BARS)
    while i < n:
        if not trend[i]:
            i += 1
            continue
        s = i
        while i + 1 < n and trend[i + 1]:
            i += 1
        e = i
        run = e - s + 1
        seg = slice(s, e + 1)
        if c[e] > h[s - BO_REF_BARS:s].max():
            # 起点含缺口：跳空 BO 的起点是缺口底部（前一根收盘），不是 BO K 线自己的低点（ch11）
            origin, top = float(min(l[seg].min(), pc[s])), float(h[seg].max())
            if run >= 3 or int(big[seg].sum()) >= 2:
                out.append(Breakout(sign, s, e, origin, top, True, int(e)))
            elif huge[seg].any():
                ci = None
                for j in range(e + 1, min(n, e + 1 + FT_WINDOW)):
                    if c[j] < origin:
                        break
                    if trend[j] and c[j] > c[e]:
                        ci = j
                        break
                out.append(Breakout(sign, s, e, origin, top, ci is not None, ci))
        i = e + 1
    return out


@dataclass
class Pullback:
    start: int          # PB 第一根
    end: int            # PB 最后一根
    bars: int
    low: float          # 定向价
    extreme: float      # 这次 PB 之前的顺势极值
    leg_low: float      # 前一腿起点
    retrace: float      # 回撤占前一腿比例
    depth_avg: float    # 回撤幅度 / 平均 K 线
    major: bool = True  # 是否主要回调（会刷新前一腿起点）


@dataclass
class Structure:
    extreme: float
    extreme_index: int
    leg_low: float
    pullbacks: list[Pullback]
    current: Pullback | None
    broken: bool        # 当前 PB 跌破前一腿起点（出现 LL）

    @property
    def majors(self) -> list[Pullback]:
        return [pb for pb in self.pullbacks if pb.major]


def _structure(df: pd.DataFrame, sign: int, bo: Breakout, avg: np.ndarray) -> Structure:
    """从 Always In 起点逐根走，记录每次顺势腿与回调。"""
    o, h, l, c = _oriented(df, sign)
    n = len(c)
    ext_i = bo.start + int(np.argmax(h[bo.start:bo.end + 1]))
    ext, leg_low = float(h[ext_i]), bo.origin
    pbs: list[Pullback] = []
    for i in range(ext_i + 1, n):
        if h[i] > ext:
            if i - ext_i > 1:
                seg = slice(ext_i + 1, i)
                low = float(l[seg].min())
                leg = max(ext - leg_low, 1e-12)
                major = (ext - low) >= MAJOR_PB_MULT * avg[ext_i]
                pbs.append(Pullback(ext_i + 1, i - 1, i - ext_i - 1, low, ext, leg_low,
                                    (ext - low) / leg, (ext - low) / avg[ext_i], major))
                if major:
                    leg_low = low
            ext, ext_i = float(h[i]), i
    current = None
    if ext_i < n - 1:
        seg = slice(ext_i + 1, n)
        low = float(l[seg].min())
        leg = max(ext - leg_low, 1e-12)
        current = Pullback(ext_i + 1, n - 1, n - 1 - ext_i, low, ext, leg_low,
                           (ext - low) / leg, (ext - low) / avg[ext_i])
    # 破位按收盘判断：盘中刺穿前一腿起点又收回，是失败的反向 BO，不是更低低点
    broken = bool(current and float(c[ext_i + 1:].min()) < leg_low)
    return Structure(ext, ext_i, leg_low, pbs, current, broken)


def _bar_count(df: pd.DataFrame, sign: int, st: Structure) -> dict[str, Any]:
    """当前 PB 里的 H/L 计数（ch09）。

    高点越过前一根高点 = 一次顺势恢复尝试；之后价格跌破该次信号 K 线低点，
    说明恢复失败、开始新一推，下一次越过才计入下一个数字。
    """
    o, h, l, c = _oriented(df, sign)
    n = len(c)
    marks: list[dict[str, Any]] = []
    count, armed, sig_low = 0, True, None
    if st.current is not None:
        for i in range(st.extreme_index + 1, n):
            if armed and h[i] > h[i - 1]:
                count += 1
                marks.append({"signal": i - 1, "entry": i, "n": count})
                armed, sig_low = False, l[i - 1]
            elif not armed and l[i] < sig_low:
                armed = True
    pending = st.current is not None and armed
    return {
        "count": count,
        "marks": marks,
        "pending": pending,
        "next": count + 1 if pending else None,
        "signal_quality": _signal_quality(o, h, l, c, n - 1),
    }


def _gap_bars(df: pd.DataFrame, ema20: np.ndarray, sign: int) -> dict[str, Any]:
    """均线同侧连续根数与首次回踩（A8，ch14）。"""
    o, h, l, c = _oriented(df, sign)
    e = sign * ema20
    run = 0
    for i in range(len(c) - 2, -1, -1):
        if not np.isfinite(e[i]) or l[i] <= e[i]:
            break
        run += 1
    first_touch = run >= GAP_BARS and np.isfinite(e[-1]) and l[-1] <= e[-1]
    above_now = 0
    for i in range(len(c) - 1, -1, -1):
        if not np.isfinite(e[i]) or l[i] <= e[i]:
            break
        above_now += 1
    return {"gap_bars": above_now, "first_touch": bool(first_touch), "prior_run": run}


# ------------------------------------------------------------------ MTR（做多方向）


def _mtr_long(df: pd.DataFrame, ema20: np.ndarray, avg: np.ndarray, atr: float,
              window_start: int, bear_tight: bool) -> dict[str, Any]:
    """AIS 中的 HL / LL MTR 买入清单（cheatsheet §4，D1）。

    第 4 条"背景支持"（MM 目标、楔形第 3 推、Final Flag、连续高潮的汇合）需要综合判断，
    没有实现，清单里明确标出。
    """
    o, h, l, c = (df[k].to_numpy(dtype=float) for k in ("open", "high", "low", "close"))
    n = len(c)
    w0 = max(0, window_start)
    i0 = w0 + int(np.argmin(l[w0:]))
    low0 = float(l[i0])

    def rally_ok(a: int, b: int) -> tuple[bool, str]:
        seg = slice(a, b + 1)
        strong, _ = _strong_bars(o, h, l, c, avg)
        n_strong = int(strong[seg].sum())
        bars = b - a + 1
        crossed = bool(np.any(c[seg] > ema20[seg]))
        ok = (n_strong >= MTR_STRONG_BARS or bars >= MTR_WEAK_BARS) and crossed
        return bool(ok), (f"反弹腿 {bars} 根、其中强阳线 {n_strong} 根，"
                    f"{'已' if crossed else '未'}收上 20 EMA")

    case = None
    if i0 < n - 2:
        iR = i0 + 1 + int(np.argmax(h[i0 + 1:]))
        if iR < n - 1:
            iT = iR + 1 + int(np.argmin(l[iR + 1:]))
            R, T = float(h[iR]), float(l[iT])
            retr = (R - T) / max(R - low0, 1e-12)
            ok1, txt1 = rally_ok(i0, iR)
            case = {"kind": "HL", "rally": (i0, iR), "R": R, "test": T, "test_index": iT,
                    "base": low0, "retrace": retr, "rally_ok": ok1, "rally_text": txt1,
                    "test_ok": bool(retr >= MTR_TEST_RETRACE and T > low0)}
    if (case is None or not (case["rally_ok"] and case["test_ok"])) and i0 > w0 + 1:
        iR = w0 + int(np.argmax(h[w0:i0]))
        if iR > w0:
            iP = w0 + int(np.argmin(l[w0:iR]))
            P, R = float(l[iP]), float(h[iR])
            ok1, txt1 = rally_ok(iP, iR)
            near = low0 >= P - atr          # "旧低附近失败"：新低不超过前低 1 ATR（推演）
            ll = {"kind": "LL", "rally": (iP, iR), "R": R, "test": low0, "test_index": i0,
                  "base": P, "retrace": (R - low0) / max(R - P, 1e-12), "rally_ok": ok1,
                  "rally_text": txt1, "test_ok": bool(near and low0 < P)}
            if case is None or (ll["rally_ok"] and ll["test_ok"]):
                case = ll

    q = _signal_quality(o, h, l, c, n - 1)
    if case is None:
        items = [
            {"ok": False, "text": "空头趋势后还没有像样的反弹腿，谈不上 MTR"},
        ]
    else:
        items = [
            {"ok": case["rally_ok"],
             "text": f"1. 强力突破空头趋势线与均线：{case['rally_text']}"
                     f"（需 >= {MTR_STRONG_BARS} 根强阳线或 >= {MTR_WEAK_BARS} 根，ch21/ch39）"},
            {"ok": case["test_ok"],
             "text": (f"2. 测试前低的第二腿（{case['kind']}）：测试低点 {case['test']:.2f}，"
                      f"回撤反弹腿 {case['retrace']:.0%}（需 >= 33%，ch22）")},
            {"ok": q == "strong",
             "text": f"3. 信号 K 线：最后一根质量{SIGNAL_LABELS[q]}（强信号或第二入场）"},
            {"ok": not bear_tight,
             "text": "5. 不在紧密空头通道里（ch38）" + ("" if not bear_tight else "：空头通道仍然紧密")},
            {"ok": None, "text": "4. 背景支持（MM / 楔形第 3 推 / 连续高潮汇合）：需人工判断，未实现"},
        ]
    valid = case is not None and all(it["ok"] for it in items if it["ok"] is not None)
    return {"case": case, "items": items, "valid": bool(valid), "signal_quality": q}


# ------------------------------------------------------------------ 主入口


def _round(v: float | None) -> float | None:
    return None if v is None or not np.isfinite(v) else round(float(v), 2)


def _equation(p: float, entry: float, stop: float, target: float | None) -> dict[str, Any]:
    risk = entry - stop
    if target is None or risk <= 0:
        return {"p": p, "risk": _round(risk), "reward": None, "rr": None, "ev_r": None,
                "min_rr": MIN_RR[p], "ok": False,
                "text": "目标或止损缺失，交易者方程式无法计算"}
    reward = target - entry
    rr = reward / risk
    ev = p * rr - (1 - p)
    # 按展示精度比较：入场与止损各差 1 tick 会让 1:1 算成 0.998，显示 1.00 却判不通过
    ok = ev > 0 and round(rr, 2) >= MIN_RR[p]
    return {
        "p": p, "risk": _round(risk), "reward": _round(reward), "rr": round(rr, 2),
        "ev_r": round(ev, 2), "min_rr": MIN_RR[p], "ok": bool(ok),
        "text": (f"{p:.0%} × {rr:.2f}R − {1 - p:.0%} × 1R = {ev:+.2f}R"
                 f"（{p:.0%} 概率要求回报 >= {MIN_RR[p]:g} 倍风险）"),
    }


def _plan(*, setup: str, code: str, trigger: str, entry: float | None, stop: float | None,
          stop_basis: str, t1: float | None, t2: float | None, p: float | None,
          t1_basis: str, cancel_if: list[str], avg_range: float,
          blocked: list[str] | None = None, cautions: list[str] | None = None) -> dict[str, Any]:
    blocked = list(blocked or [])
    cautions = list(cautions or [])
    if t2 is not None and (t1 is None or t2 <= t1):
        t2 = None   # MM 落在 T1 之内或入场之下，没有意义
    eq = None
    scale = 1.0
    if entry is not None and stop is not None and p is not None:
        eq = _equation(p, entry, stop, t1)
        if not eq["ok"]:
            blocked.append(f"交易者方程式不成立：{eq['text']}")
        dist = (entry - stop) / max(avg_range, 1e-12)
        if dist > VERY_WIDE_STOP_MULT:
            scale = 1 / 3
        elif dist > WIDE_STOP_MULT:
            scale = 0.5
        if scale < 1:
            cautions.append(f"止损距离 {dist:.1f} 倍平均 K 线，仓位降到 {scale:.0%}，"
                            "金额风险不变（ch33）")
    elif entry is None:
        blocked.append("没有可用的入场位")
    return {
        "key": "brooks",
        "title": f"D Brooks · {setup}",
        "setup": setup,
        "setup_code": code,
        "trigger": trigger,
        "entry": _round(entry),
        "stop": _round(stop),
        "stop_basis": stop_basis,
        "t1": _round(t1),
        "t1_basis": t1_basis,
        "t2": _round(t2),
        "rr": eq["rr"] if eq else None,
        "probability": p,
        "equation": eq,
        "position_scale": round(scale, 3),
        "tranche_text": "Brooks：入场即挂真实止损；2 倍处平 50%，3 倍处平 25%，余下跟踪（ch36）",
        "cancel_if": cancel_if,
        "executable": not blocked,
        "blocked_by": blocked,
        "cautions": cautions,
        "entry_level": None,
        "direction": "long",
    }


def _no_plan(setup: str, code: str, reason: str, wait: str, avg_range: float) -> dict[str, Any]:
    return _plan(setup=setup, code=code, trigger=wait, entry=None, stop=None,
                 stop_basis="—", t1=None, t2=None, p=None, t1_basis="—",
                 cancel_if=[], avg_range=avg_range, blocked=[reason])


def brooks_analysis(
    df: pd.DataFrame,
    ema20: pd.Series,
    *,
    atr: float,
    event_mode: bool = False,
    event_reason: str | None = None,
) -> dict[str, Any]:
    """df：日线（小写列），最后一根为最近已收盘 bar；ema20 与 df 同索引、用完整历史算出。"""
    n = len(df)
    if n < 2 * AVG_RANGE_BARS + BO_REF_BARS:
        return {"available": False, "reason": f"日线仅 {n} 根，不足以做 Brooks 背景判断"}

    idx = df.index
    e20 = ema20.reindex(idx).to_numpy(dtype=float)
    avg = _avg_range(df)
    avg_now = float(avg[-1])
    close = float(df["close"].iloc[-1])
    start = max(0, n - AI_LOOKBACK)

    events = _breakouts(df, 1, avg, start) + _breakouts(df, -1, avg, start)
    confirmed = sorted((e for e in events if e.confirmed), key=lambda e: e.confirm_index)
    pending = [e for e in events if not e.confirmed and e.end >= n - FT_WINDOW]

    ai, regime_bo, latest_bo, ai_note, fail_index = 0, None, None, None, None
    if confirmed:
        latest_bo = confirmed[-1]
        ai = latest_bo.sign
        regime_bo = latest_bo
        for e in reversed(confirmed):
            if e.sign != ai:
                break
            regime_bo = e
        _, _, _, c_or = _oriented(df, ai)
        after = c_or[latest_bo.confirm_index + 1:]
        if len(after) and after.min() < regime_bo.origin:
            fail_index = latest_bo.confirm_index + 1 + int(np.argmax(after < regime_bo.origin))
            ai_note = (f"最近一次{'多头' if ai > 0 else '空头'} BO（"
                       f"{idx[regime_bo.start].date()}）已被收盘反穿起点，BO 失败（ch15）")
            ai = 0
    else:
        ai_note = f"最近 {AI_LOOKBACK} 根内没有确认的强 BO（ch13：判断不出时看最近一次清晰 BO）"

    orient = ai if ai != 0 else (latest_bo.sign if latest_bo else 1)
    st = _structure(df, orient, regime_bo, avg) if regime_bo is not None else None
    count = _bar_count(df, orient, st) if st is not None else None
    gap = _gap_bars(df, e20, orient)
    trend_bars = (n - 1 - regime_bo.start) if regime_bo is not None else 0

    # ---------------- 市场周期阶段
    state_reason = ""
    tight_stats = None
    if ai == 0:
        state, state_reason = "trading_range", "Always In 不明 → 按 TR 处理（拿不准趋势还是 TR 就当 TR，ch13）"
    elif st.broken:
        state, state_reason = "trading_range", (
            "回调收盘跌破前一腿起点（出现更低低点），不再是多头趋势（ch45）" if ai > 0
            else "反弹收盘越过前一腿起点（出现更高高点），不再是空头趋势（ch45/ch46）")
    elif st.current is not None and st.current.bars >= TR_PB_BARS:
        state, state_reason = "trading_range", f"当前{'回调' if ai > 0 else '反弹'}已 {st.current.bars} 根（>= {TR_PB_BARS}），视为 TR（ch09/ch18）"
    elif count and count["count"] >= MAX_COUNT:
        state, state_reason = "trading_range", f"回调已数到 {'H' if ai > 0 else 'L'}{count['count']}，停止计数，按 TR 处理（ch09）"
    elif not st.pullbacks and (st.current is None or st.current.bars <= 1) \
            and (n - 1 - latest_bo.end) <= BREAKOUT_FRESH_BARS:
        state, state_reason = "breakout", "BO 之后回调迟迟没来（cheatsheet §1）"
    else:
        recent = [pb for pb in st.pullbacks if pb.end >= n - CHANNEL_WINDOW]
        if st.current is not None:
            recent.append(st.current)
        if not recent:
            state, state_reason = "tight_channel", "BO 后还没有完整回调，拿不准紧密还是宽幅就当紧密（ch12）"
        else:
            short_share = sum(pb.bars <= TIGHT_PB_BARS for pb in recent) / len(recent)
            max_retr = max(pb.retrace for pb in recent)
            max_depth = max(pb.depth_avg for pb in recent)
            tight_stats = {"pullbacks": len(recent), "short_share": round(short_share, 2),
                           "max_retrace": round(max_retr, 2), "max_depth_avg": round(max_depth, 2)}
            broad_reasons = []
            if short_share < 2 / 3:
                broad_reasons.append(f"只有 {short_share:.0%} 的回调在 {TIGHT_PB_BARS} 根以内")
            if max_retr > TIGHT_RETRACE:
                broad_reasons.append(f"最深回调 {max_retr:.0%} 前一腿")
            if max_depth > TIGHT_PB_RANGE_MULT:
                broad_reasons.append(f"最深回调 {max_depth:.1f} 倍平均 K 线")
            if broad_reasons:
                state = "broad_channel"
                state_reason = "；".join(broad_reasons) + "（宽幅通道，ch45/ch46）"
            else:
                state = "tight_channel"
                state_reason = (f"最近 {len(recent)} 次回调都短而浅：{short_share:.0%} 在 "
                                f"{TIGHT_PB_BARS} 根以内、最深 {max_retr:.0%}（ch17/ch43）")

    tr_box = None
    if state == "trading_range":
        # 区间要把趋势极值本身包进来：回调已 69 根时只看最近 60 根会漏掉区间上沿。
        # BO 失败时从失败那一根算起——再往前是被反转掉的那段趋势，不是区间
        if fail_index is not None:
            win = n - fail_index
        elif ai != 0 and st is not None and st.current is not None:
            win = st.current.bars + 1
        else:
            win = CHANNEL_WINDOW
        win = min(AI_LOOKBACK, max(TR_PB_BARS, win))
        seg = df.tail(win)
        hi, lo = float(seg["high"].max()), float(seg["low"].min())
        if hi - lo < TTR_HEIGHT_MULT * avg_now:
            state = "tight_trading_range"
            state_reason += f"；区间高度不到 {TTR_HEIGHT_MULT:g} 倍平均 K 线（TTR，ch17/ch47）"
        tr_box = {"high": round(hi, 2), "low": round(lo, 2), "bars": win,
                  "from": seg.index[0].isoformat(),
                  "position": (close - lo) / max(hi - lo, 1e-12)}

    # ---------------- 形态提示
    o_or, h_or, l_or, c_or = _oriented(df, orient)
    climax = False
    if regime_bo is not None and trend_bars >= CLIMAX_TREND_BARS and ai != 0:
        rng = h_or - l_or
        seg = slice(regime_bo.start, n)
        biggest = regime_bo.start + int(np.argmax(rng[seg] * (c_or[seg] > o_or[seg])))
        climax = biggest >= n - CLIMAX_RECENT_BARS
    wedge = False
    if st is not None and st.current is not None and len(st.majors) >= 2 and state == "broad_channel":
        p1, p2 = st.majors[-2], st.majors[-1]
        # 三推：p1.extreme、p2.extreme、当前极值；第 2 推后的回调越过第 1 推极值（重叠，ch24）
        wedge = p2.low < p1.extreme and p1.extreme < p2.extreme < st.extreme

    def P(v: float) -> float:   # 定向价 → 真实价
        return orient * v

    # ---------------- MM 目标（ch20）
    mm_leg = mm_bo = None
    if st is not None and ai != 0:   # BO 已失败时，从它推出来的 MM 不再成立
        leg = st.extreme - st.leg_low
        if st.current is not None:
            mm_leg = P(st.current.low + leg)
        mm_bo = P(latest_bo.top + (latest_bo.top - latest_bo.origin))
    mm_tr = None
    if tr_box:
        mm_tr = tr_box["high"] + (tr_box["high"] - tr_box["low"])

    # ---------------- D 计划（只做多）
    hi_last = float(df["high"].iloc[-1])
    sq = _signal_quality(*_oriented(df, 1), n - 1)   # 计划只做多，信号 K 线一律按多头口径评
    common_block = [f"事件模式：{event_reason or '窗口内有未定价事件'}"] if event_mode else []
    for e in pending:
        if e.sign < 0 and close < -e.origin:
            common_block.append(
                f"{idx[e.start].date()} 出现巨大阴线 BO，还在等 FT（ch13/ch15）：不在 BO 途中接刀，"
                f"{FT_WINDOW} 根内 FT 不来再按 BO Mode 处理")
    mtr = None

    if ai > 0 and state == "breakout":
        entry, stop = close, P(regime_bo.origin) - TICK
        risk = entry - stop
        blocked = list(common_block)
        if climax:
            blocked.append("趋势 20 根以上后出现最大阳线，更像买入高潮；高潮后 50/50、多数先走 3–10 根 TR（ch29）")
        plan = _plan(
            setup="强 BO 收盘买入", code="B1",
            trigger=f"强 BO + FT 已成立，按收盘价 {close:.2f} 买小仓（Buy The Close，ch12/ch41）",
            entry=entry, stop=stop, stop_basis="BO 起点下方 1 tick（ch33）",
            t1=entry + risk, t2=mm_bo, p=0.6, t1_basis="1 倍实际风险（强 BO 后约 60%，ch43）",
            cancel_if=["下一根出现大阴线收在低点：BO 可能失败，前提改变就离场（ch33）",
                       "回到 BO 起点下方：BO 失败"],
            avg_range=avg_now, blocked=blocked)
    elif ai > 0 and state in ("tight_channel", "broad_channel"):
        tight = state == "tight_channel"
        leg = st.extreme - st.leg_low
        if st.current is not None:
            triggered = not count["pending"]
            nxt = count["count"] if triggered else count["next"]
            if triggered:
                m = count["marks"][-1]
                trig = float(df["high"].iloc[m["signal"]]) + TICK
                entry = max(trig, close)    # 晚入场按现价算实际风险（ch34，B2）
            else:
                entry = hi_last + TICK
            if tight:
                stop, basis = st.leg_low - TICK, "最后一段强势腿起点下方（拿不准就用更远的，ch43）"
                p, t1, t1b = 0.6, None, "1 倍实际风险（紧密通道 PB 约 60%，ch43）"
            else:
                stop, basis = st.current.low - TICK, "本次回调低点（将成为主要 HL）下方（ch45）"
                r = st.current.retrace
                p = 0.6 if r <= 0.5 else (0.5 if r <= 2 / 3 else 0.4)
                t1, t1b = st.extreme, f"测试前高（回撤 {r:.0%}，{p:.0%}，ch30/ch45/ch12）"
            if t1 is None:
                t1 = entry + (entry - stop)
            blocked = list(common_block)
            if tight and climax:
                blocked.append("趋势 20 根以上后刚出现最大阳线，更像高潮；等 TR 或二次回调（ch29）")
            cs = []
            if sq == "weak" and not triggered:
                cs.append("最后一根是弱信号 K 线（阴线收在低点），Brooks 建议等第二入场（ch08）")
            if wedge:
                cs.append("前面三推构成楔形顶候选：楔形突破至少 75% 朝反方向，买入要等更强信号（ch24）")
            if gap["first_touch"]:
                cs.append(f"均线上方 {gap['prior_run']} 根后首次回踩 20 EMA，A8 买点（ch14）")
            if triggered:
                trigger = (f"H{nxt} 已于 {idx[m['entry']].date()} 触发（{trig:.2f}），尚未创新高；"
                           + (f"现在追入按现价 {entry:.2f} 计算实际风险（ch34）" if close > trig
                              else f"收盘回到触发价下方，仍按触发价 {trig:.2f} 计算"))
            else:
                trigger = (f"下一根高点越过 {hi_last:.2f}（本根信号 K 线质量{SIGNAL_LABELS[sq]}），"
                           f"止损单 {entry:.2f} 买入 = H{nxt}（ch09）")
            plan = _plan(
                setup=f"{'紧密' if tight else '宽幅'}通道 H{nxt}" + ("（已触发）" if triggered else ""),
                code="A5" if tight else ("A2" if nxt == 2 else "A6"),
                trigger=trigger,
                entry=entry, stop=stop, stop_basis=basis,
                t1=t1, t2=mm_leg, p=p, t1_basis=t1b,
                cancel_if=[f"跌破 {P(st.leg_low):.2f}（前一腿起点）：出现更低低点，不再是多头趋势（ch45）",
                           "H2 触发后又出现空头 BO：该 H2 失败，重新计数（ch09）",
                           "回调持续到 20 根以上：按 TR 处理"],
                avg_range=avg_now, blocked=blocked, cautions=cs)
        else:
            frac = 1 / 3 if tight else 0.5
            entry = st.extreme - frac * leg
            stop = st.leg_low - TICK
            t1 = st.extreme if not tight else entry + (entry - stop)
            plan = _plan(
                setup=f"{'紧密' if tight else '宽幅'}通道 · 回调限价买", code="A5" if tight else "A3",
                trigger=(f"刚创新高、尚无回调。等回调到前一腿的 {frac:.0%}（约 {entry:.2f}）挂限价买"
                         f"（{'ch43：紧密通道 33–50% PB' if tight else 'ch45：宽幅通道 50% PB'}）"),
                entry=entry, stop=stop, stop_basis="前一腿起点下方（ch43/ch45）",
                t1=t1, t2=mm_leg or mm_bo, p=0.6,
                t1_basis="测试前高（50% PB 约 60%，ch30）" if not tight else "1 倍实际风险（ch43）",
                cancel_if=["回调演变成 >= 20 根：按 TR 处理", "跌破前一腿起点：不再是多头趋势"],
                avg_range=avg_now, blocked=list(common_block),
                cautions=["不要在前高上方追买：通道外的顺势 BO 75% 在 5 根内失败（ch16/ch45）"])
    elif state == "trading_range":
        hi, lo = tr_box["high"], tr_box["low"]
        third = (hi - lo) / 3
        mid = (hi + lo) / 2
        pos = tr_box["position"]
        if pos <= 1 / 3:
            entry = hi_last + TICK
            stop = lo - TICK
            cs = []
            if sq == "weak":
                cs.append("最后一根是弱信号 K 线，等下一根更强的反转信号或第二入场（ch08）")
            if ai < 0:
                cs.append("Always In 仍偏空：TR 低买只当刮头皮，不当波段（ch13）")
            plan = _plan(
                setup="TR 下 1/3 低买", code="C1",
                trigger=f"价格在区间下 1/3（{lo:.2f}–{lo + third:.2f}），下一根越过 {hi_last:.2f} 买入（BLSHS，ch47）",
                entry=entry, stop=stop, stop_basis="区间低点下方（ch47）",
                t1=mid, t2=hi - TICK, p=0.6, t1_basis="区间中部（上下 1/3 约 60%，ch30/ch47）",
                cancel_if=["出现大阴线跌破区间 + FT：TR 结束，停止逆势加仓",
                           "被止损后最多再入一次，不做第 3 次（ch47）"],
                avg_range=avg_now, blocked=list(common_block), cautions=cs)
        else:
            # 挂在下 1/3 的中位：在下 1/3 上沿买，到区间中部只有 0.5 倍风险（推演）
            entry = lo + third / 2
            plan = _plan(
                setup="TR · 等回到下 1/3", code="C1",
                trigger=(f"价格在区间{'中部' if pos <= 2 / 3 else '上 1/3'}，按 BLSHS 不追买；"
                         f"等回到下 1/3（<= {lo + third:.2f}）出现反转信号再买，参考价取下 1/3 中位"
                         f" {entry:.2f}（ch47）"),
                entry=entry, stop=lo - TICK, stop_basis="区间低点下方（ch47）",
                t1=mid, t2=hi - TICK, p=0.6, t1_basis="区间中部（ch47）",
                cancel_if=["强 BO + FT 突破区间：TR 结束，改按突破处理"],
                avg_range=avg_now,
                blocked=common_block + [f"现价位于区间{'中部（约 50%，不做）' if pos <= 2 / 3 else '上 1/3（只卖不买）'}（ch30）"])
    elif state == "tight_trading_range":
        plan = _no_plan("TTR · 不交易", "C2", "紧密交易区间：多数 K 线重叠、区间太窄，止损单在里面只会买高卖低（ch17/ch47）",
                        "等强 BO + FT 离开区间后再按方向找 setup", avg_now)
    else:
        plan = None

    # AIS：只描述，做多只等 MTR
    if ai < 0 and state in ("breakout", "tight_channel", "broad_channel"):
        mtr = _mtr_long(df, e20, avg, atr, regime_bo.start, bear_tight=state in ("breakout", "tight_channel"))
        if mtr["valid"]:
            cse = mtr["case"]
            entry = hi_last + TICK
            stop = cse["test"] - TICK
            risk = entry - stop
            plan = _plan(
                setup=f"{cse['kind']} MTR 买入", code="D1",
                trigger=f"MTR 条件齐备，下一根越过 {hi_last:.2f} 买入；仍处 AIS，属早入场（ch22/ch39）",
                entry=entry, stop=stop, stop_basis=f"{cse['kind']} 测试低点下方（ch39）",
                t1=entry + 2 * risk, t2=cse["test"] + (cse["R"] - cse["base"]), p=0.4,
                t1_basis="2 倍实际风险（MTR 早入场约 40%，ch22/ch39）",
                cancel_if=["跌破测试低点：MTR 失败，原趋势通常再走一个 MM（ch22）",
                           "反弹只有 2–4 根普通 K 线：买压不足，只是次要反转（ch21）"],
                avg_range=avg_now, blocked=list(common_block))
        else:
            plan = _no_plan(
                "AIS · 不做多", "D1",
                "Always In Short，只做多的口径下没有顺势 setup；MTR 条件未齐（强趋势里 80% 的反转尝试失败，ch15）",
                "等 MTR：" + ("；".join(it["text"] for it in mtr["items"] if it["ok"] is False) or "等信号"),
                avg_now)
    if plan is None:
        plan = _no_plan("无 setup", "—", "当前背景没有对应的做多 setup", "等待", avg_now)

    # ---------------- 判断段落
    dir_word = {1: "多头", -1: "空头"}
    pbw = "回调" if orient > 0 else "反弹"
    bg: list[str] = []
    if regime_bo is not None:
        bo_txt = (f"最近一次清晰的{dir_word[regime_bo.sign]} BO 始于 {idx[regime_bo.start].date()}"
                  f"（{regime_bo.end - regime_bo.start + 1} 根强趋势 K 线，"
                  f"起点 {P(regime_bo.origin):.2f}），已持续 {trend_bars} 根")
        bg.append(bo_txt)
    if ai_note:
        bg.append(ai_note)
    bg.append(f"{AI_LABELS[ai]}；市场周期处于「{STATE_LABELS[state]}」：{state_reason}")
    side = "上方" if close >= e20[-1] else "下方"
    bg.append(f"收盘 {close:.2f} 在 20 EMA（{e20[-1]:.2f}）{side}"
              + (f"，连续 {gap['gap_bars']} 根 K 线不碰均线" if gap["gap_bars"] >= 5 else ""))
    if ai == 0:
        pass  # Always In 不明时 H/L 计数没有意义（ch09：H4 之后别再数）
    elif count and count["marks"]:
        lab = "H" if orient > 0 else "L"
        bg.append(f"当前{pbw} {st.current.bars} 根、回撤前一腿 {st.current.retrace:.0%}，"
                  + (f"已数过 {lab}{MAX_COUNT}，停止计数（ch09）" if count["count"] >= MAX_COUNT else
                     f"已出现 {lab}{count['count']}" + (f"，下一个是 {lab}{count['next']}" if count["pending"] else "")))
    elif st is not None and st.current is not None and ai != 0:
        bg.append(f"当前{pbw} {st.current.bars} 根、回撤前一腿 {st.current.retrace:.0%}，尚未出现恢复尝试")
    if pending:
        pe = pending[-1]
        bg.append(f"最后一根是潜在{dir_word[pe.sign]} BO，需要下一根 FT 确认（ch13）")

    bull: list[str] = []
    bear: list[str] = []
    if ai != 0:
        (bull if ai > 0 else bear).append(f"{AI_LABELS[ai]}，顺势方占优（ch13）")
    if close > e20[-1]:
        bull.append("收在 20 EMA 上方")
    else:
        bear.append("收在 20 EMA 下方")
    if gap["gap_bars"] >= GAP_BARS:
        (bull if orient > 0 else bear).append(f"连续 {gap['gap_bars']} 根不碰均线，趋势强（ch14）")
    if gap["first_touch"]:
        (bull if orient > 0 else bear).append("长期远离均线后首次回踩，顺势方的买点（A8）")
    if state in ("breakout", "tight_channel"):
        (bull if orient > 0 else bear).append(f"{STATE_LABELS[state]}：逆势单即使 70% 胜率也亏钱（ch17）")
    if state == "broad_channel":
        (bear if orient > 0 else bull).append("宽幅通道本质是倾斜的 TR，75% 最终演变成 TR（ch16）")
    if count and count["pending"] and count["next"] == 2:
        (bull if orient > 0 else bear).append("回调第二推结束在即（H2/L2），逆势方会在这里离场（ch09）")
    if climax:
        (bear if orient > 0 else bull).append("趋势 20 根以上后刚出现最大趋势 K 线，更像高潮/衰竭（ch29/ch42）")
    if wedge:
        (bear if orient > 0 else bull).append("三推重叠，楔形顶候选（ch24）")
    if ai != 0 and st.current is not None and st.current.retrace > 2 / 3:
        (bear if orient > 0 else bull).append(f"{pbw}已达前一腿 {st.current.retrace:.0%}，深{pbw}（ch12/ch30）")
    if ai != 0 and st.broken:
        (bear if orient > 0 else bull).append(
            "回调收盘跌破前一腿起点，多头结构转弱" if orient > 0 else "反弹收盘越过前一腿起点，空头结构转弱")
    if tr_box:
        bull.append(f"区间下沿 {tr_box['low']:.2f} 附近有买盘（TR 内 80% 的 BO 失败，ch15）")
        bear.append(f"区间上沿 {tr_box['high']:.2f} 附近有卖盘")
    if mtr is not None:
        ok = sum(1 for it in mtr["items"] if it["ok"])
        bull.append(f"MTR 清单满足 {ok}/4 条" + ("，条件齐备" if mtr["valid"] else ""))

    if state in ("trading_range", "tight_trading_range"):
        lean_side, lean_p = "neutral", 0.5
        lean_txt = "多空约 50/50：TR 里两边都有合理理由，低买高卖、不追突破（ch18/ch47）"
    elif ai != 0:
        lean_side = "bull" if ai > 0 else "bear"
        lean_p = 0.6
        if state == "broad_channel" and st.current is not None and st.current.retrace > 0.5:
            lean_p = 0.5
        if climax or wedge:
            lean_p = 0.5
        if mtr is not None and mtr["valid"]:
            lean_txt = f"空方仍约 60%，但 MTR 条件齐备：多方反转约 40%，需 >= 2 倍风险回报（ch22/ch39）"
        else:
            lean_txt = (f"{dir_word[ai]}约 {lean_p:.0%}："
                        + ({"breakout": "强 BO 后顺势追入约 60%（ch43）",
                            "tight_channel": "紧密通道顺势 PB 约 60%（ch43）",
                            "broad_channel": "宽幅通道顺势约 60/40（ch45），深回调降到约 50%（ch12）"}[state])
                        + ("；高潮/楔形迹象让概率回到约 50%" if (climax or wedge) else ""))
    else:
        lean_side, lean_p, lean_txt = "neutral", 0.5, "多空约 50/50"

    wait: list[str] = []
    if ai > 0 and st is not None:
        wait.append(f"收盘跌破 {P(st.leg_low):.2f}（前一腿起点）→ 出现更低低点，多头趋势结束")
        if state in ("breakout", "tight_channel"):
            wait.append("回调持续 >= 20 根或数到 H4 → 转为 TR")
    if ai < 0 and st is not None:
        wait.append(f"收盘站上 {P(st.leg_low):.2f}（前一腿起点）→ 出现更高高点，空头趋势结束")
    if tr_box:
        wait.append(f"强 BO + FT 收出区间（> {tr_box['high']:.2f} 或 < {tr_box['low']:.2f}）→ TR 结束")
    if ai == 0 and latest_bo is None:
        wait.append("等一次清晰的强 BO 确立 Always In 方向")

    summary = f"{AI_SHORT[ai]} · {STATE_LABELS[state]}"
    if count and count["pending"] and ai != 0:
        summary += f" · 等 {'H' if orient > 0 else 'L'}{count['next']}"

    # ---------------- 图上标记
    def ts(i: int) -> str:
        return idx[i].isoformat()

    bo_marks = [
        {"t": ts(e.start), "t_end": ts(e.end), "dir": "bull" if e.sign > 0 else "bear",
         "origin": round(e.sign * e.origin, 2), "top": round(e.sign * e.top, 2),
         "confirmed": bool(e.confirmed),
         "status": "confirmed" if e.confirmed else ("pending" if e in pending else "no_ft")}
        for e in sorted(events, key=lambda e: e.start)[-8:]
    ]
    count_marks = []
    if count and ai != 0:   # Always In 不明时计数没有意义；H4 之后也不再数（ch09）
        lab = "H" if orient > 0 else "L"
        for m in count["marks"][:MAX_COUNT]:
            i = m["signal"]
            count_marks.append({"t": ts(i), "label": f"{lab}{m['n']}",
                                "price": round(float(df["low"].iloc[i] if orient > 0 else df["high"].iloc[i]), 2),
                                "side": "below" if orient > 0 else "above"})

    return {
        "available": True,
        "always_in": {"direction": ai, "label": AI_LABELS[ai], "short": AI_SHORT[ai], "note": ai_note},
        "state": state,
        "state_label": STATE_LABELS[state],
        "state_reason": state_reason,
        "channel_stats": tight_stats,
        "trend_bars": trend_bars,
        "ema20": _round(e20[-1]),
        "avg_bar_range": _round(avg_now),
        "bar_count": ({"count": count["count"], "next": count["next"], "pending": count["pending"],
                       "prefix": "H" if orient > 0 else "L", "signal_quality": count["signal_quality"]}
                      if count and ai != 0 else None),
        "pullback": ({"bars": st.current.bars, "retrace": round(st.current.retrace, 3),
                      "low": _round(P(st.current.low)), "extreme": _round(P(st.extreme)),
                      "leg_start": _round(P(st.leg_low))}
                     if st is not None and st.current is not None else None),
        "range": tr_box,
        "climax": bool(climax),
        "wedge": bool(wedge),
        "gap": gap,
        "measured_moves": {"leg": _round(mm_leg), "breakout": _round(mm_bo), "range": _round(mm_tr)},
        "mtr": mtr,
        "plan": plan,
        "judgment": {
            "summary": summary,
            "background": bg,
            "bull": bull,
            "bear": bear,
            "lean": {"side": lean_side, "p": lean_p, "text": lean_txt},
            "wait": wait,
        },
        "marks": {"breakouts": bo_marks, "counts": count_marks},
        "disclaimer": (
            "Brooks 层把课程规则直接代码化：阈值取课程原文数字，没有按本仓库数据调参；"
            "概率是课程给的经验值，不是本仓库回测结果。课程例子以日内图为主，用在个股日线上属于推演。"
            "D 只做多，不受大盘状态开关约束，也不改变看板分组。"
        ),
    }
