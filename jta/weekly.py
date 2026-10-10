"""周报计算层：按板块看一周的趋势。

日报回答"明天在哪个价位做什么"，周报回答"这一周趋势有没有变、谁在领涨、谁在掉队"。
两者的时间框架不同（周线 + 1/4/13 周窗口 vs 日线 + 4H），所以这里不读看板快照，
直接把日线重采样成周线，任意一周都能回放。

关注列表里的票本来就是各板块的强势票，用它们合成的"板块走势"天然偏强。
因此相对强弱分两层：
  板块 ETF / QQQ —— 板块本身强不强；
  个股 / 板块 ETF —— 它还是不是领头的那只。
强势股最该盯的是第二层：价格还在涨、却跑不赢板块，往往是领涨地位松动的第一个信号。

趋势标签是描述性的，阈值在看结果之前写定，**没有回测过预测力**，不按结果调参。
叙事层（LLM）只读这里的 JSON，不自行计算任何数字。
"""

from __future__ import annotations

import html as _html
import json
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Sequence
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from .analyze import swing_sequence
from .dashboard import DISPLAY_NAME, SECTOR_MAP
from .data.provider import DataUnavailable, utcnow
from .indicators.atr import atr as atr_series
from .indicators.ema import WARMUP_MULTIPLE, ema
from .indicators.price_action import VOLUME_SPIKE_MULT, volume_baseline
from .indicators.swing import DEFAULT_K, detect_swings
from .regime import benchmark_regime

ET = ZoneInfo("America/New_York")

#: 大盘基准：所有个股与板块 ETF 都对它算第二条 RS 线
MARKET = "QQQ"

#: 指数一节的成员及各自的参照（SPY 是最宽的基准，没有参照）
INDEX_MEMBERS: dict[str, str | None] = {"QQQ": "SPY", "SPY": None, "SOXX": "QQQ"}

#: 板块参照 ETF。光模块没有纯主题 ETF，用 SOXX；单只成类的并进"其他"，只对比 QQQ
SECTOR_REFERENCE: dict[str, str] = {
    "存储": "DRAM", "光模块": "SOXX", "半导体设计": "SOXX", "云计算": "SKYY",
}

#: 输出顺序；SECTOR_MAP 里未列出的板块并进"其他"
SECTION_ORDER = ["指数", "存储", "光模块", "半导体设计", "云计算", "其他"]

#: 看板里归在"指数/ETF"的标的不作为板块成员（DRAM 是存储的参照）
_ETF_SECTOR = "指数/ETF"

#: 多周窗口
LOOKBACK_WEEKS = (1, 4, 13)

#: 周线 EMA 与 ATR
EMA_FAST, EMA_SLOW = 10, 20
ATR_WEEKS = 14

#: RS 线创新高的回看周数、斜率的回看周数
RS_HIGH_WEEKS = 13
RS_SLOPE_WEEKS = 4

#: 52 周高点
HIGH_52W = 52

#: 热力图展示最近几周的逐周涨跌
HEATMAP_WEEKS = 8

#: 重点标的（叙事层只对这些补搜个股消息）：强弱方向翻转、周幅 >= 1 倍周 ATR、
#: 预警 >= 2 条、RS 排名前后各 3 名；按理由条数排序，最多 8 只
FOCUS_MOVE_ATR = 1.0
FOCUS_MIN_WARNINGS = 2
FOCUS_RANK_EDGE = 3
FOCUS_MAX = 8
#: 标签的强弱方向。休整 / 整理 / 过热这类中性状态之间的来回不算翻转
LABEL_TONE = {"加速": 1, "延续": 1, "转弱": -1, "下行": -1, "冲高回落": -1}

# ---- 下周预测：叙事层（LLM）给方向，计算层按下一周的实际涨跌记分 ----
#: 每个板块用哪个标的代表；光模块不用 SOXX（已代表半导体设计），用两只成员等权合成；
#: "其他"各只不同质，不做预测
PREDICTION_TARGETS: dict[str, tuple[str, ...]] = {
    "指数": ("QQQ",), "存储": ("DRAM",), "光模块": ("LITE", "COHR"),
    "半导体设计": ("SOXX",), "云计算": ("SKYY",),
}
#: 周涨跌在 ±0.3 倍上周周 ATR 以内算震荡（看结果之前写定）
PREDICTION_FLAT_ATR = 0.3
DIRECTION_LABELS = {"up": "看涨", "down": "看跌", "flat": "震荡"}
CONFIDENCE_LABELS = {"low": "低", "mid": "中", "high": "高"}

# ---- 趋势标签阈值：看结果之前写定，不调参 ----
#: 加速：周收盘位于当周振幅上 1/4，且周涨幅 >= 1.5 倍上周的周 ATR，且对 QQQ 的 RS 创 13 周新高
ACCEL_CLOSE_POS = 0.75
ACCEL_MOVE_ATR = 1.5
#: 休整：从 4 周最高点回撤不到 1 倍周 ATR
PAUSE_MAX_PULLBACK_ATR = 1.0
#: 过热：收盘高于 10 周 EMA 超过 3 倍周 ATR。预设值，不是 Brooks 课程原文数字
OVERHEAT_EMA_ATR = 3.0
#: 冲高回落：周高点高于上周，却收在当周振幅下 1/4，且收盘低于上周收盘
REVERSAL_CLOSE_POS = 0.25

#: 健康度预警"跑输参照"：RS 线 4 周变化低于 -3% 才报，小幅落后不算
UNDERPERFORM_RS_PCT = -3.0

#: 周内放量下跌日：下跌且成交量 >= 前 20 日均量的 1.5 倍（与 price_action 的放量口径相同）
DISTRIBUTION_VOLUME_MULT = VOLUME_SPIKE_MULT
#: 一周内出现几个放量下跌日就报预警
DISTRIBUTION_WARN_DAYS = 2

#: 标签优先级：结构已坏的排在最前，"过热"压过"加速"
LABEL_ORDER = ["转弱", "下行", "冲高回落", "过热", "加速", "延续", "休整", "整理", "历史不足"]

WEEKDAY_CN = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]


# --------------------------------------------------------------------------- 周的界定


def parse_week(spec: str) -> date:
    """'2026-W41' → 该 ISO 周的周一。"""
    y, _, w = spec.upper().partition("-W")
    return date.fromisocalendar(int(y), int(w), 1)


def week_id(monday: date) -> str:
    y, w, _ = monday.isocalendar()
    return f"{y}-W{w:02d}"


def week_close_time(monday: date) -> datetime:
    """该周最后一个可能的收盘时刻（周五 16:00 ET）。"""
    fri = monday + timedelta(days=4)
    return datetime(fri.year, fri.month, fri.day, 16, 0, tzinfo=ET)


def default_week(now: datetime | None = None) -> date:
    """最近一个已经走完的交易周：本周五收盘之前取上一周。"""
    now = (now or utcnow()).astimezone(ET)
    monday = now.date() - timedelta(days=now.weekday())
    if now < week_close_time(monday):
        monday -= timedelta(days=7)
    return monday


# --------------------------------------------------------------------------- 周线


def to_weekly(daily: pd.DataFrame) -> pd.DataFrame:
    """日线 → 周线（周一至周五，节假日按实际交易日聚合）。

    索引是该周最后一个交易日；`monday` 列是 ISO 周一，用来跨标的对齐。
    """
    if daily.empty:
        return daily.assign(sessions=[], monday=[])
    naive = daily.index.tz_localize(None).normalize()
    monday = naive - pd.to_timedelta(naive.weekday, unit="D")
    g = daily.groupby(monday.values)
    w = pd.DataFrame({
        "open": g["open"].first(),
        "high": g["high"].max(),
        "low": g["low"].min(),
        "close": g["close"].last(),
        "volume": g["volume"].sum(),
        "sessions": g["close"].size(),
    })
    w["monday"] = [pd.Timestamp(m).date() for m in w.index]
    w.index = pd.DatetimeIndex(daily.index.to_series().groupby(monday.values).last().values)
    return w


def _slice_through(daily: pd.DataFrame, monday: date) -> pd.DataFrame:
    """只保留到该周周五（含）为止的日线。"""
    fri = monday + timedelta(days=4)
    return daily[daily.index.tz_localize(None).normalize() <= pd.Timestamp(fri)]


def _round(x: Any, nd: int = 4) -> float | None:
    if x is None:
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return round(v, nd) if np.isfinite(v) else None


def _pct(a: float, b: float) -> float | None:
    return _round((a / b - 1) * 100, 2) if b else None


# --------------------------------------------------------------------------- 相对强弱


def rs_metrics(sym_w: pd.DataFrame, ref_w: pd.DataFrame | None) -> dict[str, Any] | None:
    """RS 线 = 个股周收盘 / 参照周收盘，按 ISO 周对齐。只陈述方向与是否创新高。"""
    if ref_w is None or ref_w.empty or sym_w.empty:
        return None
    a = sym_w.set_index("monday")["close"]
    b = ref_w.set_index("monday")["close"]
    rs = (a / b).dropna()
    if rs.empty or rs.index[-1] != sym_w["monday"].iloc[-1]:
        return None
    last = float(rs.iloc[-1])

    def chg(n: int) -> float | None:
        return _pct(last, float(rs.iloc[-1 - n])) if len(rs) > n else None

    window = rs.iloc[-RS_HIGH_WEEKS:]
    start = float(rs.iloc[max(0, len(rs) - 13)])
    return {
        "weeks": len(rs),
        "chg_1w_pct": chg(1),
        "chg_4w_pct": chg(RS_SLOPE_WEEKS),
        "chg_13w_pct": chg(13),
        # 不足 13 周时按已有周数判定，并在 weeks 里写明
        "at_13w_high": bool(last >= float(window.max()) * (1 - 1e-9)),
        # 最近 13 周的 RS 线，以窗口第一周为 1
        "series": [_round(v / start, 4) for v in rs.iloc[-13:]],
        "series_weeks": [week_id(m) for m in rs.index[-13:]],
    }


# --------------------------------------------------------------------------- 单标的


def _intraweek(daily: pd.DataFrame, monday: date, week_atr: float | None) -> dict[str, Any]:
    """周内路径：收盘价看不出一周是怎么走的。"""
    naive = daily.index.tz_localize(None).normalize()
    in_week = (naive >= pd.Timestamp(monday)) & (naive <= pd.Timestamp(monday + timedelta(days=4)))
    base = volume_baseline(daily)
    prev_close = daily["close"].shift(1)
    d = daily[in_week]
    if d.empty:
        return {}
    pc = prev_close[in_week]
    vol_ratio = pd.Series(base, index=daily.index)[in_week]
    vol_ratio = d["volume"] / vol_ratio.replace(0, np.nan)
    chg = (d["close"] / pc - 1) * 100

    up, down = d[chg > 0], d[chg < 0]
    dist = d[(chg < 0) & (vol_ratio >= DISTRIBUTION_VOLUME_MULT)]
    peak = d["high"].cummax()
    dd = float((peak - d["low"]).max())

    sessions = [{
        "date": ts.strftime("%Y-%m-%d"),
        "weekday": WEEKDAY_CN[ts.weekday()],
        "close": _round(row["close"]),
        "chg_pct": _round(chg.loc[ts], 2),
        "volume_ratio": _round(vol_ratio.loc[ts], 2),
    } for ts, row in d.iterrows()]
    return {
        "sessions": sessions,
        "high_day": WEEKDAY_CN[d["high"].idxmax().weekday()],
        "low_day": WEEKDAY_CN[d["low"].idxmin().weekday()],
        "max_drawdown_atr": _round(dd / week_atr, 2) if week_atr else None,
        "up_down_volume_ratio": (
            _round(up["volume"].mean() / down["volume"].mean(), 2)
            if len(up) and len(down) else None
        ),
        "distribution_days": [ts.strftime("%Y-%m-%d") for ts in dist.index],
        "biggest_day": (
            {"date": chg.abs().idxmax().strftime("%Y-%m-%d"),
             "chg_pct": _round(chg.loc[chg.abs().idxmax()], 2)}
            if chg.notna().any() else None
        ),
    }


def _week_atr(w: pd.DataFrame) -> tuple[pd.Series, int]:
    """周 ATR。不足 14 周时退到已有周数（至少 4 周），并返回实际使用的周期。"""
    period = min(ATR_WEEKS, len(w) - 1)
    if period < 4:
        return pd.Series(np.nan, index=w.index), period
    return atr_series(w, period), period


def classify(m: dict[str, Any]) -> tuple[str, list[str]]:
    """按 LABEL_ORDER 返回第一个命中的标签，以及所有命中条件的说明。"""
    hits: dict[str, str] = {}
    c, ema10, ema20 = m["close"], m["ema10"], m["ema20"]
    prev_low, prev_high = m["prev_low"], m["prev_high"]
    atr_ = m["atr_prev"]

    if prev_low is not None and c < prev_low:
        hits["转弱"] = f"收盘 {c:g} 跌破上周低点 {prev_low:g}"
    if ema10 is not None and c < ema10:
        if ema20 is not None and ema10 < ema20:
            hits["下行"] = "收盘在 10 周 EMA 下方，且 10 周 EMA 低于 20 周 EMA"
        else:
            hits.setdefault("转弱", f"收盘 {c:g} 落到 10 周 EMA {ema10:g} 下方")
    if m["dist_ema10_atr"] is not None and m["dist_ema10_atr"] >= OVERHEAT_EMA_ATR:
        hits["过热"] = f"收盘高于 10 周 EMA {m['dist_ema10_atr']:g} 倍周 ATR（阈值 {OVERHEAT_EMA_ATR:g}）"
    pc = m.get("prev_close")
    if (prev_high is not None and pc is not None and m["high"] > prev_high
            and m["close_pos"] is not None and m["close_pos"] < REVERSAL_CLOSE_POS and c < pc):
        hits["冲高回落"] = (f"周高点 {m['high']:g} 高于上周 {prev_high:g}，却收在当周振幅 "
                         f"{m['close_pos']:.0%} 处，低于上周收盘 {pc:g}")
    rs_high = (m.get("rs_market") or {}).get("at_13w_high")
    if (m["close_pos"] is not None and m["close_pos"] >= ACCEL_CLOSE_POS
            and m["move_atr"] is not None and m["move_atr"] >= ACCEL_MOVE_ATR and rs_high):
        hits["加速"] = (f"收在当周振幅 {m['close_pos']:.0%} 处，周涨幅 {m['move_atr']:g} 倍周 ATR，"
                      "对 QQQ 的 RS 创 13 周新高")
    above = ema10 is not None and c > ema10 and (ema20 is None or ema10 > ema20)
    if above and prev_high is not None and m["high"] > prev_high:
        hits["延续"] = "收在 10 周 EMA 上方，周高点高于上周"
    if (above and prev_high is not None and m["high"] <= prev_high
            and m["low"] >= prev_low and m["pullback_4w_atr"] is not None
            and m["pullback_4w_atr"] < PAUSE_MAX_PULLBACK_ATR):
        hits["休整"] = (f"没创新高但守住上周低点，离 4 周高点 {m['pullback_4w_atr']:g} 倍周 ATR")
    if not hits:
        if ema10 is None or atr_ is None:
            hits["历史不足"] = f"只有 {m['weeks']} 周数据，EMA 或 ATR 无法计算"
        else:
            hits["整理"] = "未满足以上任何一条"
    label = next(l for l in LABEL_ORDER if l in hits)
    return label, [f"{k}：{v}" for k, v in sorted(hits.items(), key=lambda kv: LABEL_ORDER.index(kv[0]))]


def symbol_week(
    daily: pd.DataFrame,
    monday: date,
    *,
    ref_daily: pd.DataFrame | None = None,
    market_daily: pd.DataFrame | None = None,
) -> dict[str, Any] | None:
    """一个标的在给定周的周线指标。数据里没有该周时返回 None。"""
    d = _slice_through(daily, monday)
    w = to_weekly(d)
    if w.empty or w["monday"].iloc[-1] != monday:
        return None
    atr_s, atr_period = _week_atr(w)
    cl = w["close"]
    e10 = ema(cl, EMA_FAST) if len(w) >= EMA_FAST else None
    e20 = ema(cl, EMA_SLOW) if len(w) >= EMA_SLOW else None
    last, n = w.iloc[-1], len(w)
    prev = w.iloc[-2] if n >= 2 else None
    atr_prev = _round(atr_s.iloc[-2]) if n >= 2 else None
    atr_prev = atr_prev if atr_prev else None

    rng = float(last["high"] - last["low"])
    close_pos = _round((last["close"] - last["low"]) / rng, 3) if rng > 0 else None
    body = _round(abs(last["close"] - last["open"]) / rng, 3) if rng > 0 else None
    bar_type = None
    if prev is not None:
        if last["high"] <= prev["high"] and last["low"] >= prev["low"]:
            bar_type = "inside"
        elif last["high"] > prev["high"] and last["low"] < prev["low"]:
            bar_type = "outside"

    def ret(k: int) -> float | None:
        return _pct(float(last["close"]), float(cl.iloc[-1 - k])) if n > k else None

    ema10 = _round(e10.iloc[-1]) if e10 is not None else None
    ema20 = _round(e20.iloc[-1]) if e20 is not None else None
    hi4 = float(w["high"].iloc[-4:].max())
    hi13 = float(w["high"].iloc[-13:].max())
    hi52 = float(w["high"].iloc[-HIGH_52W:].max())

    swings = detect_swings(w, k=DEFAULT_K["1wk"]) if n >= 2 * DEFAULT_K["1wk"] + 1 else []
    seq = swing_sequence(swings, atr_prev or rng or 1.0, n=4)
    # 最近一个已确认的摆动高点若是 LH，只有之后还没被突破才算结构转弱
    highs = [x for x in swings if x.kind == "high"]
    lh_open = None
    if highs and seq and any(x["kind"] == "high" for x in seq):
        last_h = next(x for x in reversed(seq) if x["kind"] == "high")
        after = w["high"][w.index > highs[-1].ts]
        if last_h["label"] == "LH" and (after.empty or float(after.max()) <= highs[-1].price):
            lh_open = {"ts": last_h["ts"][:10], "price": last_h["price"]}

    ref_w = to_weekly(_slice_through(ref_daily, monday)) if ref_daily is not None else None
    mkt_w = to_weekly(_slice_through(market_daily, monday)) if market_daily is not None else None

    m: dict[str, Any] = {
        "week": week_id(monday),
        "week_end": w.index[-1].strftime("%Y-%m-%d"),
        "sessions": int(last["sessions"]),
        "weeks": n,
        "open": _round(last["open"]), "high": _round(last["high"]),
        "low": _round(last["low"]), "close": _round(last["close"]),
        "prev_high": _round(prev["high"]) if prev is not None else None,
        "prev_low": _round(prev["low"]) if prev is not None else None,
        "prev_close": _round(prev["close"]) if prev is not None else None,
        "ret_pct": {f"{k}w": ret(k) for k in LOOKBACK_WEEKS},
        "atr_prev": atr_prev,
        "atr_weeks": atr_period,
        "move_atr": (_round((last["close"] - prev["close"]) / atr_prev, 2)
                     if prev is not None and atr_prev else None),
        "close_pos": close_pos,
        "body_ratio": body,
        "bar_type": bar_type,
        "higher_high": bool(prev is not None and last["high"] > prev["high"]),
        "higher_low": bool(prev is not None and last["low"] > prev["low"]),
        "new_13w_high": bool(last["high"] >= hi13),
        "from_52w_high_pct": _pct(float(last["close"]), hi52),
        "pullback_4w_atr": _round((hi4 - last["close"]) / atr_prev, 2) if atr_prev else None,
        "ema10": ema10, "ema20": ema20,
        "ema_reliable": {
            "ema10": n >= EMA_FAST * WARMUP_MULTIPLE,
            "ema20": n >= EMA_SLOW * WARMUP_MULTIPLE,
        },
        "dist_ema10_atr": (_round((last["close"] - ema10) / atr_prev, 2)
                           if ema10 is not None and atr_prev else None),
        "swing_sequence": seq,
        "lower_high_unbroken": lh_open,
        "rs_reference": rs_metrics(w, ref_w),
        "rs_market": rs_metrics(w, mkt_w),
        "intraweek": _intraweek(d, monday, atr_prev),
        "spark": [_round(v) for v in cl.iloc[-13:]],
        "weekly_returns": [
            {"week": week_id(mo), "ret_pct": _pct(float(cl.iloc[i]), float(cl.iloc[i - 1]))}
            for i, mo in zip(range(max(1, n - HEATMAP_WEEKS), n),
                             w["monday"].iloc[max(1, n - HEATMAP_WEEKS):])
        ],
    }
    m["label"], m["label_reasons"] = classify(m)
    m["warnings"] = health_warnings(m)
    return m


def health_warnings(m: dict[str, Any]) -> list[str]:
    """强势股健康度预警。只陈述事实，不给操作建议。"""
    out: list[str] = []
    ref = m.get("rs_reference") or {}
    if m["new_13w_high"] and ref and not ref.get("at_13w_high"):
        out.append("价格创 13 周新高，但相对参照的 RS 没有创新高（背离）")
    if ((m["ret_pct"].get("4w") or 0) > 0
            and (ref.get("chg_4w_pct") or 0) < UNDERPERFORM_RS_PCT):
        out.append(f"4 周上涨 {m['ret_pct']['4w']:g}%，但相对参照的 RS 4 周 {ref['chg_4w_pct']:g}%（跑输）")
    dist = (m.get("intraweek") or {}).get("distribution_days") or []
    if len(dist) >= DISTRIBUTION_WARN_DAYS:
        out.append(f"本周 {len(dist)} 个放量下跌日（{', '.join(dist)}）")
    lh = m.get("lower_high_unbroken")
    if lh:
        out.append(f"周线更低高点 LH {lh['price']:g}（{lh['ts']} 那周）至今未被突破")
    if m["label"] == "过热":
        out.append("离 10 周 EMA 过远，按 Brooks 的说法更像高潮，高潮后常见横盘或回调")
    if m["prev_low"] is not None and m["close"] < m["prev_low"]:
        out.append("周收盘跌破上周低点")
    return out


# --------------------------------------------------------------------------- 汇总


@dataclass
class Universe:
    sections: dict[str, dict[str, Any]]   # 名称 → {reference, members}
    symbols: list[str]


def build_universe(sector_map: dict[str, str] | None = None) -> Universe:
    sector_map = sector_map or SECTOR_MAP
    sections: dict[str, dict[str, Any]] = {
        "指数": {"reference": None, "members": list(INDEX_MEMBERS)},
    }
    for sym, sec in sector_map.items():
        if sec == _ETF_SECTOR:
            continue
        name = sec if sec in SECTOR_REFERENCE else "其他"
        sections.setdefault(name, {"reference": SECTOR_REFERENCE.get(name), "members": []})
        sections[name]["members"].append(sym)
    ordered = {k: sections[k] for k in SECTION_ORDER if k in sections}
    syms = {MARKET, *INDEX_MEMBERS, *(v for v in INDEX_MEMBERS.values() if v)}
    for sec in ordered.values():
        syms.update(sec["members"])
        if sec["reference"]:
            syms.add(sec["reference"])
    return Universe(sections=ordered, symbols=sorted(syms))


def _section(
    name: str, spec: dict[str, Any], monday: date, data: dict[str, pd.DataFrame],
) -> dict[str, Any]:
    ref = spec["reference"]
    mkt = data.get(MARKET)
    prev_monday = monday - timedelta(days=7)

    def calc(sym: str, wk: date) -> dict[str, Any] | None:
        if sym not in data:
            return None
        r = INDEX_MEMBERS.get(sym) if name == "指数" else (ref or MARKET)
        return symbol_week(data[sym], wk, ref_daily=data.get(r) if r else None,
                           market_daily=None if sym == MARKET else mkt)

    members: list[dict[str, Any]] = []
    missing: list[str] = []
    for sym in spec["members"]:
        cur = calc(sym, monday)
        if cur is None:
            missing.append(sym)
            continue
        before = calc(sym, prev_monday)
        members.append({
            "symbol": sym,
            "display": DISPLAY_NAME.get(sym, sym),
            "reference": INDEX_MEMBERS.get(sym) if name == "指数" else (ref or MARKET),
            **cur,
            "prev_label": before["label"] if before else None,
            "prev_ret_4w_pct": before["ret_pct"]["4w"] if before else None,
        })

    out: dict[str, Any] = {"name": name, "reference": ref, "members": members, "missing": missing}
    if ref and ref in data:
        cur = symbol_week(data[ref], monday, ref_daily=mkt, market_daily=mkt)
        before = symbol_week(data[ref], prev_monday, ref_daily=mkt, market_daily=mkt)
        if cur:
            out["reference_metrics"] = {**cur, "prev_label": before["label"] if before else None}
    if name != "指数" and members:
        out["aggregate"] = aggregate(members, out.get("reference_metrics"))
    return out


def aggregate(members: list[dict[str, Any]], ref: dict[str, Any] | None) -> dict[str, Any]:
    """板块内的广度、分化与领涨。成员是精选强势票，合成值天然偏强，只作内部比较。"""
    def r(m: dict[str, Any], k: str) -> float | None:
        return m["ret_pct"].get(k)

    with_4w = [m for m in members if r(m, "4w") is not None]
    by_4w = sorted(with_4w, key=lambda m: r(m, "4w"), reverse=True)
    prev_ranked = sorted([m for m in members if m["prev_ret_4w_pct"] is not None],
                         key=lambda m: m["prev_ret_4w_pct"], reverse=True)
    above = [m for m in members if m["ema10"] is not None and m["close"] > m["ema10"]]
    ref_1w = r(ref, "1w") if ref else None
    ref_4w = r(ref, "4w") if ref else None

    def mean(k: str) -> float | None:
        v = [r(m, k) for m in members if r(m, k) is not None]
        return _round(np.mean(v), 2) if v else None

    return {
        "n": len(members),
        "above_ema10": len(above),
        "new_13w_high": sum(1 for m in members if m["new_13w_high"]),
        "labels": {l: [m["symbol"] for m in members if m["label"] == l]
                   for l in LABEL_ORDER if any(m["label"] == l for m in members)},
        "composite_ret_pct": {"1w": mean("1w"), "4w": mean("4w"), "13w": mean("13w")},
        "beat_reference": {
            "1w": sum(1 for m in members if ref_1w is not None and (r(m, "1w") or -1e9) > ref_1w),
            "4w": sum(1 for m in members if ref_4w is not None and (r(m, "4w") or -1e9) > ref_4w),
        } if ref else None,
        "dispersion_4w_pct": _round(r(by_4w[0], "4w") - r(by_4w[-1], "4w"), 2) if len(by_4w) >= 2 else None,
        "leader": by_4w[0]["symbol"] if by_4w else None,
        "laggard": by_4w[-1]["symbol"] if len(by_4w) >= 2 else None,
        "prev_leader": prev_ranked[0]["symbol"] if prev_ranked else None,
        "leader_changed": bool(by_4w and prev_ranked and by_4w[0]["symbol"] != prev_ranked[0]["symbol"]),
    }


def rotation(sections: list[dict[str, Any]]) -> dict[str, Any]:
    """全体个股按 4 周对 QQQ 的 RS 变化排名，并与上周名次比较。"""
    rows = []
    for sec in sections:
        if sec["name"] == "指数":
            continue
        for m in sec["members"]:
            rs = m.get("rs_market") or {}
            if rs.get("chg_4w_pct") is None:
                continue
            rows.append({"symbol": m["symbol"], "section": sec["name"],
                         "rs_4w_pct": rs["chg_4w_pct"], "rs_1w_pct": rs.get("chg_1w_pct"),
                         "label": m["label"]})
    rows.sort(key=lambda x: x["rs_4w_pct"], reverse=True)
    for i, x in enumerate(rows, 1):
        x["rank"] = i
    return {"basis": "个股 / QQQ 的 RS 线 4 周变化", "ranking": rows}


def focus_list(sections: list[dict[str, Any]], ranking: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """按事先写定的规则挑出本周重点标的，并写明入选原因。"""
    n = len(ranking)
    rank = {r["symbol"]: r["rank"] for r in ranking}
    out = []
    for sec in sections:
        for m in sec["members"]:
            why = []
            t0, t1 = LABEL_TONE.get(m.get("prev_label") or ""), LABEL_TONE.get(m["label"])
            if t0 and t1 and t0 != t1:
                why.append(f"强弱翻转 {m['prev_label']} → {m['label']}")
            if m["move_atr"] is not None and abs(m["move_atr"]) >= FOCUS_MOVE_ATR:
                why.append(f"周幅 {m['move_atr']:+g} 倍周 ATR")
            if len(m["warnings"]) >= FOCUS_MIN_WARNINGS:
                why.append(f"{len(m['warnings'])} 条预警")
            r = rank.get(m["symbol"])
            if r is not None and (r <= FOCUS_RANK_EDGE or r > n - FOCUS_RANK_EDGE):
                why.append(f"RS 排名 {r}/{n}")
            if why:
                edge = min(r - 1, n - r) if r is not None else n
                out.append({"symbol": m["symbol"], "section": sec["name"],
                            "label": m["label"], "reasons": why, "_key": (-len(why), edge)})
    out.sort(key=lambda x: x.pop("_key"))
    return out[:FOCUS_MAX]


def sector_table(sections: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """板块轮动总表：板块 ETF 自身的标签与对 QQQ 的 RS，加上成员的广度。"""
    out = []
    for sec in sections:
        if sec["name"] == "指数":
            continue
        ref = sec.get("reference_metrics") or {}
        agg = sec.get("aggregate") or {}
        out.append({
            "section": sec["name"],
            "reference": sec["reference"],
            "reference_label": ref.get("label"),
            "reference_prev_label": ref.get("prev_label"),
            "reference_ret_pct": ref.get("ret_pct"),
            "reference_rs_market_4w_pct": (ref.get("rs_market") or {}).get("chg_4w_pct"),
            "composite_ret_pct": agg.get("composite_ret_pct"),
            "breadth": f"{agg.get('above_ema10', 0)}/{agg.get('n', 0)}",
            "leader": agg.get("leader"),
            "laggard": agg.get("laggard"),
            "leader_changed": agg.get("leader_changed"),
        })
    return out


def regimes(data: dict[str, pd.DataFrame], monday: date) -> dict[str, Any]:
    """日线大盘开关（REGIME_GATE 同口径）在周初与周末的状态。"""
    out = {}
    for sym in INDEX_MEMBERS:
        if sym not in data:
            continue
        end = benchmark_regime(_slice_through(data[sym], monday))
        start = benchmark_regime(_slice_through(data[sym], monday - timedelta(days=7)))
        out[sym] = {"start": start["state"], "end": end["state"], "end_label": end["label"],
                    "reason": end["reason"]}
    return out


def _symbol_metrics(payload: dict[str, Any], sym: str) -> dict[str, Any] | None:
    """在周报 JSON 里找一个标的的周线指标：板块参照或成员都算。"""
    for sec in payload.get("sections", []):
        ref = sec.get("reference_metrics")
        if sec.get("reference") == sym and ref:
            return ref
        for m in sec.get("members", []):
            if m["symbol"] == sym:
                return m
    return None


def target_move(payload: dict[str, Any], section: str) -> dict[str, Any] | None:
    """预测对象这一周的涨跌：多只时取等权平均。缺任何一只就返回 None。"""
    syms = PREDICTION_TARGETS.get(section)
    if not syms:
        return None
    rets, moves = [], []
    for sym in syms:
        m = _symbol_metrics(payload, sym)
        if not m or m["ret_pct"].get("1w") is None or m.get("move_atr") is None:
            return None
        rets.append(m["ret_pct"]["1w"])
        moves.append(m["move_atr"])
    move = float(np.mean(moves))
    return {"target": "+".join(syms), "ret_pct": _round(np.mean(rets), 2),
            "move_atr": _round(move, 2), "direction": realized_direction(move)}


def realized_direction(move_atr: float) -> str:
    if abs(move_atr) < PREDICTION_FLAT_ATR:
        return "flat"
    return "up" if move_atr > 0 else "down"


def score_predictions(
    predictions: list[dict[str, Any]], made_on: dict[str, Any], outcome: dict[str, Any],
) -> list[dict[str, Any]]:
    """一周的预测对一周的结果。made_on 是做预测那一周的 JSON（动量基准用它的方向）。"""
    rows = []
    for pr in predictions:
        sec = pr["section"]
        res = target_move(outcome, sec)
        base = target_move(made_on, sec)
        row = {"section": sec, "target": "+".join(PREDICTION_TARGETS.get(sec, ())),
               "direction": pr["direction"], "confidence": pr.get("confidence")}
        if res is None:
            row["status"] = "无数据"
        else:
            row.update({
                "status": "已判定",
                "actual": res["direction"], "ret_pct": res["ret_pct"], "move_atr": res["move_atr"],
                "hit": pr["direction"] == res["direction"],
                "momentum": base["direction"] if base else None,
                "momentum_hit": bool(base) and base["direction"] == res["direction"],
                "all_up_hit": res["direction"] == "up",
            })
        rows.append(row)
    return rows


def scorecard(history_dir: Path, payload: dict[str, Any]) -> dict[str, Any] | None:
    """上一周预测的判定，加上至今全部已判定预测的累计命中率与两个基准。

    只读磁盘上的周报 JSON 与叙事；当前这一周的结果用传入的 payload。
    """
    from .narrative import load_predictions

    cur = payload["week"]
    weeks = sorted(f.stem for f in history_dir.glob("*-W*.json")
                   if not f.stem.endswith((".narrative", ".news", ".sources")))
    allrows: list[dict[str, Any]] = []
    last: dict[str, Any] | None = None
    for wk in weeks + ([cur] if cur not in weeks else []):
        if wk >= cur:
            break
        preds = load_predictions(history_dir, wk)
        if not preds:
            continue
        nxt = week_id(parse_week(wk) + timedelta(days=7))
        if nxt == cur:
            outcome = payload
        elif (history_dir / f"{nxt}.json").exists():
            outcome = json.loads((history_dir / f"{nxt}.json").read_text(encoding="utf-8"))
        else:
            continue
        made_on = json.loads((history_dir / f"{wk}.json").read_text(encoding="utf-8"))
        rows = score_predictions(preds, made_on, outcome)
        allrows += [r for r in rows if r["status"] == "已判定"]
        if nxt == cur:
            last = {"made_in": wk, "rows": rows}
    if not allrows and not last:
        return None
    n = len(allrows)
    by_conf = {}
    for c in CONFIDENCE_LABELS:
        sub = [r for r in allrows if r.get("confidence") == c]
        if sub:
            by_conf[c] = {"n": len(sub), "hits": sum(r["hit"] for r in sub)}
    return {
        "last_week": last,
        "cumulative": {
            "n": n,
            "hits": sum(r["hit"] for r in allrows),
            "momentum_hits": sum(r["momentum_hit"] for r in allrows),
            "all_up_hits": sum(r["all_up_hit"] for r in allrows),
            "hit_pct": _round(100 * sum(r["hit"] for r in allrows) / n, 1) if n else None,
            "momentum_pct": _round(100 * sum(r["momentum_hit"] for r in allrows) / n, 1) if n else None,
            "all_up_pct": _round(100 * sum(r["all_up_hit"] for r in allrows) / n, 1) if n else None,
            "by_confidence": by_conf,
        },
        "note": "样本很少时命中率没有意义；三选一随机猜的期望约 33%",
    }


def build_weekly(
    monday: date,
    *,
    provider: Any,
    universe: Universe | None = None,
    now: datetime | None = None,
    history_dir: Path | None = None,
) -> dict[str, Any]:
    """history_dir 给出时，读上一周的预测并按本周结果记分（scorecard）。"""
    universe = universe or build_universe()
    data: dict[str, pd.DataFrame] = {}
    failed: dict[str, str] = {}
    stale: list[str] = []
    for sym in universe.symbols:
        try:
            o = provider.fetch(sym, "1d")
        except (DataUnavailable, Exception) as exc:  # noqa: BLE001
            failed[sym] = f"{type(exc).__name__}: {exc}"
            continue
        data[sym] = o.df
        if o.meta.stale or o.meta.too_old:
            stale.append(sym)

    now = (now or utcnow()).astimezone(ET)
    partial = now < week_close_time(monday)
    sections = [_section(n, s, monday, data) for n, s in universe.sections.items()]
    rot = rotation(sections)
    payload = {
        "week": week_id(monday),
        "monday": monday.isoformat(),
        "friday": (monday + timedelta(days=4)).isoformat(),
        "partial": partial,
        "generated_at": now.isoformat(timespec="seconds"),
        "market": MARKET,
        "regimes": regimes(data, monday),
        "sector_table": sector_table(sections),
        "rotation": rot,
        "focus": focus_list(sections, rot["ranking"]),
        "sections": sections,
        "data_health": {"failed": failed, "stale": stale},
        "rules": {
            "lookback_weeks": list(LOOKBACK_WEEKS), "high_weeks": HIGH_52W,
            "ema": [EMA_FAST, EMA_SLOW], "atr_weeks": ATR_WEEKS,
            "rs_high_weeks": RS_HIGH_WEEKS, "rs_slope_weeks": RS_SLOPE_WEEKS,
            "accel": {"close_pos": ACCEL_CLOSE_POS, "move_atr": ACCEL_MOVE_ATR},
            "pause_max_pullback_atr": PAUSE_MAX_PULLBACK_ATR,
            "overheat_ema_atr": OVERHEAT_EMA_ATR,
            "reversal_close_pos": REVERSAL_CLOSE_POS,
            "underperform_rs_pct": UNDERPERFORM_RS_PCT,
            "distribution": {"volume_mult": DISTRIBUTION_VOLUME_MULT,
                             "warn_days": DISTRIBUTION_WARN_DAYS},
            "label_order": LABEL_ORDER,
            "focus": {"move_atr": FOCUS_MOVE_ATR, "min_warnings": FOCUS_MIN_WARNINGS,
                      "rank_edge": FOCUS_RANK_EDGE, "max": FOCUS_MAX},
            "prediction": {"targets": {k: list(v) for k, v in PREDICTION_TARGETS.items()},
                           "flat_atr": PREDICTION_FLAT_ATR},
            "note": "标签只描述状态，阈值事先写定，未回测预测力；成员是精选强势票，板块合成值天然偏强",
        },
    }
    if history_dir is not None:
        payload["scorecard"] = scorecard(Path(history_dir), payload)
    return payload


# --------------------------------------------------------------------------- 文本


def _fmt(v: Any, suffix: str = "") -> str:
    return "—" if v is None else f"{v:+g}{suffix}" if isinstance(v, (int, float)) else str(v)


def format_weekly(p: dict[str, Any]) -> str:
    L: list[str] = []
    head = f"周报 {p['week']}（{p['monday']} – {p['friday']}）"
    L.append(head + ("  [未完成周]" if p["partial"] else ""))
    L.append("")
    L.append("大盘开关（日线，周初 → 周末）")
    for sym, r in p["regimes"].items():
        arrow = "" if r["start"] == r["end"] else "  ← 切换"
        L.append(f"  {sym:<5} {r['start']} → {r['end']}{arrow}   {r['reason']}")
    L.append("")
    L.append("板块轮动总表")
    L.append(f"  {'板块':<6}{'参照':<6}{'参照标签':<12}{'参照1w':>8}{'4w':>8}{'对QQQ 4w':>10}"
             f"{'成员1w':>8}{'4w':>8}{'广度':>6}  领涨/落后")
    for s in p["sector_table"]:
        rr = s["reference_ret_pct"] or {}
        cr = s["composite_ret_pct"] or {}
        lab = s["reference_label"] or "—"
        if s["reference_prev_label"] and s["reference_prev_label"] != s["reference_label"]:
            lab = f"{s['reference_prev_label']}→{lab}"
        L.append(f"  {s['section']:<6}{s['reference'] or '—':<6}{lab:<12}"
                 f"{_fmt(rr.get('1w')):>8}{_fmt(rr.get('4w')):>8}"
                 f"{_fmt(s['reference_rs_market_4w_pct']):>10}"
                 f"{_fmt(cr.get('1w')):>8}{_fmt(cr.get('4w')):>8}{s['breadth']:>6}  "
                 f"{s['leader'] or '—'}/{s['laggard'] or '—'}"
                 + ("  领涨换人" if s["leader_changed"] else ""))
    for sec in p["sections"]:
        L.append("")
        ref = sec.get("reference_metrics")
        title = f"【{sec['name']}】" + (f" 参照 {sec['reference']}：{ref['label']}"
                                       f"（上周 {ref['prev_label']}）" if ref else "")
        L.append(title)
        L.append(f"  {'标的':<7}{'标签':<10}{'1w%':>7}{'4w%':>7}{'13w%':>7}{'周幅ATR':>8}"
                 f"{'收位':>6}{'距EMA10':>8}{'RS参照4w':>9}{'RS QQQ4w':>9}  周内")
        for m in sec["members"]:
            lab = m["label"] if not m["prev_label"] or m["prev_label"] == m["label"] \
                else f"{m['prev_label']}→{m['label']}"
            iw = m.get("intraweek") or {}
            rr, rm = m.get("rs_reference") or {}, m.get("rs_market") or {}
            pos = "—" if m["close_pos"] is None else f"{m['close_pos']:.0%}"
            L.append(
                f"  {m['symbol']:<7}{lab:<10}{_fmt(m['ret_pct']['1w']):>7}{_fmt(m['ret_pct']['4w']):>7}"
                f"{_fmt(m['ret_pct']['13w']):>7}{_fmt(m['move_atr']):>8}{pos:>6}"
                f"{_fmt(m['dist_ema10_atr']):>8}{_fmt(rr.get('chg_4w_pct')):>9}"
                f"{_fmt(rm.get('chg_4w_pct')):>9}  "
                f"高{iw.get('high_day', '—')} 低{iw.get('low_day', '—')}"
                + (f" 放量跌{len(iw['distribution_days'])}" if iw.get("distribution_days") else "")
            )
            for w in m["warnings"]:
                L.append(f"      ! {w}")
        if sec.get("aggregate"):
            a = sec["aggregate"]
            L.append(f"  广度 {a['above_ema10']}/{a['n']} 在 10 周 EMA 上方，"
                     f"{a['new_13w_high']} 只创 13 周新高；4 周分化 {_fmt(a['dispersion_4w_pct'], '%')}"
                     + (f"；跑赢参照 1w {a['beat_reference']['1w']}/{a['n']}、"
                        f"4w {a['beat_reference']['4w']}/{a['n']}" if a["beat_reference"] else ""))
        if sec["missing"]:
            L.append(f"  无本周数据：{', '.join(sec['missing'])}")
    L.append("")
    L.append("RS 排名（个股 / QQQ，4 周变化）")
    for x in p["rotation"]["ranking"]:
        L.append(f"  {x['rank']:>2}. {x['symbol']:<6}{x['section']:<6}{x['rs_4w_pct']:>+8.2f}%  "
                 f"本周 {_fmt(x['rs_1w_pct'], '%')}  {x['label']}")
    h = p["data_health"]
    if h["failed"] or h["stale"]:
        L.append("")
        L.append(f"数据问题：失败 {list(h['failed'])}，回退缓存 {h['stale']}")
    return "\n".join(L)


# --------------------------------------------------------------------------- HTML

TEMPLATE = Path(__file__).with_name("templates") / "weekly.html"


def _render(payload: dict[str, Any], title: str) -> str:
    from .chart import _json_safe

    blob = json.dumps(_json_safe(payload), ensure_ascii=False, default=str).replace("</", "<\\/")
    return (TEMPLATE.read_text(encoding="utf-8")
            .replace("__TITLE__", _html.escape(title, quote=False))
            .replace("__PAYLOAD__", blob))


def _summary(p: dict[str, Any]) -> str:
    """列表页上一行：各板块参照 ETF 的标签。"""
    parts = [f"{s['section']} {s['reference_label']}" for s in p.get("sector_table", [])
             if s.get("reference_label")]
    return " · ".join(parts)


def load_narrative(out_dir: Path, week: str) -> dict[str, Any] | None:
    """叙事单独存放（{week}.narrative.json），重跑计算层不会覆盖它。"""
    f = out_dir / f"{week}.narrative.json"
    return json.loads(f.read_text(encoding="utf-8")) if f.exists() else None


def write_site(payload: dict[str, Any], out_dir: Path) -> Path:
    """写入本周 JSON，并用目录里所有周的 JSON 重新渲染各周页面与列表页。

    全部重渲染是为了让上一周的"下一周"链接跟着更新；每周一页，量很小。
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{payload['week']}.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=1, default=str), encoding="utf-8")

    weeks = sorted(f.stem for f in out_dir.glob("*-W*.json")
                   if not f.stem.endswith((".narrative", ".news", ".sources")))
    index = []
    for i, wk in enumerate(weeks):
        p = json.loads((out_dir / f"{wk}.json").read_text(encoding="utf-8"))
        p["narrative"] = load_narrative(out_dir, wk)
        if p["narrative"]:
            from .narrative import data_fingerprint
            p["narrative_stale"] = p["narrative"].get("data_fingerprint") != data_fingerprint(p)
        p["nav"] = {"prev": f"{weeks[i - 1]}.html" if i > 0 else None,
                    "next": f"{weeks[i + 1]}.html" if i + 1 < len(weeks) else None}
        (out_dir / f"{wk}.html").write_text(_render(p, f"johnny-ta 周报 {wk}"), encoding="utf-8")
        index.append({"week": wk, "file": f"{wk}.html", "monday": p["monday"],
                      "friday": p["friday"], "summary": _summary(p),
                      "narrative": bool(p["narrative"])})
    (out_dir / "index.html").write_text(
        _render({"index": list(reversed(index))}, "johnny-ta 周报"), encoding="utf-8")
    return out_dir / f"{payload['week']}.html"
