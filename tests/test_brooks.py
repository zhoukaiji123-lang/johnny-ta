"""Brooks 价格行为层测试（不联网）。

合成 K 线逐根构造，验证的是规则落地是否与课程口径一致，不是收益。
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from jta.analyze import analyze
from jta.brooks import (
    _zh,
    _avg_range,
    _breakouts,
    _equation,
    _oriented,
    _trend_bars,
    brooks_analysis,
)
from jta.indicators.ema import ema
from tests.test_pipeline import FakeProvider

TZ = "America/New_York"


def frame(rows: list[tuple[float, float, float, float]]) -> pd.DataFrame:
    idx = pd.DatetimeIndex(
        list(pd.date_range("2025-01-02", periods=len(rows), freq="B", tz=TZ))
    )
    o, h, l, c = zip(*rows)
    return pd.DataFrame(
        {"open": o, "high": h, "low": l, "close": c, "volume": [1000.0] * len(rows)},
        index=idx,
    )


def steps(seq: list[float], start: float = 100.0, wick: float = 0.2) -> list[tuple]:
    rows, c = [], start
    for st in seq:
        o, c = c, c + st
        rows.append((o, max(o, c) + wick, min(o, c) - wick, c))
    return rows


FLAT = [0.3, -0.3] * 30
BREAKOUT = [3.0, 3.0, 3.0]
STAIRS = [1.5, 1.5, -0.8] * 6


def uptrend_rows(extra: list[float] | None = None) -> list[tuple]:
    rows = steps(FLAT, wick=0.5)
    rows += steps(BREAKOUT + STAIRS + (extra or []), start=rows[-1][3])
    return rows


def run(rows, **kw):
    df = frame(rows)
    return brooks_analysis(df, ema(df["close"], 20), atr=2.0, **kw)


# ------------------------------------------------------------------ K 线口径


def test_gap_counts_as_part_of_the_trend_bar():
    """ch11：缺口本身就是 BO 趋势 K 线。跳空低开再收阳的暴跌日是阴线，不是阳线。"""
    df = frame([(100, 101, 99, 100)] * 25 + [(90, 92, 89, 91.5)])
    avg = _avg_range(df)
    bull, _, _ = _trend_bars(*_oriented(df, 1), avg)
    bear, big, _ = _trend_bars(*_oriented(df, -1), avg)
    assert not bull[-1]
    assert bear[-1] and big[-1]


def test_three_small_trend_bars_form_a_confirmed_breakout():
    """ch13 第③种：3–5 根较小的同向 K 线。"""
    df = frame(steps(FLAT, wick=0.5) + steps([0.8, 0.8, 0.8], start=100.0, wick=0.1))
    ev = _breakouts(df, 1, _avg_range(df), 0)
    assert len(ev) == 1 and ev[0].confirmed


def test_single_huge_bar_waits_for_follow_through():
    """ch13：单根巨大 K 线要再来一根 FT；最后一根上的 BO 记为未确认。"""
    base = steps(FLAT, wick=0.5)
    pending = frame(base + steps([5.0], start=100.0))
    ev = _breakouts(pending, 1, _avg_range(pending), 0)
    assert len(ev) == 1 and not ev[0].confirmed

    with_ft = frame(base + steps([5.0, -0.3, 0.6], start=100.0))
    ev = _breakouts(with_ft, 1, _avg_range(with_ft), 0)
    assert ev[0].confirmed and ev[0].confirm_index == len(with_ft) - 1


def test_gap_breakout_origin_is_the_bottom_of_the_gap():
    """跳空 BO 的起点是前一根收盘（缺口底部），止损"BO 起点下方"才不会被放进缺口里。"""
    base = steps(FLAT, wick=0.5)
    last = base[-1][3]
    gap = [(last + 4.0, last + 6.2, last + 3.9, last + 6.0), (last + 6.0, last + 9.2, last + 5.9, last + 9.0)]
    df = frame(base + gap)
    ev = _breakouts(df, 1, _avg_range(df), 0)
    assert ev[-1].confirmed and ev[-1].origin == last


# ------------------------------------------------------------------ 状态判定


def test_uptrend_is_always_in_long_tight_channel_with_h1_setup():
    b = run(uptrend_rows())
    assert b["always_in"]["direction"] == 1
    assert b["state"] == "tight_channel"
    assert b["bar_count"]["next"] == 1
    plan = b["plan"]
    assert plan["direction"] == "long" and plan["setup_code"] == "A5"
    assert plan["executable"], plan["blocked_by"]
    assert plan["entry"] > plan["stop"] and plan["t1"] > plan["entry"]


def test_h1_failure_rearms_the_count_to_h2():
    """ch09：H1 之后跌破其信号 K 线低点 = 新的一推，下一次越过前高才是 H2。"""
    rows = uptrend_rows([1.5, 1.5])
    x = rows[-1][3]
    rows += [
        (x, x + 0.05, x - 1.2, x - 1.0),          # A：回调第一根
        (x - 1.0, x + 0.1, x - 1.1, x - 0.1),     # B：越过 A 高点 = H1，未创新高
        (x - 0.1, x - 0.05, x - 1.6, x - 1.5),    # C：跌破 A 低点，重新开始计数
        (x - 1.5, x - 1.4, x - 1.8, x - 1.7),     # D：最后一根，等 H2
    ]
    b = run(rows)
    assert b["bar_count"] == {"count": 1, "next": 2, "pending": True, "prefix": "H",
                              "signal_quality": b["bar_count"]["signal_quality"]}
    assert "H2" in b["plan"]["setup"]
    assert [m["label"] for m in b["marks"]["counts"]] == ["H1"]


def test_downtrend_gives_no_long_plan_without_mtr():
    """只做多：AIS 且 MTR 条件不齐时，D 不给入场位。"""
    rows = steps(FLAT, start=200.0, wick=0.5)
    rows += steps([-s for s in BREAKOUT + STAIRS], start=rows[-1][3])
    b = run(rows)
    assert b["always_in"]["direction"] == -1
    assert b["plan"]["entry"] is None and not b["plan"]["executable"]
    assert b["mtr"] is not None and not b["mtr"]["valid"]
    assert b["judgment"]["lean"]["side"] == "bear"


def test_flat_market_is_a_tight_trading_range_and_not_traded():
    b = run(steps([0.3, -0.3] * 60, wick=0.5))
    assert b["always_in"]["direction"] == 0
    assert b["state"] == "tight_trading_range"
    assert not b["plan"]["executable"]
    assert b["judgment"]["lean"]["p"] == 0.5


def test_event_mode_blocks_the_plan():
    b = run(uptrend_rows(), event_mode=True, event_reason="3 日后财报")
    assert not b["plan"]["executable"]
    assert any("事件模式" in x for x in b["plan"]["blocked_by"])


def test_wide_stop_scales_position_down():
    """ch33：正确止损超过平均 K 线 2 倍，仓位降到 1/2–1/3，金额风险不变。"""
    plan = run(uptrend_rows())["plan"]
    assert plan["position_scale"] < 1
    assert any("ch33" in c for c in plan["cautions"])


def test_short_history_is_declared_unavailable():
    b = run(steps([0.3, -0.3] * 10))
    assert b == {"available": False, "reason": b["reason"]} and "不足" in b["reason"]


# ------------------------------------------------------------------ 交易者方程式


def test_equation_one_to_one_survives_tick_rounding():
    """入场与止损各差 1 tick 会让 1:1 算成 0.998：按展示精度判，不能显示 1.00 却判不通过。"""
    eq = _equation(0.6, entry=100.0, stop=94.99, target=105.0)
    assert eq["rr"] == 1.0 and eq["ok"]


def test_equation_rejects_insufficient_reward_for_lower_probability():
    eq = _equation(0.5, entry=100.0, stop=95.0, target=107.5)   # 1.5R，50% 要求 2R
    assert not eq["ok"] and eq["min_rr"] == 2.0


# ------------------------------------------------------------------ 编排


def test_analyze_carries_brooks_as_strict_json():
    """不能夹带 numpy 类型：dashboard 用 default=str 兜底，会把 np.True_ 悄悄变成字符串。"""
    r = analyze("TEST", provider=FakeProvider(), include_events=False,
                include_market_cap=False, account=100_000)
    b = r["brooks"]
    assert b["available"]
    json.dumps(b, allow_nan=False)
    assert b["plan"]["key"] == "brooks"
    assert all(p["key"] != "brooks" for p in r["plans"])     # A/B/C 与回测口径不受影响


def test_brooks_respects_as_of():
    as_of = pd.Timestamp("2025-06-02", tz=TZ)
    r = analyze("TEST", provider=FakeProvider(), as_of=as_of,
                include_events=False, include_market_cap=False)
    marks = r["brooks"]["marks"]
    for m in marks["breakouts"] + marks["counts"]:
        assert pd.Timestamp(m["t"]) <= as_of


# ------------------------------------------------------------------ 术语注释


def test_terms_get_chinese_notes_once_per_sentence():
    assert _zh("强 BO 后，BO 失败") == "强 BO（强势突破）后，BO（突破）失败"
    assert _zh("等 H2") == "等 H2（第 2 次回调买点）"
    assert _zh("TTR 不是 TR") == "TTR（紧密交易区间）不是 TR（交易区间）"   # TTR 里的 TR 不单独匹配


def test_term_notes_merge_into_existing_parens_and_are_idempotent():
    s = _zh("收盘在 20 EMA（171.57）上方")
    assert s == "收盘在 20 EMA（指数移动平均线，171.57）上方"
    assert _zh(s) == s


def test_brooks_output_is_annotated():
    b = run(uptrend_rows())
    assert "（" in b["always_in"]["label"] and "AIL（多方主导）" in b["judgment"]["summary"]
    assert b["plan"]["setup_code"] == "A5"          # 代码字段不加注释
