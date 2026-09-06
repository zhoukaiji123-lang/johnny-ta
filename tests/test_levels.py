"""关键位计算层测试。

重点不在"算得出数字"，而在于锚点自由度是否真的被锁死——
这是整个改造的目的。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from jta.indicators.swing import SwingPoint, detect_swings
from jta.levels.fib import (
    MAX_NESTED_ANCHORS,
    Swing,
    extension,
    local_navigation,
    nested_anchors,
    primary_rebound,
    primary_retracement,
    select_dominant_swing,
    select_recent_swing,
    visible_levels,
)
from jta.levels.pivots import (
    gaps,
    horizontal_pivots,
    prior_session_levels,
    round_numbers,
)
from jta.levels.trendline import fit_trendlines, parallel_channel

TZ = "America/New_York"


def idx(n: int) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(
        list(pd.date_range("2026-01-05", periods=n, freq="B", tz=TZ)), name="Date"
    )


def frame(closes, highs=None, lows=None, volumes=None) -> pd.DataFrame:
    n = len(closes)
    return pd.DataFrame(
        {
            "open": closes,
            "high": highs if highs is not None else [c + 1 for c in closes],
            "low": lows if lows is not None else [c - 1 for c in closes],
            "close": closes,
            "volume": volumes if volumes is not None else [1000.0] * n,
        },
        index=idx(n),
        dtype="float64",
    )


def sp(bar: int, kind: str, price: float, *, base: bool = False) -> SwingPoint:
    ts = idx(500)[bar]
    conf = idx(500)[bar + 3]
    return SwingPoint(ts, kind, price, conf, bar, base, conf if base else None)


# ------------------------------------------------------------------ Fib 数学


def test_retracement_formula():
    s = Swing(sp(0, "low", 100.0), sp(10, "high", 200.0), "up", "1d")
    levels = {round(f.ratio, 3): f.price for f in primary_retracement(s)}
    assert levels[0.5] == pytest.approx(150.0)
    assert levels[0.618] == pytest.approx(138.2)
    assert levels[0.236] == pytest.approx(176.4)


def test_rebound_formula_is_measured_from_the_low():
    s = Swing(sp(0, "low", 100.0), sp(10, "high", 200.0), "down", "1d")
    levels = {round(f.ratio, 3): f.price for f in primary_rebound(s)}
    assert levels[0.5] == pytest.approx(150.0)
    assert levels[0.382] == pytest.approx(138.2)
    assert all(f.side == "resistance" for f in primary_rebound(s))


def test_half_ratio_note_guards_against_the_zero_point_five_percent_bug():
    """原博文把 0.5 回撤写成 '0.5%'。输出必须自带澄清。"""
    s = Swing(sp(0, "low", 100.0), sp(10, "high", 200.0), "up", "1d")
    note = next(f.note for f in primary_retracement(s) if f.ratio == 0.5)
    assert "50.0%" in note and "非 0.5%" in note


def test_extension_targets_beyond_the_swing():
    s = Swing(sp(0, "low", 100.0), sp(10, "high", 200.0), "up", "1d")
    got = {round(f.ratio, 3): f.price for f in extension(s)}
    assert got[1.618] == pytest.approx(261.8)
    assert got[2.618] == pytest.approx(361.8)


def test_local_navigation_marks_current_price_as_neutral_axis():
    s = Swing(sp(0, "low", 100.0), sp(10, "high", 200.0), "up", "1d")
    lv = local_navigation(s, current_price=150.0)
    sides = {f.ratio: f.side for f in lv}
    assert sides[0.5] == "neutral"          # 现价所在比例是中轴
    assert sides[0.236] == "resistance"     # 176.4 在现价之上
    assert sides[0.786] == "support"        # 121.4 在现价之下


# ------------------------------------------------------------------ 锚点自由度


def test_dominant_swing_picks_largest_span_not_the_latest():
    pts = [
        sp(0, "low", 100.0),
        sp(10, "high", 300.0),   # 幅度 200，最大
        sp(20, "low", 250.0),
        sp(30, "high", 280.0),   # 幅度 30，更近但更小
    ]
    s = select_dominant_swing(pts, "up", timeframe="1d")
    assert s.low.price == 100.0 and s.high.price == 300.0


def test_recent_swing_requires_alternating_kinds():
    assert select_recent_swing([sp(0, "low", 1.0)], timeframe="1d") is None
    assert (
        select_recent_swing([sp(0, "high", 2.0), sp(5, "high", 3.0)], timeframe="1d")
        is None
    )


def test_nested_anchors_rejects_non_base_steps():
    """未通过突破基座校验的低点不得充当台阶——否则可以凑出任意价位。"""
    apex = sp(50, "high", 1000.0)
    steps = [sp(10, "low", 200.0), sp(20, "low", 400.0)]  # 均非 base
    assert nested_anchors(apex, steps, timeframe="1d") == []


def test_nested_anchors_rejects_steps_after_the_apex():
    apex = sp(50, "high", 1000.0)
    later = sp(60, "low", 700.0, base=True)
    assert nested_anchors(apex, [later], timeframe="1d") == []


def test_nested_anchors_are_capped():
    apex = sp(200, "high", 1000.0)
    steps = [sp(i * 10, "low", 100.0 + i * 10, base=True) for i in range(1, 12)]
    out = nested_anchors(apex, steps, timeframe="1d")
    assert len(out) == MAX_NESTED_ANCHORS


def test_nested_anchor_math_and_recency_selection():
    apex = sp(100, "high", 1000.0)
    steps = [sp(10, "low", 200.0, base=True), sp(50, "low", 500.0, base=True)]
    out = nested_anchors(apex, steps, timeframe="1d")
    prices = sorted(round(f.price, 2) for f in out)
    # 1000 - 0.618*(1000-500) = 691.0 ; 1000 - 0.618*(1000-200) = 505.6
    assert prices == [505.6, 691.0]


def test_visible_levels_filters_by_anchor_confirmation():
    apex = sp(100, "high", 1000.0)
    steps = [sp(50, "low", 500.0, base=True)]
    lv = nested_anchors(apex, steps, timeframe="1d")
    conf = pd.Timestamp(lv[0].confirmed_at)
    assert visible_levels(lv, conf - pd.Timedelta(days=1)) == []
    assert len(visible_levels(lv, conf)) == 1


# ------------------------------------------------------------------ 水平结构


def test_pivot_cluster_radius_scales_with_local_atr():
    """低价区的历史摆动不能被现价的绝对 ATR 糊成一个枢轴。"""
    n = 300
    closes = list(np.linspace(100, 1000, n))
    df = frame(closes)
    swings = [sp(20, "low", 100.0), sp(40, "high", 108.0), sp(280, "low", 950.0)]
    piv = horizontal_pivots(swings, df, 1000.0, timeframe="1d", min_touches=1)
    prices = sorted(round(p.price) for p in piv)
    assert 100 in prices and 108 in prices  # 相距 8 元，在低价区必须分开


def test_pivot_requires_minimum_touches():
    df = frame(list(np.linspace(100, 200, 100)))
    one = [sp(10, "low", 120.0)]
    assert horizontal_pivots(one, df, 200.0, timeframe="1d") == []


def test_pivot_flags_support_resistance_flip():
    df = frame(list(np.linspace(100, 200, 100)))
    swings = [sp(10, "low", 150.0), sp(30, "high", 150.2)]
    piv = horizontal_pivots(swings, df, 200.0, timeframe="1d")
    assert piv and piv[0].detail["flipped"] is True


def test_prior_session_levels_use_the_second_to_last_bar():
    df = frame([10, 20, 30])
    lv = {l.source: l.price for l in prior_session_levels(df, 30.0, timeframe="1d")}
    assert lv["prev_close"] == 20.0
    assert lv["prev_high"] == 21.0 and lv["prev_low"] == 19.0


def test_filled_gaps_are_excluded():
    n = 40
    closes = [100.0] * n
    highs = [101.0] * n
    lows = [99.0] * n
    # 第 20 根向上跳空，随后第 30 根完全回补
    closes[20:] = [130.0] * (n - 20)
    highs[20:] = [131.0] * (n - 20)
    lows[20:] = [129.0] * (n - 20)
    lows[30] = 90.0
    df = frame(closes, highs, lows)
    assert gaps(df, 130.0, timeframe="1d") == []


def test_unfilled_gap_is_reported_with_edge_price():
    n = 40
    closes = [100.0] * n
    highs = [101.0] * n
    lows = [99.0] * n
    closes[20:] = [130.0] * (n - 20)
    highs[20:] = [131.0] * (n - 20)
    lows[20:] = [129.0] * (n - 20)
    df = frame(closes, highs, lows)
    g = gaps(df, 130.0, timeframe="1d")
    assert g and g[0].detail["gap_kind"] == "gap_up"
    assert g[0].price == pytest.approx(101.0)  # 缺口下沿 = 前一根的 high


def test_round_numbers_scale_with_price_magnitude():
    small = [l.price for l in round_numbers(42.0, 2.0, timeframe="1d")]
    big = [l.price for l in round_numbers(910.0, 68.0, timeframe="1d")]
    assert 40.0 in small and 45.0 in small
    assert 900.0 in big and 950.0 in big
    assert all(abs(p - 910.0) <= 3 * 68.0 for p in big)


def test_round_numbers_reject_bad_inputs():
    assert round_numbers(0.0, 1.0, timeframe="1d") == []
    assert round_numbers(100.0, float("nan"), timeframe="1d") == []


# ------------------------------------------------------------------ 趋势线


def sawtooth(legs: int, up: int, down: int, step: float, start: float = 100.0):
    """锯齿走势：回调低点严格等距递增，因此必然共线。

    正弦类合成数据的局部极小并不共线，连线会在中途被穿透——那是数据的问题，
    不是拟合规则的问题，所以测试必须用几何上确定的序列。
    """
    prices = [start]
    for _ in range(legs):
        for _ in range(up):
            prices.append(prices[-1] + step)
        for _ in range(down):
            prices.append(prices[-1] - step)
    return prices


def rising_market() -> pd.DataFrame:
    return frame(sawtooth(legs=6, up=10, down=8, step=2.0))


def falling_market() -> pd.DataFrame:
    return frame(sawtooth(legs=6, up=8, down=10, step=2.0, start=300.0))


def test_uptrend_line_has_positive_slope_and_touches():
    df = rising_market()
    pts = detect_swings(df, k=2, min_atr_mult=0)
    lines = fit_trendlines(df, pts, "up")
    assert lines
    assert lines[0].slope > 0
    assert lines[0].touches >= 2


def test_trendline_value_is_dynamic_and_labelled_approximate():
    df = rising_market()
    pts = detect_swings(df, k=2, min_atr_mult=0)
    line = fit_trendlines(df, pts, "up")[0]
    d = line.to_dict()
    assert d["dynamic"] is True and d["display"].startswith("约")
    assert line.value_at(line.x0 + 10) != line.value_at(line.x0)
    assert d["as_of_bar"] == df.index[-1].isoformat()


def test_line_pierced_before_second_anchor_is_rejected():
    """两点之间就被穿透，说明这条线画错了，不能靠事后挑锚点保留。"""
    closes = [100.0, 90.0, 60.0, 95.0, 120.0, 130.0]
    df = frame(closes)
    a = sp(0, "low", 99.0)
    b = sp(4, "low", 119.0)
    from jta.levels.trendline import _fit
    from jta.indicators.atr import atr as atr_series

    assert _fit(df, a, b, "up", atr_series(df).to_numpy(), 0.25) is None


def test_downtrend_line_connects_lower_highs():
    df = falling_market()
    pts = detect_swings(df, k=2, min_atr_mult=0)
    lines = fit_trendlines(df, pts, "down")
    assert lines
    assert lines[0].slope < 0
    assert lines[0].touches >= 2


def test_parallel_channel_envelopes_opposite_extreme():
    df = rising_market()
    pts = detect_swings(df, k=2, min_atr_mult=0)
    line = fit_trendlines(df, pts, "up")[0]
    ch = parallel_channel(df, line, pts)
    assert ch and ch["role"] == "channel_upper"
    assert ch["current_value"] > line.current_value


def test_insufficient_anchors_yield_no_lines():
    df = rising_market()
    assert fit_trendlines(df, [sp(0, "low", 1.0)], "up") == []
