"""指标层测试。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from jta.indicators.atr import atr, true_range
from jta.indicators.ema import ema, ema_set, ema_stack, vegas_zone, warmup_status
from jta.indicators.swing import detect_swings, visible_at
from jta.indicators.td import SETUP_LENGTH, latest_td_signal, td_setup

TZ = "America/New_York"


def frame(closes, highs=None, lows=None, volumes=None) -> pd.DataFrame:
    n = len(closes)
    idx = pd.DatetimeIndex(
        list(pd.date_range("2026-01-05", periods=n, freq="B", tz=TZ)), name="Date"
    )
    return pd.DataFrame(
        {
            "open": closes,
            "high": highs if highs is not None else [c + 1 for c in closes],
            "low": lows if lows is not None else [c - 1 for c in closes],
            "close": closes,
            "volume": volumes if volumes is not None else [1000] * n,
        },
        index=idx,
        dtype="float64",
    )


# ------------------------------------------------------------------ EMA


def test_ema_matches_manual_recursion():
    """alpha = 2/(n+1) 的递推，首值等于首个价格（adjust=False）。"""
    s = pd.Series([10.0, 11.0, 12.0])
    out = ema(s, 3)
    alpha = 2 / 4
    expected = [10.0]
    for v in [11.0, 12.0]:
        expected.append(alpha * v + (1 - alpha) * expected[-1])
    np.testing.assert_allclose(out.to_numpy(), expected)


def test_ema_set_columns_and_stack():
    df = frame(list(range(100, 200)))
    e = ema_set(df["close"])
    assert list(e.columns) == ["ema8", "ema13", "ema21", "ema144", "ema169"]
    # 单调上涨序列必然是多头排列
    assert ema_stack(e.iloc[-1]) == "bull"


def test_ema_stack_detects_bear_and_unknown():
    df = frame(list(range(200, 100, -1)))
    assert ema_stack(ema_set(df["close"]).iloc[-1]) == "bear"
    assert ema_stack(pd.Series({"ema8": np.nan, "ema13": 1, "ema21": 2})) == "unknown"


def test_warmup_flags_insufficient_history():
    st = warmup_status(300)
    assert st["ema21"]["reliable"] is True
    assert st["ema169"]["reliable"] is False  # 需要 845 根
    assert st["ema169"]["bars_needed"] == 845


def test_vegas_zone_orders_bounds():
    z = vegas_zone(pd.Series({"ema144": 700.0, "ema169": 690.0}))
    assert z == {"upper": 700.0, "lower": 690.0}
    assert vegas_zone(pd.Series({"ema144": np.nan, "ema169": 1.0})) is None


# ------------------------------------------------------------------ ATR


def test_true_range_uses_previous_close():
    df = frame([10, 20], highs=[11, 21], lows=[9, 19])
    tr = true_range(df)
    assert tr.iloc[0] == 2.0            # 首根退化为当根振幅
    assert tr.iloc[1] == 21 - 10        # |high - prev_close| 胜出


def test_atr_first_value_is_simple_mean_then_wilder():
    n = 20
    df = frame([100.0] * n, highs=[102.0] * n, lows=[98.0] * n)
    a = atr(df, period=14)
    assert np.isnan(a.iloc[12])
    # 恒定振幅 4，但首根 TR 也是 4，故 ATR 恒为 4
    assert a.iloc[13] == pytest.approx(4.0)
    assert a.iloc[-1] == pytest.approx(4.0)


def test_atr_wilder_recursion():
    df = frame([100.0] * 16, highs=[102.0] * 16, lows=[98.0] * 16)
    a = atr(df, 14)
    prev = a.iloc[13]
    tr_last = 4.0
    assert a.iloc[14] == pytest.approx((prev * 13 + tr_last) / 14)


def test_atr_returns_nan_when_history_too_short():
    assert atr(frame([1.0] * 5), 14).isna().all()


# ------------------------------------------------------------------ TD Setup


def test_buy_setup_requires_bearish_price_flip():
    # 前 6 根上涨制造 bearish flip 前提，随后连续 9 根低于 4 根前收盘
    closes = [10, 11, 12, 13, 14, 15] + [9, 8, 7, 6, 5, 4, 3, 2, 1]
    t = td_setup(frame(closes))
    assert t["td_buy_setup"].max() == SETUP_LENGTH
    assert t["td_sell_setup"].iloc[-1] == 0


def test_setup_resets_when_condition_breaks():
    # 计数走到 3 后被一根大阳打断，必须归零而不是接着数
    closes = [10, 11, 12, 13, 14, 15, 9, 8, 7, 99, 6, 5]
    counts = td_setup(frame(closes))["td_buy_setup"].tolist()
    assert counts[6:10] == [1, 2, 3, 0]
    assert max(counts) < SETUP_LENGTH


def test_buy_and_sell_setups_are_mutually_exclusive():
    rng = np.random.RandomState(7)
    closes = (100 + np.cumsum(rng.randn(200))).tolist()
    t = td_setup(frame(closes))
    both = (t["td_buy_setup"] > 0) & (t["td_sell_setup"] > 0)
    assert not both.any()


def test_perfection_checked_only_on_completed_nine():
    closes = [10, 11, 12, 13, 14, 15] + [9, 8, 7, 6, 5, 4, 3, 2, 1]
    t = td_setup(frame(closes))
    nine = t[t["td_buy_setup"] == SETUP_LENGTH]
    assert len(nine) == 1
    assert bool(nine["td_buy_9_perfected"].iloc[0]) is True  # 持续新低必然完美
    assert not t.loc[t["td_buy_setup"] < SETUP_LENGTH, "td_buy_9_perfected"].any()


def test_latest_td_signal_window():
    closes = [10, 11, 12, 13, 14, 15] + [9, 8, 7, 6, 5, 4, 3, 2, 1] + [2, 3, 4, 5]
    t = td_setup(frame(closes))
    assert latest_td_signal(t, within=2) is None
    sig = latest_td_signal(t, within=10)
    assert sig and sig["kind"] == "buy_setup_9"


# ------------------------------------------------------------------ Swing


def zigzag(legs: list[int], step: float = 10.0) -> pd.DataFrame:
    """按给定腿长生成锯齿价格序列。"""
    prices, cur, up = [100.0], 100.0, True
    for leg in legs:
        for _ in range(leg):
            cur += step if up else -step
            prices.append(cur)
        up = not up
    return frame(prices)


def test_detect_swings_alternates_high_low():
    pts = detect_swings(zigzag([5, 5, 5, 5, 5]), k=2, min_atr_mult=0)
    assert len(pts) >= 3
    kinds = [p.kind for p in pts]
    assert all(a != b for a, b in zip(kinds, kinds[1:]))


def test_swing_confirmed_at_is_k_bars_later():
    df = zigzag([5, 5, 5, 5])
    k = 2
    for p in detect_swings(df, k=k, min_atr_mult=0):
        assert df.index.get_loc(p.confirmed_at) == p.bar_index + k


def test_visible_at_blocks_lookahead():
    """as_of 当天不得看到尚未确认的摆动点——这是前向测试成立的前提。"""
    df = zigzag([5, 5, 5, 5])
    pts = detect_swings(df, k=2, min_atr_mult=0)
    assert pts
    target = pts[-1]
    just_before = df.index[df.index.get_loc(target.confirmed_at) - 1]
    assert target not in visible_at(pts, just_before)
    assert target in visible_at(pts, target.confirmed_at)
    assert visible_at(pts, None) == pts


def test_atr_filter_drops_small_swings():
    # 大腿夹小腿：小幅摆动应被 ATR 门槛滤掉
    df = zigzag([6, 6, 1, 1, 6, 6], step=10.0)
    loose = detect_swings(df, k=1, min_atr_mult=0)
    tight = detect_swings(df, k=1, min_atr_mult=3.0)
    assert len(tight) < len(loose)


def test_short_history_returns_no_swings():
    assert detect_swings(frame([1.0, 2.0, 3.0]), k=3) == []


def test_base_flag_requires_rally_and_volume():
    """无放量则不认定为突破基座，嵌套锚点 Fib 因此拿不到台阶。"""
    legs = [4, 4, 4, 4]
    quiet = zigzag(legs)
    quiet["volume"] = 1000.0
    pts = detect_swings(quiet, k=2, min_atr_mult=0)
    assert not any(p.is_base for p in pts)
