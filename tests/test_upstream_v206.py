"""上游 v2.0.4–v2.0.6 移植部分的行为锁定：proximity 选点、L 左侧支撑直入、方案状态、摆动序列。"""

from __future__ import annotations

import pandas as pd

from jta.analyze import select_key_levels, swing_sequence
from jta.backtest import TradeRules, simulate
from jta.indicators.swing import SwingPoint
from jta.levels.candidates import Candidate, make_source
from jta.plans import build_plans, sides
from tests.test_backtest import daily_frame, series

ATR = 10.0


def cand(price, hits, families=("fib",), reaction=False, now=100.0):
    c = Candidate(
        price=price, side="support", display=price,
        sources=[make_source(f, f, timeframe="1d") for f in families],
    )
    c.distance_atr = abs(now - price) / ATR
    factors = {"price_action": {"hit": reaction}}
    return c, {"hits": hits, "band": "", "factors": factors, "caveat": ""}


# ------------------------------------------------------------------ 选点


def lite_like():
    """近处结构命中数低、深处 Fib 密集区命中数高——LITE 2026-10-05 的形状。"""
    return [
        cand(99.0, 3, ("prev_session",)),        # 0.1 ATR，前日低点
        cand(97.0, 3, ("ema",)),                 # 无真实反应
        cand(91.0, 3, ("pivot",)),               # 0.9 ATR，摆动枢轴
        cand(84.0, 6, ("fib", "pivot")),         # 深处 Fib 密集区
        cand(78.0, 6, ("fib",)),
        cand(72.0, 6, ("fib",)),
    ]


def test_legacy_ranks_by_hits_and_pushes_s1_deep():
    picked, meta = select_key_levels(lite_like(), "support", ATR, 100.0, "legacy")
    assert meta["mode"] == "legacy"
    assert [c.display for c, _ in picked] == [84.0, 78.0, 72.0]


def test_proximity_keeps_near_structure_and_deep_coverage():
    picked, meta = select_key_levels(lite_like(), "support", ATR, 100.0, "proximity")
    assert meta["mode"] == "proximity"
    assert [c.display for c, _ in picked] == [99.0, 91.0, 84.0]
    assert [c.label for c, _ in picked] == ["S1", "S2", "S3"]


def test_proximity_skips_formula_only_levels_for_s1_s2():
    """S2 跳过只有 EMA、没有真实反应的 97，取 0.5 ATR 外最近的枢轴。"""
    picked, _ = select_key_levels(lite_like(), "support", ATR, 100.0, "proximity")
    assert 97.0 not in [c.display for c, _ in picked]


def test_proximity_price_action_counts_as_reaction():
    scored = [cand(98.0, 3, ("ema",)), cand(97.0, 3, ("fib",), reaction=True), cand(80.0, 6)]
    picked, _ = select_key_levels(scored, "support", ATR, 100.0, "proximity")
    assert picked[0][0].display == 97.0


def test_proximity_falls_back_when_nothing_has_a_reaction():
    scored = [cand(98.0, 3, ("ema",)), cand(90.0, 4, ("fib",)), cand(80.0, 6)]
    picked, _ = select_key_levels(scored, "support", ATR, 100.0, "proximity")
    assert [c.display for c, _ in picked] == [98.0, 90.0, 80.0]


def test_proximity_respects_min_hits_and_separation():
    scored = [cand(99.5, 2, ("pivot",)), cand(99.0, 3, ("pivot",)), cand(97.0, 3, ("pivot",))]
    picked, _ = select_key_levels(scored, "support", ATR, 100.0, "proximity")
    # 99.5 命中数不足；97 离 99 只有 0.2 ATR，不能成为独立一档
    assert [c.display for c, _ in picked] == [99.0]


# ------------------------------------------------------------------ L 方案


def level(label, price, distance, side="support", raw=None):
    return {"label": label, "display": price, "raw_price": raw if raw is not None else price,
            "distance_atr": distance, "confirmation": "c", "invalidation": "i", "side": side}


def plans_at(price, sups=None, ress=None, **kw):
    sups = sups if sups is not None else [level("S1", 100.0, 0.5, raw=100.23), level("S2", 90.0, 1.5)]
    ress = ress if ress is not None else [level("R1", 130.0, 2.5, "resistance"), level("R2", 140.0, 3.5, "resistance")]
    return {p["key"]: p for p in build_plans(sups, ress, atr=ATR, zone="at_support",
                                             current_price=price, **kw)}


def test_left_stop_is_one_percent_below_support_basis():
    left = plans_at(105.0)["left"]
    # S 取 min(展示值, 原值) = 100；0.99×100 = 99，不加 ATR 缓冲
    assert left["support_basis"] == 100.0
    assert left["stop"] == 99.0
    assert left["entry"] == 100.0
    assert left["validated"] is False
    assert left["rr"] == 30.0


def test_left_stop_rounds_toward_earlier_exit():
    left = plans_at(105.0, sups=[level("S1", 101.37, 0.4)])["left"]
    # 0.99 × 101.37 = 100.3563 → 向上取整 100.36，不向下扩大距离
    assert left["stop"] == 100.36


def test_plan_a_is_gone_and_l_takes_the_first_support():
    """2026-10-07 删除 A：它与 L 同档同入场价，回测里 87% 的成交与 L 重叠。"""
    p = plans_at(105.0)
    assert set(p) == {"left", "deep", "breakout"}
    assert p["left"]["entry_level"] == "S1"
    assert p["deep"]["entry_level"] == "S2"


def test_status_waiting_vs_ready():
    far = plans_at(110.0)
    assert far["left"]["status"] == "waiting_price"
    assert far["breakout"]["status"] == "waiting_trigger"
    at = plans_at(100.5)
    assert at["left"]["status"] == "ready"


def test_status_ineligible_and_no_data():
    blocked = plans_at(105.0, regime={"state": "down", "label": "向下", "reason": "x"})
    assert blocked["left"]["status"] == "ineligible"
    assert "大盘状态" in blocked["left"]["status_reason"]
    no_target = plans_at(105.0, ress=[])
    assert no_target["left"]["status"] == "no_data"


def test_sides_always_reports_both():
    sd = sides(list(plans_at(110.0).values()))
    assert sd["left"]["plan"] == "left" and sd["right"]["plan"] == "breakout"
    assert sd["left"]["status_label"] == "等待到位"
    empty = sides([])
    assert empty["left"]["status"] == "no_data" and empty["right"]["status"] == "no_data"


# ------------------------------------------------------------------ L 回测


def lplan(entry=100.0, stop=99.0, t1=110.0, t2=120.0):
    return {"key": "left", "entry": entry, "stop": stop, "t1": t1, "t2": t2,
            "rr": 10.0, "executable": True}


def sim(bars, p=None):
    intraday = series(bars)
    ts = intraday.index[0] - pd.Timedelta(hours=4)
    return simulate(p or lplan(), intraday, daily_frame(), ts, symbol="T",
                    index_bullish=True, rules=TradeRules())


def test_left_fills_at_support_without_confirmation():
    # 一路阴跌到 100、没有任何止跌信号——A 不会成交，L 直接成交
    r = sim([(103, 103.5, 100.0, 100.3), (100.3, 100.8, 99.6, 100.5)] + [(101, 111, 100.5, 110.5)] * 3)
    assert r.entry_fill == 100.0
    assert r.outcome in ("t1_then_timeout", "target", "t1_then_stop")


def test_left_same_bar_entry_and_stop_counts_as_stop():
    r = sim([(103, 103.5, 98.0, 98.5)] + [(98.5, 99, 98, 98.5)] * 3)
    assert r.entry_fill == 100.0
    assert r.outcome == "stopped"
    assert r.r_multiple == -1.0


def test_left_gap_below_stop_cancels():
    r = sim([(98.0, 98.5, 97.0, 97.5)] * 3)
    assert r.outcome == "cancelled" and r.entry_fill is None


def test_left_gap_between_entry_and_stop_fills_at_open():
    r = sim([(99.5, 99.8, 99.2, 99.6)] + [(99.6, 100, 99.3, 99.8)] * 2)
    assert r.entry_fill == 99.5


def test_fill_bar_high_before_limit_fill_does_not_count_as_target():
    """开在限价上方、先冲到 T1 再回落成交：T1 发生在成交之前，不能记成已兑现。"""
    r = sim([(105, 111, 100.0, 100.5)] + [(100.5, 101, 100.2, 100.6)] * 3)
    assert r.entry_fill == 100.0
    assert r.outcome == "timeout"


# ------------------------------------------------------------------ 摆动序列


def sp(day, kind, price):
    ts = pd.Timestamp(f"2026-01-{day:02d}", tz="America/New_York")
    return SwingPoint(ts=ts, kind=kind, price=price, confirmed_at=ts + pd.Timedelta(days=3),
                      bar_index=day)


def test_swing_sequence_labels():
    swings = [sp(1, "low", 90), sp(3, "high", 110), sp(5, "low", 95), sp(7, "high", 120),
              sp(9, "low", 94.5), sp(11, "high", 115)]
    seq = swing_sequence(swings, atr_value=10.0, n=6)
    assert [x["label"] for x in seq] == [None, None, "HL", "HH", "EL", "LH"]
    assert seq[4]["label_text"] == "等低"


def test_swing_sequence_keeps_last_n():
    swings = [sp(i, "low" if i % 2 else "high", 100 + i) for i in range(1, 15)]
    assert len(swing_sequence(swings, atr_value=1.0, n=6)) == 6
