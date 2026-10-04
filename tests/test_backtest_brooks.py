"""Brooks 计划 D 回测的行为锁定（不联网）。"""

from __future__ import annotations

import numpy as np
import pandas as pd

from jta.backtest import (
    align_to_daily,
    TRIGGERED,
    TradeRules,
    distance_matched,
    run_brooks_backtest,
    simulate_brooks,
    summarise_brooks,
)
from jta.forward import ReplayProvider
from tests.test_backtest import daily_frame
from tests.test_pipeline import FakeProvider

TZ = "America/New_York"


def sessions(rows: list[tuple[float, float, float, float]], start="2026-01-05") -> pd.DataFrame:
    """真实交易时段：每个交易日两根 4H（09:30 / 13:30）。rows 两两一天。"""
    days = pd.date_range(start, periods=(len(rows) + 1) // 2, freq="B", tz=TZ)
    idx = []
    for d in days:
        idx += [d + pd.Timedelta(hours=9, minutes=30), d + pd.Timedelta(hours=13, minutes=30)]
    idx = pd.DatetimeIndex(idx[: len(rows)])
    o, h, l, c = zip(*rows)
    return pd.DataFrame({"open": o, "high": h, "low": l, "close": c,
                         "volume": [1000.0] * len(rows)}, index=idx)


def plan(order="stop", entry=101.0, stop=97.0, t1=105.0, t2=None):
    return {"key": "brooks", "setup_code": "A2", "order_type": order, "entry": entry,
            "stop": stop, "t1": t1, "t2": t2, "rr": 1.0, "executable": True}


#: 计划日 = 2026-01-02（周五）收盘后
PLAN_TS = pd.Timestamp("2026-01-02 23:00", tz=TZ)


def sim(rows, p):
    return simulate_brooks(p, sessions(rows), daily_frame(), PLAN_TS, symbol="T", rules=TradeRules())


FLAT = (100, 100.5, 99.5, 100)


def test_stop_order_fills_on_break_and_gap_fills_at_open():
    r = sim([(100, 101.5, 99.8, 101.2), FLAT] + [(104, 106, 103.5, 105.5)] * 4, plan())
    assert r.entry_fill == 101.0 and r.outcome in TRIGGERED
    gap = sim([(102.5, 103, 102, 102.8), FLAT] + [(104, 106, 103.5, 105.5)] * 4, plan())
    assert gap.entry_fill == 102.5          # 跳空高开越过入场价，按开盘价成交


def test_order_only_lives_for_the_next_session():
    """Brooks 的信号 K 线每天在变：次日没成交就作废，第三天再涨也不算。"""
    r = sim([FLAT, FLAT, (104, 106, 103, 105)] + [FLAT] * 4, plan())
    assert r.outcome == "not_triggered" and "未成交" in r.note


def test_limit_order_fills_at_better_of_open_and_limit_and_skips_gap_below_stop():
    r = sim([(100, 100.2, 98.5, 99), FLAT] + [(104, 106, 103, 105)] * 4, plan("limit", entry=99.0, stop=96.0))
    assert r.entry_fill == 99.0
    gapdown = sim([(95, 95.5, 94, 95), FLAT] + [FLAT] * 4, plan("limit", entry=99.0, stop=96.0))
    assert gapdown.entry_fill is None and "止损位之下" in gapdown.note


def test_market_order_fills_at_next_open():
    r = sim([(100.3, 101, 100, 100.8), FLAT] + [(104, 106, 103, 105)] * 4, plan("market"))
    assert r.entry_fill == 100.3


def test_plan_day_bars_never_fill():
    """计划用了当天收盘，当天盘中的 4H 不能参与成交。"""
    rows = sessions([(100, 103, 99, 102), (102, 103, 101, 102)] + [FLAT] * 6, start="2026-01-02")
    r = simulate_brooks(plan(), rows, daily_frame(), PLAN_TS, symbol="T", rules=TradeRules())
    assert r.entry_fill is None


def test_control_shifts_every_price_by_the_same_offset():
    p = plan(t2=110.0)
    q = distance_matched(p, atr=4.0, rng=np.random.RandomState(0))
    off = q["control_offset"]
    assert 2.0 <= abs(off) <= 4.0
    for k in ("entry", "stop", "t1", "t2"):
        assert abs((q[k] - p[k]) - off) < 1e-9
    assert q["order_type"] == p["order_type"]


def _replay_from_daily() -> ReplayProvider:
    """FakeProvider 的 4H 只覆盖到 2024 年中；用日线拆成每天两根 4H，保证回测窗口有成交数据。"""
    import dataclasses

    from jta.data.provider import OHLCV

    d = FakeProvider().fetch("TEST", "1d")
    df, rows, idx = d.df, [], []
    for ts, r in df.iterrows():
        day = ts.normalize()
        mid = (r["open"] + r["close"]) / 2
        rows += [(r["open"], r["high"], min(r["open"], mid) - 0.2, mid),
                 (mid, max(mid, r["close"]) + 0.2, r["low"], r["close"])]
        idx += [day + pd.Timedelta(hours=9, minutes=30), day + pd.Timedelta(hours=13, minutes=30)]
    o, h, l, c = zip(*rows)
    four = pd.DataFrame({"open": o, "high": h, "low": l, "close": c, "volume": 500.0},
                        index=pd.DatetimeIndex(idx))
    meta = dataclasses.replace(d.meta, interval="4h", rows=len(four))
    return ReplayProvider({("TEST", "1d"): d, ("TEST", "4h"): OHLCV(df=four, meta=meta)})


def test_runner_never_holds_two_positions_and_summarises():
    df = run_brooks_backtest([("TEST", None)], _replay_from_daily(), start="2025-01-01",
                             end="2025-12-31", seeds=(0, 1))
    trig = df[df["outcome"].isin(TRIGGERED)]
    assert len(trig[trig["stream"] == "real_all"]) > 0      # 不能是空跑
    assert set(df["stream"]) <= {"real_all", "real_exe", "ctrl_0", "ctrl_1"}
    for _, g in trig.groupby("stream"):
        g = g.sort_values("entry_date")
        entries = pd.to_datetime(g["entry_date"], utc=True).tolist()
        exits = pd.to_datetime(g["exit_date"], utc=True).tolist()
        assert all(entries[i + 1] >= exits[i] for i in range(len(g) - 1))
    s = summarise_brooks(df, start="2025-01-01", end="2025-12-31")
    assert "merge_recommended" in s and set(s["criteria"]) == {
        "1_real_positive", "2_beats_control", "3_both_halves", "4_regime_gate_needed"}


def test_intraday_is_scaled_to_the_daily_adjustment():
    """4H 未做分红复权时整体偏高：按当天日线收盘缩放后，两边同口径。"""
    four = sessions([(110, 111, 109, 110.5), (110.5, 112, 110, 111.1)] * 3)
    days = sorted({t.normalize() for t in four.index})
    daily = pd.DataFrame({"open": 100.0, "high": 102.0, "low": 99.0, "close": 101.0,
                          "volume": 1.0}, index=pd.DatetimeIndex(days))
    out = align_to_daily(four, daily)
    last = out["close"].groupby(out.index.normalize()).last()
    assert np.allclose(last.to_numpy(), 101.0)
    assert np.allclose(out["high"] / four["high"], 101.0 / 111.1)
