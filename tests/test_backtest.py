"""回测模拟的行为锁定。

这一层直接决定策略结论，任何一条判定规则松掉都会让期望值凭空好看。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from jta.backtest import TRIGGERED, TradeRules, simulate, summarise_trades

TZ = "America/New_York"


def series(rows: list[tuple[float, float, float, float]], start="2026-01-05 09:30", freq="4h"):
    """rows = [(open, high, low, close), ...]"""
    idx = pd.DatetimeIndex(list(pd.date_range(start, periods=len(rows), freq=freq, tz=TZ)))
    return pd.DataFrame(
        {"open": [r[0] for r in rows], "high": [r[1] for r in rows],
         "low": [r[2] for r in rows], "close": [r[3] for r in rows],
         "volume": [1000.0] * len(rows)},
        index=idx,
    )


def daily_frame(n=40, price=100.0, rng=4.0):
    idx = pd.DatetimeIndex(list(pd.date_range("2025-11-03", periods=n, freq="B", tz=TZ)))
    return pd.DataFrame(
        {"open": [price] * n, "high": [price + rng] * n, "low": [price - rng] * n,
         "close": [price] * n, "volume": [1000.0] * n},
        index=idx,
    )


def plan(key="aggressive", entry=100.0, stop=95.0, t1=110.0, t2=120.0):
    return {"key": key, "entry": entry, "stop": stop, "t1": t1, "t2": t2,
            "rr": 2.0, "executable": True}


def run(bars, *, p=None, rules=None, daily=None, plan_ts=None):
    intraday = series(bars)
    d = daily if daily is not None else daily_frame()
    ts = plan_ts or intraday.index[0] - pd.Timedelta(hours=4)
    return simulate(p or plan(), intraday, d, ts, symbol="T",
                    index_bullish=True, rules=rules or TradeRules())


# 长下影 = 止跌确认信号
def wick(o, c, low_extra=6.0):
    return (o, max(o, c) + 0.1, min(o, c) - low_extra, c)


def test_never_touched_is_not_triggered():
    r = run([(130, 132, 128, 131)] * 10)
    assert r.outcome == "not_triggered" and r.r_multiple is None


def test_plan_without_a_first_target_is_not_tradeable():
    """没有目标位就只能靠超时平仓，结果由持仓期长度决定而不是策略。"""
    r = run([wick(100, 101)] * 6, p=plan(t1=None, t2=None))
    assert r.outcome == "not_triggered"
    assert "第一目标" in r.note


def test_touch_without_confirmation_does_not_fill():
    # 一路阴跌，没有任何止跌信号
    r = run([(100, 100.2, 96, 96.5), (96, 96.2, 94, 94.5), (94, 94.2, 92, 92.5)])
    assert r.entry_fill is None
    assert "确认" in r.note or "跌破" in r.note


def test_limit_order_does_not_chase_a_gap():
    """确认后跳空高开，限价单不该成交——按开盘价成交会凭空多付一大截。"""
    bars = [wick(100, 100.5)] + [(140, 142, 139, 141)] * 5
    r = run(bars)
    assert r.entry_fill is None
    assert "未回到限价" in r.note


def test_limit_fills_at_the_limit_when_price_dips_back():
    bars = [wick(100, 100.5), (108, 109, 100.2, 108.5)] + [(108, 112, 107, 111)] * 6
    r = run(bars)
    assert r.entry_fill is not None
    # daily_frame 的日振幅是 8，故 ATR=8、限价容差 0.8
    assert r.entry_fill <= 100.0 + TradeRules().touch_tol_atr * 8.0 + 1e-9


def test_stop_loss_is_minus_one_r():
    bars = [wick(100, 100.5), (100, 100.5, 99.5, 100), (99, 99.5, 90, 91)]
    r = run(bars)
    assert r.outcome == "stopped"
    assert r.r_multiple == pytest.approx(-1.0, abs=0.05)


def test_gap_through_stop_loses_more_than_one_r():
    """跳空越过止损必须按开盘价结算，假设按止损价成交会系统性低估亏损。"""
    bars = [wick(100, 100.5), (100, 100.5, 99.5, 100), (85, 86, 84, 84.5)]
    r = run(bars)
    assert r.outcome == "stopped" and r.gapped
    assert r.r_multiple < -1.5


def test_t1_takes_half_and_trailing_stop_protects_the_rest():
    bars = [wick(100, 100.5), (100, 100.5, 99.8, 100), (105, 112, 104, 111), (111, 112, 95, 96)]
    r = run(bars)
    assert r.outcome == "t1_then_stop"
    assert r.r_multiple > 0            # T1 已落袋，剩余被跟踪止损保护
    assert r.r_multiple < 2.0


def test_trailing_stop_protects_profit_on_a_reversal():
    """只把止损抬到成本价、然后持有到超时，会把已有利润全数还回去。"""
    ramp = [wick(100, 100.5), (100, 100.5, 99.8, 100)] + [
        (100 + 8 * i, 108 + 8 * i, 99 + 8 * i, 107 + 8 * i) for i in range(1, 12)
    ]
    crash = [(190, 191, 100, 101), (101, 102, 99, 100)]     # 冲高后崩回起点
    bars = ramp + crash
    trail = run(bars, p=plan(t2=None), rules=TradeRules(trail_atr=2.0))
    none_ = run(bars, p=plan(t2=None), rules=TradeRules(trail_atr=0.0))
    assert trail.r_multiple > none_.r_multiple


def test_cancelled_when_price_breaks_the_stop_while_waiting_to_confirm():
    """触及后没等到确认、反而跌穿止损——计划作废，不再挂着等。"""
    # 首根刻意不带长下影，否则会被当成止跌确认而直接成交
    bars = [(100.5, 100.6, 99.5, 99.6), (99, 99.2, 90, 90.5), (90, 91, 89, 90)]
    r = run(bars, p=plan(entry=100.0, stop=95.0))
    assert r.outcome == "cancelled"
    assert "作废" in r.note


def test_breakout_requires_a_daily_close_above_the_level():
    d = daily_frame()                      # 日线收盘恒为 100，从未站上 110
    bars = [wick(110, 110.5)] * 6
    r = run(bars, p=plan(key="breakout", entry=110.0, stop=105.0, t1=120.0),
            daily=d)                       # 用默认 plan_ts，保证 ATR 有足够历史
    assert r.outcome == "not_triggered"
    assert "日线未收盘站上" in r.note


def test_holding_window_forces_an_exit():
    bars = [wick(100, 100.5), (100, 100.5, 99.9, 100)] + [(100, 100.4, 99.7, 100)] * 30
    r = run(bars, rules=TradeRules(max_holding_bars=6))
    assert r.outcome == "timeout"
    assert r.bars_held is not None and r.bars_held <= 6


def test_summary_reports_expectancy_with_its_standard_error():
    df = pd.DataFrame([
        {"executable": True, "plan_key": "aggressive", "index_bullish": True,
         "outcome": o, "r_multiple": v, "gapped": False}
        for o, v in [("stopped", -1.0), ("target", 2.0), ("stopped", -1.0), ("target", 3.0)]
    ])
    s = summarise_trades(df)
    assert s["overall"]["expectancy_r"] == pytest.approx(0.75)
    assert s["overall"]["stderr_r"] is not None      # 没有标准误的期望值不可解读
    assert s["overall"]["win_rate"] == 0.5
    assert any("滑点" in c for c in s["caveats"])


def test_summary_handles_no_trades():
    df = pd.DataFrame([{"executable": True, "plan_key": "deep", "index_bullish": None,
                        "outcome": "not_triggered", "r_multiple": None, "gapped": False}])
    s = summarise_trades(df)
    assert s["overall"]["triggered"] == 0 and s["overall"]["expectancy_r"] is None
