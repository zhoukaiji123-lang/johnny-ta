"""A/B/C 回测的两处口径锁定（不联网）：计划日当天的 4H 不参与成交；回放中 4H 与日线同口径。"""

from __future__ import annotations

import numpy as np
import pandas as pd

from jta.backtest import TRIGGERED, TradeRules, collect_plans, run_backtest, simulate
from jta.data.provider import OHLCV
from jta.data.resample import align_to_daily
from jta.forward import ReplayProvider
from tests.test_backtest import daily_frame, plan, wick
from tests.test_backtest_brooks import _replay_from_daily, sessions

TZ = "America/New_York"
FLAT = (130, 131, 129, 130)


def test_abc_plans_are_timestamped_after_the_close():
    """计划用了当天收盘，plan_ts 必须晚于当天最后一根 4H。"""
    p = _replay_from_daily()
    idx = p.fetch("TEST", "4h").df.index
    items = collect_plans("TEST", p, start="2025-01-01", end="2025-06-30")
    assert items
    for it in items:
        before, after = idx[idx <= it["ts"]], idx[idx > it["ts"]]
        # ts 落在两个交易时段之间：它之前最后一根与之后第一根不属于同一天
        assert before[-1].date() != after[0].date()


def test_plan_day_bars_never_trigger_an_entry():
    """计划日当天下探入场位并收出长下影，之后再不回来：不能算成交。"""
    rows = [wick(101, 101.5), (101.5, 102, 100.5, 101.8)] + [FLAT] * 20
    four = sessions(rows, start="2026-01-02")
    after_close = pd.Timestamp("2026-01-02 23:00", tz=TZ)
    r = simulate(plan(), four, daily_frame(), after_close, symbol="T",
                 index_bullish=True, rules=TradeRules())
    assert r.outcome == "not_triggered" and r.entry_fill is None
    # 对照：按旧口径（当天 00:00）会把当天的下影算成触及 + 确认
    old = simulate(plan(), four, daily_frame(), after_close.normalize(), symbol="T",
                   index_bullish=True, rules=TradeRules())
    assert old.entry_fill is not None


def test_backtest_entries_never_land_on_the_plan_day():
    df = run_backtest([("TEST", None)], _replay_from_daily(), start="2025-01-01", end="2025-12-31")
    trig = df[df["outcome"].isin(TRIGGERED)]
    assert len(trig)                      # 不能是空跑
    plan_day = pd.to_datetime(trig["plan_date"], utc=True).dt.tz_convert(TZ).dt.date
    entry_day = pd.to_datetime(trig["entry_date"], utc=True).dt.tz_convert(TZ).dt.date
    assert (entry_day > plan_day).all()


def _unadjusted_pair(scale: float = 1.08) -> tuple[OHLCV, OHLCV]:
    """日线已做分红复权，4H 没做：4H 整体高 scale 倍。"""
    p = _replay_from_daily()
    d, f = p._series[("TEST", "1d")], p._series[("TEST", "4h")]
    raw = f.df.copy()
    for c in ("open", "high", "low", "close"):
        raw[c] = raw[c] * scale
    return d, OHLCV(df=raw, meta=f.meta)


def test_replay_provider_puts_4h_on_the_daily_scale():
    d, f = _unadjusted_pair()
    p = ReplayProvider({("TEST", "1d"): d, ("TEST", "4h"): f})
    as_of = pd.Timestamp("2025-03-14 23:00", tz=TZ)
    four = p.fetch("TEST", "4h", as_of=as_of).df
    daily = p.fetch("TEST", "1d", as_of=as_of).df
    last = four["close"].groupby(four.index.normalize()).last()
    dclose = pd.Series(daily["close"].to_numpy(), index=daily.index.normalize())
    assert np.allclose(last.to_numpy(), dclose.reindex(last.index).to_numpy())
    raw = ReplayProvider({("TEST", "1d"): d, ("TEST", "4h"): f}, align=False)
    assert np.allclose(raw.fetch("TEST", "4h", as_of=as_of).df["close"] / four["close"], 1.08)


def test_alignment_is_idempotent():
    d, f = _unadjusted_pair()
    once = align_to_daily(f.df, d.df)
    assert np.allclose(align_to_daily(once, d.df)[["open", "high", "low", "close"]],
                       once[["open", "high", "low", "close"]])


def test_single_bar_day_keeps_the_previous_factor():
    """只有 1 根 4H 的日子（半日市 / 缺下午数据）收盘对不上日线，不能拿来估因子。"""
    four = sessions([(110, 111, 109, 110.0), (110, 112, 110, 110.0)] * 2
                    + [(110, 111, 109, 105.0)])          # 第 3 天只有上午一根
    days = sorted({t.normalize() for t in four.index})
    daily = pd.DataFrame({"open": 100.0, "high": 102.0, "low": 99.0, "close": 100.0,
                          "volume": 1.0}, index=pd.DatetimeIndex(days))
    out = align_to_daily(four, daily)
    assert np.allclose(out["close"].iloc[:4], 100.0)
    assert np.isclose(out["close"].iloc[-1], 105.0 * 100.0 / 110.0)   # 沿用前一天的 100/110


def test_breakout_pullback_search_starts_after_the_confirming_close():
    """日线收盘站上后，确认日当天的 4H 不能算回踩成交——收盘确认要到 16:00 才成立。"""
    d = daily_frame()
    confirm_day = pd.Timestamp("2026-01-05", tz=TZ)
    d.loc[confirm_day] = {"open": 100.0, "high": 113.0, "low": 99.0, "close": 112.0, "volume": 1000.0}
    rows = [(111, 112, 103, 111.5), (111.5, 112, 110.5, 112)] + [(120, 121, 119, 120)] * 20
    four = sessions(rows, start="2026-01-05")
    p = plan(key="breakout", entry=110.0, stop=105.0, t1=130.0, t2=None)
    r = simulate(p, four, d, pd.Timestamp("2026-01-02 23:00", tz=TZ), symbol="T",
                 index_bullish=True, rules=TradeRules())
    assert r.entry_fill is None and "未触及" in r.note
