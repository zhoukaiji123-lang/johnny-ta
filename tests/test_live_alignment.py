"""实盘数据层的口径锁定（不联网）：4H 复权口径、量能分时段、财报反应日。"""

from __future__ import annotations

import numpy as np
import pandas as pd

from jta.analyze import analyze, fetch_timeframes
from jta.data.provider import OHLCV
from jta.events import _reaction_session
from jta.forward import ReplayProvider
from jta.indicators.price_action import bar_signals
from tests.test_backtest_brooks import _replay_from_daily, sessions

TZ = "America/New_York"


def _raw_provider(scale: float) -> ReplayProvider:
    """不经 ReplayProvider 对齐，模拟实盘：4H 比日线整体高 scale 倍。"""
    base = _replay_from_daily()._series
    f = base[("TEST", "4h")]
    df = f.df.copy()
    for c in ("open", "high", "low", "close"):
        df[c] = df[c] * scale
    return ReplayProvider({("TEST", "1d"): base[("TEST", "1d")],
                           ("TEST", "4h"): OHLCV(df=df, meta=f.meta)}, align=False)


def test_fetch_timeframes_puts_4h_on_the_daily_scale():
    daily, four = fetch_timeframes(_raw_provider(1.08), "TEST")
    last = four.df["close"].groupby(four.df.index.normalize()).last()
    dclose = pd.Series(daily.df["close"].to_numpy(), index=daily.df.index.normalize())
    assert np.allclose(last.to_numpy(), dclose.reindex(last.index).to_numpy())


def test_analyze_is_invariant_to_the_4h_adjustment_basis():
    """4H 复权基准不同，关键位与计划不该跟着变。"""
    as_of = pd.Timestamp("2025-10-15 23:00", tz=TZ)
    kw = dict(as_of=as_of, include_events=False, include_market_cap=False)
    a = analyze("TEST", provider=_raw_provider(1.0), **kw)
    b = analyze("TEST", provider=_raw_provider(1.08), **kw)
    for side in ("supports", "resistances"):
        assert [round(x["raw_price"], 2) for x in a[side]] == [round(x["raw_price"], 2) for x in b[side]]
    assert [p["entry"] for p in a["plans"]] == [p["entry"] for p in b["plans"]]
    assert "intraday_scale" in a["data"]


def test_afternoon_4h_bar_is_not_flagged_as_volume_dry_up_by_construction():
    """下午那根 4H 只有 2.5 小时、量天然更小：只和同一时段比，不能因此被标成缩量。"""
    rows = [(100, 101, 99, 100.5), (100.5, 101, 100, 100.2)] * 40
    four = sessions(rows)
    four["volume"] = np.where(four.index.hour == 9, 1000.0, 600.0)
    sig = bar_signals(four)
    assert not sig["volume_dry_up"].any() and not sig["volume_spike"].any()


def test_volume_baseline_excludes_the_current_bar():
    idx = pd.DatetimeIndex(pd.date_range("2026-01-05", periods=30, freq="B", tz=TZ))
    df = pd.DataFrame({"open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0,
                       "volume": [100.0] * 29 + [150.0]}, index=idx)
    # 150 >= 1.5 × 前 20 根均量 100；把自己算进均量（102.5）就到不了门槛
    assert bool(bar_signals(df)["volume_spike"].iloc[-1])


def test_earnings_reaction_session_depends_on_release_time():
    idx = pd.DatetimeIndex(pd.date_range("2026-01-26", periods=5, freq="B", tz=TZ))
    bmo = pd.Timestamp("2026-01-28 07:00", tz=TZ)
    amc = pd.Timestamp("2026-01-28 16:05", tz=TZ)
    date_only = pd.Timestamp("2026-01-28", tz=TZ)
    assert idx[_reaction_session(idx, bmo)].date().isoformat() == "2026-01-28"
    assert idx[_reaction_session(idx, amc)].date().isoformat() == "2026-01-29"
    assert idx[_reaction_session(idx, date_only)].date().isoformat() == "2026-01-29"
