"""4A 市值里程碑映射测试（不联网，除显式 mock 的 yfinance 调用外）。"""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
import pytest

from jta.levels.candidates import Candidate, note_market_cap_confluence
from jta.marketcap import (
    _bracket_anchors,
    _classify_weekly_state,
    _completed_weekly_closes,
    fetch_market_cap_context,
)

TZ = "America/New_York"


# ------------------------------------------------------------------ 整数锚点


def test_bracket_anchors_matches_upstream_example():
    # 上游原文例子：1T → 2T，0.382 中间位 1.382T
    below, above = _bracket_anchors(1.2e12)
    assert below == 1e12
    assert above == 2e12


def test_bracket_anchors_works_for_smaller_caps():
    below, above = _bracket_anchors(3.5e9)
    assert below == 2e9
    assert above == 5e9


def test_bracket_anchors_at_exact_ladder_point():
    # 市值恰好落在阶梯点上时，below 是自己，above 是下一档
    below, above = _bracket_anchors(2e12)
    assert below == 2e12
    assert above == 5e12


# ------------------------------------------------------------------ 周线聚合与状态机


def _daily_df(weekly_closes: list[float], *, last_week_partial_days: int = 0) -> pd.DataFrame:
    """构造整数周的合成日线：每周一到周五各一根，收盘价按周给定。

    last_week_partial_days>0 时，最后一周只保留前 N 根（模拟"这周还没走完"）。
    """
    rows = []
    # 从周一开始，每周 5 根工作日 bar
    start = pd.Timestamp("2026-08-03", tz=TZ)  # 一个周一
    for wi, close in enumerate(weekly_closes):
        week_start = start + pd.Timedelta(weeks=wi)
        days = 5
        if wi == len(weekly_closes) - 1 and last_week_partial_days:
            days = last_week_partial_days
        for d in range(days):
            ts = week_start + pd.Timedelta(days=d)
            rows.append(
                {
                    "ts": ts,
                    "open": close, "high": close + 1, "low": close - 1, "close": close,
                    "volume": 1000,
                }
            )
    df = pd.DataFrame(rows).set_index("ts")
    df.index.name = "Date"
    return df


def test_completed_weekly_closes_drops_the_still_open_week():
    df = _daily_df([100, 110, 120], last_week_partial_days=3)  # 最后一周只到周三
    weekly = _completed_weekly_closes(df)
    assert len(weekly) == 2
    assert list(weekly.values) == [100, 110]


def test_completed_weekly_closes_keeps_friday_closed_week():
    df = _daily_df([100, 110, 120])  # 每周都走满到周五
    weekly = _completed_weekly_closes(df)
    assert len(weekly) == 3
    assert list(weekly.values) == [100, 110, 120]


def test_classify_weekly_state_accepted():
    df = _daily_df([90, 105])  # 上周 90（未站上），这周 105（站上 100）
    state = _classify_weekly_state(df, target_price=100.0, tolerance=0.1)
    assert state["state"] == "周线接受"


def test_classify_weekly_state_lost_after_confirmation():
    df = _daily_df([105, 95])  # 上周站上，这周跌破
    state = _classify_weekly_state(df, target_price=100.0, tolerance=0.1)
    assert state["state"] == "确认后失守"


def test_classify_weekly_state_false_breakout():
    # 最近一个完整周收在 95（未站上），但盘中 high 摸到过 102
    df = _daily_df([80, 95])
    df.iloc[-1, df.columns.get_loc("high")] = 102.0
    state = _classify_weekly_state(df, target_price=100.0, tolerance=0.1)
    assert state["state"] == "初始周线假突破"


def test_classify_weekly_state_boundary_uncertain():
    df = _daily_df([100.05])
    state = _classify_weekly_state(df, target_price=100.0, tolerance=0.5)
    assert state["state"] == "边界不确定"


def test_classify_weekly_state_no_data():
    df = _daily_df([100], last_week_partial_days=2)  # 唯一一周还没走完
    state = _classify_weekly_state(df, target_price=100.0, tolerance=0.1)
    assert state["state"] == "数据不足"


def test_classify_weekly_state_flags_in_progress_attempt():
    # 上周未站上、这周（还没收盘）盘中已经摸到目标价
    df = _daily_df([90, 95], last_week_partial_days=3)
    df.iloc[-1, df.columns.get_loc("high")] = 101.0
    state = _classify_weekly_state(df, target_price=100.0, tolerance=0.1)
    assert state["state"] == "未触及"  # 上周的正式状态不受影响
    assert state["in_progress_attempt"] is True


# ------------------------------------------------------------------ fetch_market_cap_context 门槛


class _FakeTicker:
    def __init__(self, info: dict) -> None:
        self._info = info

    def get_info(self) -> dict:
        return self._info


def _mk_df(close: float = 200.0) -> pd.DataFrame:
    idx = pd.DatetimeIndex(
        list(pd.date_range("2026-08-03", periods=25, freq="B", tz=TZ)), name="Date"
    )
    return pd.DataFrame(
        {"open": close, "high": close + 1, "low": close - 1, "close": close, "volume": 1000},
        index=idx,
    )


def test_fetch_market_cap_context_disabled_for_historical_replay(monkeypatch):
    df = _mk_df()
    as_of = datetime(2020, 1, 1, tzinfo=timezone.utc)
    ctx = fetch_market_cap_context("X", df, as_of=as_of)
    assert ctx["available"] is False
    assert "历史" in ctx["reason"]


def test_fetch_market_cap_context_disabled_without_shares(monkeypatch):
    monkeypatch.setattr(
        "yfinance.Ticker", lambda symbol: _FakeTicker({"sharesOutstanding": None})
    )
    ctx = fetch_market_cap_context("QQQ", _mk_df())
    assert ctx["available"] is False


def test_fetch_market_cap_context_disabled_for_etf(monkeypatch):
    # ETF 有份额数但不是公司市值——上游原文明确写了 ETF 默认跳过
    monkeypatch.setattr(
        "yfinance.Ticker",
        lambda symbol: _FakeTicker(
            {"sharesOutstanding": 7_650_000, "currency": "USD", "quoteType": "ETF"}
        ),
    )
    ctx = fetch_market_cap_context("SOXX", _mk_df())
    assert ctx["available"] is False
    assert "ETF" in ctx["reason"]


def test_fetch_market_cap_context_disabled_for_non_usd(monkeypatch):
    monkeypatch.setattr(
        "yfinance.Ticker",
        lambda symbol: _FakeTicker({"sharesOutstanding": 1_000_000_000, "currency": "KRW"}),
    )
    ctx = fetch_market_cap_context("X", _mk_df())
    assert ctx["available"] is False
    assert "KRW" in ctx["reason"]


def test_fetch_market_cap_context_computes_anchors(monkeypatch):
    # 现价 200，股数 60 亿 → 市值 1.2T → below=1T, above=2T, mid=1.382T
    monkeypatch.setattr(
        "yfinance.Ticker",
        lambda symbol: _FakeTicker({"sharesOutstanding": 6_000_000_000, "currency": "USD"}),
    )
    ctx = fetch_market_cap_context("X", _mk_df(close=200.0))
    assert ctx["available"] is True
    labels = {a["label"]: a["equivalent_price"] for a in ctx["anchors"]}
    assert labels["低位整数锚点"] == pytest.approx(1e12 / 6e9)
    assert labels["高位整数锚点"] == pytest.approx(2e12 / 6e9)
    assert labels["0.382 中间位"] == pytest.approx(1.382e12 / 6e9)


# ------------------------------------------------------------------ 与候选位的共振标注


def _candidate(price: float, side: str = "support") -> Candidate:
    return Candidate(price=price, side=side, sources=[{"family": "fib", "label": "x"}])


def test_note_market_cap_confluence_tags_nearby_candidate():
    ctx = {
        "available": True,
        "anchors": [
            {"label": "低位整数锚点", "market_cap": 1e12, "equivalent_price": 100.0,
             "weekly": {"state": "周线接受"}},
        ],
    }
    near = _candidate(100.3)
    far = _candidate(200.0)
    note_market_cap_confluence([near, far], atr_value=2.0, market_cap_context=ctx)
    assert near.market_cap_note is not None
    assert near.market_cap_note["weekly_state"] == "周线接受"
    assert far.market_cap_note is None


def test_note_market_cap_confluence_noop_when_unavailable():
    c = _candidate(100.0)
    note_market_cap_confluence([c], atr_value=2.0, market_cap_context={"available": False})
    assert c.market_cap_note is None
