"""twelvedata provider 与主备切换逻辑测试（不联网）。"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

import pandas as pd
import pytest

from jta.data.fallback_provider import FallbackProvider, build_provider
from jta.data.provider import DataUnavailable, OHLCV, SeriesMeta
from jta.data.twelvedata_provider import TwelveDataAuthError, TwelveDataProvider

TZ = "America/New_York"


def _series(
    interval: str = "1d", *, too_old: bool = False, age_sessions: int | None = None,
    last_bar=None, source="fake",
) -> OHLCV:
    idx = pd.DatetimeIndex(
        list(pd.date_range("2026-08-18", periods=3, freq="D", tz=TZ)), name="Date"
    )
    df = pd.DataFrame(
        {"open": [1.0, 2.0, 3.0], "high": [1.5, 2.5, 3.5], "low": [0.5, 1.5, 2.5],
         "close": [1.2, 2.2, 3.2], "volume": [100.0, 200.0, 300.0]},
        index=idx,
    )
    if age_sessions is None:
        age_sessions = 5 if too_old else 0
    meta = SeriesMeta(
        symbol="X", interval=interval, adjust="back", tz=TZ, source=source,
        fetched_at=datetime.now(timezone.utc), too_old=too_old, age_sessions=age_sessions,
        last_bar=last_bar or idx[-1].to_pydatetime(), first_bar=idx[0].to_pydatetime(),
        rows=3, warnings=[],
    )
    return OHLCV(df=df, meta=meta)


class FakeProvider:
    def __init__(self, name: str, result=None, exc: Exception | None = None) -> None:
        self.name = name
        self._result = result
        self._exc = exc
        self.calls = 0

    def fetch(self, symbol, interval, **kwargs):
        self.calls += 1
        if self._exc is not None:
            raise self._exc
        return self._result


# ------------------------------------------------------------------ FallbackProvider


def test_fallback_returns_primary_when_fresh():
    primary = FakeProvider("p", _series())
    secondary = FakeProvider("s", _series())
    fb = FallbackProvider(primary, secondary)
    got = fb.fetch("X", "1d")
    assert got.meta.source == "fake"
    assert secondary.calls == 0


def test_fallback_switches_on_primary_failure():
    primary = FakeProvider("p", exc=DataUnavailable("挂了"))
    secondary = FakeProvider("s", _series(source="secondary"))
    fb = FallbackProvider(primary, secondary)
    got = fb.fetch("X", "1d")
    assert got.meta.source == "secondary"
    assert "p 不可用" in got.meta.warnings[0]


def test_fallback_raises_when_both_fail():
    primary = FakeProvider("p", exc=DataUnavailable("primary 挂了"))
    secondary = FakeProvider("s", exc=DataUnavailable("secondary 挂了"))
    fb = FallbackProvider(primary, secondary)
    with pytest.raises(DataUnavailable) as exc_info:
        fb.fetch("X", "1d")
    assert "primary 挂了" in str(exc_info.value)
    assert "secondary 挂了" in str(exc_info.value)


def test_fallback_switches_when_primary_too_old_and_secondary_fresher():
    old_bar = datetime(2026, 8, 1, tzinfo=timezone.utc)
    new_bar = datetime(2026, 9, 10, tzinfo=timezone.utc)
    primary = FakeProvider("p", _series(too_old=True, last_bar=old_bar))
    secondary = FakeProvider("s", _series(source="secondary", last_bar=new_bar))
    fb = FallbackProvider(primary, secondary)
    got = fb.fetch("X", "1d")
    assert got.meta.source == "secondary"
    assert "数据滞后" in got.meta.warnings[0]


def test_fallback_switches_on_one_session_lag_even_when_not_too_old():
    """收盘后好几小时，yfinance 仍停在上一交易日：这是 age=1、too_old=False，
    但正是用户反馈的"经常滞后"场景，必须能触发切换，不能被 too_old 的宽松阈值挡住。
    """
    old_bar = datetime(2026, 9, 11, tzinfo=timezone.utc)  # 上周五
    new_bar = datetime(2026, 9, 14, tzinfo=timezone.utc)  # 本周一，已收盘
    primary = FakeProvider("p", _series(too_old=False, age_sessions=1, last_bar=old_bar))
    secondary = FakeProvider("s", _series(source="secondary", age_sessions=0, last_bar=new_bar))
    fb = FallbackProvider(primary, secondary)
    got = fb.fetch("X", "1d")
    assert got.meta.source == "secondary"
    assert "数据滞后" in got.meta.warnings[0]


def test_fallback_skips_secondary_call_when_primary_already_current():
    primary = FakeProvider("p", _series(age_sessions=0))
    secondary = FakeProvider("s", _series(source="secondary"))
    fb = FallbackProvider(primary, secondary)
    fb.fetch("X", "1d")
    assert secondary.calls == 0


def test_fallback_keeps_primary_when_secondary_not_fresher():
    same_bar = datetime(2026, 8, 1, tzinfo=timezone.utc)
    primary = FakeProvider("p", _series(too_old=True, last_bar=same_bar))
    secondary = FakeProvider("s", _series(source="secondary", too_old=True, last_bar=same_bar))
    fb = FallbackProvider(primary, secondary)
    got = fb.fetch("X", "1d")
    assert got.meta.source == "fake"


def test_fallback_keeps_primary_when_secondary_errors_too():
    primary = FakeProvider("p", _series(too_old=True))
    secondary = FakeProvider("s", exc=RuntimeError("no key"))
    fb = FallbackProvider(primary, secondary)
    got = fb.fetch("X", "1d")
    assert got.meta.source == "fake"


def test_build_provider_selects_explicit_source():
    assert build_provider("yfinance").name == "yfinance"
    assert build_provider("twelvedata").name == "twelvedata"
    assert build_provider("auto").name == "auto"


# ------------------------------------------------------------------ TwelveDataProvider


def _payload(dates: list[str], closes: list[float]) -> dict[str, Any]:
    return {
        "status": "ok",
        "meta": {"symbol": "X", "exchange_timezone": "America/New_York"},
        "values": [
            {"datetime": d, "open": c, "high": c + 1, "low": c - 1, "close": c, "volume": "1000"}
            for d, c in zip(dates, closes)
        ],
    }


def test_twelvedata_requires_api_key(tmp_path):
    from jta.data.cache import ParquetCache

    p = TwelveDataProvider(cache=ParquetCache(tmp_path), api_key=None)
    with pytest.raises(DataUnavailable):
        p.fetch("X", "1d")


def test_twelvedata_parses_daily_bars(tmp_path, monkeypatch):
    from jta.data.cache import ParquetCache

    p = TwelveDataProvider(cache=ParquetCache(tmp_path), api_key="dummy")
    calls = []

    def fake_get(url, params=None, timeout=None):
        calls.append(params)
        payload = _payload(["2026-08-18", "2026-08-19", "2026-08-20"], [100.0, 101.0, 102.0])

        class R:
            def raise_for_status(self):
                return None

            def json(self):
                return payload

        return R()

    monkeypatch.setattr("jta.data.twelvedata_provider.requests.get", fake_get)
    got = p.fetch("X", "1d", as_of=datetime(2026, 8, 25, 20, 0, tzinfo=timezone.utc))
    assert got.meta.source == "twelvedata"
    assert str(got.df.index.tz) == TZ
    assert list(got.df["close"]) == [100.0, 101.0, 102.0]
    assert calls[0]["adjust"] == "all"


def test_twelvedata_downgrades_adjust_when_all_rejected(tmp_path, monkeypatch):
    from jta.data.cache import ParquetCache

    p = TwelveDataProvider(cache=ParquetCache(tmp_path), api_key="dummy")
    seen_adjust = []

    def fake_get(url, params=None, timeout=None):
        seen_adjust.append(params["adjust"])

        class R:
            def raise_for_status(self):
                return None

            def json(self):
                if params["adjust"] == "all":
                    return {"status": "error", "message": "需要更高计划"}
                return _payload(["2026-08-18"], [100.0])

        return R()

    monkeypatch.setattr("jta.data.twelvedata_provider.requests.get", fake_get)
    got = p.fetch("X", "1d", as_of=datetime(2026, 8, 25, 20, 0, tzinfo=timezone.utc))
    assert seen_adjust == ["all", "splits"]
    assert any("降级" in w for w in got.meta.warnings)


def test_twelvedata_4h_resamples_from_1h(tmp_path, monkeypatch):
    from jta.data.cache import ParquetCache

    p = TwelveDataProvider(cache=ParquetCache(tmp_path), api_key="dummy")

    def fake_get(url, params=None, timeout=None):
        assert params["interval"] == "1h"
        hours = ["09:30:00", "10:30:00", "11:30:00", "12:30:00", "13:30:00", "14:30:00"]
        payload = {
            "status": "ok",
            "values": [
                {"datetime": f"2026-08-18 {h}", "open": 1, "high": 2, "low": 0.5, "close": 1.5, "volume": "10"}
                for h in hours
            ],
        }

        class R:
            def raise_for_status(self):
                return None

            def json(self):
                return payload

        return R()

    monkeypatch.setattr("jta.data.twelvedata_provider.requests.get", fake_get)
    got = p.fetch("X", "4h", as_of=datetime(2026, 8, 25, 20, 0, tzinfo=timezone.utc))
    assert got.meta.bar_alignment is not None
    assert len(got.df) == 2  # 09:30-13:30 / 13:30-16:00
