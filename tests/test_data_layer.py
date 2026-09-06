"""缓存与 provider 契约测试（不联网）。"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from jta.data.cache import ParquetCache
from jta.data.provider import OHLCV, SeriesMeta, normalize_index, truncate_as_of

TZ = "America/New_York"


def sample_df(n: int = 5) -> pd.DataFrame:
    # 显式丢弃 freq：真实行情（含节假日缺口）从来不是规则频率
    idx = pd.DatetimeIndex(list(pd.date_range("2026-08-18 09:30", periods=n, freq="D", tz=TZ)))
    return pd.DataFrame(
        {
            "open": range(100, 100 + n),
            "high": range(101, 101 + n),
            "low": range(99, 99 + n),
            "close": range(100, 100 + n),
            "volume": [1000] * n,
        },
        index=pd.DatetimeIndex(idx, name="Date"),
        dtype="float64",
    )


# ------------------------------------------------------------------ cache


def test_cache_roundtrip_preserves_tz_and_values(tmp_path):
    c = ParquetCache(tmp_path)
    key = c.key("yfinance", "MU", "1d", "back")
    df = sample_df()
    c.write(key, df, {"fetched_at": datetime.now(timezone.utc).isoformat(), "splits": []})
    got = c.read(key)
    assert got is not None
    back, meta = got
    assert str(back.index.tz) == TZ
    pd.testing.assert_frame_equal(normalize_index(back), normalize_index(df))
    assert meta["splits"] == []


def test_cache_miss_returns_none(tmp_path):
    c = ParquetCache(tmp_path)
    assert c.read(c.key("yfinance", "NOPE", "1d", "back")) is None
    assert c.age_seconds("nope") is None
    assert c.is_fresh("nope", "1d") is False


def test_cache_freshness_uses_interval_ttl(tmp_path):
    c = ParquetCache(tmp_path)
    key = c.key("yfinance", "MU", "60m", "back")
    old = datetime.now(timezone.utc) - timedelta(hours=2)
    c.write(key, sample_df(), {"fetched_at": old.isoformat(), "splits": []})
    assert c.is_fresh(key, "60m") is False
    c.write(key, sample_df(), {"fetched_at": datetime.now(timezone.utc).isoformat(), "splits": []})
    assert c.is_fresh(key, "60m") is True


def test_corrupt_cache_is_treated_as_miss(tmp_path):
    c = ParquetCache(tmp_path)
    key = c.key("yfinance", "MU", "1d", "back")
    c.write(key, sample_df(), {"fetched_at": datetime.now(timezone.utc).isoformat()})
    (tmp_path / f"{key}.parquet").write_bytes(b"not a parquet file")
    assert c.read(key) is None


def test_cache_key_sanitises_symbol(tmp_path):
    c = ParquetCache(tmp_path)
    assert "/" not in c.key("yfinance", "BRK/B", "1d", "back")


# ------------------------------------------------------------------ as_of


def test_truncate_as_of_is_inclusive_and_tz_safe():
    df = sample_df(5)
    cutoff = df.index[2]
    assert len(truncate_as_of(df, cutoff)) == 3
    # naive datetime 按 UTC 解释，不得抛错
    naive = datetime(2026, 8, 20, 23, 59)
    assert len(truncate_as_of(df, naive)) >= 1
    assert truncate_as_of(df, None) is df


def test_truncate_as_of_before_history_returns_empty():
    df = sample_df()
    assert truncate_as_of(df, datetime(2000, 1, 1, tzinfo=timezone.utc)).empty


def test_normalize_index_converts_ms_to_ns():
    df = sample_df()
    ms = df.copy()
    ms.index = ms.index.as_unit("ms")
    assert normalize_index(ms).index.unit == "ns"


# ------------------------------------------------------------------ OHLCV 契约


def meta() -> SeriesMeta:
    return SeriesMeta(
        symbol="MU",
        interval="1d",
        adjust="back",
        tz=TZ,
        source="test",
        fetched_at=datetime.now(timezone.utc),
    )


def test_ohlcv_rejects_naive_index():
    df = sample_df()
    df.index = df.index.tz_localize(None)
    with pytest.raises(ValueError):
        OHLCV(df=df, meta=meta())


def test_ohlcv_rejects_missing_columns():
    with pytest.raises(ValueError):
        OHLCV(df=sample_df().drop(columns=["volume"]), meta=meta())


def test_ohlcv_rejects_unsorted_index():
    df = sample_df().iloc[::-1]
    with pytest.raises(ValueError):
        OHLCV(df=df, meta=meta())


def test_series_meta_serialises_datetimes():
    d = meta().to_dict()
    assert isinstance(d["fetched_at"], str)
    assert d["as_of"] is None
    json.dumps(d)  # 必须可直接进 JSON 报告


# ------------------------------------------------------------------ 未完成 bar


def test_bar_end_respects_session_close():
    from jta.data.provider import bar_end

    day = pd.Timestamp("2026-08-25 00:00", tz=TZ)
    assert bar_end(day, "1d") == pd.Timestamp("2026-08-25 16:00", tz=TZ)
    # 下午的 4H bar 只到收盘，不是满 4 小时
    pm = pd.Timestamp("2026-08-25 13:30", tz=TZ)
    assert bar_end(pm, "4h") == pd.Timestamp("2026-08-25 16:00", tz=TZ)
    am = pd.Timestamp("2026-08-25 09:30", tz=TZ)
    assert bar_end(am, "4h") == pd.Timestamp("2026-08-25 13:30", tz=TZ)


def test_intraday_run_marks_today_bar_incomplete():
    from jta.data.provider import is_bar_complete

    ts = pd.Timestamp("2026-08-25 00:00", tz=TZ)
    mid_session = datetime(2026, 8, 25, 13, 51, tzinfo=timezone.utc)  # 09:51 ET
    after_close = datetime(2026, 8, 25, 20, 30, tzinfo=timezone.utc)  # 16:30 ET
    assert is_bar_complete(ts, "1d", mid_session) is False
    assert is_bar_complete(ts, "1d", after_close) is True


def test_split_incomplete_removes_only_the_live_bar():
    """盘中的当日 bar 会污染 ATR、摆动点与枢轴，必须移出结构计算。"""
    from jta.data.provider import split_incomplete

    df = sample_df(3)
    df.index = pd.DatetimeIndex(
        [
            pd.Timestamp("2026-08-21 00:00", tz=TZ),
            pd.Timestamp("2026-08-24 00:00", tz=TZ),
            pd.Timestamp("2026-08-25 00:00", tz=TZ),
        ]
    )
    mid_session = datetime(2026, 8, 25, 13, 51, tzinfo=timezone.utc)
    kept, live = split_incomplete(df, "1d", mid_session)
    assert len(kept) == 2
    assert live is not None and live["ts"].startswith("2026-08-25")
    assert live["close"] == df["close"].iloc[-1]

    kept2, live2 = split_incomplete(df, "1d", datetime(2026, 8, 26, 20, 0, tzinfo=timezone.utc))
    assert len(kept2) == 3 and live2 is None


def test_split_incomplete_handles_empty_frame():
    from jta.data.provider import split_incomplete

    empty = sample_df(0)
    kept, live = split_incomplete(empty, "1d")
    assert kept.empty and live is None
