"""4H 切分规则的回归测试。

时间戳偏移和精度单位是这一层最容易静默出错的地方：错了不会抛异常，
只会让后面所有 Fib / EMA / 确认口径落在错误的 bar 上。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from jta.data.resample import audit_4h, bars_per_day, to_4h

TZ = "America/New_York"


def make_60m(days: list[str], bars_per_session: int = 7, start_hm: str = "09:30") -> pd.DataFrame:
    rows, idx = [], []
    price = 100.0
    for d in days:
        open_ts = pd.Timestamp(f"{d} {start_hm}", tz=TZ)
        n = bars_per_session if isinstance(bars_per_session, int) else bars_per_session[d]
        for i in range(n):
            ts = open_ts + pd.Timedelta(hours=i)
            idx.append(ts)
            rows.append(
                {
                    "open": price,
                    "high": price + 2,
                    "low": price - 1,
                    "close": price + 1,
                    "volume": 100 * (i + 1),
                }
            )
            price += 1
    return pd.DataFrame(rows, index=pd.DatetimeIndex(idx, name="Datetime"))


def test_regular_session_yields_two_bars_at_0930_and_1330():
    df = to_4h(make_60m(["2026-08-24", "2026-08-25"]))
    assert len(df) == 4
    times = sorted({t.strftime("%H:%M") for t in df.index})
    assert times == ["09:30", "13:30"]
    assert df.index.tz is not None and str(df.index.tz) == TZ


def test_ohlcv_aggregation_matches_constituent_bars():
    src = make_60m(["2026-08-24"])
    out = to_4h(src)
    first = src.iloc[:4]  # 09:30-12:30 归入 09:30 bar
    bar = out.iloc[0]
    assert bar["open"] == first["open"].iloc[0]
    assert bar["high"] == first["high"].max()
    assert bar["low"] == first["low"].min()
    assert bar["close"] == first["close"].iloc[-1]
    assert bar["volume"] == first["volume"].sum()


def test_half_day_session_yields_single_bar():
    # 半日市 09:30-13:00，只有 4 根 60m bar（含 12:30 那根）
    df = to_4h(make_60m(["2026-11-27"], bars_per_session=4))
    assert len(df) == 1
    assert df.index[0].strftime("%H:%M") == "09:30"
    assert audit_4h(df) and "半日市" in audit_4h(df)[0]


def test_millisecond_precision_input_is_normalised():
    """datetime64[ms] 输入必须得到与 ns 输入完全相同的结果。

    yfinance 1.6 与 parquet 往返都会给出 ms 精度；若按 ns 解释整数时间戳，
    时间会塌缩到 1970 年附近而不会报错。
    """
    src = make_60m(["2026-08-24", "2026-08-25"])
    ms = src.copy()
    ms.index = ms.index.as_unit("ms")
    assert ms.index.unit == "ms"
    pd.testing.assert_frame_equal(to_4h(ms), to_4h(src))
    assert to_4h(ms).index[0].year == 2026


def test_timestamps_are_bar_start_not_bar_end():
    df = to_4h(make_60m(["2026-08-24"]))
    assert df.index[0] == pd.Timestamp("2026-08-24 09:30", tz=TZ)
    assert df.index[1] == pd.Timestamp("2026-08-24 13:30", tz=TZ)


def test_unsorted_input_is_handled():
    src = make_60m(["2026-08-24", "2026-08-25"])
    shuffled = src.iloc[np.random.RandomState(0).permutation(len(src))]
    pd.testing.assert_frame_equal(to_4h(shuffled), to_4h(src))


def test_dst_boundary_day_keeps_local_session_anchor():
    # 2026-11-01 美国夏令时结束；11-02 是切换后第一个交易日
    df = to_4h(make_60m(["2026-10-30", "2026-11-02"]))
    assert {t.strftime("%H:%M") for t in df.index} == {"09:30", "13:30"}
    assert bars_per_day(df).tolist() == [2, 2]


def test_empty_input_returns_empty():
    empty = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    empty.index = pd.DatetimeIndex([], tz=TZ)
    assert to_4h(empty).empty


def test_naive_index_rejected():
    src = make_60m(["2026-08-24"])
    src.index = src.index.tz_localize(None)
    with pytest.raises(ValueError):
        to_4h(src)
