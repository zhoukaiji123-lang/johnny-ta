"""前向记录与评估测试。

这一层的正确性直接决定验证结论可不可信，因此 hold/break/untested 的判定
必须用手工构造的、结果确定的行情来钉死。
"""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from jta.data.provider import OHLCV, SeriesMeta
from jta.forward import (
    MIN_SAMPLES,
    ReplayProvider,
    _compare,
    _rate,
    evaluate,
    format_summary,
    random_controls,
    summarise,
)

TZ = "America/New_York"


def bars(rows: list[tuple[float, float, float]], start: str = "2026-01-05") -> pd.DataFrame:
    """rows = [(high, low, close), ...]"""
    idx = pd.DatetimeIndex(
        list(pd.date_range(start, periods=len(rows), freq="B", tz=TZ)), name="Date"
    )
    return pd.DataFrame(
        {
            "open": [r[2] for r in rows],
            "high": [r[0] for r in rows],
            "low": [r[1] for r in rows],
            "close": [r[2] for r in rows],
            "volume": [1000.0] * len(rows),
        },
        index=idx,
    )


def provider_from(df: pd.DataFrame, symbol: str = "TEST") -> ReplayProvider:
    meta = SeriesMeta(
        symbol=symbol, interval="1d", adjust="back", tz=TZ, source="test",
        fetched_at=datetime.now(timezone.utc), rows=len(df),
        first_bar=df.index[0].to_pydatetime(), last_bar=df.index[-1].to_pydatetime(),
    )
    return ReplayProvider({(symbol, "1d"): OHLCV(df=df, meta=meta)})


def record(price: float, side: str = "support", *, atr: float = 10.0,
           day: str = "2026-01-05", distance: float = 1.0) -> pd.DataFrame:
    return pd.DataFrame(
        [{
            "symbol": "TEST",
            "run_date": pd.Timestamp(day, tz=TZ),
            "kind": "level",
            "label": "S1",
            "side": side,
            "role": "executable_pullback",
            "price": price,
            "display": price,
            "hits": 4,
            "band": "moderate",
            "distance_atr": distance,
            "atr": atr,
            "price_at_record": price + atr if side == "support" else price - atr,
        }]
    )


# ------------------------------------------------------------------ ReplayProvider


def test_replay_provider_slices_by_as_of():
    df = bars([(101, 99, 100)] * 10)
    p = provider_from(df)
    cut = df.index[4]
    assert len(p.fetch("TEST", "1d", as_of=cut).df) == 5
    assert len(p.fetch("TEST", "1d").df) == 10


def test_replay_provider_rejects_unknown_symbol():
    with pytest.raises(KeyError):
        provider_from(bars([(1, 1, 1)])).fetch("NOPE", "1d")


def test_replay_provider_raises_when_slice_is_empty():
    from jta.data.provider import DataUnavailable

    df = bars([(101, 99, 100)] * 5)
    with pytest.raises(DataUnavailable):
        provider_from(df).fetch("TEST", "1d", as_of=pd.Timestamp("2000-01-01", tz=TZ))


# ------------------------------------------------------------------ 结果判定


def test_untested_when_price_never_reaches_the_level():
    df = bars([(101, 99, 100)] + [(121, 119, 120)] * 10)
    out = evaluate(record(90.0), provider_from(df), horizon=10)
    assert out["touched"].iloc[0] is np.False_ or out["touched"].iloc[0] is False
    assert out["outcome"].iloc[0] == "untested"


def test_support_holds_when_it_bounces_a_full_atr():
    # 触及 90 后反弹到 101（>= 90 + 1 ATR），期间收盘从未跌破 85
    df = bars([(101, 99, 100)] + [(92, 89.5, 91)] + [(101, 95, 100)] * 5)
    out = evaluate(record(90.0), provider_from(df), horizon=10)
    assert out["outcome"].iloc[0] == "hold"
    assert out["bars_to_touch"].iloc[0] == 0


def test_support_breaks_on_close_below_the_threshold():
    # 触及后收盘 84 < 90 - 0.5 ATR = 85
    df = bars([(101, 99, 100)] + [(92, 89.5, 84)] + [(86, 80, 82)] * 5)
    out = evaluate(record(90.0), provider_from(df), horizon=10)
    assert out["outcome"].iloc[0] == "break"


def test_whichever_comes_first_decides():
    """先破后弹算 break，不能因为后面涨回来就改判。"""
    df = bars(
        [(101, 99, 100)]
        + [(92, 89.5, 84)]      # 先收盘破位
        + [(105, 95, 104)] * 5  # 之后才反弹
    )
    out = evaluate(record(90.0), provider_from(df), horizon=10)
    assert out["outcome"].iloc[0] == "break"


def test_inconclusive_when_neither_threshold_is_reached():
    df = bars([(101, 99, 100)] + [(92, 89.5, 91)] * 6)
    out = evaluate(record(90.0), provider_from(df), horizon=10)
    assert out["outcome"].iloc[0] == "inconclusive"


def test_resistance_uses_mirrored_logic():
    # 压力 110：触及后回落到 99（<= 110 - 1 ATR）算"守住"
    df = bars([(101, 99, 100)] + [(110.5, 108, 109)] + [(105, 98, 99)] * 5)
    out = evaluate(record(110.0, side="resistance"), provider_from(df), horizon=10)
    assert out["outcome"].iloc[0] == "hold"


def test_horizon_limits_the_observation_window():
    df = bars([(101, 99, 100)] * 5 + [(92, 89.5, 91)] + [(101, 95, 100)] * 5)
    short = evaluate(record(90.0), provider_from(df), horizon=2)
    long = evaluate(record(90.0), provider_from(df), horizon=10)
    assert short["outcome"].iloc[0] == "untested"
    assert long["touched"].iloc[0]


def test_evaluation_never_uses_bars_before_the_record_date():
    """记录日之前的触及不算数，否则就是用未来数据回头找答案的镜像错误。"""
    df = bars([(92, 89.0, 91)] + [(121, 119, 120)] * 8)
    rec = record(90.0, day=str(df.index[3].date()))
    out = evaluate(rec, provider_from(df), horizon=10)
    assert out["outcome"].iloc[0] == "untested"


# ------------------------------------------------------------------ 对照组


def test_controls_avoid_real_levels_and_match_side():
    rec = pd.concat([record(90.0), record(80.0)], ignore_index=True)
    ctrl = random_controls(rec, provider_from(bars([(101, 99, 100)] * 5)), seed=1)
    assert len(ctrl) == len(rec)
    assert set(ctrl["kind"]) == {"control"}
    for _, c in ctrl.iterrows():
        assert (np.abs(rec["price"].to_numpy() - c["price"]) > 0.5 * c["atr"]).all()
        assert c["side"] in {"support", "resistance"}


def test_controls_carry_no_evidence_flags():
    ctrl = random_controls(record(90.0), provider_from(bars([(101, 99, 100)] * 5)), seed=1)
    flags = [c for c in ctrl.columns if c.startswith("f_")]
    assert flags and not ctrl[flags].any().any()


def test_controls_on_empty_input():
    assert random_controls(pd.DataFrame(), provider_from(bars([(1, 1, 1)]))).empty


# ------------------------------------------------------------------ 统计


def make_outcomes(n_hold: int, n_break: int, kind: str = "level", distance: float = 1.0):
    rows = []
    for i in range(n_hold + n_break):
        rows.append({
            "kind": kind, "side": "support", "hits": 4, "distance_atr": distance,
            "touched": True, "outcome": "hold" if i < n_hold else "break",
            "mfe_atr": 1.0, "mae_atr": 0.5,
        })
    return pd.DataFrame(rows)


def test_rate_excludes_inconclusive_from_the_denominator():
    df = make_outcomes(6, 4)
    df.loc[len(df)] = {**df.iloc[0].to_dict(), "outcome": "inconclusive"}
    r = _rate(df)
    assert r["decided"] == 10 and r["hold_rate"] == 0.6
    assert r["touched"] == 11


def test_compare_refuses_small_samples():
    small = _rate(make_outcomes(3, 2))
    big = _rate(make_outcomes(60, 40))
    assert _compare(small, big)["z"] is None
    assert str(MIN_SAMPLES) in _compare(small, big)["note"]


def test_compare_reports_direction_and_z():
    a = _rate(make_outcomes(70, 30))    # 70%
    b = _rate(make_outcomes(50, 50))    # 50%
    cmp = _compare(a, b)
    assert cmp["diff"] == pytest.approx(0.2, abs=0.01)
    assert cmp["z"] > 2


def test_summarise_flags_multiple_comparisons():
    real = make_outcomes(30, 30)
    for f in ("primary_fib", "ema_cluster", "trendline", "horizontal",
              "extension_fib", "price_action", "td9", "index"):
        # 交替赋值，让两组各自都有 hold 和 break——全 hold / 全 break 会让
        # 标准误为 0，z 无法计算，测不到多重比较逻辑
        real[f"f_{f}"] = [True, False] * 30
    ctrl = make_outcomes(30, 30, kind="control")
    for f in ("primary_fib", "ema_cluster", "trendline", "horizontal",
              "extension_fib", "price_action", "td9", "index"):
        ctrl[f"f_{f}"] = False
    s = summarise(pd.concat([real, ctrl], ignore_index=True))
    assert s["multiple_comparisons"]["tests"] >= 7
    assert s["multiple_comparisons"]["z_threshold"] == 2.69
    assert any("外推" in c for c in s["caveats"])


def test_summarise_warns_that_overall_is_uncontrolled():
    s = summarise(pd.concat([make_outcomes(5, 5), make_outcomes(5, 5, kind="control")]))
    assert "未控制距离" in s["overall"]["warning"]


def test_summarise_on_empty_input():
    assert "error" in summarise(pd.DataFrame())
    assert format_summary({"error": "无记录"}) == "无记录"
