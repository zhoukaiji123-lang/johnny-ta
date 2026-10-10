"""周报计算层测试（不联网）。"""

from __future__ import annotations

from datetime import date, datetime

import numpy as np
import pandas as pd
import pytest

from jta.data.provider import OHLCV, SeriesMeta
from jta.weekly import (
    ET, LABEL_ORDER, build_universe, build_weekly, classify, default_week,
    parse_week, rs_metrics, symbol_week, to_weekly, week_id,
)

TZ = "America/New_York"


def _daily(closes, start="2025-01-06", vol=None) -> pd.DataFrame:
    idx = pd.bdate_range(start, periods=len(closes), tz=TZ)
    c = np.asarray(closes, dtype=float)
    return pd.DataFrame({
        "open": c, "high": c * 1.01, "low": c * 0.99, "close": c,
        "volume": vol if vol is not None else np.full(len(c), 1000.0),
    }, index=idx)


def test_parse_and_default_week():
    assert parse_week("2026-W41") == date(2026, 10, 5)
    assert week_id(date(2026, 10, 5)) == "2026-W41"
    # 周六取刚走完的这一周；周五盘中还没收盘，取上一周
    assert default_week(datetime(2026, 10, 10, 9, tzinfo=ET)) == date(2026, 10, 5)
    assert default_week(datetime(2026, 10, 9, 15, tzinfo=ET)) == date(2026, 9, 28)
    assert default_week(datetime(2026, 10, 9, 16, 30, tzinfo=ET)) == date(2026, 10, 5)


def test_to_weekly_handles_holiday_week():
    # 2025-01-20 是马丁·路德·金纪念日，那一周只有 4 个交易日
    idx = pd.bdate_range("2025-01-13", "2025-01-24", tz=TZ)
    idx = idx[idx.normalize() != pd.Timestamp("2025-01-20", tz=TZ)]
    c = np.arange(len(idx), dtype=float) + 100
    df = pd.DataFrame({"open": c, "high": c + 1, "low": c - 1, "close": c,
                       "volume": np.ones(len(c))}, index=idx)
    w = to_weekly(df)
    assert list(w["sessions"]) == [5, 4]
    assert w["monday"].iloc[1] == date(2025, 1, 20)
    assert w["open"].iloc[1] == 105 and w["close"].iloc[1] == 108
    assert w["high"].iloc[1] == 109 and w["low"].iloc[0] == 99
    assert w.index[1].strftime("%Y-%m-%d") == "2025-01-24"


def test_symbol_week_ignores_data_after_target_week():
    d = _daily(np.linspace(100, 200, 300))
    monday = date(2025, 6, 2)
    a = symbol_week(d, monday)
    b = symbol_week(d[d.index.tz_localize(None) <= pd.Timestamp("2025-06-06")], monday)
    assert a["close"] == b["close"] and a["ema10"] == b["ema10"]
    assert a["week_end"] == "2025-06-06"


def test_symbol_week_returns_none_when_week_missing():
    d = _daily(np.linspace(100, 200, 50))
    assert symbol_week(d, date(2030, 1, 7)) is None


def test_rs_metrics_alignment_and_high():
    up = to_weekly(_daily(np.linspace(100, 300, 150)))
    flat = to_weekly(_daily(np.full(150, 100.0)))
    rs = rs_metrics(up, flat)
    assert rs["at_13w_high"] and rs["chg_4w_pct"] > 0
    rs_down = rs_metrics(flat, up)
    assert not rs_down["at_13w_high"] and rs_down["chg_4w_pct"] < 0


def _m(**kw):
    base = dict(close=110.0, high=111.0, low=105.0, prev_high=109.0, prev_low=104.0,
                ema10=100.0, ema20=95.0, atr_prev=5.0, dist_ema10_atr=2.0,
                close_pos=0.5, move_atr=0.5, prev_close=108.0, pullback_4w_atr=0.2, weeks=60,
                rs_market={"at_13w_high": False})
    base.update(kw)
    return base


@pytest.mark.parametrize("kw,label", [
    ({}, "延续"),
    ({"close": 103.0, "dist_ema10_atr": 0.6}, "转弱"),                 # 跌破上周低点
    ({"close": 99.0, "low": 98.0, "prev_low": 97.0, "dist_ema10_atr": -0.2}, "转弱"),  # 落到 EMA10 下
    ({"close": 90.0, "ema10": 95.0, "ema20": 100.0, "low": 89.0, "prev_low": 88.0,
      "dist_ema10_atr": -1.0}, "下行"),
    ({"dist_ema10_atr": 3.5, "close_pos": 0.9, "move_atr": 2.0,
      "rs_market": {"at_13w_high": True}}, "过热"),                     # 过热压过加速
    ({"close_pos": 0.9, "move_atr": 2.0, "rs_market": {"at_13w_high": True}}, "加速"),
    ({"high": 108.0, "pullback_4w_atr": 0.5}, "休整"),
    ({"close_pos": 0.1, "prev_close": 112.0}, "冲高回落"),           # 新高后收在周低位
    ({"close_pos": 0.1, "prev_close": 108.0}, "延续"),               # 收盘仍高于上周，不算
    ({"close_pos": 0.1, "prev_close": 112.0, "dist_ema10_atr": 3.5}, "冲高回落"),  # 压过过热
    ({"high": 108.0, "pullback_4w_atr": 1.5}, "整理"),
    ({"ema10": None, "ema20": None, "atr_prev": None, "dist_ema10_atr": None,
      "prev_high": None, "prev_low": None, "weeks": 3}, "历史不足"),
])
def test_classify(kw, label):
    got, reasons = classify(_m(**kw))
    assert got == label
    assert reasons and reasons[0].startswith(label)
    assert got in LABEL_ORDER


def test_universe_sections():
    u = build_universe()
    assert list(u.sections)[0] == "指数"
    assert u.sections["存储"]["reference"] == "DRAM"
    assert "DRAM" not in u.sections["存储"]["members"]
    assert {"DELL", "TSLA", "SPCX", "MRNA"} <= set(u.sections["其他"]["members"])
    assert u.sections["其他"]["reference"] is None
    assert {"QQQ", "SPY", "SOXX", "SKYY", "DRAM"} <= set(u.symbols)


class _Provider:
    name = "fake"

    def __init__(self, frames):
        self.frames = frames

    def fetch(self, symbol, interval, **kw):
        df = self.frames[symbol]
        return OHLCV(df=df, meta=SeriesMeta(symbol=symbol, interval=interval, adjust="back",
                                            tz=TZ, source="fake", fetched_at=datetime.now(ET)))


def test_build_weekly_end_to_end():
    rng = np.random.RandomState(3)
    u = build_universe({"MU": "存储", "WDC": "存储", "DELL": "硬件"})
    frames = {s: _daily(100 * np.cumprod(1 + rng.randn(400) * 0.01 + 0.001)) for s in u.symbols}
    monday = date(2026, 5, 4)
    p = build_weekly(monday, provider=_Provider(frames), universe=u,
                     now=datetime(2026, 5, 9, tzinfo=ET))
    assert p["week"] == "2026-W19" and not p["partial"]
    assert not p["data_health"]["failed"]
    names = [s["name"] for s in p["sections"]]
    assert names == ["指数", "存储", "其他"]
    store = p["sections"][1]
    assert store["aggregate"]["n"] == 2 and store["reference_metrics"]["label"] in LABEL_ORDER
    assert {r["symbol"] for r in p["rotation"]["ranking"]} == {"MU", "WDC", "DELL"}
    qqq = p["sections"][0]["members"][0]
    assert qqq["symbol"] == "QQQ" and qqq["rs_market"] is None
    assert set(p["regimes"]) == {"QQQ", "SPY", "SOXX"}


def _weekly_path(points):
    """按周给定收盘路径，每周 5 根相同的日线。"""
    closes = np.repeat(np.interp(np.arange(points[-1][0] + 1), *zip(*points)), 5)
    return _daily(closes, start="2025-01-06")


def test_lower_high_warning_only_while_unbroken():
    # 高点 200 → 回落 120 → 反弹到 170 形成 LH → 回落
    base = [(0, 100), (10, 200), (20, 120), (28, 170), (36, 130)]
    d = _weekly_path(base + [(40, 140)])
    m = symbol_week(d, to_weekly(d)["monday"].iloc[-1])
    assert m["lower_high_unbroken"] and abs(m["lower_high_unbroken"]["price"] - 170 * 1.01) < 1
    assert any("LH" in w for w in m["warnings"])
    # 之后涨回 LH 上方：预警消失
    d2 = _weekly_path(base + [(40, 180), (44, 175)])
    m2 = symbol_week(d2, to_weekly(d2)["monday"].iloc[-1])
    assert m2["lower_high_unbroken"] is None
    assert not any("LH" in w for w in m2["warnings"])


def test_underperform_warning_threshold():
    from jta.weekly import health_warnings
    base = {"new_13w_high": False, "ret_pct": {"4w": 5.0}, "intraweek": {},
            "lower_high_unbroken": None, "label": "延续", "prev_low": 90.0, "close": 100.0}
    assert not any("跑输" in w for w in health_warnings({**base, "rs_reference": {"chg_4w_pct": -2.9}}))
    assert any("跑输" in w for w in health_warnings({**base, "rs_reference": {"chg_4w_pct": -3.1}}))
