"""structure 口径：日线结构支点（上游 7.1 节）进入候选并占第三档。"""

from __future__ import annotations

from types import SimpleNamespace

import pandas as pd

from jta.analyze import _swing_sources, select_key_levels
from jta.indicators.swing import SwingPoint, visible_at
from jta.levels.candidates import Candidate, make_source

ATR = 10.0
NOW = 100.0


def cand(price, hits, families=("fib",), reaction=False, side="support"):
    sources = []
    for f in families:
        detail = {"price": price, "relation": "HL", "ts": "2026-09-28"} if f == "swing" else {}
        sources.append(make_source(f, f, timeframe="1d", detail=detail))
    c = Candidate(price=price, side=side, display=price, sources=sources)
    c.distance_atr = abs(NOW - price) / ATR
    return c, {"hits": hits, "band": "", "factors": {"price_action": {"hit": reaction}}, "caveat": ""}


def base():
    """proximity 会选 99 / 91 / 84；结构支点在 70。"""
    return [
        cand(99.0, 3, ("prev_session",)),
        cand(91.0, 3, ("pivot",)),
        cand(84.0, 6, ("fib", "pivot")),
        cand(70.0, 4, ("swing",)),
    ]


def test_proximity_ignores_structure_pivot_slot():
    picked, meta = select_key_levels(base(), "support", ATR, NOW, "proximity")
    assert [c.display for c, _ in picked] == [99.0, 91.0, 84.0]
    assert "structure_pivot" not in meta


def test_structure_pivot_takes_third_slot():
    picked, meta = select_key_levels(base(), "support", ATR, NOW, "structure")
    assert [c.display for c, _ in picked] == [99.0, 91.0, 70.0]
    assert [c.label for c, _ in picked] == ["S1", "S2", "S3"]
    assert picked[2][0].role == "structure_pivot_low"
    assert meta["structure_pivot"]["placed"] is True
    assert meta["structure_pivot"]["relation"] == "HL"
    displaced = [x for x in meta["crowded_out"] if "结构支点" in x["reason"]]
    assert [x["price"] for x in displaced] == [84.0]


def test_structure_keeps_s1_s2_identical_to_proximity():
    prox, _ = select_key_levels(base(), "support", ATR, NOW, "proximity")
    struct, _ = select_key_levels(base(), "support", ATR, NOW, "structure")
    assert [c.display for c, _ in prox[:2]] == [c.display for c, _ in struct[:2]]


def test_pivot_already_selected_changes_nothing():
    scored = [cand(99.0, 3, ("prev_session",)), cand(91.0, 4, ("swing",)), cand(84.0, 6)]
    picked, meta = select_key_levels(scored, "support", ATR, NOW, "structure")
    assert [c.display for c, _ in picked] == [99.0, 91.0, 84.0]
    assert meta["structure_pivot"]["placed"] is False
    assert meta["structure_pivot"]["note"] == "结构支点已经入选"


def test_pivot_below_min_hits_is_not_forced_in():
    scored = base()[:3] + [cand(70.0, 2, ("swing",))]
    picked, meta = select_key_levels(scored, "support", ATR, NOW, "structure")
    assert [c.display for c, _ in picked] == [99.0, 91.0, 84.0]
    assert meta["structure_pivot"]["placed"] is False
    assert "证据不足" in meta["structure_pivot"]["note"]


def test_pivot_too_close_to_s2_is_represented_by_s2():
    # 88 离 S2（91）只有 0.3 ATR：不能成为独立一档
    scored = base()[:3] + [cand(88.0, 4, ("swing",))]
    picked, meta = select_key_levels(scored, "support", ATR, NOW, "structure")
    assert [c.display for c, _ in picked] == [99.0, 91.0, 84.0]
    assert meta["structure_pivot"]["placed"] is False


def test_pivot_fills_empty_third_slot_without_displacing():
    scored = [cand(99.0, 3, ("prev_session",)), cand(91.0, 3, ("pivot",)), cand(70.0, 4, ("swing",))]
    picked, meta = select_key_levels(scored, "support", ATR, NOW, "structure")
    assert [c.display for c, _ in picked] == [99.0, 91.0, 70.0]
    assert not [x for x in meta["crowded_out"] if "结构支点" in x["reason"]]


# ------------------------------------------------------------------ 来源


def _pt(day, kind, price, confirm_lag=3):
    ts = pd.Timestamp(day, tz="America/New_York")
    return SwingPoint(ts, kind, price, ts + pd.Timedelta(days=confirm_lag), 0)


def _ctx(swings):
    return SimpleNamespace(swings=swings, atr=ATR, interval="1d")


def test_swing_sources_take_latest_intact_low_and_high():
    swings = [
        _pt("2026-08-01", "low", 80.0), _pt("2026-08-10", "high", 110.0),
        _pt("2026-08-20", "low", 90.0), _pt("2026-09-01", "high", 105.0),
    ]
    out = _swing_sources(_ctx(swings), NOW)
    by_kind = {s["detail"]["relation"]: p for p, s in out}
    assert by_kind == {"HL": 90.0, "LH": 105.0}
    assert all(s["family"] == "swing" and s["timeframe"] == "1d" for _, s in out)


def test_swing_sources_skip_broken_low():
    """最近的低点 102 已在现价上方（被跌破），取更早的 90。"""
    swings = [_pt("2026-08-20", "low", 90.0), _pt("2026-09-01", "high", 115.0),
              _pt("2026-09-10", "low", 102.0)]
    lows = [p for p, s in _swing_sources(_ctx(swings), NOW) if p < NOW]
    assert lows == [90.0]


def test_swing_sources_respect_as_of():
    """9/28 的低点要到 10/1 才确认；as_of=9/29 时不能用它。"""
    swings = [_pt("2026-09-10", "low", 85.0), _pt("2026-09-20", "high", 110.0),
              _pt("2026-09-28", "low", 95.0)]
    visible = visible_at(swings, pd.Timestamp("2026-09-29", tz="America/New_York"))
    lows = [p for p, s in _swing_sources(_ctx(visible), NOW) if p < NOW]
    assert lows == [85.0]


# ------------------------------------------------------------------ 端到端


def test_analyze_structure_mode_only_touches_third_slot():
    from jta.analyze import analyze
    from tests.test_pipeline import FakeProvider

    prox = analyze("TEST", provider=FakeProvider(), include_events=False,
                   include_market_cap=False, selection="proximity")
    struct = analyze("TEST", provider=FakeProvider(), include_events=False,
                     include_market_cap=False, selection="structure")
    assert struct["selection"]["support"]["mode"] == "structure"
    assert "structure_pivot" in struct["selection"]["support"]
    for side in ("supports", "resistances"):
        assert [lv["display_text"] for lv in prox[side][:2]] == \
            [lv["display_text"] for lv in struct[side][:2]]
    # proximity 口径下不产生 swing 证据
    for side in ("supports", "resistances"):
        assert all("swing" not in lv["evidence_families"] for lv in prox[side])
