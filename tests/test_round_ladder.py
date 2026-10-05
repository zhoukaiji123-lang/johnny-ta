"""整数关口穿插：只做展示，不动关键位筛选与计划。"""

from __future__ import annotations

import pandas as pd

from jta.analyze import analyze, build_ladder
from jta.levels.pivots import round_ladder
from tests.test_live_alignment import _raw_provider

TZ = "America/New_York"


def test_round_ladder_scales_with_price_magnitude():
    assert [x["price"] for x in round_ladder(1720, 100, 1570, 1950)] == [1900, 1800, 1700, 1600]
    ko = round_ladder(85.65, 1.0, 81.3, 87.5)
    assert [x["price"] for x in ko] == [87, 86, 85, 84, 83, 82]
    assert {x["price"]: x["tier"] for x in ko}[85] == "半整数"
    assert {x["price"]: x["tier"] for x in round_ladder(1075, 60, 905, 1100)}[1000] == "大整数"


def test_round_ladder_skips_tiers_finer_than_half_an_atr():
    # ATR 300 时 100 一档太密（0.33 ATR），从 500 起
    assert [x["price"] for x in round_ladder(1720, 300, 1000, 2600)] == [2500, 2000, 1500, 1000]


def _lv(label, raw, lo, hi, side_text=None):
    return {"label": label, "raw_price": raw, "display": hi if label.startswith("S") else lo,
            "range_low": lo, "range_high": hi, "display_text": side_text or f"{lo}–{hi}"}


def test_build_ladder_interleaves_and_merges_only_inside_a_range():
    sup = [_lv("S1", 1716.3, 1710, 1720), _lv("S2", 1659.4, 1650, 1660)]
    res = [_lv("R1", 1729.6, 1720, 1740), _lv("R2", 1787.0, 1780, 1800)]
    rounds, ladder = build_ladder(sup, res, 1719.99, 100.0)
    # 1800 落在 R2 的展示区间里 → 标注在 R2 上；1700 单独穿插在 S1 与 S2 之间
    assert res[1]["round_number"] == {"price": 1800.0, "tier": "整数"}
    assert [r["price"] for r in rounds] == [1700.0]
    assert rounds[0]["between"] == "S1 与 S2"
    # 排序按原值：S1 展示 1720 高于现价，但原值 1716.3 在现价之下
    assert [it["label"] for it in ladder] == ["R2", "R1", "现价", "S1", "整数", "S2"]


def test_round_levels_do_not_change_key_levels_or_plans():
    as_of = pd.Timestamp("2025-10-15 23:00", tz=TZ)
    r = analyze("TEST", provider=_raw_provider(1.0), as_of=as_of,
                include_events=False, include_market_cap=False)
    assert "ladder" in r and "round_levels" in r and r["round_levels_note"]
    keys = {lv["label"] for lv in r["supports"] + r["resistances"]}
    assert {it["label"] for it in r["ladder"] if it["kind"] in ("support", "resistance")} == keys
    prices = [it["price"] for it in r["ladder"]]
    for rd in r["round_levels"]:
        assert rd["price"] in prices and rd["between"]
