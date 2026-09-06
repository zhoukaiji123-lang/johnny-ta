"""候选构建、共振检查与编排层测试（不联网）。"""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from jta.analyze import MIN_HITS, MIN_SEPARATION_ATR, analyze, select_key_levels
from jta.data.provider import OHLCV, SeriesMeta
from jta.indicators.swing import SwingPoint
from jta.levels.candidates import (
    Candidate,
    annotate_resonance,
    build_candidates,
    make_source,
    note_pivot_confluence,
)
from jta.report import position_size, render_text
from jta.scoring import evaluate, index_alignment

TZ = "America/New_York"


def src(family: str, tf: str = "1d") -> dict:
    return make_source(family, family, timeframe=tf, confirmed_at="2026-01-01T00:00:00-05:00")


# ------------------------------------------------------------------ 候选构建


def test_only_near_price_candidates_survive():
    raw = [(100.0, src("fib")), (1000.0, src("fib"))]
    got = build_candidates(raw, current_price=100.0, atr_value=5.0)
    assert [round(c.price) for c in got] == [100]


def test_identical_values_merge_but_nearby_ones_do_not():
    """0.05 ATR 内视为同一个数字；0.3 ATR 内是两个独立结构，必须都留着。"""
    atr = 10.0
    raw = [
        (90.0, src("fib")),
        (90.2, src("pivot")),   # 0.02 ATR —— 同一个数字
        (93.0, src("ema")),     # 0.30 ATR —— 独立结构
    ]
    got = build_candidates(raw, current_price=100.0, atr_value=atr)
    assert len(got) == 2
    merged = got[0]
    assert {s["family"] for s in merged.sources} == {"fib", "pivot"}


def test_candidates_on_opposite_sides_never_merge():
    atr = 10.0
    raw = [(99.9, src("fib")), (100.1, src("pivot"))]
    got = build_candidates(raw, current_price=100.0, atr_value=atr)
    assert len(got) == 2
    assert {c.side for c in got} == {"support", "resistance"}


def test_build_rejects_invalid_atr():
    assert build_candidates([(1.0, src("fib"))], 1.0, 0.0) == []
    assert build_candidates([(1.0, src("fib"))], 1.0, float("nan")) == []


def test_resonance_counts_neighbours_within_wider_radius():
    atr = 10.0
    raw = [(90.0, src("fib")), (92.0, src("pivot")), (99.0, src("ema"))]
    got = build_candidates(raw, current_price=100.0, atr_value=atr)
    annotate_resonance(got, atr)
    near = next(c for c in got if abs(c.price - 90.0) < 0.01)
    # 92.0 在 0.25 ATR 内 → 计入共振；99.0 不在
    assert set(near.resonance["families"]) == {"fib", "pivot"}


def test_confluence_is_recorded_without_rewriting_the_display_price():
    """重合只进证据栏。改写展示价会让两行显示同一个数字。"""
    atr = 10.0
    raw = [(90.0, src("fib")), (91.5, src("pivot"))]
    got = build_candidates(raw, current_price=100.0, atr_value=atr)
    note_pivot_confluence(got, atr)
    fib_c = next(c for c in got if "fib" in c.source_families)
    assert fib_c.snap["raw_price"] == 90.0
    assert fib_c.snap["confluent_pivot"] == 91.5
    assert fib_c.price == 90.0 and fib_c.display == 90.0


def test_every_displayed_price_stays_distinct():
    atr = 10.0
    raw = [(90.0, src("fib")), (91.5, src("pivot")), (93.0, src("ema"))]
    got = build_candidates(raw, current_price=100.0, atr_value=atr)
    note_pivot_confluence(got, atr)
    displays = [c.display for c in got]
    assert len(displays) == len(set(displays))


def test_pivot_candidates_are_not_annotated_against_themselves():
    got = build_candidates([(91.5, src("pivot"))], 100.0, 10.0)
    note_pivot_confluence(got, 10.0)
    assert got[0].snap is None


# ------------------------------------------------------------------ 共振检查


def make_df(n: int = 200) -> pd.DataFrame:
    idx = pd.DatetimeIndex(
        list(pd.date_range("2026-01-05", periods=n, freq="B", tz=TZ)), name="Date"
    )
    closes = np.linspace(100, 140, n)
    return pd.DataFrame(
        {
            "open": closes,
            "high": closes + 1,
            "low": closes - 1,
            "close": closes,
            "volume": np.full(n, 1000.0),
        },
        index=idx,
    )


def test_evaluate_maps_families_to_factors():
    c = Candidate(price=120.0, side="support", sources=[src("fib"), src("trendline")])
    c.resonance = {"families": ["fib", "trendline", "vegas"]}
    out = evaluate(
        c, df=make_df(), swings=[], atr_value=2.0, td_signal=None, index_state=None
    )
    f = out["factors"]
    assert f["primary_fib"]["hit"] and f["trendline"]["hit"] and f["ema_cluster"]["hit"]
    assert not f["extension_fib"]["hit"]
    assert out["hits"] >= 3


def test_td_signal_direction_must_match_the_side():
    buy9 = {"kind": "buy_setup_9", "ts": "2026-01-01", "perfected": True, "meaning": ""}
    support = Candidate(price=120.0, side="support", sources=[src("fib")])
    support.resonance = {"families": ["fib"]}
    resistance = Candidate(price=120.0, side="resistance", sources=[src("fib")])
    resistance.resonance = {"families": ["fib"]}
    kw = dict(df=make_df(), swings=[], atr_value=2.0, index_state=None)
    assert evaluate(support, td_signal=buy9, **kw)["factors"]["td9"]["hit"]
    assert not evaluate(resistance, td_signal=buy9, **kw)["factors"]["td9"]["hit"]


def test_index_alignment_is_directional():
    bull = {"symbol": "SOXX", "ema_stack": "bull"}
    assert index_alignment("support", bull)["aligned"] is True
    assert index_alignment("resistance", bull)["aligned"] is False
    assert index_alignment("support", None) is None


def test_score_carries_the_no_weighting_caveat():
    c = Candidate(price=120.0, side="support", sources=[src("fib")])
    c.resonance = {"families": ["fib"]}
    out = evaluate(c, df=make_df(), swings=[], atr_value=2.0, td_signal=None, index_state=None)
    assert "没有统计含义" in out["caveat"]


# ------------------------------------------------------------------ 筛选


def scored_candidate(price: float, side: str, hits: int, atr: float = 10.0):
    c = Candidate(price=price, side=side, sources=[src("fib")], display=price)
    c.distance_atr = abs(price - 100.0) / atr
    return c, {"hits": hits, "band": "strong", "factors": {}, "caveat": ""}


def test_selection_enforces_minimum_separation():
    atr = 10.0
    scored = [
        scored_candidate(95.0, "support", 6),
        scored_candidate(94.0, "support", 5),   # 距离 0.1 ATR，无法成为独立一档
        scored_candidate(80.0, "support", 5),
    ]
    picked, meta = select_key_levels(scored, "support", atr, 100.0)
    assert [c.display for c, _ in picked] == [95.0, 80.0]
    assert meta["crowded_out"] and meta["crowded_out"][0]["price"] == 94.0


def test_selection_rejects_weak_evidence_and_reports_the_gap():
    scored = [scored_candidate(95.0, "support", MIN_HITS - 1)]
    picked, meta = select_key_levels(scored, "support", 10.0, 100.0)
    assert picked == []
    assert "不用弱点补足数量" in meta["gap_note"]


def test_selection_caps_at_three_per_side():
    scored = [scored_candidate(100.0 - 10 * i, "support", 6) for i in range(1, 8)]
    picked, _ = select_key_levels(scored, "support", 10.0, 100.0)
    assert len(picked) == 3


def test_labels_follow_display_value_not_raw_price():
    """贴合会让展示值与原始值分离；编号必须跟着读者看到的数字走。"""
    atr = 10.0
    a, sa = scored_candidate(88.0, "support", 6)
    a.display = 70.0          # 贴合到更远的枢轴
    b, sb = scored_candidate(80.0, "support", 6)
    b.display = 80.0
    picked, _ = select_key_levels([(a, sa), (b, sb)], "support", atr, 100.0)
    assert [c.label for c, _ in picked] == ["S1", "S2"]
    assert [c.display for c, _ in picked] == [80.0, 70.0]


def test_roles_are_assigned_after_selection():
    scored = [scored_candidate(100.0 - 20 * i, "support", 6) for i in range(1, 4)]
    picked, _ = select_key_levels(scored, "support", 10.0, 100.0)
    assert [c.role for c, _ in picked] == [
        "immediate_defense",
        "executable_pullback",
        "core_defense",
    ]


# ------------------------------------------------------------------ 端到端


class FakeProvider:
    """确定性合成行情，用于在不联网的情况下验证编排与 as_of 语义。"""

    name = "fake"

    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def _frame(self, interval: str) -> pd.DataFrame:
        n = 600 if interval == "1d" else 900
        freq = "B" if interval == "1d" else "4h"
        idx = pd.DatetimeIndex(
            list(pd.date_range("2024-01-02 09:30", periods=n, freq=freq, tz=TZ))
        )
        x = np.arange(n)
        closes = 100 + 40 * np.sin(x / 60.0) + x * 0.08
        rng = np.random.RandomState(11)
        noise = rng.randn(n) * 0.4
        closes = closes + noise
        return pd.DataFrame(
            {
                "open": closes,
                "high": closes + 1.5,
                "low": closes - 1.5,
                "close": closes,
                "volume": 1000 + rng.randint(0, 800, n),
            },
            index=idx,
        )

    def fetch(self, symbol, interval, *, as_of=None, **kw) -> OHLCV:
        self.calls.append((symbol, interval))
        df = self._frame(interval)
        if as_of is not None:
            df = df[df.index <= pd.Timestamp(as_of).tz_convert(df.index.tz)]
        return OHLCV(
            df=df,
            meta=SeriesMeta(
                symbol=symbol,
                interval=interval,
                adjust="back",
                tz=TZ,
                source=self.name,
                fetched_at=datetime.now(timezone.utc),
                as_of=as_of,
                rows=len(df),
                first_bar=df.index[0].to_pydatetime(),
                last_bar=df.index[-1].to_pydatetime(),
            ),
        )


def test_live_price_and_structure_use_different_vintages():
    """盘中现价来自未完成 bar，但结构必须只用已收盘的 bar。"""
    r = analyze("TEST", provider=FakeProvider())
    assert "price_is_live" in r and "structure_as_of" in r
    if r["price_is_live"]:
        assert r["current_price"] == pytest.approx(r["live_bar"]["close"])
        assert pd.Timestamp(r["structure_as_of"]) < pd.Timestamp(r["live_bar"]["ts"])


def test_analyze_returns_complete_schema():
    r = analyze("TEST", provider=FakeProvider())
    for key in (
        "schema_version", "symbol", "current_price", "data", "market_state",
        "supports", "resistances", "selection", "position_zone",
        "position_sizing", "known_gaps",
    ):
        assert key in r
    assert r["market_state"]["phase"] in {
        "downtrend", "stabilizing", "reversal_candidate", "trend_improving", "range_or_mixed"
    }
    assert len(r["supports"]) <= 3 and len(r["resistances"]) <= 3


def test_analyze_declares_unimplemented_parts():
    r = analyze("TEST", provider=FakeProvider())
    gaps = " ".join(r["known_gaps"])
    assert "Countdown" in gaps and "基本面" in gaps and "市值" in gaps


def test_analyze_respects_as_of_everywhere():
    cutoff = pd.Timestamp("2025-06-02 16:00", tz=TZ)
    r = analyze("TEST", as_of=cutoff, provider=FakeProvider())
    assert pd.Timestamp(r["data"]["daily"]["last_bar"]) <= cutoff
    for side in ("supports", "resistances"):
        for lv in r[side]:
            for s in lv["sources"]:
                if s.get("confirmed_at"):
                    assert pd.Timestamp(s["confirmed_at"]) <= cutoff
                for key in ("anchor_low", "anchor_high"):
                    a = (s.get("detail") or {}).get(key)
                    if a:
                        assert pd.Timestamp(a["ts"]) <= cutoff


def test_benchmark_is_fetched_when_requested():
    p = FakeProvider()
    r = analyze("TEST", benchmark="BENCH", provider=p)
    assert ("BENCH", "1d") in p.calls
    assert r["benchmark"]["symbol"] == "BENCH"


def test_render_text_surfaces_data_vintage_and_caveats():
    r = analyze("TEST", provider=FakeProvider())
    text = render_text(r, account=100_000)
    assert "复权=back" in text and "已知缺口" in text
    assert "数量 = (账户净值" in position_size(1e5, 0.01, 100.0, 95.0)["formula"]


def test_position_size_math_and_guard():
    ps = position_size(100_000, 0.01, 100.0, 95.0)
    assert ps["risk_amount"] == 1000.0 and ps["quantity"] == 200
    assert "error" in position_size(100_000, 0.01, 100.0, 100.0)


# ------------------------------------------------------------------ 展示粒度


def test_display_step_follows_one_two_five_ladder():
    from jta.levels.candidates import display_step

    assert display_step(183.76) == 20.0    # 0.1 ATR = 18.4
    assert display_step(23.06) == 2.0      # 0.1 ATR = 2.3
    assert display_step(67.63) == 5.0      # 0.1 ATR = 6.8
    assert display_step(2.3) == pytest.approx(0.2)
    assert display_step(0.45) == pytest.approx(0.05)


def test_display_step_degrades_safely():
    from jta.levels.candidates import MIN_DISPLAY_STEP, display_step

    assert display_step(0.0) == MIN_DISPLAY_STEP
    assert display_step(float("nan")) == MIN_DISPLAY_STEP
    assert display_step(0.0001) == MIN_DISPLAY_STEP  # 不得比最小跳动更细


def test_quantize_rounds_to_step():
    from jta.levels.candidates import quantize

    assert quantize(1185.7, 20.0) == 1180.0
    assert quantize(517.5, 2.0) == 518.0
    assert quantize(3.276, 0.05) == pytest.approx(3.3)
    assert quantize(100.0, 0.0) == 100.0


def test_raw_price_survives_quantisation():
    """取整只影响展示；原始计算值必须能反查。"""
    atr = 183.76
    got = build_candidates([(1185.7, src("fib"))], current_price=1200.0, atr_value=atr)
    c = got[0]
    assert c.display == 1180.0
    assert c.price == pytest.approx(1185.7)
    assert c.to_dict()["raw_price"] == pytest.approx(1185.7)


def test_selected_levels_never_collide_after_quantisation():
    """最小间距是 0.5 ATR，展示步长是 0.1 ATR，取整不可能让两档撞成同一个数。"""
    atr = 183.76
    scored = [scored_candidate(1200.0 - 100 * i, "support", 6, atr) for i in range(1, 4)]
    from jta.levels.candidates import display_step, quantize

    step = display_step(atr)
    for c, _ in scored:
        c.display = quantize(c.price, step)
    picked, _ = select_key_levels(scored, "support", atr, 1200.0)
    displays = [c.display for c, _ in picked]
    assert len(displays) == len(set(displays))


def test_report_exposes_display_granularity():
    r = analyze("TEST", provider=FakeProvider())
    assert "display_step" in r["data"] and "伪精度" in r["data"]["display_step_note"]
    assert "展示粒度" in render_text(r)


def test_recursive_indicators_use_full_history_not_the_truncated_window():
    """EMA169 需要 845 根 warmup。若在 tail(500) 上起算，它对任何标的都不可信——
    那不是历史不够，是自己把历史丢了。"""
    from jta.analyze import DAILY_LOOKBACK, _prepare
    from jta.indicators.ema import ema_set

    p = FakeProvider()
    series = p.fetch("TEST", "1d")
    ctx = _prepare(series, DAILY_LOOKBACK, None)
    assert ctx.full_bars == len(series.df)
    assert len(ctx.df) <= DAILY_LOOKBACK
    # EMA 必须与在完整序列上递推的结果一致
    expected = ema_set(series.df["close"]).iloc[-1]
    for col in ("ema8", "ema169"):
        assert ctx.emas[col].iloc[-1] == pytest.approx(expected[col])


def test_warmup_is_judged_against_full_history():
    from jta.analyze import DAILY_LOOKBACK, _prepare, market_state

    p = FakeProvider()
    daily = _prepare(p.fetch("TEST", "1d"), DAILY_LOOKBACK, None)
    intraday = _prepare(p.fetch("TEST", "4h"), DAILY_LOOKBACK, None)
    st = market_state(daily, intraday)
    assert st["ema_warmup"]["ema8"]["bars_available"] == daily.full_bars


def test_vegas_verdict_withheld_when_warmup_is_short():
    """50 根数据算出的 EMA169 只是把起始价拖了一段，不能拿来判长期牛熊。"""
    from jta.analyze import market_state
    from jta.indicators.ema import ema_set
    from jta.analyze import TimeframeContext

    p = FakeProvider()
    series = p.fetch("TEST", "1d")
    short = series.df.tail(50)
    ctx = TimeframeContext(
        interval="1d", series=series, df=short, atr=1.0, swings=[],
        emas=ema_set(short["close"]), td=pd.DataFrame(index=short.index), full_bars=50,
    )
    st = market_state(ctx, ctx)
    assert st["long_term_vs_vegas"] == "insufficient_history"
    assert st["vegas"] is None and st["vegas_reliable"] is False
    assert "ema169" in st["unreliable_emas"]


# --------------------------------------------- 验证结论反馈进计划生成


def level(label, price, distance, side="support"):
    return {
        "label": label, "display": price, "raw_price": price, "distance_atr": distance,
        "confirmation": f"{label} 确认", "invalidation": f"{label} 失效", "side": side,
    }


def test_distance_filter_is_off_by_default():
    """P5 支持过滤，P7 计划级回测推翻了它：砍掉 35% 机会而单笔期望不变。"""
    from jta.plans import MIN_ENTRY_DISTANCE_ATR, build_plans

    assert MIN_ENTRY_DISTANCE_ATR == 0.0
    sups = [level("S1", 99.0, 0.1), level("S2", 92.0, 0.8)]
    ress = [level("R1", 101.0, 0.1, "resistance"), level("R2", 112.0, 1.2, "resistance")]
    by_key = {p["key"]: p for p in build_plans(sups, ress, atr=10.0, zone="at_support")}
    assert by_key["aggressive"]["entry"] == 99.0        # 仍然用最近的 S1
    assert by_key["aggressive"]["entry_note"] is None
    assert by_key["breakout"]["entry"] == 101.0


def test_distance_filter_still_works_when_explicitly_enabled():
    """机制保留：有了新数据可以重新开启，不必改回代码。"""
    from jta.plans import build_plans

    sups = [level("S1", 99.0, 0.1), level("S2", 92.0, 0.8), level("S3", 85.0, 1.5)]
    ress = [level("R1", 101.0, 0.1, "resistance"), level("R2", 112.0, 1.2, "resistance")]
    by_key = {
        p["key"]: p
        for p in build_plans(sups, ress, atr=10.0, zone="at_support",
                             min_entry_distance_atr=0.5)
    }
    assert by_key["aggressive"]["entry"] == 92.0        # 跳过 S1
    assert by_key["deep"]["entry"] == 85.0
    assert by_key["breakout"]["entry"] == 112.0
    assert "S1" in by_key["aggressive"]["entry_note"]


def test_plan_blocked_when_every_level_is_filtered_out():
    from jta.plans import build_plans

    sups = [level("S1", 99.0, 0.1), level("S2", 98.0, 0.2)]
    plans = build_plans(sups, [], atr=10.0, zone="at_support", min_entry_distance_atr=0.5)
    assert all(not p["executable"] for p in plans)
    assert any("ATR 的可用档位" in b for b in plans[0]["blocked_by"])


def test_adverse_index_warns_without_cutting_size_by_default():
    """回测里指数非多头的期望反而更高（+0.871R vs +0.388R），不该减半。"""
    from jta.plans import ADVERSE_INDEX_SCALE, build_plans

    assert ADVERSE_INDEX_SCALE == 1.0
    sups = [level("S1", 92.0, 0.8), level("S2", 85.0, 1.5)]
    ress = [level("R1", 112.0, 1.2, "resistance"), level("R2", 125.0, 2.5, "resistance")]
    good = build_plans(sups, ress, atr=10.0, zone="at_support", index_bullish=True)
    bad = build_plans(sups, ress, atr=10.0, zone="at_support",
                      index_bullish=False, index_symbol="SOXX")
    assert good[0]["position_scale"] == 1.0 and not good[0]["cautions"]
    assert bad[0]["position_scale"] == 1.0            # 提示但不缩减
    assert "SOXX" in bad[0]["cautions"][0]
    assert "不缩减仓位" in bad[0]["cautions"][0]
    assert "缩减" not in bad[0]["tranche_text"]


def test_index_scaling_still_works_when_explicitly_enabled():
    from jta.plans import build_plans

    sups = [level("S1", 92.0, 0.8), level("S2", 85.0, 1.5)]
    ress = [level("R1", 112.0, 1.2, "resistance")]
    bad = build_plans(sups, ress, atr=10.0, zone="at_support", index_bullish=False,
                      index_symbol="SOXX", adverse_index_scale=0.5)
    assert bad[0]["position_scale"] == 0.5
    assert "缩减" in bad[0]["tranche_text"]


def test_unknown_index_leaves_sizing_untouched():
    from jta.plans import build_plans

    sups = [level("S1", 92.0, 0.8), level("S2", 85.0, 1.5)]
    plans = build_plans(sups, [level("R1", 112.0, 1.2, "resistance")],
                        atr=10.0, zone="at_support", index_bullish=None)
    assert plans[0]["position_scale"] == 1.0


def test_sizing_warns_when_notional_exceeds_the_account():
    r = analyze("TEST", provider=FakeProvider(), account=1000.0, include_events=False)
    for p in r["plans"]:
        sz = p.get("sizing")
        if sz and sz.get("notional", 0) > 1000.0:
            assert "超过账户净值" in sz["warning"]
            break


def test_past_earnings_date_does_not_trigger_event_mode():
    """数据源的日程可能滞后。把已发生的财报钳成"0 个交易日后"会让事件模式
    一直挂着，而那时风险其实已经随财报释放。"""
    import numpy as np
    import pandas as pd

    from jta.events import fetch_events
    from jta.indicators.atr import atr

    idx = pd.DatetimeIndex(
        list(pd.date_range("2026-08-01", periods=40, freq="B", tz="America/New_York"))
    )
    df = pd.DataFrame(
        {"open": 100.0, "high": 102.0, "low": 98.0, "close": 100.0, "volume": 1000.0},
        index=idx,
    )

    class FakeTicker:
        calendar = {"Earnings Date": [pd.Timestamp("2026-09-03").date()]}
        earnings_dates = None
        news = []

    import jta.events as ev

    real = __import__("yfinance")
    orig = real.Ticker
    real.Ticker = lambda *a, **k: FakeTicker()
    try:
        # 最后一根 bar 在财报日之后 → 应识别为已发生，不触发事件模式
        out = ev.fetch_events("X", df, atr(df))
    finally:
        real.Ticker = orig

    if out.get("available"):
        ne = out["next_earnings"]
        assert ne["sessions_away"] < 0 and ne["already_reported"] is True
        assert out["event_mode"] is False
        assert any("已过去" in w for w in out["warnings"])
