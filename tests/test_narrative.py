"""叙事数字校验测试（不联网）。"""

from __future__ import annotations

import json

from jta.narrative import data_fingerprint, extract_numbers, save_narrative, validate

PAYLOAD = {
    "week": "2026-W41", "monday": "2026-10-05", "friday": "2026-10-09",
    "generated_at": "2026-10-10T09:00:00-04:00",
    "rules": {"lookback_weeks": [1, 4, 13], "ema": [10, 20], "underperform_rs_pct": -3.0},
    "sections": [{"name": "存储", "members": [{
        "symbol": "MU", "close": 1029.0, "high": 1089.22, "close_pos": 0.23,
        "ret_pct": {"1w": -4.27, "4w": 5.51}, "move_atr": -0.38, "dist_ema10_atr": 0.23,
        "rs_reference": {"chg_4w_pct": 9.03}, "warnings": [],
        "label_reasons": ["转弱：收盘 169.03 跌破上周低点 179.31"],
        "aggregate_hint": {"above_ema10": 1, "n": 4},
    }]}],
}


def _flags(md, sources=None):
    return validate(md, PAYLOAD, sources)[1]


def test_numbers_found_in_payload_pass():
    md = ("MU 本周下跌 4.27%，4 周仍涨 5.5%，对 DRAM 的 RS 4 周 +9.03%。"
          "收在当周振幅 23% 处，离 10 周 EMA 0.23 倍周 ATR。")
    assert _flags(md) == []


def test_rounding_and_sign_are_tolerated():
    assert _flags("MU 本周 -4.3%。") == []        # -4.27 四舍五入
    assert _flags("MU 本周跌了 4%。") == []         # 整数位
    assert _flags("MU 本周 −4.27%。") == []        # Unicode 减号


def test_fabricated_number_is_flagged_and_sentence_marked():
    md = "MU 本周下跌 4.27%。HBM 价格本周上涨 18%。"
    marked, flags = validate(md, PAYLOAD)
    assert [f["number"] for f in flags] == ["18"]
    assert "==HBM 价格本周上涨 18%。==" in marked
    assert "==MU" not in marked


def test_typed_matching_percent_vs_price():
    # 1029 是价格，不能当百分比用
    assert _flags("MU 本周涨了 1029%。")
    assert _flags("MU 收在 1029。") == []
    # 0.38 是 ATR 倍数；冒充百分比不行
    assert _flags("MU 周幅 0.38 倍周 ATR。") == []
    assert _flags("MU 周涨 0.38%。")


def test_numbers_in_text_fields_count():
    assert _flags("SKHY 收盘 169.03，跌破上周低点 179.31。") == []


def test_source_facts_extend_allowed_numbers():
    src = [{"id": "1", "title": "Micron results", "url": "https://example.com",
            "facts": ["HBM 合约价上涨 18%"]}]
    assert _flags("同期消息：HBM 合约价上涨 18%（[来源 1](https://example.com)）。", src) == []


def test_dates_links_and_identifiers_are_ignored():
    md = ("2026-10-05 至 10-09（W41），EMA10 与 H2 不算数字；"
          "见 [来源 7](https://example.com/a-2026-99)。Q3 财报在 10 月 23 日。")
    assert _flags(md) == []


def test_headings_and_list_items_keep_prefix_when_flagged():
    marked, flags = validate("## 存储 跌 77%\n- MU 涨 66%", PAYLOAD)
    assert marked.splitlines() == ["## ==存储 跌 77%==", "- ==MU 涨 66%=="]
    assert len(flags) == 2


def test_existing_marks_are_reset():
    marked, flags = validate("==MU 本周下跌 4.27%。==", PAYLOAD)
    assert flags == [] and "==" not in marked


def test_extract_kinds():
    kinds = {n.raw.strip(): n.kind for n in extract_numbers("涨 4.3%，周幅 1.2 倍 ATR，价格 1029")}
    assert kinds == {"4.3": "pct", "1.2": "atr", "1029": "plain"}


def test_save_narrative_and_fingerprint(tmp_path):
    doc = save_narrative(tmp_path, PAYLOAD, "MU 跌 4.27%。编的 55.5%。", model="claude-opus-5-5")
    on_disk = json.loads((tmp_path / "2026-W41.narrative.json").read_text(encoding="utf-8"))
    assert on_disk["flags"] == doc["flags"] and len(doc["flags"]) == 1
    assert on_disk["data_fingerprint"] == data_fingerprint(PAYLOAD)
    # 只有生成时间变化，指纹不变；数字变化，指纹就变
    assert data_fingerprint({**PAYLOAD, "generated_at": "x"}) == data_fingerprint(PAYLOAD)
    changed = json.loads(json.dumps(PAYLOAD))
    changed["sections"][0]["members"][0]["close"] = 1030.0
    assert data_fingerprint(changed) != data_fingerprint(PAYLOAD)


# ---------------------------------------------------------------- 下周预测

import pytest

from jta.narrative import PredictionError, validate_predictions
from jta.weekly import realized_direction, score_predictions, scorecard, target_move


def _wk(week, monday, moves):
    """moves: {symbol: (ret_pct_1w, move_atr)}；DRAM/SOXX/SKYY 作参照，其余作成员。"""
    refs = {"DRAM": "存储", "SOXX": "半导体设计", "SKYY": "云计算"}
    secs = [{"name": "指数", "reference": None, "members": []}]
    for sym, sec in refs.items():
        if sym in moves:
            r, a = moves[sym]
            secs.append({"name": sec, "reference": sym, "members": [],
                         "reference_metrics": {"ret_pct": {"1w": r}, "move_atr": a, "close": 57.19}})
    lit = {"name": "光模块", "reference": "SOXX", "members": []}
    for sym in ("QQQ", "LITE", "COHR"):
        if sym in moves:
            r, a = moves[sym]
            m = {"symbol": sym, "ret_pct": {"1w": r}, "move_atr": a, "close": 100.0}
            (secs[0] if sym == "QQQ" else lit)["members"].append(m)
    secs.append(lit)
    return {"week": week, "monday": monday, "friday": monday, "generated_at": "x",
            "sections": secs, "rules": {}}


def test_validate_predictions_accepts_good_and_rejects_bad():
    p = _wk("2026-W41", "2026-10-05", {"DRAM": (-7.4, -1.0)})
    good = [{"section": "存储", "direction": "down", "confidence": "mid",
             "rationale": "DRAM 本周跌 7.4%。", "invalidation": {"symbol": "DRAM", "side": "above", "price": 57.19}}]
    out, flags = validate_predictions(good, p)
    assert flags == [] and out[0]["direction"] == "down"

    with pytest.raises(PredictionError, match="板块只能是"):
        validate_predictions([{**good[0], "section": "其他"}], p)
    with pytest.raises(PredictionError, match="direction"):
        validate_predictions([{**good[0], "direction": "bullish"}], p)
    with pytest.raises(PredictionError, match="不在周报 JSON 里"):
        validate_predictions([{**good[0], "invalidation": {"symbol": "DRAM", "side": "above", "price": 60}}], p)
    with pytest.raises(PredictionError, match="只能有一条"):
        validate_predictions(good + good, p)


def test_prediction_rationale_numbers_are_flagged():
    p = _wk("2026-W41", "2026-10-05", {"DRAM": (-7.4, -1.0)})
    pr = [{"section": "存储", "direction": "down", "confidence": "low",
           "rationale": "DRAM 跌 7.4%，HBM 价格跌 33%。",
           "invalidation": {"symbol": "DRAM", "side": "above", "price": 57.19}}]
    out, flags = validate_predictions(pr, p)
    assert [f["number"] for f in flags] == ["33"] and flags[0]["where"].startswith("预测 #1")
    assert out[0]["rationale"] == "==DRAM 跌 7.4%，HBM 价格跌 33%。=="   # 按句号分句，整句标红


def test_target_move_and_direction():
    p = _wk("2026-W42", "2026-10-12", {"LITE": (4.0, 0.5), "COHR": (-1.0, -0.1), "SKYY": (0.2, 0.1)})
    m = target_move(p, "光模块")
    assert m["target"] == "LITE+COHR" and m["ret_pct"] == 1.5 and m["move_atr"] == 0.2
    assert m["direction"] == "flat"                    # 0.2 < 0.3
    assert target_move(p, "云计算")["direction"] == "flat"
    assert target_move(p, "存储") is None              # 缺数据
    assert realized_direction(0.3) == "up" and realized_direction(-0.31) == "down"


def test_scorecard_scores_last_week_and_accumulates(tmp_path):
    w40 = _wk("2026-W40", "2026-09-28", {"DRAM": (2.0, 0.4), "SKYY": (1.0, 0.5)})
    w41 = _wk("2026-W41", "2026-10-05", {"DRAM": (-7.4, -1.0), "SKYY": (4.0, 0.9)})
    w42 = _wk("2026-W42", "2026-10-12", {"DRAM": (3.0, 0.6), "SKYY": (-0.1, -0.05)})
    for p in (w40, w41):
        (tmp_path / f"{p['week']}.json").write_text(json.dumps(p), encoding="utf-8")
    preds40 = [{"section": "存储", "direction": "up", "confidence": "high"},
               {"section": "云计算", "direction": "up", "confidence": "low"}]
    preds41 = [{"section": "存储", "direction": "down", "confidence": "mid"},
               {"section": "云计算", "direction": "flat", "confidence": "mid"}]
    for wk, pr in (("2026-W40", preds40), ("2026-W41", preds41)):
        (tmp_path / f"{wk}.narrative.json").write_text(json.dumps({"predictions": pr}), encoding="utf-8")

    sc = scorecard(tmp_path, w42)
    last = sc["last_week"]
    assert last["made_in"] == "2026-W41"
    rows = {r["section"]: r for r in last["rows"]}
    assert rows["存储"]["actual"] == "up" and rows["存储"]["hit"] is False
    assert rows["存储"]["momentum"] == "down" and rows["存储"]["momentum_hit"] is False
    assert rows["云计算"]["actual"] == "flat" and rows["云计算"]["hit"] is True
    # W40 的两条按 W41 判定：存储 up→down 未中，云计算 up→up 命中
    c = sc["cumulative"]
    assert c["n"] == 4 and c["hits"] == 2 and c["all_up_hits"] == 2
    assert c["by_confidence"]["mid"] == {"n": 2, "hits": 1}


def test_scorecard_none_without_predictions(tmp_path):
    assert scorecard(tmp_path, _wk("2026-W41", "2026-10-05", {})) is None


def test_save_narrative_rejects_bad_predictions_without_writing(tmp_path):
    p = _wk("2026-W41", "2026-10-05", {"DRAM": (-7.4, -1.0)})
    with pytest.raises(PredictionError):
        save_narrative(tmp_path, p, "正文", predictions=[{"section": "火星"}])
    assert not (tmp_path / "2026-W41.narrative.json").exists()
