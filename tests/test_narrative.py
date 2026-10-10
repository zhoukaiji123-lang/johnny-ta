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
