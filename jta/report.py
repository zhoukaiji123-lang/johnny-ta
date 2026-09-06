"""报告渲染。

JSON 是给 skill 叙事层消费的唯一接口；文本渲染只为人工检视。
两者都必须把口径、数据时间和已知缺口摆在显眼位置——
一份看不出数据是什么时候的关键位表，比没有更危险。
"""

from __future__ import annotations

from typing import Any

PHASE_LABELS = {
    "downtrend": "下跌趋势",
    "stabilizing": "止跌阶段",
    "reversal_candidate": "反转候选",
    "trend_improving": "趋势改善",
    "range_or_mixed": "震荡/结构混合",
}

ZONE_LABELS = {
    "at_support": "贴近支撑",
    "at_resistance": "贴近压力",
    "between": "支撑与压力中间（默认不交易区）",
}

VEGAS_LABELS = {
    "above_vegas": "位于 Vegas 隧道上方",
    "below_vegas": "位于 Vegas 隧道下方",
    "inside_vegas": "位于 Vegas 隧道内",
    "unknown": "数据不足",
    "insufficient_history": "历史不足，无法判断长期位置",
}


def position_size(account: float, risk_pct: float, entry: float, stop: float) -> dict[str, Any]:
    distance = abs(entry - stop)
    if distance <= 0:
        return {"error": "入场价与止损价相同，无法反推仓位"}
    risk_amount = account * risk_pct
    qty = risk_amount / distance
    return {
        "risk_amount": round(risk_amount, 2),
        "stop_distance": round(distance, 4),
        "quantity": int(qty),
        "notional": round(int(qty) * entry, 2),
        "formula": "数量 = (账户净值 × 单笔风险比例) ÷ |入场价 - 止损价|",
    }


def render_text(r: dict[str, Any], account: float | None = None, risk_pct: float = 0.01) -> str:
    lines: list[str] = []
    d = r["data"]
    daily = d["daily"]

    live_tag = "（盘中，未收盘）" if r.get("price_is_live") else ""
    lines.append(
        f"{r['symbol']}  现价 {r['current_price']:,.2f}{live_tag}  日 ATR {d['atr_daily']:,.2f}"
    )
    if r.get("price_is_live"):
        lines.append(
            f"⚠ 结构计算截至 {r['structure_as_of'][:10]} 收盘；当日 bar 未走完，"
            "任何「日线收盘确认」都要等收盘后复核"
        )
    lines.append(
        f"数据：{daily['source']} · 复权={daily['adjust']} · 最后 bar {daily['last_bar']}"
        f" · 抓取于 {daily['fetched_at']}" + ("  [stale]" if daily["stale"] else "")
    )
    lines.append(
        f"展示粒度：{d['display_step']}（约 0.1 ATR）— {d['display_step_note']}"
    )
    health = r.get("data_health") or {}
    if health.get("stale"):
        lines.append("⚠ 抓取失败已回退缓存，本次结论基于旧数据，不是最新收盘")
    if health.get("too_old"):
        lines.append(
            f"⚠ 数据陈旧：最后一根 bar 距今 {health['age_sessions']} 个交易日"
            f"（阈值 {health['max_age_sessions']}）——数据源可能返回旧响应，或标的已停牌"
        )
    if d["intraday"].get("bar_alignment"):
        lines.append(f"4H 对齐：{d['intraday']['bar_alignment']}")

    ms = r["market_state"]
    lines.append("")
    lines.append(
        f"状态：{PHASE_LABELS.get(ms['phase'], ms['phase'])}"
        f" · 日线 EMA {ms['daily_ema_stack']} · 4H EMA {ms['intraday_ema_stack']}"
        f" · {VEGAS_LABELS.get(ms['long_term_vs_vegas'])}"
    )
    lines.append(f"结构：日线 {ms['daily_structure']} / 4H {ms['intraday_structure']}")
    if ms.get("unreliable_emas"):
        lines.append(
            f"⚠ warmup 不足：{'、'.join(ms['unreliable_emas'])} 数值不可信，"
            "含这些来源的关键位证据须降级看待"
        )
    if r.get("benchmark"):
        b = r["benchmark"]
        lines.append(f"基准 {b['symbol']}：EMA 排列 {b['ema_stack']}（{b['as_of_bar'][:10]}）")
    if r.get("td_signal"):
        t = r["td_signal"]
        lines.append(f"TD：{t['kind']} @ {t['ts'][:10]} 完美={t['perfected']} — {t['meaning']}")
    lines.append(f"当前位置：{ZONE_LABELS.get(r['position_zone'], r['position_zone'])}")

    for key, title in (("resistances", "压力"), ("supports", "支撑")):
        rows = r[key]
        lines.append("")
        lines.append(f"—— {title} ——")
        if not rows:
            lines.append("  （无达标关键位）")
        for lv in rows:
            sc = lv["scoring"]
            lines.append(
                f"  {lv['label']}  {lv['display_text']:>12}  {lv['role_label']}"
                f"   证据 {sc['hits']}/8 [{sc['band']}]  距现价 {lv['distance_atr']} ATR"
                f"   (原值 {lv['raw_price']:,.2f})"
            )
            lines.append(f"      来源：{', '.join(lv['evidence_families'])}")
            if lv.get("snap"):
                s = lv["snap"]
                lines.append(
                    f"      重合：Fib 原值 {s['raw_price']:,.2f} 与枢轴 "
                    f"{s['confluent_pivot']:,.2f} 相距 {s['distance_atr']} ATR"
                )
            if lv.get("dynamic"):
                lines.append("      动态位：数值随每根 bar 变化，须按数据时间复核")
            lines.append(f"      确认：{lv['confirmation']}")
            lines.append(f"      失效：{lv['invalidation']}")

        meta = r["selection"]["resistance" if key == "resistances" else "support"]
        if meta.get("gap_note"):
            lines.append(f"  [缺口] {meta['gap_note']}")
        for co in meta.get("crowded_out", [])[:3]:
            lines.append(f"  [未入选] {co['price']:,.2f}（{co['hits']}/8）— {co['reason']}")

    if account:
        # 仓位不再单独反推：三套计划各自带 sizing，且用的是计划自己的止损。
        # 保留一个独立区块会给出与计划矛盾的止损和入场档位。
        lines.append("")
        lines.append(
            f"—— 仓位口径（账户 {account:,.0f}，单笔风险 {risk_pct:.2%}）——"
        )
        lines.append(f"  {r['position_sizing']['formula']}")
        lines.append("  各计划的股数与名义金额见下方计划区块，止损以该计划自身的失效位为准")

    lines.append("")
    lines.append("—— 三套交易计划 ——")
    for p in r.get("plans", []):
        mark = "✓ 可执行" if p["executable"] else "✗ 不可执行"
        lines.append(f"  {p['title']}   {mark}")
        lines.append(
            f"      入场 {p['entry']}（{p['entry_level'] or '—'}） / 止损 {p['stop']}"
            f" / T1 {p['t1']} / T2 {p['t2']} / 收益风险比 {p['rr']}"
        )
        lines.append(f"      止损依据：{p['stop_basis']}")
        lines.append(f"      触发：{p['trigger']}")
        if p.get("entry_note"):
            lines.append(f"      [入场位选择] {p['entry_note']}")
        for c in p.get("cautions", []):
            lines.append(f"      [降级] {c}")
        sz = p.get("sizing")
        if sz and "quantity" in sz:
            lines.append(
                f"      仓位 {sz['quantity']} 股 · 名义 {sz['notional']:,.0f}"
                f" · 风险 {sz['risk_amount']:,.0f} · {p['tranche_text']}"
            )
            if sz.get("warning"):
                lines.append(f"      [警告] {sz['warning']}")
        for b in p["blocked_by"]:
            lines.append(f"      [阻断] {b}")

    ev = r.get("events") or {}
    lines.append("")
    lines.append("—— 事件 ——")
    if not ev.get("available"):
        lines.append(f"  {ev.get('reason', '无事件数据')}")
    else:
        if ev.get("event_mode"):
            lines.append(f"  ⚠ 事件模式：{ev.get('event_mode_reason')}")
            lines.append(f"  {ev.get('note', '')}")
        ne = ev.get("next_earnings")
        if ne:
            lines.append(f"  下次财报 {ne['date']}（{ne['sessions_away']} 个交易日后）")
        moves = ev.get("historical_earnings_moves") or []
        if moves:
            txt = "、".join(
                f"{m['earnings_date'][5:]} {m['move_pct']:+}%({m['move_atr']} ATR)"
                for m in moves[:4]
            )
            lines.append(f"  历史财报后次日：{txt}")

    lines.append("")
    lines.append("—— 已知缺口 ——")
    for g in r["known_gaps"]:
        lines.append(f"  · {g}")
    lines.append("")
    lines.append(f"评分说明：{r['supports'][0]['scoring']['caveat'] if r['supports'] else ''}")
    return "\n".join(lines)
