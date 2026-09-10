"""关注列表看板。

一次跑完整个列表，产出一个汇总页加各标的的独立图。设计上只回答一个问题：
**今天有没有值得看的东西**。多数交易日答案是"没有"，页面就该让这件事一眼可见，
而不是把十一行同等权重地铺开。

无人值守运行时，抓取失败必须显性化：看板顶部挂红条，并保留上次成功的数据与时间。
静默展示过期数据比任务失败本身危险得多。
"""

from __future__ import annotations

import html as _html
import json
import traceback
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .analyze import analyze
from .chart import _json_safe, build_payload, render_html
from .data.provider import MAX_DATA_AGE_SESSIONS, DataUnavailable
from .data.yf_provider import YFinanceProvider

TEMPLATE = Path(__file__).with_name("templates") / "dashboard.html"

#: 分组顺序即优先级：需要注意的排在前面，"没事"压到最后
GROUP_ORDER = ["actionable", "event", "quiet", "failed"]
GROUP_LABELS = {
    "actionable": "有可执行计划",
    "event": "事件模式 · 计划已阻断",
    "quiet": "无可执行计划",
    "failed": "抓取失败",
}

PHASE_LABELS = {
    "downtrend": "下跌趋势", "stabilizing": "止跌", "reversal_candidate": "反转候选",
    "trend_improving": "趋势改善", "range_or_mixed": "震荡",
}
ZONE_LABELS = {"at_support": "贴近支撑", "at_resistance": "贴近压力", "between": "中间区"}
STACK_LABELS = {"bull": "多头", "bear": "空头", "mixed": "交错", "unknown": "—"}

#: 侧栏按行业分组的顺序；未登记的标的落进"其他"，排在最后
SECTOR_ORDER = ["指数/ETF", "存储", "光模块", "半导体设计", "云计算", "汽车", "航天", "其他"]
SECTOR_MAP = {
    "MU": "存储", "SNDK": "存储", "SKHY": "存储",
    "LITE": "光模块", "COHR": "光模块",
    "AVGO": "半导体设计", "AMD": "半导体设计", "INTC": "半导体设计",
    "AMZN": "云计算", "GOOGL": "云计算", "NOW": "云计算", "NET": "云计算",
    "TSLA": "汽车",
    "SPCX": "航天",
    "SOXX": "指数/ETF", "QQQ": "指数/ETF",
}

#: 侧栏显示名——非直观的代码（比如境外股票代码）换成人读得懂的名字，
#: 完整代码仍保留在链接的 title 提示里，不丢信息
DISPLAY_NAME = {
    "SKHY": "海力士",
}


@dataclass
class Row:
    symbol: str
    benchmark: str | None
    group: str = "quiet"
    sector: str = "其他"
    display: str = ""
    price: float | None = None
    prev_close: float | None = None
    change_pct: float | None = None
    atr: float | None = None
    phase: str = ""
    zone: str = ""
    daily_stack: str = ""
    intraday_stack: str = ""
    benchmark_stack: str = ""
    supports: list[dict[str, Any]] = field(default_factory=list)
    resistances: list[dict[str, Any]] = field(default_factory=list)
    plans: list[dict[str, Any]] = field(default_factory=list)
    best_rr: float | None = None
    event_mode: bool = False
    event_reason: str | None = None
    earnings: str | None = None
    stale: bool = False
    too_old: bool = False
    age_sessions: int = 0
    fetched_at: str | None = None
    structure_as_of: str | None = None
    price_is_live: bool = False
    unreliable_emas: list[str] = field(default_factory=list)
    detail_page: str | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _slug(symbol: str) -> str:
    return "".join(c for c in symbol.lower() if c.isalnum() or c in "-_")


def _brief(level: dict[str, Any]) -> dict[str, Any]:
    return {
        "label": level["label"],
        "display": level.get("display"),
        "text": level.get("display_text"),
        "hits": level["scoring"]["hits"],
        "role": level.get("role_label"),
        "distance_atr": level.get("distance_atr"),
    }


def _plan_brief(plan: dict[str, Any]) -> dict[str, Any]:
    sizing = plan.get("sizing") or {}
    return {
        "key": plan["key"],
        "title": plan["title"],
        "executable": bool(plan.get("executable")),
        "entry": plan.get("entry"),
        "stop": plan.get("stop"),
        "t1": plan.get("t1"),
        "rr": plan.get("rr"),
        "entry_level": plan.get("entry_level"),
        "trigger": plan.get("trigger"),
        "quantity": sizing.get("quantity"),
        "notional": sizing.get("notional"),
        "sizing_warning": sizing.get("warning"),
    }


def collect_row(
    symbol: str,
    benchmark: str | None,
    *,
    provider: Any,
    account: float | None,
    risk_pct: float,
) -> Row:
    row = Row(
        symbol=symbol, benchmark=benchmark,
        sector=SECTOR_MAP.get(symbol.upper(), "其他"),
        display=DISPLAY_NAME.get(symbol.upper(), symbol),
    )
    try:
        r = analyze(
            symbol, benchmark=benchmark, provider=provider,
            account=account, risk_pct=risk_pct,
        )
    except (DataUnavailable, Exception) as exc:  # noqa: BLE001
        row.group = "failed"
        row.error = f"{type(exc).__name__}: {exc}"
        return row

    daily = r["data"]["daily"]
    m = r["market_state"]
    events = r.get("events") or {}

    row.price = r["current_price"]
    row.atr = r["data"]["atr_daily"]
    row.phase = m["phase"]
    row.zone = r["position_zone"]
    row.daily_stack = m["daily_ema_stack"]
    row.intraday_stack = m["intraday_ema_stack"]
    row.benchmark_stack = (r.get("benchmark") or {}).get("ema_stack", "")
    row.supports = [_brief(l) for l in r["supports"]]
    row.resistances = [_brief(l) for l in r["resistances"]]
    row.plans = [_plan_brief(p) for p in r["plans"]]
    health = r.get("data_health") or {}
    row.stale = bool(health.get("stale"))
    row.too_old = bool(health.get("too_old"))
    row.age_sessions = int(health.get("age_sessions") or 0)
    row.fetched_at = daily.get("fetched_at")
    row.structure_as_of = r["structure_as_of"]
    row.price_is_live = r["price_is_live"]
    row.unreliable_emas = m.get("unreliable_emas", [])
    row.event_mode = bool(events.get("event_mode"))
    row.event_reason = events.get("event_mode_reason")
    row.earnings = (events.get("next_earnings") or {}).get("date")

    executable = [p for p in r["plans"] if p["executable"]]
    rrs = [p["rr"] for p in r["plans"] if p.get("rr") is not None]
    row.best_rr = max(rrs) if rrs else None
    row.group = "actionable" if executable else ("event" if row.event_mode else "quiet")

    # 与结构日之前一根的收盘比较；盘中则与最后一根完整 bar 比
    try:
        df = provider.fetch(symbol, "1d").df
        if len(df) >= 2:
            row.prev_close = round(float(df["close"].iloc[-2]), 4)
            base = float(df["close"].iloc[-1]) if not row.price_is_live else row.prev_close
            ref = row.prev_close if row.price_is_live else float(df["close"].iloc[-2])
            row.change_pct = round((row.price / ref - 1) * 100, 2) if ref else None
            row.prev_close = round(ref, 4)
    except Exception:  # noqa: BLE001
        pass
    return row


def build_watchlist(
    pairs: list[tuple[str, str | None]],
    *,
    account: float | None = None,
    risk_pct: float = 0.01,
    out_dir: Path | None = None,
    write_details: bool = True,
    provider: Any | None = None,
    refresh: bool = True,
) -> dict[str, Any]:
    """跑完整个关注列表，可选地把各标的的详情页写到 out_dir。"""
    provider = provider or YFinanceProvider(force_refresh=refresh)
    rows: list[Row] = []
    for symbol, bm in pairs:
        row = collect_row(
            symbol, bm, provider=provider, account=account, risk_pct=risk_pct
        )
        if write_details and out_dir is not None and row.error is None:
            try:
                page = f"{_slug(symbol)}.html"
                payload = build_payload(
                    symbol, benchmark=bm, account=account,
                    risk_pct=risk_pct, provider=provider,
                )
                (out_dir / page).write_text(
                    render_html(payload, standalone=True), encoding="utf-8"
                )
                row.detail_page = page
            except Exception as exc:  # noqa: BLE001
                row.error = f"详情页生成失败: {exc}"
        rows.append(row)

    order = {g: i for i, g in enumerate(GROUP_ORDER)}
    rows.sort(key=lambda r: (order.get(r.group, 9), -(r.best_rr or 0), r.symbol))

    dates = {r.structure_as_of[:10] for r in rows if r.structure_as_of}
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "data_dates": sorted(dates),
        "any_stale": any(r.stale for r in rows),
        "any_too_old": any(r.too_old for r in rows),
        "max_age_sessions": MAX_DATA_AGE_SESSIONS,
        "any_failed": any(r.group == "failed" for r in rows),
        "account": account,
        "risk_pct": risk_pct,
        "counts": {g: sum(1 for r in rows if r.group == g) for g in GROUP_ORDER},
        "rows": [r.to_dict() for r in rows],
        "labels": {
            "group": GROUP_LABELS, "phase": PHASE_LABELS,
            "zone": ZONE_LABELS, "stack": STACK_LABELS,
        },
        "sector_order": SECTOR_ORDER,
    }


def render_dashboard(payload: dict[str, Any]) -> str:
    blob = json.dumps(_json_safe(payload), ensure_ascii=False, default=str).replace("</", "<\\/")
    title = _html.escape("johnny-ta 看板", quote=False)
    return (
        TEMPLATE.read_text(encoding="utf-8")
        .replace("__TITLE__", title)
        .replace("__PAYLOAD__", blob)
    )


def parse_pairs(specs: list[str]) -> list[tuple[str, str | None]]:
    out: list[tuple[str, str | None]] = []
    for spec in specs:
        sym, _, bm = spec.partition(":")
        out.append((sym.strip().upper(), bm.strip().upper() or None))
    return out
