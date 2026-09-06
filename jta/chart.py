"""交互图的数据载荷。

图上的每一个数字都必须来自计算层。这里只做取数与整形，不做任何推断——
一旦允许渲染层"补"一个点，整套确定性计算就白做了。
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .analyze import DAILY_LOOKBACK, INTRADAY_LOOKBACK, _prepare, analyze
from .data.yf_provider import YFinanceProvider
from .indicators.ema import ALL_SPANS, SHORT_SPANS, ema_set
from .indicators.swing import DEFAULT_K, detect_swings, visible_at
from .indicators.td import SETUP_LENGTH, td_setup
from .levels import fib as fibmod
from .levels.trendline import fit_trendlines

#: 图上显示的 bar 数。SVG 逐根渲染，太多既慢又看不清
DAILY_BARS = 180
INTRADAY_BARS = 120


#: 图上纵向可接受的价格跨度倍数。超过这个倍数，关键位会被压进图顶一小条里
MAX_PRICE_RATIO = 2.5


def adaptive_window(
    df: pd.DataFrame, *, max_ratio: float = MAX_PRICE_RATIO, min_bars: int = 70, max_bars: int = DAILY_BARS
) -> int:
    """自适应显示窗口。

    固定 180 根对一年涨了 8 倍的标的是灾难：y 轴被历史低点拉开，
    现价附近的全部关键位挤成图顶的一条线，图就白画了。
    从最近往前扩，直到纵向跨度超过上限为止。
    """
    n = best = min(min_bars, len(df))
    while n <= min(max_bars, len(df)):
        t = df.tail(n)
        lo, hi = float(t["low"].min()), float(t["high"].max())
        if lo <= 0 or hi / lo > max_ratio:
            break
        best, n = n, n + 10
    return min(best, len(df))


def _bars(df: pd.DataFrame, n: int) -> list[dict[str, Any]]:
    tail = df.tail(n)
    return [
        {
            "t": ts.isoformat(),
            "o": round(float(r["open"]), 4),
            "h": round(float(r["high"]), 4),
            "l": round(float(r["low"]), 4),
            "c": round(float(r["close"]), 4),
            "v": float(r["volume"]),
        }
        for ts, r in tail.iterrows()
    ]


def _ema_series(emas: pd.DataFrame, n: int, spans) -> dict[str, list[float | None]]:
    """直接取编排层算好的 EMA：在这里重算一遍会重蹈截断后起算的覆辙。"""
    cols = [f"ema{s}" for s in spans if f"ema{s}" in emas.columns]
    e = emas[cols].tail(n)
    return {
        col: [None if not np.isfinite(v) else round(float(v), 4) for v in e[col]]
        for col in e.columns
    }


def _fib_layer(swing: fibmod.Swing | None, levels: list[fibmod.FibLevel], label: str):
    if swing is None or not levels:
        return None
    return {
        "label": label,
        "anchors": swing.describe(),
        "levels": [
            {"ratio": f.ratio, "price": round(f.price, 4), "kind": f.kind}
            for f in levels
        ],
    }


def build_payload(
    symbol: str,
    *,
    benchmark: str | None = None,
    account: float | None = None,
    risk_pct: float = 0.01,
    holding: dict[str, Any] | None = None,
    daily_bars: int = DAILY_BARS,
    intraday_bars: int = INTRADAY_BARS,
    provider: Any | None = None,
    include_events: bool = True,
) -> dict[str, Any]:
    provider = provider or YFinanceProvider()
    result = analyze(
        symbol,
        benchmark=benchmark,
        provider=provider,
        account=account,
        risk_pct=risk_pct,
        holding=holding,
        include_events=include_events,
    )

    daily = _prepare(provider.fetch(symbol, "1d"), DAILY_LOOKBACK, None)
    daily_bars = adaptive_window(daily.df, max_bars=daily_bars)
    intraday = _prepare(provider.fetch(symbol, "4h"), INTRADAY_LOOKBACK, None)
    price = result["current_price"]

    # Fib 图层用完整比例集（含未入选的），因为图要表达"逐级路径"而不只是入选点
    up = fibmod.select_dominant_swing(daily.swings, "up", timeframe="1d")
    down = fibmod.select_dominant_swing(daily.swings, "down", timeframe="1d")
    local = fibmod.select_recent_swing(intraday.swings, timeframe="4h")

    fibs = []
    if up:
        fibs.append(_fib_layer(up, fibmod.primary_retracement(up), "主升回撤 Fib（日线）"))
    if down:
        fibs.append(_fib_layer(down, fibmod.primary_rebound(down), "主跌反弹 Fib（日线）"))
    if local:
        fibs.append(
            _fib_layer(local, fibmod.local_navigation(local, price), "局部导航 Fib（4H）")
        )
    fibs = [f for f in fibs if f]

    lines = []
    for kind in ("up", "down"):
        for tl in fit_trendlines(daily.df, daily.swings, kind, top_n=1):
            a, b = tl.anchors
            lines.append(
                {
                    "kind": kind,
                    "from": {"t": a["ts"], "price": a["price"]},
                    "to": {"t": b["ts"], "price": b["price"]},
                    "current": round(tl.current_value, 4),
                    "slope_per_bar": round(tl.slope, 6),
                    "anchor_index": tl.x0,
                    "touches": tl.touches,
                    "broken_at": tl.broken_at,
                    "as_of_bar": tl.as_of_bar,
                }
            )

    td = td_setup(daily.df).tail(daily_bars)
    td_marks = [
        {
            "t": ts.isoformat(),
            "kind": "buy" if row["td_buy_setup"] == SETUP_LENGTH else "sell",
            "perfected": bool(
                row["td_buy_9_perfected"] or row["td_sell_9_perfected"]
            ),
        }
        for ts, row in td.iterrows()
        if row["td_buy_setup"] == SETUP_LENGTH or row["td_sell_setup"] == SETUP_LENGTH
    ]

    swing_marks = [
        {"t": p.ts.isoformat(), "price": round(p.price, 4), "kind": p.kind, "base": p.is_base}
        for p in visible_at(daily.swings, None)
        if p.ts >= daily.df.tail(daily_bars).index[0]
    ]

    shown = daily.df.tail(daily_bars)
    span_ratio = float(shown["high"].max() / shown["low"].min())

    return {
        "analysis": result,
        "chart": {
            "window": {
                "daily_bars": daily_bars,
                "price_span_ratio": round(span_ratio, 2),
                # 保底 70 根仍然可能压缩得厉害；与其悄悄画一张失真的图，不如说出来
                "compressed": span_ratio > MAX_PRICE_RATIO,
                "note": (
                    f"纵向跨度 {span_ratio:.1f} 倍，关键位区域被压缩，"
                    "读数请以关键位表为准"
                    if span_ratio > MAX_PRICE_RATIO else None
                ),
            },
            "daily": {
                "bars": _bars(daily.df, daily_bars),
                "ema": _ema_series(daily.emas, daily_bars, ALL_SPANS),
                "td": td_marks,
                "swings": swing_marks,
                "trendlines": lines,
            },
            "intraday": {
                "bars": _bars(intraday.df, intraday_bars),
                "ema": _ema_series(intraday.emas, intraday_bars, SHORT_SPANS),
            },
            "fibs": fibs,
        },
    }


TEMPLATE = __file__.replace("chart.py", "templates/chart.html")


#: 去掉 Google Fonts 后的替代字体栈。模板里本来就写了 fallback，
#: 这里只是把首选项换成系统字体，避免在无法访问 Google 的网络里空等
OFFLINE_FONT_UI = '"PingFang SC", "Hiragino Sans GB", "Microsoft YaHei", system-ui, sans-serif'
OFFLINE_FONT_NUM = 'ui-monospace, "SF Mono", Menlo, Consolas, "Courier New", monospace'


def render_html(payload: dict[str, Any], *, standalone: bool = False) -> str:
    """把载荷注入模板。

    standalone=True 产出一份完整、可离线打开的 HTML 文档：
    补上 doctype 与 head/body 骨架（Artifact 平台平时会代劳），
    并去掉 Google Fonts 外链——发给访问不了 Google 的人时，
    外链只会让页面空等，而模板本来就备了系统字体栈。
    """
    import html as _html
    import json
    import re
    from pathlib import Path

    blob = json.dumps(payload, ensure_ascii=False, default=str).replace("</", "<\\/")
    # 标题要带标的：多张图并列时，一个通用名让人分不出哪张是哪只票。
    # symbol 进 HTML 前必须转义——它最终来自命令行参数，不是可信输入。
    title = _html.escape(f"{payload['analysis']['symbol']} 技术交易地图", quote=False)
    body = (
        Path(TEMPLATE)
        .read_text(encoding="utf-8")
        .replace("__TITLE__", title)
        .replace("__PAYLOAD__", blob)
    )
    if not standalone:
        return body

    body = re.sub(r'\s*<link rel="preconnect"[^>]*>', "", body)
    body = re.sub(r'\s*<link rel="stylesheet" href="https://fonts\.googleapis[^>]*>', "", body)
    body = body.replace(
        '--font-ui: "Archivo", "PingFang SC", "Hiragino Sans GB", "Microsoft YaHei", system-ui, sans-serif;',
        f"--font-ui: {OFFLINE_FONT_UI};",
    )
    body = body.replace(
        '--font-num: "IBM Plex Mono", ui-monospace, "SF Mono", Menlo, monospace;',
        f"--font-num: {OFFLINE_FONT_NUM};",
    )
    body = body.replace(f"<title>{title}</title>", "", 1).replace(
        '<meta charset="utf-8">', "", 1
    )
    return (
        "<!doctype html>\n"
        '<html lang="zh-CN">\n<head>\n'
        '<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        f"<title>{title}</title>\n"
        "</head>\n<body>\n"
        f"{body.strip()}\n"
        "</body>\n</html>\n"
    )
