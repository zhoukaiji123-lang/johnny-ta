"""事件与事件模式。

技术分析不覆盖尚未定价的事件。财报、指引、监管动作会让价格直接跳过关键位，
此时"支撑守住"的历史统计不适用——所以有事件在即时，必须降级而不是照常给点位。

边界：本模块只报**可核验的日程**（财报日、除息日）和**外部新闻标题**。
新闻标题是未核验的外部内容，只列出并标注来源时间，不做判断、不参与任何计算。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

#: 未来多少个交易日内有财报就进入事件模式
EVENT_HORIZON_BARS = 10

#: 新闻最多列出几条
MAX_NEWS = 8

EVENT_MODE_NOTE = (
    "事件模式：窗口内存在尚未定价的重大日程。价格可能跳空越过关键位与止损，"
    "技术信号置信度下调；不建立依赖精确入场价的新仓，已有持仓按事件风险缩减。"
)

NEWS_NOTE = "以下为外部新闻标题，未经核验，不参与任何计算；请自行点开来源确认。"


@dataclass(frozen=True)
class EarningsMove:
    earnings_date: str
    next_session: str
    move_pct: float
    move_atr: float


def _to_ts(v: Any, tz: str) -> pd.Timestamp | None:
    if v is None:
        return None
    if isinstance(v, list):
        v = v[0] if v else None
    if v is None:
        return None
    ts = pd.Timestamp(v)
    return ts.tz_localize(tz) if ts.tz is None else ts.tz_convert(tz)


def historical_earnings_moves(
    df: pd.DataFrame, earnings_dates: pd.DatetimeIndex, atr: pd.Series, limit: int = 6
) -> list[EarningsMove]:
    """历史财报后第一个交易日的跳空幅度。

    用来回答"这只票财报后通常动多少"——决定事件模式该降级多少，
    比笼统说一句"注意财报风险"有用。
    """
    out: list[EarningsMove] = []
    idx = df.index
    closes = df["close"].to_numpy(dtype=float)
    for ed in earnings_dates:
        ed = ed.tz_convert(idx.tz) if ed.tz is not None else ed.tz_localize(idx.tz)
        pos = idx.searchsorted(ed, side="right")
        if pos <= 0 or pos >= len(idx):
            continue
        prev, cur = closes[pos - 1], closes[pos]
        a = float(atr.iloc[pos - 1]) if np.isfinite(atr.iloc[pos - 1]) else np.nan
        out.append(
            EarningsMove(
                earnings_date=ed.date().isoformat(),
                next_session=idx[pos].date().isoformat(),
                move_pct=round((cur - prev) / prev * 100, 2),
                move_atr=round(abs(cur - prev) / a, 2) if np.isfinite(a) and a > 0 else float("nan"),
            )
        )
        if len(out) >= limit:
            break
    return out


def fetch_events(
    symbol: str,
    df: pd.DataFrame,
    atr: pd.Series,
    *,
    as_of: pd.Timestamp | None = None,
    horizon_bars: int = EVENT_HORIZON_BARS,
) -> dict[str, Any]:
    """取事件日程与新闻。

    as_of 明显早于今天时直接放弃：Yahoo 只给当前时点的日程和新闻，
    没有历史快照，硬拿会把今天才知道的信息塞进历史回放里。
    """
    tz = str(df.index.tz)
    today = pd.Timestamp.now(tz=tz).normalize()
    if as_of is not None and pd.Timestamp(as_of).tz_convert(tz).normalize() < today:
        return {
            "available": False,
            "reason": "事件与新闻无法按历史时点回放（数据源只提供当前快照）",
            "event_mode": False,
        }

    result: dict[str, Any] = {"available": True, "event_mode": False, "warnings": []}
    try:
        import logging

        import yfinance as yf

        # ETF 与指数没有基本面接口，yfinance 会把 404 和 "may be delisted"
        # 直接打到终端。那是正常情况，不该看起来像故障。
        logging.getLogger("yfinance").setLevel(logging.CRITICAL)
        ticker = yf.Ticker(symbol)
        cal = ticker.calendar or {}
    except Exception:  # noqa: BLE001
        return {
            "available": False,
            "reason": "该标的没有财报日程（ETF / 指数），或事件接口暂不可用",
            "event_mode": False,
        }
    if not cal:
        return {
            "available": False,
            "reason": "该标的没有财报日程（ETF / 指数）",
            "event_mode": False,
        }

    last_bar = df.index[-1]
    earn = _to_ts(cal.get("Earnings Date"), tz)
    if earn is not None:
        # 用交易日而不是自然日计算距离：10 个自然日和 10 个交易日差两周。
        # 不要钳到 0——负值表示财报已经发生，把它压成 0 会让过期日程一直
        # 触发事件模式，而那时风险其实已经释放。
        sessions = int(np.busday_count(last_bar.date(), earn.date()))
        past = sessions < 0
        result["next_earnings"] = {
            "date": earn.date().isoformat(),
            "sessions_away": sessions,
            "already_reported": past,
            "eps_estimate": cal.get("Earnings Average"),
            "eps_range": [cal.get("Earnings Low"), cal.get("Earnings High")],
        }
        if past:
            result["warnings"].append(
                f"数据源给出的财报日 {earn.date().isoformat()} 已过去 "
                f"{-sessions} 个交易日，日程尚未刷新；下次财报日未知"
            )
        elif sessions <= horizon_bars:
            result["event_mode"] = True
            result["event_mode_reason"] = (
                f"财报在 {sessions} 个交易日后（{earn.date().isoformat()}）"
            )

    for key, name in (("Ex-Dividend Date", "ex_dividend"), ("Dividend Date", "dividend")):
        ts = _to_ts(cal.get(key), tz)
        if ts is not None:
            result[name] = ts.date().isoformat()

    try:
        ed = getattr(ticker, "earnings_dates", None)
        if ed is not None and len(ed):
            past = ed[ed.index <= last_bar].index
            result["historical_earnings_moves"] = [
                m.__dict__ for m in historical_earnings_moves(df, past, atr)
            ]
    except Exception as exc:  # noqa: BLE001
        result["warnings"].append(f"历史财报反应不可用: {exc}")

    try:
        items = []
        for n in (ticker.news or [])[:MAX_NEWS]:
            c = n.get("content", n)
            items.append(
                {
                    "title": c.get("title"),
                    "published": c.get("pubDate") or c.get("providerPublishTime"),
                    "publisher": (c.get("provider") or {}).get("displayName")
                    if isinstance(c.get("provider"), dict)
                    else c.get("publisher"),
                    "link": (c.get("canonicalUrl") or {}).get("url")
                    if isinstance(c.get("canonicalUrl"), dict)
                    else c.get("link"),
                }
            )
        result["news"] = items
        result["news_note"] = NEWS_NOTE
    except Exception as exc:  # noqa: BLE001
        result["warnings"].append(f"新闻不可用: {exc}")

    if result["event_mode"]:
        result["note"] = EVENT_MODE_NOTE
    return result
