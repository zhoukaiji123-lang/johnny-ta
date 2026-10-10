"""周报叙事层的确定性部分：数字校验、消息采集、叙事落盘。

叙事由 LLM 在 Claude Code 会话里写（johnny-weekly skill），本模块不调用任何模型 API。
它只做两件可以由代码保证的事：
  1. 叙事里的每个数字必须能在计算层 JSON 或叙事引用的来源里找到。找不到的句子
     用 ==…== 标出，页面标红照常发布——不悄悄删，也不让模型自己说了算。
  2. 采集可核验的外部信息（Yahoo 新闻标题、财报日程），附来源与时间，不做判断。

校验按类型对照：带 % 的只和百分比字段比，带 "倍 / ATR" 的只和 ATR 倍数字段比，
其余数字和全部数值比。这样"+4.3%"不会因为某个价格恰好是 4.3 而蒙混过关。
校验挡得住编造和抄错，挡不住"用了一个真实存在、但放错地方的数字"——那需要人看。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable

#: 叙事里引用新闻时的数字也算合法来源，但只认 sources 里登记过的
SOURCE_TEXT_FIELDS = ("title", "facts")

_DATE_PATTERNS = [
    r"\d{4}-W\d{1,2}",          # 2026-W41
    r"\bW\d{1,2}\b",            # W41
    r"\d{4}-\d{1,2}-\d{1,2}",   # 2026-10-05
    r"\d{4}/\d{1,2}/\d{1,2}",
    r"(?<!\d)\d{1,2}-\d{1,2}(?!\d)",  # 10-05
    r"\d{1,2}\s*月\s*\d{1,2}\s*日",
    r"\d{4}\s*年",
    r"\bQ[1-4]\b",
    r"\bFY\d{2,4}\b",
]
_DATE_RE = re.compile("|".join(_DATE_PATTERNS))
_LINK_RE = re.compile(r"\[([^\]]*)\]\((https?://[^\s)]+)\)")
# 前面紧挨 ASCII 字母数字的不算（EMA10、H2）；中文字符后紧跟数字照常抽取
_NUM_RE = re.compile(r"(?<![A-Za-z0-9_.])([+\-−]?)(\d+(?:,\d{3})*(?:\.\d+)?)")
_ATR_HINT = re.compile(r"^\s*(?:倍|x|×)?\s*(?:周\s*)?ATR|^\s*倍")
_SENT_SPLIT = re.compile(r"(?<=[。！？；])")


@dataclass(frozen=True)
class Number:
    value: float
    decimals: int
    kind: str          # pct / atr / plain
    raw: str


# --------------------------------------------------------------------------- 允许的数字


def _walk(obj: Any, path: tuple[str, ...] = ()) -> Iterable[tuple[tuple[str, ...], float]]:
    if isinstance(obj, bool) or obj is None:
        return
    if isinstance(obj, (int, float)):
        yield path, float(obj)
    elif isinstance(obj, dict):
        for k, v in obj.items():
            yield from _walk(v, path + (str(k),))
    elif isinstance(obj, list):
        for v in obj:
            yield from _walk(v, path)
    elif isinstance(obj, str):
        # 标签说明等字符串里也带数字（"收盘 169.03 跌破上周低点 179.31"）
        for n in extract_numbers(obj):
            yield path + ("text",), n.value


def allowed_numbers(payload: dict[str, Any], sources: list[dict[str, Any]] | None = None) -> dict[str, set[float]]:
    out: dict[str, set[float]] = {"pct": set(), "atr": set(), "plain": set()}
    for path, v in _walk(payload):
        keys = " ".join(path).lower()
        v = abs(v)
        out["plain"].add(v)
        if "pct" in keys or "ret" in keys or "text" in keys or "rules" in keys:
            out["pct"].add(v)
        if "atr" in keys or "text" in keys or "rules" in keys:
            out["atr"].add(v)
        if path and path[-1] == "close_pos":
            out["pct"].add(round(v * 100, 4))
    for src in sources or []:
        for field in SOURCE_TEXT_FIELDS:
            val = src.get(field)
            texts = val if isinstance(val, list) else [val]
            for t in texts:
                for n in extract_numbers(str(t or "")):
                    for k in out:
                        out[k].add(abs(n.value))
    return out


# --------------------------------------------------------------------------- 抽取


def _strip(text: str) -> str:
    """去掉不参与校验的部分：链接（引用标签与 URL）、日期与周号、强调符号。"""
    text = _LINK_RE.sub(" ", text)
    text = _DATE_RE.sub(" ", text)
    return text.replace("**", "").replace("==", "")


def extract_numbers(text: str) -> list[Number]:
    s = _strip(text)
    out = []
    for m in _NUM_RE.finditer(s):
        raw = m.group(2).replace(",", "")
        val = float(raw) * (-1 if m.group(1) in "-−" and m.group(1) else 1)
        dec = len(raw.split(".")[1]) if "." in raw else 0
        tail = s[m.end():m.end() + 8]
        if tail.lstrip().startswith("%"):
            kind = "pct"
        elif _ATR_HINT.match(tail):
            kind = "atr"
        else:
            kind = "plain"
        out.append(Number(val, dec, kind, m.group(0)))
    return out


def _matches(n: Number, pool: set[float]) -> bool:
    tol = 0.5 * 10 ** -n.decimals + 1e-9
    x = abs(n.value)
    return any(abs(v - x) <= tol for v in pool)


# --------------------------------------------------------------------------- 校验


def validate(markdown: str, payload: dict[str, Any],
             sources: list[dict[str, Any]] | None = None) -> tuple[str, list[dict[str, Any]]]:
    """返回（标注后的 markdown，未通过的数字列表）。

    逐句检查；一句里有任何一个数字找不到出处，整句包进 ==…==。
    """
    pool = allowed_numbers(payload, sources)
    flags: list[dict[str, Any]] = []
    lines_out = []
    for line in markdown.replace("==", "").split("\n"):
        m = re.match(r"^(\s*(?:#{1,6}\s+|[-*]\s+|\d+\.\s+)?)(.*)$", line)
        prefix, body = m.group(1), m.group(2)
        parts = []
        for sent in _SENT_SPLIT.split(body):
            if not sent:
                continue
            bad = [n for n in extract_numbers(sent) if not _matches(n, pool[n.kind])]
            if bad:
                core = sent.rstrip()
                trail = sent[len(core):]
                parts.append(f"=={core}=={trail}")
                for n in bad:
                    flags.append({"number": n.raw.strip(), "kind": n.kind,
                                  "sentence": _LINK_RE.sub(r"\1", core).strip()})
            else:
                parts.append(sent)
        lines_out.append(prefix + "".join(parts))
    return "\n".join(lines_out), flags


# --------------------------------------------------------------------------- 落盘

#: 不参与数据指纹的字段：生成时间、页面导航、叙事本身
_VOLATILE = ("generated_at", "nav", "narrative", "narrative_stale")


def data_fingerprint(payload: dict[str, Any]) -> str:
    """计算层数字的指纹。重跑同一周、数字没变时指纹不变，叙事就不算过期。"""
    import hashlib

    core = {k: v for k, v in payload.items() if k not in _VOLATILE}
    blob = json.dumps(core, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]



def save_narrative(
    out_dir: Path,
    payload: dict[str, Any],
    markdown: str,
    *,
    sources: list[dict[str, Any]] | None = None,
    model: str | None = None,
) -> dict[str, Any]:
    marked, flags = validate(markdown, payload, sources)
    doc = {
        "week": payload["week"],
        "markdown": marked,
        "flags": flags,
        "sources": sources or [],
        "model": model,
        "written_at": datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M %Z"),
        # 计算层重跑后数字可能变化，页面据此提示叙事是否基于当前数据
        "data_generated_at": payload.get("generated_at"),
        "data_fingerprint": data_fingerprint(payload),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{payload['week']}.narrative.json").write_text(
        json.dumps(doc, ensure_ascii=False, indent=1), encoding="utf-8")
    return doc


# --------------------------------------------------------------------------- 消息采集

NEWS_PER_SYMBOL = 10
#: Yahoo 把部分新闻挂在同一公司的另一个代码上
TICKER_ALIASES = {"GOOGL": {"GOOG"}, "GOOG": {"GOOGL"}}
NEWS_NOTE = "外部新闻标题，未经核验，不参与任何计算。"


def collect_news(payload: dict[str, Any], symbols: list[str] | None = None,
                 now: datetime | None = None) -> dict[str, Any]:
    """Yahoo 新闻标题（周一 00:00 ET 起的 7 天）与全部成员的下一次财报日程。

    新闻只取 symbols（默认是周报的重点标的），财报日程查全部成员，供"下周关注"用。

    只能取到"现在"的快照：Yahoo 只返回最近十来条新闻，回补很久以前的周拿不到当周消息。
    """
    import logging

    import yfinance as yf

    logging.getLogger("yfinance").setLevel(logging.CRITICAL)
    from .weekly import ET

    monday = date.fromisoformat(payload["monday"])
    start = datetime(monday.year, monday.month, monday.day, tzinfo=ET)
    end = start + timedelta(days=7)
    members = [m["symbol"] for s in payload["sections"] for m in s["members"]]
    symbols = symbols or members
    now = now or datetime.now(timezone.utc)

    news: dict[str, list[dict[str, Any]]] = {}
    earnings: dict[str, Any] = {}
    errors: dict[str, str] = {}
    for sym in symbols:
        try:
            items = yf.Search(sym, news_count=NEWS_PER_SYMBOL).news or []
        except Exception as exc:  # noqa: BLE001
            errors[sym] = f"新闻: {type(exc).__name__}: {exc}"
            items = []
        rows = []
        for it in items:
            ts = it.get("providerPublishTime")
            if not ts:
                continue
            t = datetime.fromtimestamp(int(ts), tz=timezone.utc)
            if not (start <= t < end):
                continue
            related = set(it.get("relatedTickers") or [sym])
            if not ({sym} | TICKER_ALIASES.get(sym, set())) & related:
                continue
            rows.append({"title": it.get("title"), "publisher": it.get("publisher"),
                         "published": t.isoformat(timespec="minutes"), "url": it.get("link")})
        news[sym] = rows
    for sym in members:
        try:
            cal = yf.Ticker(sym).calendar or {}
            ed = cal.get("Earnings Date")
            ed = ed[0] if isinstance(ed, list) and ed else ed
            if ed:
                d = ed if isinstance(ed, date) else date.fromisoformat(str(ed)[:10])
                earnings[sym] = {"date": d.isoformat(),
                                 "days_away": (d - now.date()).days}
        except Exception as exc:  # noqa: BLE001
            errors[sym] = (errors.get(sym, "") + f" 财报日程: {type(exc).__name__}").strip()
    return {
        "week": payload["week"],
        "collected_at": now.isoformat(timespec="minutes"),
        "window": [start.isoformat(timespec="minutes"), end.isoformat(timespec="minutes")],
        "note": NEWS_NOTE,
        "news": news,
        "earnings": dict(sorted(earnings.items(), key=lambda kv: kv[1]["date"])),
        "errors": errors,
    }
