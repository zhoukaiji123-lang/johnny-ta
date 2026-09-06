"""前向记录与结果评估。

回答两个原 skill 无法回答的问题：
  1. 共振命中数与实际价格反应有没有相关性？
  2. 八项证据里哪几项真正贡献信息？

方法：逐日用 `as_of` 回放（输出只能看到当天为止的数据），记录当时给出的关键位，
再用**之后**的行情评估反应。关键在于有随机对照组——
不和"同样远近但没有结构依据的价位"比较，任何命中率都说明不了问题。
原 skill 自己引用的研究正是这么做的，结论是 Fib 区域与随机区域没有显著差异。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import pandas as pd

from .analyze import analyze
from .data.provider import OHLCV, SeriesMeta
from .data.yf_provider import YFinanceProvider
from .scoring import FACTORS

#: 观察窗口：记录之后多少根 bar 内算"被触及"
DEFAULT_HORIZON = 20

#: 触及判定容差（ATR 倍数）
TOUCH_ATR = 0.1

#: 判定"守住"所需的反向离开幅度（ATR 倍数）
REACT_ATR = 1.0

#: 判定"失守"所需的收盘穿越幅度（ATR 倍数）
BREAK_ATR = 0.5

#: 同一个价位在多少 ATR 内视为同一次观察（去重）
DEDUP_ATR = 0.25

#: 随机对照位与任何真实候选的最小距离（ATR 倍数）
CONTROL_MIN_GAP_ATR = 0.5

#: 随机对照位的距离范围（ATR 倍数）
CONTROL_DISTANCE_RANGE = (0.3, 3.0)


class ReplayProvider:
    """把一次性抓来的全量行情按 as_of 切片，回放期间不再触网。"""

    name = "replay"

    def __init__(self, series: dict[tuple[str, str], OHLCV]) -> None:
        self._series = series

    @classmethod
    def prefetch(
        cls, symbols: Sequence[str], intervals: Sequence[str] = ("1d", "4h")
    ) -> "ReplayProvider":
        upstream = YFinanceProvider()
        store: dict[tuple[str, str], OHLCV] = {}
        for sym in symbols:
            for iv in intervals:
                store[(sym, iv)] = upstream.fetch(sym, iv)
        return cls(store)

    def fetch(self, symbol, interval, *, as_of=None, **kw) -> OHLCV:
        base = self._series.get((symbol, interval))
        if base is None:
            raise KeyError(f"回放数据里没有 {symbol} {interval}")
        df = base.df
        if as_of is not None:
            ts = pd.Timestamp(as_of)
            if ts.tz is None:
                ts = ts.tz_localize("UTC")
            df = df[df.index <= ts.tz_convert(df.index.tz)]
        if df.empty:
            from .data.provider import DataUnavailable

            raise DataUnavailable(f"{symbol} {interval} 在 as_of={as_of} 之前没有数据")
        m = base.meta
        meta = SeriesMeta(
            symbol=m.symbol, interval=m.interval, adjust=m.adjust, tz=m.tz,
            source=self.name, fetched_at=m.fetched_at, as_of=as_of, rows=len(df),
            first_bar=df.index[0].to_pydatetime(), last_bar=df.index[-1].to_pydatetime(),
            splits=m.splits,
        )
        return OHLCV(df=df, meta=meta)

    def daily(self, symbol: str) -> pd.DataFrame:
        return self._series[(symbol, "1d")].df


# ------------------------------------------------------------------ 记录


def replay(
    symbol: str,
    provider: ReplayProvider,
    *,
    start: str,
    end: str,
    benchmark: str | None = None,
    dedup_atr: float = DEDUP_ATR,
) -> pd.DataFrame:
    """逐个交易日回放，记录当天输出的关键位。

    去重：同一个价位在观察窗口内反复出现只算一次观察，取最早那天，
    否则一个长期有效的支撑会被重复计数几十次，把统计彻底污染。
    """
    daily = provider.daily(symbol)
    days = daily.loc[(daily.index >= start) & (daily.index <= end)].index

    rows: list[dict[str, Any]] = []
    active: list[tuple[str, float, float, pd.Timestamp]] = []  # side, price, atr, seen
    for day in days:
        as_of = day + pd.Timedelta(hours=23)
        try:
            r = analyze(
                symbol, as_of=as_of, benchmark=benchmark,
                provider=provider, include_events=False,
            )
        except Exception:
            continue
        atr = r["data"]["atr_daily"]
        if not atr or not np.isfinite(atr):
            continue

        active = [a for a in active if (day - a[3]).days <= 45]
        for side_key, side in (("supports", "support"), ("resistances", "resistance")):
            for lv in r[side_key]:
                price = lv["raw_price"]
                if any(
                    s == side and abs(p - price) <= dedup_atr * atr for s, p, _, _ in active
                ):
                    continue
                active.append((side, price, atr, day))
                factors = {
                    f"f_{k}": bool(v["hit"]) for k, v in lv["scoring"]["factors"].items()
                }
                rows.append(
                    {
                        "symbol": symbol,
                        "run_date": day,
                        "kind": "level",
                        "label": lv["label"],
                        "side": side,
                        "role": lv["role"],
                        "price": price,
                        "display": lv["display"],
                        "hits": lv["scoring"]["hits"],
                        "band": lv["scoring"]["band"],
                        "distance_atr": lv["distance_atr"],
                        "atr": atr,
                        "price_at_record": r["current_price"],
                        **factors,
                    }
                )
    return pd.DataFrame(rows)


def random_controls(
    records: pd.DataFrame,
    provider: ReplayProvider,
    *,
    seed: int = 0,
    per_record: int = 1,
) -> pd.DataFrame:
    """为每条真实记录生成同样远近、但没有结构依据的对照位。

    距离现价多远本身就强烈影响"会不会被触及"，所以对照组必须控制这个变量，
    否则比较的是距离而不是结构。
    """
    if records.empty:
        return records
    rng = np.random.RandomState(seed)
    lo, hi = CONTROL_DISTANCE_RANGE
    out: list[dict[str, Any]] = []
    for (symbol, day), group in records.groupby(["symbol", "run_date"]):
        atr = float(group["atr"].iloc[0])
        price = float(group["price_at_record"].iloc[0])
        real = group["price"].to_numpy(dtype=float)
        for _, row in group.iterrows():
            for _ in range(per_record):
                for _attempt in range(20):
                    dist = rng.uniform(lo, hi) * atr
                    cand = price - dist if row["side"] == "support" else price + dist
                    if np.min(np.abs(real - cand)) > CONTROL_MIN_GAP_ATR * atr:
                        break
                else:
                    continue
                out.append(
                    {
                        "symbol": symbol,
                        "run_date": day,
                        "kind": "control",
                        "label": "CTRL",
                        "side": row["side"],
                        "role": None,
                        "price": float(cand),
                        "display": float(cand),
                        "hits": np.nan,
                        "band": "control",
                        "distance_atr": float(dist / atr),
                        "atr": atr,
                        "price_at_record": price,
                        **{f"f_{k}": False for k in FACTORS},
                    }
                )
    return pd.DataFrame(out)


# ------------------------------------------------------------------ 评估


def evaluate(
    records: pd.DataFrame,
    provider: ReplayProvider,
    *,
    horizon: int = DEFAULT_HORIZON,
    touch_atr: float = TOUCH_ATR,
    react_atr: float = REACT_ATR,
    break_atr: float = BREAK_ATR,
) -> pd.DataFrame:
    """用记录之后的行情评估每个价位实际发生了什么。

    outcome 取值：
      untested     窗口内未被触及
      hold         触及后向反方向离开 >= react_atr，且期间未收盘穿越 break_atr
      break        先出现收盘穿越 break_atr
      inconclusive 触及了，但窗口内两个条件都没达成
    """
    if records.empty:
        return records
    out = records.copy()
    for c in ("bars_to_touch", "mfe_atr", "mae_atr"):
        out[c] = np.nan
    # 日期列必须单独初始化：往 float64 列里写 Timestamp 会直接抛 TypeError
    out["touch_date"] = pd.Series(pd.NaT, index=out.index, dtype="datetime64[ns, UTC]")
    out["touched"] = False
    out["outcome"] = "untested"

    for symbol, group in out.groupby("symbol"):
        daily = provider.daily(symbol)
        idx = daily.index
        highs, lows, closes = (daily[k].to_numpy(dtype=float) for k in ("high", "low", "close"))
        for i, row in group.iterrows():
            pos = idx.searchsorted(row["run_date"], side="right")
            end = min(pos + horizon, len(idx))
            if pos >= end:
                continue
            level, atr, side = row["price"], row["atr"], row["side"]
            tol = touch_atr * atr

            probe = lows[pos:end] if side == "support" else highs[pos:end]
            hit = (probe <= level + tol) if side == "support" else (probe >= level - tol)
            where = np.flatnonzero(hit)
            if where.size == 0:
                continue
            t = pos + int(where[0])
            out.at[i, "touched"] = True
            out.at[i, "touch_date"] = idx[t].tz_convert("UTC")
            out.at[i, "bars_to_touch"] = int(t - pos)

            w_end = min(t + horizon, len(idx))
            seg_close = closes[t:w_end]
            seg_high, seg_low = highs[t:w_end], lows[t:w_end]
            if side == "support":
                broke = np.flatnonzero(seg_close <= level - break_atr * atr)
                held = np.flatnonzero(seg_high >= level + react_atr * atr)
                out.at[i, "mfe_atr"] = float((seg_high.max() - level) / atr)
                out.at[i, "mae_atr"] = float((level - seg_low.min()) / atr)
            else:
                broke = np.flatnonzero(seg_close >= level + break_atr * atr)
                held = np.flatnonzero(seg_low <= level - react_atr * atr)
                out.at[i, "mfe_atr"] = float((level - seg_low.min()) / atr)
                out.at[i, "mae_atr"] = float((seg_high.max() - level) / atr)

            first_break = broke[0] if broke.size else np.inf
            first_hold = held[0] if held.size else np.inf
            if first_hold < first_break:
                out.at[i, "outcome"] = "hold"
            elif first_break < np.inf:
                out.at[i, "outcome"] = "break"
            else:
                out.at[i, "outcome"] = "inconclusive"
    return out


# ------------------------------------------------------------------ 汇总


#: 距离分桶边界（ATR 倍数）。距现价多远强烈影响触及率与穿越概率，
#: 不分桶比较真实位与随机位，比的是距离而不是结构
DISTANCE_BUCKETS = [0.0, 0.5, 1.0, 2.0, 10.0]
BUCKET_LABELS = ["0-0.5", "0.5-1", "1-2", "2+"]

#: 单组最少样本数，低于此不报比率
MIN_SAMPLES = 20


def _rate(group: pd.DataFrame) -> dict[str, Any]:
    tested = group[group["touched"]]
    n_tested = len(tested)
    holds = int((tested["outcome"] == "hold").sum())
    breaks = int((tested["outcome"] == "break").sum())
    decided = holds + breaks
    p = holds / decided if decided else None
    return {
        "n": len(group),
        "touched": n_tested,
        "hold": holds,
        "break": breaks,
        "decided": decided,
        "hold_rate": round(p, 3) if p is not None else None,
        "stderr": round(float(np.sqrt(p * (1 - p) / decided)), 4) if decided else None,
        "mfe_atr_median": round(float(tested["mfe_atr"].median()), 3) if n_tested else None,
    }


def _compare(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    """两个比率的差异与 z 值。样本不足时明确返回 None，不给出误导性的小数。"""
    if (
        a["decided"] < MIN_SAMPLES
        or b["decided"] < MIN_SAMPLES
        or a["hold_rate"] is None
        or b["hold_rate"] is None
    ):
        return {"diff": None, "z": None, "note": f"样本不足（需各 >= {MIN_SAMPLES}）"}
    diff = a["hold_rate"] - b["hold_rate"]
    se = float(np.sqrt(a["stderr"] ** 2 + b["stderr"] ** 2))
    return {
        "diff": round(diff, 4),
        "z": round(diff / se, 2) if se > 0 else None,
        "note": None,
    }


def summarise(evaluated: pd.DataFrame) -> dict[str, Any]:
    """统计守住率，并在控制距离后与随机对照比较。

    核心比较是 by_distance——总体比较会被"真实关键位更靠近现价"这一点带偏。
    """
    if evaluated.empty:
        return {"error": "无记录"}
    df = evaluated.copy()
    df["bucket"] = pd.cut(
        df["distance_atr"], DISTANCE_BUCKETS, labels=BUCKET_LABELS, include_lowest=True
    )
    real = df[df["kind"] == "level"]
    ctrl = df[df["kind"] == "control"]

    by_distance = {}
    for b in BUCKET_LABELS:
        r, c = _rate(real[real["bucket"] == b]), _rate(ctrl[ctrl["bucket"] == b])
        by_distance[b] = {"real": r, "control": c, "comparison": _compare(r, c)}

    by_side = {}
    for side in ("support", "resistance"):
        r, c = _rate(real[real["side"] == side]), _rate(ctrl[ctrl["side"] == side])
        by_side[side] = {"real": r, "control": c, "comparison": _compare(r, c)}

    by_factor = {}
    tested_factors = 0
    for name in FACTORS:
        col = f"f_{name}"
        if col not in real.columns:
            continue
        on, off = _rate(real[real[col]]), _rate(real[~real[col]])
        cmp = _compare(on, off)
        if cmp["z"] is not None:
            tested_factors += 1
        by_factor[name] = {"with": on, "without": off, "comparison": cmp}

    return {
        "overall": {
            "real": _rate(real),
            "control": _rate(ctrl),
            "comparison": _compare(_rate(real), _rate(ctrl)),
            "warning": "总体比较未控制距离，只应参考 by_distance",
        },
        "by_hits": {int(h): _rate(g) for h, g in real.groupby("hits") if pd.notna(h)},
        "by_distance": by_distance,
        "by_side": by_side,
        "by_factor": by_factor,
        "multiple_comparisons": {
            "tests": tested_factors,
            "bonferroni_alpha": round(0.05 / tested_factors, 4) if tested_factors else None,
            "z_threshold": 2.69 if tested_factors >= 7 else None,
            "note": (
                f"逐项证据做了 {tested_factors} 次比较；单次 p<0.05（|z|>1.96）"
                "不足以宣称发现，需要 Bonferroni 校正后的阈值"
            ),
        },
        "caveats": [
            "hold_rate 的分母只含已判定样本（hold + break），不含 inconclusive",
            "样本标的高度相关时，有效样本量远小于记录条数",
            "hold/break 的判定阈值（反弹 1 ATR / 收盘穿越 0.5 ATR）是选定参数，换参数结果可能变",
            "单一时期的结果不能外推到其他市场环境",
        ],
    }


def format_summary(s: dict[str, Any]) -> str:
    """把统计结果渲染成可读文本。"""
    if "error" in s:
        return s["error"]
    L: list[str] = []
    o = s["overall"]
    L.append("=== 总体（未控制距离，仅供参考）===")
    for k, label in (("real", "真实关键位"), ("control", "随机对照")):
        v = o[k]
        L.append(
            f"  {label}: n={v['n']} 触及={v['touched']} 已判定={v['decided']} "
            f"守住率={v['hold_rate']}"
        )

    L.append("")
    L.append("=== 按距现价远近分桶（控制距离混杂）===")
    L.append(f"  {'距离':<8}{'真实 n':>8}{'守住率':>9}{'对照 n':>8}{'守住率':>9}{'差值':>9}{'z':>7}")
    for b, v in s["by_distance"].items():
        r, c, cmp = v["real"], v["control"], v["comparison"]
        if cmp["z"] is None:
            L.append(f"  {b:<8}{r['decided']:>8}{'':>9}{c['decided']:>8}{'':>9}   {cmp['note']}")
            continue
        L.append(
            f"  {b:<8}{r['decided']:>8}{r['hold_rate']:>9.1%}{c['decided']:>8}"
            f"{c['hold_rate']:>9.1%}{cmp['diff']:>+9.1%}{cmp['z']:>7.2f}"
        )

    L.append("")
    L.append("=== 按证据命中数（真实关键位）===")
    for h, v in sorted(s["by_hits"].items()):
        L.append(
            f"  {h}/8: n={v['n']:<5} 已判定={v['decided']:<5} 守住率={v['hold_rate']}"
        )

    L.append("")
    L.append("=== 各项证据的边际贡献 ===")
    L.append(f"  {'证据':<18}{'命中':>7}{'守住率':>9}{'未命中':>8}{'守住率':>9}{'差值':>9}{'z':>7}")
    for k, v in s["by_factor"].items():
        on, off, cmp = v["with"], v["without"], v["comparison"]
        if cmp["z"] is None:
            L.append(f"  {k:<18}{on['decided']:>7}   {cmp['note']}")
            continue
        L.append(
            f"  {k:<18}{on['decided']:>7}{on['hold_rate']:>9.1%}{off['decided']:>8}"
            f"{off['hold_rate']:>9.1%}{cmp['diff']:>+9.1%}{cmp['z']:>7.2f}"
        )
    mc = s["multiple_comparisons"]
    if mc["z_threshold"]:
        L.append(f"  多重比较：{mc['note']}（|z| > {mc['z_threshold']}）")

    L.append("")
    L.append("=== 局限 ===")
    for c in s["caveats"]:
        L.append(f"  · {c}")
    return "\n".join(L)
