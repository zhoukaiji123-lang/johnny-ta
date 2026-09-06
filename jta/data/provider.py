"""数据源抽象层。

所有 provider 必须返回统一 schema，并把口径信息（复权方式、时区、bar 对齐规则、
数据时间、是否 stale）随数据一起带出。下游只信 meta，不再自行猜测口径。
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime, time, timedelta, timezone
from typing import Protocol, Literal, Any

import pandas as pd

Interval = Literal["1wk", "1d", "4h", "60m", "30m", "15m"]

#: 统一列名，顺序固定
COLUMNS = ["open", "high", "low", "close", "volume"]

#: 复权口径。back = 后复权（历史价调整，最新价保持真实成交价）
Adjust = Literal["back", "raw"]


@dataclass(frozen=True)
class SeriesMeta:
    symbol: str
    interval: str
    adjust: Adjust
    tz: str
    source: str
    fetched_at: datetime
    stale: bool = False
    bar_alignment: str | None = None
    as_of: datetime | None = None
    rows: int = 0
    first_bar: datetime | None = None
    last_bar: datetime | None = None
    splits: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    last_bar_complete: bool = True
    live_bar: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        for k in ("fetched_at", "as_of", "first_bar", "last_bar"):
            v = d.get(k)
            d[k] = v.isoformat() if isinstance(v, datetime) else v
        return d


@dataclass(frozen=True)
class OHLCV:
    df: pd.DataFrame
    meta: SeriesMeta

    def __post_init__(self) -> None:
        missing = [c for c in COLUMNS if c not in self.df.columns]
        if missing:
            raise ValueError(f"OHLCV 缺少列: {missing}")
        if not isinstance(self.df.index, pd.DatetimeIndex):
            raise TypeError("OHLCV.df 必须使用 DatetimeIndex")
        if self.df.index.tz is None:
            raise ValueError("OHLCV.df 索引必须带时区")
        if not self.df.index.is_monotonic_increasing:
            raise ValueError("OHLCV.df 索引必须按时间升序")

    def __len__(self) -> int:
        return len(self.df)

    @property
    def empty(self) -> bool:
        return self.df.empty


class DataUnavailable(RuntimeError):
    """行情抓取失败且无可用缓存。绝不静默返回空数据。"""


class Provider(Protocol):
    name: str

    def fetch(
        self,
        symbol: str,
        interval: Interval,
        *,
        start: str | datetime | None = None,
        end: str | datetime | None = None,
        adjust: Adjust = "back",
        as_of: datetime | None = None,
        force_refresh: bool = False,
    ) -> OHLCV:
        """取回统一 schema 的 OHLCV。

        as_of 用于前向测试：只返回 <= as_of 的 bar，杜绝 look-ahead。
        """
        ...


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


#: 美股常规盘收盘时间（交易所本地时区）
RTH_CLOSE = time(16, 0)

#: 各周期的名义长度，用于推算 bar 结束时刻
BAR_DURATION = {
    "1wk": timedelta(days=7),
    "1d": timedelta(days=1),
    "4h": timedelta(hours=4),
    "60m": timedelta(hours=1),
    "30m": timedelta(minutes=30),
    "15m": timedelta(minutes=15),
}


def bar_end(ts: "pd.Timestamp", interval: str) -> "pd.Timestamp":
    """bar 的结束时刻。日内 bar 不会越过当日收盘。"""
    session_close = ts.normalize() + pd.Timedelta(
        hours=RTH_CLOSE.hour, minutes=RTH_CLOSE.minute
    )
    if interval in ("1d", "1wk"):
        return session_close
    return min(ts + BAR_DURATION.get(interval, timedelta(hours=1)), session_close)


def is_bar_complete(ts: "pd.Timestamp", interval: str, now: datetime | None = None) -> bool:
    """该 bar 是否已经走完。

    盘中运行时 yfinance 会返回一根未完成的当日 bar：high/low/volume 都还在变。
    把它喂给 ATR、摆动点、枢轴或"日线收盘确认"，得到的全是会自己变化的结论。
    半日市（13:00 收盘）当天下午会被保守地判为未完成，代价可以接受。
    """
    now_ts = pd.Timestamp(now or utcnow())
    if now_ts.tz is None:
        now_ts = now_ts.tz_localize("UTC")
    return now_ts.tz_convert(ts.tz) >= bar_end(ts, interval)


def split_incomplete(
    df: "pd.DataFrame", interval: str, now: datetime | None = None
) -> tuple["pd.DataFrame", dict[str, Any] | None]:
    """把未完成的末根 bar 从结构计算数据中分离出来。

    它仍然有用——收盘价就是当前价——但只能用于"现价在哪"，
    不能参与任何依赖完整 OHLC 的计算。
    """
    if df.empty or is_bar_complete(df.index[-1], interval, now):
        return df, None
    row = df.iloc[-1]
    live = {
        "ts": df.index[-1].isoformat(),
        "bar_end": bar_end(df.index[-1], interval).isoformat(),
        **{k: float(row[k]) for k in COLUMNS},
        "note": "未完成 bar：high/low/volume 仍在变化，已排除在结构计算之外",
    }
    return df.iloc[:-1], live


def normalize_index(df: pd.DataFrame) -> pd.DataFrame:
    """把索引统一到纳秒精度。

    yfinance 1.6 与 parquet 往返都可能返回 datetime64[ms]。任何依赖 asi8 的整数
    时间算术都会因此按错误单位解释，必须在数据入口统一，而不是在每个下游补救。
    """
    if isinstance(df.index, pd.DatetimeIndex) and df.index.unit != "ns":
        df = df.copy()
        df.index = df.index.as_unit("ns")
    return df


def truncate_as_of(df: pd.DataFrame, as_of: datetime | None) -> pd.DataFrame:
    """按 as_of 截断。比较使用 bar 的收盘时间概念上的开始时间戳。"""
    if as_of is None:
        return df
    ts = pd.Timestamp(as_of)
    if ts.tz is None:
        ts = ts.tz_localize("UTC")
    return df[df.index <= ts.tz_convert(df.index.tz)]
