"""yfinance provider（美股）。

口径说明：
- adjust="back" 使用 yfinance auto_adjust=True，即历史价按拆股/股息向下调整、
  最新价保持真实成交价。这与整数关口、真实枢轴同口径，避免 Fib 算在复权价、
  水平位算在原始价上的错配。
- adjust="raw" 返回未调整价，仅用于锚点敏感性检查。
- 4H 非原生，由 60m 按 RTH 规则聚合，见 resample.BAR_ALIGNMENT_4H。
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import pandas as pd

from .cache import ParquetCache
from .provider import (
    COLUMNS,
    Adjust,
    DataUnavailable,
    Interval,
    OHLCV,
    SeriesMeta,
    normalize_index,
    split_incomplete,
    truncate_as_of,
    utcnow,
)
from .resample import BAR_ALIGNMENT_4H, audit_4h, to_4h

#: 各周期默认回溯长度。60m/30m/15m 受 Yahoo 侧硬限制（分别约 730 / 60 / 60 天）
DEFAULT_PERIOD = {
    "1wk": "15y",
    "1d": "10y",
    "60m": "729d",
    "30m": "59d",
    "15m": "59d",
}

_RENAME = {
    "Open": "open",
    "High": "high",
    "Low": "low",
    "Close": "close",
    "Volume": "volume",
}


class YFinanceProvider:
    name = "yfinance"

    def __init__(
        self,
        cache: ParquetCache | None = None,
        timeout: int = 20,
        force_refresh: bool = False,
    ) -> None:
        self.cache = cache or ParquetCache()
        self.timeout = timeout
        # 实例级强制刷新：命令行的 --refresh 需要作用到一次分析里的**每一次**抓取
        # （标的 + 基准 + 各周期），而不只是某一个调用点
        self.force_refresh = force_refresh

    # ------------------------------------------------------------------ 内部

    def _native_interval(self, interval: Interval) -> str:
        return "60m" if interval == "4h" else interval

    def _download(
        self,
        symbol: str,
        native: str,
        adjust: Adjust,
        start: Any,
        end: Any,
    ) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
        import yfinance as yf

        ticker = yf.Ticker(symbol)
        kwargs: dict[str, Any] = {
            "interval": native,
            "auto_adjust": adjust == "back",
            "actions": True,
            "prepost": False,
            "timeout": self.timeout,
            "raise_errors": True,
        }
        if start or end:
            kwargs["start"] = start
            kwargs["end"] = end
        else:
            kwargs["period"] = DEFAULT_PERIOD.get(native, "1y")

        raw = ticker.history(**kwargs)
        if raw is None or raw.empty:
            raise DataUnavailable(f"{symbol} {native} 返回空数据")

        splits: list[dict[str, Any]] = []
        if "Stock Splits" in raw.columns:
            s = raw["Stock Splits"]
            for ts, ratio in s[s != 0].items():
                splits.append({"date": ts.isoformat(), "ratio": float(ratio)})

        df = raw.rename(columns=_RENAME)[COLUMNS].copy()
        df = normalize_index(df[~df.index.duplicated(keep="last")].sort_index())
        df = df.dropna(subset=["open", "high", "low", "close"])
        return df, splits

    # ------------------------------------------------------------------ 对外

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
        native = self._native_interval(interval)
        force_refresh = force_refresh or self.force_refresh
        key = self.cache.key(self.name, symbol, native, adjust)
        warnings: list[str] = []
        stale = False
        splits: list[dict[str, Any]] = []
        fetched_at = utcnow()

        use_cache = (
            not force_refresh
            and start is None
            and end is None
            and self.cache.is_fresh(key, native)
        )
        cached = self.cache.read(key) if (use_cache or not force_refresh) else None

        if use_cache and cached is not None:
            df, meta_d = cached
            df = normalize_index(df)
            splits = meta_d.get("splits", [])
            fetched_at = datetime.fromisoformat(meta_d["fetched_at"])
        else:
            try:
                df, splits = self._download(symbol, native, adjust, start, end)
                if start is None and end is None:
                    self.cache.write(
                        key,
                        df,
                        {
                            "fetched_at": fetched_at.isoformat(),
                            "symbol": symbol,
                            "interval": native,
                            "adjust": adjust,
                            "splits": splits,
                        },
                    )
            except Exception as exc:  # noqa: BLE001 — 网络/接口异常统一降级
                if cached is None:
                    raise DataUnavailable(
                        f"{symbol} {native} 抓取失败且无可用缓存: {exc}"
                    ) from exc
                df, meta_d = cached
                df = normalize_index(df)
                splits = meta_d.get("splits", [])
                fetched_at = datetime.fromisoformat(meta_d["fetched_at"])
                stale = True
                warnings.append(f"抓取失败，回退缓存（{exc}）")

        bar_alignment = None
        if interval == "4h":
            df = to_4h(df)
            bar_alignment = BAR_ALIGNMENT_4H
            warnings.extend(audit_4h(df))
        elif native in ("60m", "30m", "15m"):
            bar_alignment = "交易所常规盘，时间戳为 bar 开始时间"

        df = truncate_as_of(df, as_of)
        df, live = split_incomplete(df, interval)
        if live:
            warnings.append(
                f"末根 {interval} bar（{live['ts']}）尚未收盘，已排除在结构计算之外；"
                "现价取该 bar 的最新成交价"
            )
        if df.empty:
            raise DataUnavailable(f"{symbol} {interval} 在 as_of={as_of} 之前没有数据")

        meta = SeriesMeta(
            symbol=symbol,
            interval=interval,
            adjust=adjust,
            tz=str(df.index.tz),
            source=self.name,
            fetched_at=fetched_at,
            stale=stale,
            bar_alignment=bar_alignment,
            as_of=as_of,
            rows=len(df),
            first_bar=df.index[0].to_pydatetime(),
            last_bar=df.index[-1].to_pydatetime(),
            splits=splits,
            warnings=warnings,
            last_bar_complete=live is None,
            live_bar=live,
        )
        return OHLCV(df=df, meta=meta)
