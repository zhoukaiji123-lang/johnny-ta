"""Twelve Data provider（美股，yfinance 的备用数据源）。

口径说明：
- adjust="back" 请求 Twelve Data 的 adjust=all（拆股+股息复权），与 yfinance
  auto_adjust=True 同口径。免费计划若拒绝 all（返回权限错误），自动降级为
  adjust=splits（仅拆股复权）并在 warnings 里显式标注——绝不静默降级口径。
- adjust="raw" 对应 adjust=none。
- 4H 与 yfinance 一样非原生：Twelve Data 的原生 4h 边界与本项目
  RTH-anchored（09:30-13:30 / 13:30-16:00）定义不一致，一律由 1h 数据
  按 resample.to_4h 聚合，保证两个数据源的 4H bar 口径完全相同。
- 免费计划无逐笔拆股事件端点，splits 恒为空列表，不影响价格本身
  （已经复权），只是 SeriesMeta.splits 这一项展示信息缺失。

key 从环境变量 TWELVEDATA_API_KEY 读取，不接受硬编码或参数明文传入。
"""

from __future__ import annotations

import os
import threading
import time
from datetime import datetime
from typing import Any

import pandas as pd
import requests

from .cache import ParquetCache
from .provider import (
    COLUMNS,
    Adjust,
    DataUnavailable,
    Interval,
    MAX_DATA_AGE_SESSIONS,
    OHLCV,
    SeriesMeta,
    data_age_sessions,
    normalize_index,
    split_incomplete,
    truncate_as_of,
    utcnow,
)
from .resample import BAR_ALIGNMENT_4H, audit_4h, to_4h

API_URL = "https://api.twelvedata.com/time_series"

_INTERVAL_MAP = {
    "1wk": "1week",
    "1d": "1day",
    "4h": "1h",
    "60m": "1h",
    "30m": "30min",
    "15m": "15min",
}

_ADJUST_MAP = {"back": "all", "raw": "none"}

#: 免费计划限速 8 次/分钟。一个看板批次几十个标的顺序调用很容易打穿，
#: 429 会被上层 FallbackProvider 当成"备用源也不可用"直接吞掉，表现成
#: 莫名其妙的"数据滞后"。进程内全局节流，跨 provider 实例共享。
_RATE_LIMIT_PER_MINUTE = 8
_rate_lock = threading.Lock()
_request_times: list[float] = []


def _throttle() -> None:
    with _rate_lock:
        now = time.monotonic()
        while _request_times and now - _request_times[0] >= 60:
            _request_times.pop(0)
        if len(_request_times) >= _RATE_LIMIT_PER_MINUTE:
            time.sleep(max(0.0, 60 - (now - _request_times[0]) + 0.1))
            now = time.monotonic()
            while _request_times and now - _request_times[0] >= 60:
                _request_times.pop(0)
        _request_times.append(time.monotonic())


class TwelveDataAuthError(RuntimeError):
    """缺少或无效的 TWELVEDATA_API_KEY。"""


class TwelveDataProvider:
    name = "twelvedata"

    def __init__(
        self,
        cache: ParquetCache | None = None,
        timeout: int = 20,
        force_refresh: bool = False,
        api_key: str | None = None,
    ) -> None:
        self.cache = cache or ParquetCache()
        self.timeout = timeout
        self.force_refresh = force_refresh
        self.api_key = api_key or os.environ.get("TWELVEDATA_API_KEY")

    # ------------------------------------------------------------------ 内部

    def _native_interval(self, interval: Interval) -> str:
        return _INTERVAL_MAP[interval]

    def _request(self, symbol: str, native: str, adjust_param: str) -> dict[str, Any]:
        if not self.api_key:
            raise TwelveDataAuthError(
                "未设置 TWELVEDATA_API_KEY：请在本地终端用环境变量注入，不要贴入对话"
            )
        params = {
            "symbol": symbol,
            "interval": native,
            "outputsize": 5000,
            "order": "ASC",
            "timezone": "America/New_York",
            "adjust": adjust_param,
            "apikey": self.api_key,
        }
        _throttle()
        resp = requests.get(API_URL, params=params, timeout=self.timeout)
        resp.raise_for_status()
        payload = resp.json()
        if payload.get("status") == "error":
            raise DataUnavailable(
                f"{symbol} {native} twelvedata 报错: {payload.get('message')}"
            )
        return payload

    def _download(
        self, symbol: str, native: str, adjust: Adjust
    ) -> tuple[pd.DataFrame, list[str]]:
        warnings: list[str] = []
        adjust_param = _ADJUST_MAP[adjust]
        try:
            payload = self._request(symbol, native, adjust_param)
        except DataUnavailable:
            if adjust_param != "all":
                raise
            # 免费计划可能拒绝股息复权，降级为仅拆股复权并显式标注口径差异
            payload = self._request(symbol, native, "splits")
            warnings.append(
                "twelvedata 免费计划拒绝股息复权（adjust=all），已降级为仅拆股复权"
                "（adjust=splits），与 yfinance auto_adjust 口径存在细微差异"
            )

        values = payload.get("values")
        if not values:
            raise DataUnavailable(f"{symbol} {native} twelvedata 返回空数据")

        df = pd.DataFrame(values)
        df["datetime"] = pd.to_datetime(df["datetime"])
        df = df.set_index("datetime").rename_axis("Date")
        for col in ("open", "high", "low", "close", "volume"):
            df[col] = pd.to_numeric(df[col])
        df.index = df.index.tz_localize("America/New_York")
        df = normalize_index(df[COLUMNS][~df.index.duplicated(keep="last")].sort_index())
        df = df.dropna(subset=["open", "high", "low", "close"])
        return df, warnings

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
        fetched_at = utcnow()

        use_cache = not force_refresh and self.cache.is_fresh(key, native)
        cached = self.cache.read(key) if (use_cache or not force_refresh) else None

        if use_cache and cached is not None:
            df, meta_d = cached
            df = normalize_index(df)
            fetched_at = datetime.fromisoformat(meta_d["fetched_at"])
        else:
            try:
                df, dl_warnings = self._download(symbol, native, adjust)
                warnings.extend(dl_warnings)
                self.cache.write(
                    key,
                    df,
                    {
                        "fetched_at": fetched_at.isoformat(),
                        "symbol": symbol,
                        "interval": native,
                        "adjust": adjust,
                        "splits": [],
                    },
                )
            except Exception as exc:  # noqa: BLE001 — 网络/接口异常统一降级
                if cached is None:
                    raise DataUnavailable(
                        f"{symbol} {native} twelvedata 抓取失败且无可用缓存: {exc}"
                    ) from exc
                df, meta_d = cached
                df = normalize_index(df)
                fetched_at = datetime.fromisoformat(meta_d["fetched_at"])
                stale = True
                warnings.append(f"抓取失败，回退缓存（{exc}）")

        bar_alignment = None
        if interval == "4h":
            df = to_4h(df)
            bar_alignment = BAR_ALIGNMENT_4H
            warnings.extend(audit_4h(df))
        elif native == "1h" or native in ("30min", "15min"):
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

        age = data_age_sessions(df.index[-1], as_of)
        too_old = age > MAX_DATA_AGE_SESSIONS
        if too_old:
            warnings.append(
                f"最后一根 {interval} bar 距参考时点已 {age} 个交易日"
                f"（阈值 {MAX_DATA_AGE_SESSIONS}）：数据源可能返回了旧响应，"
                "或该标的已停牌/退市"
            )

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
            splits=[],
            warnings=warnings,
            last_bar_complete=live is None,
            live_bar=live,
            age_sessions=age,
            too_old=too_old,
        )
        return OHLCV(df=df, meta=meta)
