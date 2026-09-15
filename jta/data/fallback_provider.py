"""主备数据源切换。

优先 yfinance；yfinance 抓取失败，或最后一根 bar 不是当天的（age_sessions > 0），
就顺手拿备用源比一下，谁的 last_bar 更新用谁。

不拿 meta.too_old（阈值 3 个交易日）做切换判断：too_old 是为了兼容"周末+连续
假期"设计的宽松阈值，原作者注释里写"正常情况是 0–1"——也就是说 age=1 在那套
语义里本来就不算异常。但 yfinance 免费源常见的滞后恰恰就是"收盘后好几小时，
最新数据仍停在上一个交易日"，用 too_old 那套阈值根本捕捉不到，必须直接比较
两个源的 last_bar。twelvedata 免费配额 800 次/天，多打这一次比价完全够用。

切换必须在 meta.warnings 里留痕——下游（尤其是看板）只信 meta，不该在数据
来源不透明的情况下继续算.

两个数据源的复权/4H 对齐口径已经在各自 provider 里对齐（见 twelvedata_provider
文档），这里不再重复处理口径问题，只负责"用哪个源的结果"。
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from .provider import Adjust, DataUnavailable, Interval, OHLCV

#: --source 的合法取值，用于 CLI 校验
SOURCES = ["auto", "yfinance", "twelvedata"]


def build_provider(source: str = "auto", *, force_refresh: bool = False) -> Any:
    """按 --source 构造 provider。auto = yfinance 优先、twelvedata 兜底。"""
    from .twelvedata_provider import TwelveDataProvider
    from .yf_provider import YFinanceProvider

    if source == "yfinance":
        return YFinanceProvider(force_refresh=force_refresh)
    if source == "twelvedata":
        return TwelveDataProvider(force_refresh=force_refresh)
    return FallbackProvider(
        YFinanceProvider(force_refresh=force_refresh),
        TwelveDataProvider(force_refresh=force_refresh),
    )


class FallbackProvider:
    def __init__(self, primary: Any, secondary: Any, name: str = "auto") -> None:
        self.primary = primary
        self.secondary = secondary
        self.name = name

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
        kwargs = dict(
            start=start, end=end, adjust=adjust, as_of=as_of, force_refresh=force_refresh
        )

        try:
            primary_result = self.primary.fetch(symbol, interval, **kwargs)
        except DataUnavailable as primary_exc:
            try:
                secondary_result = self.secondary.fetch(symbol, interval, **kwargs)
            except Exception as secondary_exc:  # noqa: BLE001 — 两个源都失败，原样上抛主源错误
                raise DataUnavailable(
                    f"{primary_exc}；备用源 {getattr(self.secondary, 'name', '?')} "
                    f"同样不可用: {secondary_exc}"
                ) from primary_exc
            secondary_result.meta.warnings.insert(
                0, f"{self.primary.name} 不可用，已切换至 {self.secondary.name}"
            )
            return secondary_result

        if primary_result.meta.age_sessions == 0:
            return primary_result  # 已经是当天数据，不可能有更新的了

        try:
            secondary_result = self.secondary.fetch(symbol, interval, **kwargs)
        except Exception:  # noqa: BLE001 — 备用源也拿不到，保留主源结果
            return primary_result

        if secondary_result.df.empty:
            return primary_result
        if secondary_result.meta.last_bar is not None and primary_result.meta.last_bar is not None:
            if secondary_result.meta.last_bar <= primary_result.meta.last_bar:
                return primary_result

        secondary_result.meta.warnings.insert(
            0,
            f"{self.primary.name} 数据滞后（{primary_result.meta.age_sessions} 个交易日），"
            f"已切换至更新的 {self.secondary.name}",
        )
        return secondary_result
