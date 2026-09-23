"""看板"数据不新鲜就拒绝发布"的保护机制测试（不联网）。"""

from __future__ import annotations

import dataclasses

import pytest

from jta.data.provider import DataNotCurrent, DataUnavailable, OHLCV
from jta.dashboard import build_watchlist
from tests.test_pipeline import FakeProvider


class StaleAwareProvider(FakeProvider):
    """在 FakeProvider 基础上，允许指定标的返回过期数据或直接抓取失败。"""

    def __init__(self, *, stale_symbols: set[str] = frozenset(), fail_symbols: set[str] = frozenset()):
        super().__init__()
        self.stale_symbols = stale_symbols
        self.fail_symbols = fail_symbols

    def fetch(self, symbol, interval, *, as_of=None, **kw) -> OHLCV:
        if symbol in self.fail_symbols:
            raise DataUnavailable(f"{symbol} 模拟抓取失败")
        result = super().fetch(symbol, interval, as_of=as_of, **kw)
        if symbol in self.stale_symbols:
            meta = dataclasses.replace(result.meta, age_sessions=2, too_old=False)
            return OHLCV(df=result.df, meta=meta)
        return result


def test_build_watchlist_blocks_publish_when_a_symbol_is_stale(tmp_path):
    provider = StaleAwareProvider(stale_symbols={"MU"})
    with pytest.raises(DataNotCurrent) as exc_info:
        build_watchlist(
            [("MU", None), ("QQQ", None)], out_dir=tmp_path, provider=provider,
        )
    assert "MU" in str(exc_info.value)
    assert list(tmp_path.iterdir()) == []  # 一个文件都不该写


def test_build_watchlist_blocks_publish_when_a_symbol_fails(tmp_path):
    provider = StaleAwareProvider(fail_symbols={"QQQ"})
    with pytest.raises(DataNotCurrent):
        build_watchlist(
            [("MU", None), ("QQQ", None)], out_dir=tmp_path, provider=provider,
        )
    assert list(tmp_path.iterdir()) == []


def test_build_watchlist_publishes_when_everything_is_current(tmp_path):
    provider = StaleAwareProvider()
    payload = build_watchlist(
        [("MU", None), ("QQQ", None)], out_dir=tmp_path, provider=provider,
    )
    assert len(payload["rows"]) == 2
    written = {p.name for p in tmp_path.iterdir()}
    assert "mu.html" in written and "qqq.html" in written


def test_allow_stale_opts_out_of_the_gate(tmp_path):
    provider = StaleAwareProvider(stale_symbols={"MU"})
    payload = build_watchlist(
        [("MU", None), ("QQQ", None)], out_dir=tmp_path, provider=provider,
        require_fresh=False,
    )
    assert len(payload["rows"]) == 2
    assert payload["any_too_old"] is False  # too_old 阈值没触发，只是 age_sessions>0


def test_build_watchlist_deduplicates_repeated_fetches(tmp_path):
    """一个标的的 collect_row + build_payload（详情页）内部各自独立调用
    analyze()，对同一个 (symbol, interval) 会发起好几次请求；配合 twelvedata
    限速，不去重会把一次批跑拖出天际。同一批次内必须只真正打一次。"""
    provider = StaleAwareProvider()
    build_watchlist([("MU", "QQQ")], out_dir=tmp_path, provider=provider)
    counts: dict[tuple[str, str], int] = {}
    for symbol, interval in provider.calls:
        counts[(symbol, interval)] = counts.get((symbol, interval), 0) + 1
    assert counts, "没有任何请求发生，测试没测到东西"
    assert {"MU", "QQQ"} <= {s for s, _ in counts}
    assert all(n == 1 for n in counts.values()), counts
