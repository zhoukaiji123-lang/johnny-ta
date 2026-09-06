"""本地 parquet 缓存。

抓取失败时可回退缓存，但回退结果一律标记 stale=True 并保留原始 fetched_at，
不允许静默把旧数据当最新数据用。
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

DEFAULT_ROOT = Path(__file__).resolve().parents[2] / "cache"

#: 各周期缓存有效期（秒）
DEFAULT_TTL = {
    "1wk": 3600,
    "1d": 1800,
    "4h": 900,
    "60m": 900,
    "30m": 600,
    "15m": 600,
}


def _safe(name: str) -> str:
    return "".join(c if c.isalnum() or c in "-._" else "_" for c in name)


class ParquetCache:
    def __init__(self, root: Path | str | None = None) -> None:
        self.root = Path(root or os.environ.get("JTA_CACHE_DIR") or DEFAULT_ROOT)
        self.root.mkdir(parents=True, exist_ok=True)

    def key(self, source: str, symbol: str, interval: str, adjust: str) -> str:
        return f"{_safe(source)}__{_safe(symbol)}__{_safe(interval)}__{_safe(adjust)}"

    def _paths(self, key: str) -> tuple[Path, Path]:
        return self.root / f"{key}.parquet", self.root / f"{key}.meta.json"

    def read(self, key: str) -> tuple[pd.DataFrame, dict[str, Any]] | None:
        pq, mj = self._paths(key)
        if not pq.exists() or not mj.exists():
            return None
        try:
            df = pd.read_parquet(pq)
            meta = json.loads(mj.read_text())
        except Exception:
            return None
        if not isinstance(df.index, pd.DatetimeIndex) or df.index.tz is None:
            return None
        return df, meta

    def write(self, key: str, df: pd.DataFrame, meta: dict[str, Any]) -> None:
        pq, mj = self._paths(key)
        tmp_pq = pq.with_suffix(".parquet.tmp")
        df.to_parquet(tmp_pq)
        tmp_pq.replace(pq)
        mj.write_text(json.dumps(meta, ensure_ascii=False, indent=2, default=str))

    def age_seconds(self, key: str) -> float | None:
        _, mj = self._paths(key)
        if not mj.exists():
            return None
        try:
            meta = json.loads(mj.read_text())
            fetched = datetime.fromisoformat(meta["fetched_at"])
        except Exception:
            return None
        if fetched.tzinfo is None:
            fetched = fetched.replace(tzinfo=timezone.utc)
        return (datetime.now(timezone.utc) - fetched).total_seconds()

    def is_fresh(self, key: str, interval: str, ttl: dict[str, int] | None = None) -> bool:
        age = self.age_seconds(key)
        if age is None:
            return False
        table = ttl or DEFAULT_TTL
        return age < table.get(interval, 900)
