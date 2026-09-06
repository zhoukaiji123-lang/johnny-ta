"""TD Setup（神奇九转）。

口径固定为标准 TD Sequential Setup，不再"按平台设置"：
  - Buy Setup：以 bearish price flip 起算，连续 9 根 close < close[-4]；
  - Sell Setup：以 bullish price flip 起算，连续 9 根 close > close[-4]；
  - 条件中断即重置计数。

Perfection（完美化）按标准定义检查：Buy Setup 的第 8 或第 9 根 low
需 <= 第 6、7 根 low；Sell Setup 对称。未完美的 9 提示计数可能延伸。

TD Countdown 13 尚未实现——它的规则分支（qualifier、recycle、cancel）远比
Setup 复杂，实现不当反而会制造伪信号。上层报告必须如实标注该字段缺失。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

SETUP_LENGTH = 9
LOOKBACK = 4

COUNTDOWN_IMPLEMENTED = False


def td_setup(df: pd.DataFrame) -> pd.DataFrame:
    close = df["close"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)
    high = df["high"].to_numpy(dtype=float)
    n = len(close)

    buy = np.zeros(n, dtype=int)
    sell = np.zeros(n, dtype=int)

    for i in range(LOOKBACK + 1, n):
        # Buy Setup：需要 bearish price flip 启动
        if close[i] < close[i - LOOKBACK]:
            if buy[i - 1] > 0:
                buy[i] = buy[i - 1] + 1
            elif close[i - 1] > close[i - 1 - LOOKBACK]:
                buy[i] = 1
        # Sell Setup：需要 bullish price flip 启动
        if close[i] > close[i - LOOKBACK]:
            if sell[i - 1] > 0:
                sell[i] = sell[i - 1] + 1
            elif close[i - 1] < close[i - 1 - LOOKBACK]:
                sell[i] = 1

    buy_perf = np.zeros(n, dtype=bool)
    sell_perf = np.zeros(n, dtype=bool)
    for i in range(n):
        if buy[i] == SETUP_LENGTH and i >= 3:
            buy_perf[i] = min(low[i], low[i - 1]) <= min(low[i - 2], low[i - 3])
        if sell[i] == SETUP_LENGTH and i >= 3:
            sell_perf[i] = max(high[i], high[i - 1]) >= max(high[i - 2], high[i - 3])

    return pd.DataFrame(
        {
            "td_buy_setup": buy,
            "td_sell_setup": sell,
            "td_buy_9_perfected": buy_perf,
            "td_sell_9_perfected": sell_perf,
        },
        index=df.index,
    )


def latest_td_signal(setup: pd.DataFrame, within: int = 3) -> dict | None:
    """最近 within 根内是否出现完成的 9。

    出现 9 只代表衰竭观察，不构成买卖信号——它在共振评分里最多算一项。
    """
    if setup.empty:
        return None
    tail = setup.tail(within)
    for ts, row in tail.iloc[::-1].iterrows():
        if row["td_buy_setup"] == SETUP_LENGTH:
            return {
                "kind": "buy_setup_9",
                "ts": ts.isoformat(),
                "perfected": bool(row["td_buy_9_perfected"]),
                "meaning": "下跌衰竭观察，需配合强支撑与反转 K 线",
            }
        if row["td_sell_setup"] == SETUP_LENGTH:
            return {
                "kind": "sell_setup_9",
                "ts": ts.isoformat(),
                "perfected": bool(row["td_sell_9_perfected"]),
                "meaning": "上涨衰竭观察，需配合强压力与止盈计划",
            }
    return None
