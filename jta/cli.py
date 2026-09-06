"""命令行入口。"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone

import pandas as pd

from .analyze import analyze
from .data.provider import DataUnavailable
from .data.yf_provider import YFinanceProvider
from .report import render_text

INTERVALS = ["1wk", "1d", "4h", "60m", "30m", "15m"]


def _parse_as_of(value: str | None) -> datetime | None:
    if not value:
        return None
    ts = pd.Timestamp(value)
    if ts.tz is None:
        # 裸日期视为该日收盘之后，避免把当天 bar 排除掉
        if ts.normalize() == ts:
            ts = ts + pd.Timedelta(hours=23, minutes=59)
        ts = ts.tz_localize("America/New_York")
    return ts.to_pydatetime().astimezone(timezone.utc)


def cmd_fetch(args: argparse.Namespace) -> int:
    provider = YFinanceProvider()
    try:
        series = provider.fetch(
            args.symbol,
            args.interval,
            adjust=args.adjust,
            as_of=_parse_as_of(args.as_of),
            force_refresh=args.refresh,
        )
    except DataUnavailable as exc:
        print(f"错误: {exc}", file=sys.stderr)
        return 2

    if args.json:
        payload = {
            "meta": series.meta.to_dict(),
            "bars": [
                {"t": t.isoformat(), **{k: float(v) for k, v in row.items()}}
                for t, row in series.df.tail(args.tail).iterrows()
            ],
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    m = series.meta
    print(f"{m.symbol}  {m.interval}  复权={m.adjust}  源={m.source}")
    print(f"  bar 数={m.rows}  区间={m.first_bar}  ->  {m.last_bar}")
    print(f"  时区={m.tz}  数据时间={m.fetched_at.isoformat()}  stale={m.stale}")
    if m.bar_alignment:
        print(f"  bar 对齐={m.bar_alignment}")
    if m.splits:
        print(f"  拆股事件={len(m.splits)} 条，最近: {m.splits[-1]}")
    for w in m.warnings:
        print(f"  [warn] {w}")
    print()
    with pd.option_context("display.width", 140, "display.max_columns", 20):
        print(series.df.tail(args.tail))
    return 0


def cmd_analyze(args: argparse.Namespace) -> int:
    try:
        result = analyze(
            args.symbol,
            as_of=_parse_as_of(args.as_of),
            benchmark=args.benchmark,
            provider=YFinanceProvider(force_refresh=args.refresh),
        )
    except DataUnavailable as exc:
        print(f"错误: {exc}", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    else:
        print(render_text(result, account=args.account, risk_pct=args.risk))
    return 0


def cmd_chart(args: argparse.Namespace) -> int:
    from .chart import build_payload, render_html

    holding = None
    if args.cost:
        holding = {"cost": args.cost, "shares": args.shares}
    try:
        payload = build_payload(
            args.symbol, benchmark=args.benchmark, account=args.account,
            risk_pct=args.risk, holding=holding,
            provider=YFinanceProvider(force_refresh=args.refresh),
        )
    except DataUnavailable as exc:
        print(f"错误: {exc}", file=sys.stderr)
        return 2

    html = render_html(payload, standalone=args.standalone)
    out = args.out or f"{args.symbol.lower()}-map.html"
    with open(out, "w", encoding="utf-8") as fh:
        fh.write(html)
    print(f"已写入 {out}（{len(html) // 1024} KB）")
    return 0


def cmd_dashboard(args: argparse.Namespace) -> int:
    from pathlib import Path

    from .dashboard import build_watchlist, parse_pairs, render_dashboard

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pairs = parse_pairs(args.pairs)

    payload = build_watchlist(
        pairs, account=args.account, risk_pct=args.risk,
        out_dir=out_dir, write_details=not args.no_details, refresh=not args.no_refresh,
    )
    index = out_dir / "index.html"
    index.write_text(render_dashboard(payload), encoding="utf-8")

    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))
        return 0

    c = payload["counts"]
    print(f"已写入 {index}")
    print(f"  有可执行计划 {c['actionable']} · 事件模式 {c['event']} · "
          f"无计划 {c['quiet']} · 失败 {c['failed']}")
    for row in payload["rows"]:
        if row["group"] == "actionable":
            plans = " / ".join(
                f"{p['title'][:1]} 入{p['entry']:g} R:R {p['rr']}"
                for p in row["plans"] if p["executable"]
            )
            print(f"  ✓ {row['symbol']:<6}{row['price']:>10.2f}  {plans}")
        elif row["group"] == "event":
            print(f"  ⚠ {row['symbol']:<6}{row['price']:>10.2f}  {row['event_reason']}")
        elif row["group"] == "failed":
            print(f"  ✗ {row['symbol']:<6}{'':>10}  {row['error']}")
    if payload["any_stale"]:
        print("  [警告] 部分标的抓取失败并回退了缓存，不是最新收盘")
    if payload["any_too_old"]:
        print(f"  [警告] 部分标的的最后一根 bar 超过 {payload['max_age_sessions']} 个交易日")
    # 无人值守时用退出码表达失败，让调度器能察觉并重试
    bad = payload["any_failed"] or payload["any_stale"] or payload["any_too_old"]
    return 1 if bad else 0


def cmd_backtest(args: argparse.Namespace) -> int:
    from .backtest import TradeRules, run_backtest, summarise_trades
    from .forward import ReplayProvider

    pairs = []
    for spec in args.pairs:
        sym, _, bm = spec.partition(":")
        pairs.append((sym.upper(), bm.upper() or None))
    provider = ReplayProvider.prefetch(sorted({s for p in pairs for s in p if s}))
    rules = TradeRules()

    df = run_backtest(
        pairs, provider, start=args.start, end=args.end,
        rules=rules, executable_only=args.executable_only,
    )
    if args.out:
        df.to_parquet(args.out)
        print(f"已写入 {args.out}（{len(df)} 行）", file=sys.stderr)

    summary = summarise_trades(df)
    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
        return 0

    o = summary["overall"]
    print(f"计划 {o['plans']} 条 · 触发 {o['triggered']} 笔（{o['trigger_rate']:.0%}）")
    if not o["triggered"]:
        print("无成交，无法统计期望")
        return 0
    print(
        f"胜率 {o['win_rate']:.1%} · 期望 {o['expectancy_r']:+.3f}R "
        f"±{o['stderr_r']} · 总计 {o['total_r']:+.1f}R"
    )
    print(f"平均盈 {o['avg_win_r']:+.2f}R · 平均亏 {o['avg_loss_r']:+.2f}R "
          f"· 最差 {o['worst_r']:+.2f}R · 跳空止损占 {o['gap_rate']:.1%}")
    print()
    print(f"{'计划类型':<12}{'触发':>6}{'胜率':>8}{'期望R':>9}{'标准误':>8}")
    for k, v in summary["by_plan"].items():
        if not v["triggered"]:
            continue
        print(f"{k:<12}{v['triggered']:>6}{v['win_rate']:>8.1%}"
              f"{v['expectancy_r']:>9.3f}{v['stderr_r'] or 0:>8.3f}")
    print()
    for c in summary["caveats"]:
        print(f"· {c}")
    return 0


def cmd_forward(args: argparse.Namespace) -> int:
    from .forward import (
        ReplayProvider,
        evaluate,
        format_summary,
        random_controls,
        replay,
        summarise,
    )

    provider = ReplayProvider.prefetch(args.symbols)
    frames = []
    for sym in args.symbols:
        rec = replay(
            sym, provider, start=args.start, end=args.end, benchmark=args.benchmark
        )
        print(f"{sym}: {len(rec)} 条记录", file=sys.stderr)
        frames.append(rec)

    records = pd.concat(frames, ignore_index=True)
    if records.empty:
        print("错误: 回放没有产生任何记录", file=sys.stderr)
        return 2
    controls = random_controls(records, provider, seed=args.seed)
    evaluated = evaluate(
        pd.concat([records, controls], ignore_index=True),
        provider,
        horizon=args.horizon,
    )
    if args.out:
        evaluated.to_parquet(args.out)
        print(f"已写入 {args.out}（{len(evaluated)} 行）", file=sys.stderr)

    summary = summarise(evaluated)
    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
    else:
        print(format_summary(summary))
    return 0


def cmd_forward_report(args: argparse.Namespace) -> int:
    from .forward import format_summary, summarise

    evaluated = pd.read_parquet(args.path)
    summary = summarise(evaluated)
    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
    else:
        print(format_summary(summary))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="jta", description="Johnny TA 计算层")
    sub = parser.add_subparsers(dest="command", required=True)

    f = sub.add_parser("fetch", help="抓取并检视 OHLCV")
    f.add_argument("symbol")
    f.add_argument("-i", "--interval", default="1d", choices=INTERVALS)
    f.add_argument("--adjust", default="back", choices=["back", "raw"])
    f.add_argument("--as-of", default=None, help="只使用该时点之前的数据（前向测试）")
    f.add_argument("--tail", type=int, default=10)
    f.add_argument("--refresh", action="store_true", help="忽略缓存强制重抓")
    f.add_argument("--json", action="store_true")
    f.set_defaults(func=cmd_fetch)

    a = sub.add_parser("analyze", help="输出关键位与共振证据")
    a.add_argument("symbol")
    a.add_argument("-b", "--benchmark", default=None, help="基准指数，如 SOXX / QQQ")
    a.add_argument("--as-of", default=None, help="只使用该时点之前的数据（前向测试）")
    a.add_argument("--account", type=float, default=None, help="账户净值，用于反推仓位")
    a.add_argument("--risk", type=float, default=0.01, help="单笔风险比例，默认 1%%")
    a.add_argument("--refresh", action="store_true", help="忽略缓存强制重抓")
    a.add_argument("--json", action="store_true")
    a.set_defaults(func=cmd_analyze)

    ch = sub.add_parser("chart", help="生成自包含的交互技术地图 HTML")
    ch.add_argument("symbol")
    ch.add_argument("-b", "--benchmark", default=None)
    ch.add_argument("--account", type=float, default=None)
    ch.add_argument("--risk", type=float, default=0.01)
    ch.add_argument("--cost", type=float, default=None, help="持仓成本价")
    ch.add_argument("--shares", type=int, default=None, help="持仓股数")
    ch.add_argument("--out", default=None)
    ch.add_argument("--refresh", action="store_true", help="忽略缓存强制重抓")
    ch.add_argument(
        "--standalone", action="store_true",
        help="产出完整 HTML 文档并去掉 Google Fonts 外链，适合转发或离线打开",
    )
    ch.set_defaults(func=cmd_chart)

    fw = sub.add_parser("forward", help="逐日回放并评估关键位的实际反应")
    fw.add_argument("symbols", nargs="+")
    fw.add_argument("--start", required=True)
    fw.add_argument("--end", required=True)
    fw.add_argument("-b", "--benchmark", default=None)
    fw.add_argument("--horizon", type=int, default=20, help="观察窗口 bar 数，默认 20")
    fw.add_argument("--seed", type=int, default=42, help="随机对照组种子")
    fw.add_argument("--out", default=None, help="把逐条记录写入 parquet")
    fw.add_argument("--json", action="store_true")
    fw.set_defaults(func=cmd_forward)

    db = sub.add_parser("dashboard", help="跑完关注列表，生成看板与各标的的技术地图")
    db.add_argument("pairs", nargs="+", help="标的或 标的:基准，例如 MU:SOXX QQQ:SPY")
    db.add_argument("--out-dir", default="docs", help="输出目录，默认 docs/")
    db.add_argument("--account", type=float, default=None)
    db.add_argument("--risk", type=float, default=0.01)
    db.add_argument("--no-details", action="store_true", help="只生成看板，不生成各标的的图")
    db.add_argument("--no-refresh", action="store_true", help="允许使用缓存（默认强制重抓）")
    db.add_argument("--json", action="store_true")
    db.set_defaults(func=cmd_dashboard)

    bt = sub.add_parser("backtest", help="按三套计划模拟成交，统计 R 倍数期望")
    bt.add_argument("pairs", nargs="+", help="标的或 标的:基准，例如 MU:SOXX QQQ:SPY")
    bt.add_argument("--start", required=True)
    bt.add_argument("--end", required=True)
    bt.add_argument("--executable-only", action="store_true", help="只模拟通过门槛的计划")
    bt.add_argument("--out", default=None, help="把逐笔结果写入 parquet")
    bt.add_argument("--json", action="store_true")
    bt.set_defaults(func=cmd_backtest)

    fr = sub.add_parser("forward-report", help="从已保存的记录重新汇总")
    fr.add_argument("path")
    fr.add_argument("--json", action="store_true")
    fr.set_defaults(func=cmd_forward_report)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
