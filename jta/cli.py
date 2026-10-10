"""命令行入口。"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone

import pandas as pd

from .analyze import SELECTION_MODES, analyze
from .data.fallback_provider import SOURCES, build_provider
from .data.provider import DataUnavailable
from .report import render_text

INTERVALS = ["1wk", "1d", "4h", "60m", "30m", "15m"]

LIVE_HELP = (
    "盘中用最新成交价作现价（默认用前一交易日收盘价定当天点位，"
    "与看板一致；结构计算始终只用已收盘的 bar）"
)


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
    provider = build_provider(args.source, force_refresh=args.refresh)
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
            provider=build_provider(args.source, force_refresh=args.refresh),
            use_live=args.live,
            selection=args.selection,
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
            provider=build_provider(args.source, force_refresh=args.refresh),
            use_live=args.live,
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

    try:
        payload = build_watchlist(
            pairs, account=args.account, risk_pct=args.risk,
            out_dir=out_dir, write_details=not args.no_details, refresh=not args.no_refresh,
            provider=build_provider(args.source, force_refresh=not args.no_refresh),
            require_fresh=not args.allow_stale,
        )
    except DataUnavailable as exc:
        # 不写任何文件——宁可看板不更新，也不能让过期数据顶着"刚生成"的时间戳发布
        print(f"错误: {exc}", file=sys.stderr)
        return 2
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
        bk = (row.get("brooks") or {}).get("plan") or {}
        if bk.get("executable"):
            print(f"  ◇ {row['symbol']:<6}{row['price']:>10.2f}  Brooks {bk['setup']} "
                  f"入{bk['entry']:g} 损{bk['stop']:g} R:R {bk['rr']}")
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
    if args.brooks:
        return _backtest_brooks(args, pairs, provider, rules)

    df = run_backtest(
        pairs, provider, start=args.start, end=args.end,
        rules=rules, executable_only=args.executable_only,
        control_seeds=tuple(range(args.control)),
        min_entry_distance_atr=args.min_entry_distance, selection=args.selection,
        jobs=args.jobs,
    )
    if args.out:
        df.to_parquet(args.out)
        print(f"已写入 {args.out}（{len(df)} 行）", file=sys.stderr)

    summary = summarise_trades(df, split=args.split)
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
    if summary.get("by_regime"):
        print()
        print(f"{'大盘状态':<12}{'触发':>6}{'胜率':>8}{'期望R':>9}{'标准误':>8}")
        for k, v in summary["by_regime"].items():
            if not v["triggered"]:
                continue
            print(f"{k:<12}{v['triggered']:>6}{v['win_rate']:>8.1%}"
                  f"{v['expectancy_r']:>9.3f}{v['stderr_r'] or 0:>8.3f}")
    def gate_line(name: str, g: dict) -> str:
        d = g.get("up_minus_blocked") or {}
        if d.get("a") is None or d.get("b") is None:
            return f"{name:<14}样本不足"
        return (f"{name:<14}向上 {d['a']:+.3f}R（{d['n_a']}）· 被拦截 {d['b']:+.3f}R（{d['n_b']}）"
                f" · 差 {(d['diff_r'] or 0):+.3f}R z={d['z']}")

    def cmp_line(name: str, c: dict) -> str:
        r, k = c["real"], c["control"]
        if not r.get("triggered") or not k.get("triggered"):
            return f"{name:<14}样本不足"
        return (f"{name:<14}真实 {r['expectancy_r']:+.3f}R（{r['triggered']}）· 对照 "
                f"{k['expectancy_r']:+.3f}R（{k['triggered']}）· 差 {(c['diff_r'] or 0):+.3f}R z={c['z_diff']}")

    print()
    print("大盘开关（REGIME_GATE）：")
    print(gate_line("真实计划", summary["regime_gate"]))
    ctrl = summary.get("control")
    if ctrl:
        print(gate_line("距离匹配对照", ctrl["regime_gate"]))
        print()
        print(f"距离匹配随机对照（{ctrl['seeds']} 个种子，标准误按单个种子样本量算）：")
        print(cmp_line("全部", ctrl["overall"]))
        for k, v in ctrl["by_plan"].items():
            print(cmp_line(k, v))
    h = summary.get("halves")
    if h:
        print()
        for key, label in (("first", f"<= {h['split']}"), ("second", f"> {h['split']}")):
            part = h[key]
            print(f"[{label}]")
            if "control" in part:
                print(cmp_line("  真实 vs 对照", part["control"]))
            print(gate_line("  大盘开关", part["regime_gate"]))
    print()
    for c in summary["caveats"]:
        print(f"· {c}")
    return 0


def _backtest_brooks(args, pairs, provider, rules) -> int:
    from .backtest import run_brooks_backtest, summarise_brooks

    df = run_brooks_backtest(pairs, provider, start=args.start, end=args.end,
                             rules=rules, jobs=args.jobs)
    if args.out:
        df.to_parquet(args.out)
        print(f"已写入 {args.out}（{len(df)} 行）", file=sys.stderr)
    s = summarise_brooks(df, start=args.start, end=args.end)
    if args.json:
        print(json.dumps(s, ensure_ascii=False, indent=2, default=str))
        return 0
    if "error" in s:
        print(s["error"])
        return 0

    def line(name, c):
        r, k = c["real"], c["control"]
        if not r.get("triggered"):
            return f"{name:<14}无成交"
        return (f"{name:<14}D {r['triggered']:>4} 笔 {r['expectancy_r']:+.3f}R ±{r['stderr_r'] or 0:.3f}"
                f" | 对照 {k.get('triggered', 0):>5} 笔 {(k.get('expectancy_r') or 0):+.3f}R"
                f" | 差 {(c['diff_r'] or 0):+.3f}R z={c['z_diff']}")

    w = s["window"]
    print(f"Brooks 计划 D（BO＝突破，TR＝交易区间）· {w['start']} 至 {w['end']}（前后两半分界 {w['split']}）")
    print(line("可执行 D", s["main"]) + f" · D 对 0 的 z={s['main']['z_real']}")
    a = s["all_plans"]
    if a.get("triggered"):
        print(f"{'含不可执行':<14}D {a['triggered']:>4} 笔 {a['expectancy_r']:+.3f}R")
    print()
    for k, v in s["halves"].items():
        print(line({"first": "前半段", "second": "后半段"}[k], v))
    print()
    for k, v in s["by_setup"].items():
        print(line(f"setup {k}", v))
    print()
    for k, v in s["by_regime"].items():
        print(line(f"大盘 {k}", v))
    print()
    for k, v in s["criteria"].items():
        mark = ("✓" if v["pass"] else "✗") if "pass" in v else ("需要" if v["value"] else "不需要")
        print(f"[{mark}] {v['rule']}")
    print(f"结论：{'建议并入「有可执行计划」分组' if s['merge_recommended'] else '不并入，继续只作独立参考'}")
    print()
    for c in s["caveats"]:
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


def cmd_weekly(args: argparse.Namespace) -> int:
    from pathlib import Path

    from .weekly import build_weekly, default_week, format_weekly, parse_week, write_site

    monday = parse_week(args.week) if args.week else default_week()
    payload = build_weekly(
        monday, provider=build_provider(args.source, force_refresh=args.refresh)
    )
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))
    else:
        print(format_weekly(payload))
    if not args.no_write:
        page = write_site(payload, Path(args.out_dir))
        print(f"已写入 {page}", file=sys.stderr if args.json else sys.stdout)
    return 1 if payload["data_health"]["failed"] else 0


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
    f.add_argument(
        "--source", default="yfinance", choices=SOURCES,
        help="数据源，默认 yfinance；auto 会在滞后/失败时自动切到 twelvedata",
    )
    f.add_argument("--json", action="store_true")
    f.set_defaults(func=cmd_fetch)

    a = sub.add_parser("analyze", help="输出关键位与共振证据")
    a.add_argument("--selection", default=None, choices=SELECTION_MODES,
                   help="关键位筛选口径：legacy 按命中数 / proximity 先近后深 / structure 先近后深 + 日线结构支点")
    a.add_argument("symbol")
    a.add_argument("-b", "--benchmark", default=None, help="基准指数，如 SOXX / QQQ")
    a.add_argument("--as-of", default=None, help="只使用该时点之前的数据（前向测试）")
    a.add_argument("--account", type=float, default=None, help="账户净值，用于反推仓位")
    a.add_argument("--risk", type=float, default=0.01, help="单笔风险比例，默认 1%%")
    a.add_argument("--refresh", action="store_true", help="忽略缓存强制重抓")
    a.add_argument(
        "--source", default="auto", choices=SOURCES,
        help="数据源，默认 auto（yfinance 优先，滞后/失败时切到 twelvedata）",
    )
    a.add_argument("--json", action="store_true")
    a.add_argument("--live", action="store_true", help=LIVE_HELP)
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
    ch.add_argument("--source", default="auto", choices=SOURCES)
    ch.add_argument("--live", action="store_true", help=LIVE_HELP)
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
    db.add_argument("--source", default="auto", choices=SOURCES)
    db.add_argument(
        "--allow-stale", action="store_true",
        help="即使没拿到最新一个已收盘交易日的数据也照常发布（默认拒绝发布，只报错退出）",
    )
    db.add_argument("--json", action="store_true")
    db.set_defaults(func=cmd_dashboard)

    bt = sub.add_parser("backtest", help="按 L/B/C 计划模拟成交，统计 R 倍数期望")
    bt.add_argument("pairs", nargs="+", help="标的或 标的:基准，例如 MU:SOXX QQQ:SPY")
    bt.add_argument("--start", required=True)
    bt.add_argument("--end", required=True)
    bt.add_argument("--executable-only", action="store_true", help="只模拟通过门槛的计划")
    bt.add_argument("--out", default=None, help="把逐笔结果写入 parquet")
    bt.add_argument("--json", action="store_true")
    bt.add_argument("--brooks", action="store_true",
                    help="回测 Brooks 计划 D，并与距离匹配的随机入场对照比较")
    bt.add_argument("--jobs", type=int, default=1, help="并行进程数（按标的拆分）")
    bt.add_argument("--control", type=int, default=0,
                    help="距离匹配随机对照的种子数（每条计划复制 N 份，入场 ±0.5–1.0 ATR 平移）")
    bt.add_argument("--split", default=None, help="按计划日切成前后两段，例如 2026-06-25")
    bt.add_argument("--min-entry-distance", type=float, default=None,
                    help="覆盖 MIN_ENTRY_DISTANCE_ATR，复现距离过滤开/关对比")
    bt.add_argument("--selection", default=None, choices=SELECTION_MODES,
                    help="关键位筛选口径，覆盖 analyze.SELECTION_MODE")
    bt.set_defaults(func=cmd_backtest)

    wk = sub.add_parser("weekly", help="周报计算层：按板块看一周的趋势")
    wk.add_argument("--week", default=None,
                    help="ISO 周，如 2026-W41；默认取最近一个已走完的交易周")
    wk.add_argument("--source", default="auto", choices=SOURCES)
    wk.add_argument("--refresh", action="store_true", help="忽略缓存强制重抓")
    wk.add_argument("--out-dir", default="docs/weekly", help="输出目录，默认 docs/weekly/")
    wk.add_argument("--no-write", action="store_true", help="只打印，不写文件")
    wk.add_argument("--json", action="store_true")
    wk.set_defaults(func=cmd_weekly)

    fr = sub.add_parser("forward-report", help="从已保存的记录重新汇总")
    fr.add_argument("path")
    fr.add_argument("--json", action="store_true")
    fr.set_defaults(func=cmd_forward_report)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
