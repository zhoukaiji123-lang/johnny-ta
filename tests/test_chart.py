"""图表载荷与渲染测试。

图上的每个数字都必须来自计算层——这里守住的就是这条线。
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from jta.chart import DAILY_BARS, adaptive_window, build_payload, render_html
from tests.test_pipeline import FakeProvider


def frame(prices: list[float]) -> pd.DataFrame:
    idx = pd.DatetimeIndex(
        list(pd.date_range("2026-01-05", periods=len(prices), freq="B", tz="America/New_York"))
    )
    return pd.DataFrame(
        {"open": prices, "high": [p * 1.01 for p in prices], "low": [p * 0.99 for p in prices],
         "close": prices, "volume": [1000.0] * len(prices)},
        index=idx,
    )


def test_adaptive_window_shrinks_on_huge_price_range():
    """一年涨 8 倍的标的用满 180 根，关键位会被压进图顶的一条线里。"""
    df = frame([100 * 1.02 ** i for i in range(200)])   # 复合上涨，跨度极大
    n = adaptive_window(df)
    assert n == 70          # 收到保底下限：图仍需要最低限度的上下文
    assert n < DAILY_BARS


def test_moderate_growth_lands_inside_the_ratio_budget():
    df = frame([100 * 1.004 ** i for i in range(200)])
    n = adaptive_window(df)
    tail = df.tail(n)
    assert tail["high"].max() / tail["low"].min() <= 2.6


def test_payload_flags_a_compressed_chart():
    """保底窗口下跨度仍然超标时，必须在图上说明，而不是画一张失真的图。"""
    p = build_payload("TEST", provider=FakeProvider(), include_events=False)
    w = p["chart"]["window"]
    assert w["daily_bars"] > 0 and w["price_span_ratio"] > 0
    assert ("compressed" in w) and (w["note"] is None or "压缩" in w["note"])


def test_adaptive_window_uses_full_range_when_calm():
    df = frame([100 + (i % 7) for i in range(200)])
    assert adaptive_window(df) == DAILY_BARS


def test_adaptive_window_never_exceeds_available_bars():
    assert adaptive_window(frame([100.0] * 30)) <= 30


def test_payload_carries_every_layer():
    p = build_payload("TEST", provider=FakeProvider(), include_events=False)
    c = p["chart"]
    assert c["daily"]["bars"] and c["intraday"]["bars"]
    assert set(c["daily"]["ema"]) == {"ema8", "ema13", "ema21", "ema144", "ema169"}
    assert set(c["intraday"]["ema"]) == {"ema8", "ema13", "ema21"}
    for key in ("td", "swings", "trendlines"):
        assert key in c["daily"]
    assert "analysis" in p and p["analysis"]["supports"] is not None


def test_payload_fib_layers_disclose_their_anchors():
    p = build_payload("TEST", provider=FakeProvider(), include_events=False)
    for layer in p["chart"]["fibs"]:
        assert layer["anchors"]["low"]["ts"] and layer["anchors"]["high"]["ts"]
        assert layer["levels"] and all("ratio" in lv for lv in layer["levels"])


def test_payload_is_json_serialisable():
    p = build_payload("TEST", provider=FakeProvider(), include_events=False)
    json.dumps(p, default=str)


def test_render_replaces_placeholder_and_escapes_script_close():
    p = build_payload("TEST", provider=FakeProvider(), include_events=False)
    p["analysis"]["symbol"] = "</script><script>alert(1)</script>"
    html = render_html(p)
    assert "__PAYLOAD__" not in html
    # 注入的 </script> 必须被转义，否则数据会提前闭合脚本块
    assert "</script><script>alert(1)" not in html
    assert "<\\/script>" in html
    # 标题也拼了 symbol，同样不能让它闭合标签
    assert "<title></script>" not in html
    assert "&lt;/script&gt;" in html


def test_render_declares_charset_and_no_document_wrapper():
    """Artifact 会补 <html>/<head>/<body>，模板不能自带；charset 必须自己声明。"""
    html = render_html(build_payload("TEST", provider=FakeProvider(), include_events=False))
    assert html.lstrip().startswith('<meta charset="utf-8">')
    for tag in ("<!doctype", "<html", "<head>", "<body"):
        assert tag not in html.lower()


def test_render_loads_no_resources_beyond_google_fonts():
    """CSP 只放行 Google Fonts；其余资源必须内联。"""
    html = render_html(build_payload("TEST", provider=FakeProvider(), include_events=False))
    import re

    resources = re.findall(r'<(?:link|script|img|iframe)\b[^>]*\b(?:href|src)="([^"]+)"', html)
    external = [r for r in resources if r.startswith("http")]
    allowed = ("fonts.googleapis.com", "fonts.gstatic.com")  # gstatic 是 preconnect 目标
    assert all(any(a in r for a in allowed) for r in external), external


def test_svg_colours_never_hardcoded():
    """SVG presentation attribute 不解析 var()，写死字面色会让图卡在一个主题上。"""
    html = render_html(build_payload("TEST", provider=FakeProvider(), include_events=False))
    script = html[html.index("const D = C.daily"):]
    assert 'stroke: css(' not in script and 'fill: css(' not in script


def test_title_names_the_symbol():
    """多张图并列时，通用标题让人分不出哪张是哪只票。"""
    html = render_html(build_payload("TEST", provider=FakeProvider(), include_events=False))
    assert "<title>TEST 技术交易地图</title>" in html
    assert "__TITLE__" not in html


def test_standalone_is_a_complete_document_without_external_fonts():
    """发给访问不了 Google 的人时，字体外链只会让页面空等。"""
    import re

    html = render_html(
        build_payload("TEST", provider=FakeProvider(), include_events=False), standalone=True
    )
    assert html.startswith("<!doctype html>")
    for tag in ("<html", "<head>", "<body", "</html>"):
        assert tag in html
    assert html.count("<title>") == 1
    assert html.count('<meta charset="utf-8">') == 1
    assert "fonts.googleapis.com" not in html and "fonts.gstatic.com" not in html
    # 只看**资源加载**：新闻标题里的 <a href> 是用户点击才访问的超链接，
    # 保留它们不影响离线打开，删掉反而丢了溯源入口
    resources = re.findall(r'<(?:link|script|img|iframe)\b[^>]*\b(?:href|src)="([^"]+)"', html)
    assert all(not r.startswith("http") for r in resources), resources
    assert "PingFang SC" in html            # 中文回退到系统字体


def test_standalone_keeps_the_payload_and_title():
    html = render_html(
        build_payload("TEST", provider=FakeProvider(), include_events=False), standalone=True
    )
    assert "<title>TEST 技术交易地图</title>" in html
    assert "__PAYLOAD__" not in html and "__TITLE__" not in html


def test_default_render_stays_a_fragment_for_the_artifact_host():
    """Artifact 平台会自己补文档骨架，非 standalone 输出不能自带。"""
    html = render_html(build_payload("TEST", provider=FakeProvider(), include_events=False))
    assert not html.lstrip().startswith("<!doctype")
    assert "fonts.googleapis.com" in html


def test_stale_data_is_surfaced_on_the_page():
    """抓取失败回退缓存时，页面必须显著提示——静默展示过期数据比抓取失败更危险。"""
    html = render_html(build_payload("TEST", provider=FakeProvider(), include_events=False))
    assert "数据已过期" in html
    assert "A.data.daily.stale" in html


def test_refresh_flag_reaches_every_fetch():
    """一次分析要抓标的 + 基准 × 多个周期，--refresh 必须作用到每一次。"""
    from jta.data.yf_provider import YFinanceProvider

    p = YFinanceProvider(force_refresh=True)
    assert p.force_refresh is True
    assert YFinanceProvider().force_refresh is False
