"""交易计划：A/B/C 三套互斥计划 + L 左侧支撑直入（上游 v2.0.5）。

计划全部由已入选的关键位确定性推导——入场取某一档，止损放在该档自身失效之处，
目标取对侧的下一档。这样每套计划的收益风险比是算出来的，不是估出来的，
达不到门槛就直接标为不可执行，而不是靠人自觉遵守纪律。

每套计划另带 status（等待到位 / 等待触发 / 到位可执行 / 不合格 / 数据不足，上游 v2.0.6），
由现价与入场位的相对位置、blocked_by 与数据缺失直接算出；sides() 把它们汇总成
上游要求的"左侧 / 右侧两项建议"。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

#: 止损在结构位之外的 ATR 缓冲
STOP_BUFFER_ATR = 0.25

#: 没有更深一档时的兜底缓冲
FALLBACK_BUFFER_ATR = 0.5

#: 收益风险比门槛：低吸类要求更高，右侧确认可以低一些。
#: L 与 A 同属支撑做多，沿用 2.0；1% 止损很紧，这个门槛基本只拦"目标贴着入场"的情形
MIN_RR = {"left": 2.0, "aggressive": 2.0, "deep": 2.0, "breakout": 1.5}

#: L 左侧支撑直入的止损：支撑基准 S 下方这个比例，P_stop = S × (1 − LEFT_STOP_PCT)。
#:
#: 来自上游 v2.0.5 第 8A 节，原文注明是"用户于 2026-09-27 明确指定的配置"，
#: 不是 Johnny 公开的统一参数，也不是所有股票的最优阈值。盘中价格达到或低于 P_stop 即离场，
#: 不等收盘、不加 ATR 缓冲；向更早保护的一侧取整（做多止损向上取整到 0.01）。
#: S 取关键位原值 raw_price：jta 的"约 965–975"只是按 0.1 ATR 取整的展示宽度，
#: 不是结构上的支撑带，取它的下沿当 S 等于凭空加宽止损。
LEFT_STOP_PCT = 0.01

#: 判断"已到位"的容差：现价距入场位不足这个 ATR 倍数视为到位
AT_ENTRY_ATR = 0.1

#: 入场位距现价的最小距离（ATR 倍数）。**当前默认关闭（0.0）。**
#:
#: 曾按 P5 前向验证设为 0.5：距现价 0–0.5 ATR 的关键位守住率 42.4%，
#: 而同距离随机价位是 60.4%（z=-2.54）。但 P7 计划级回测推翻了这个改动——
#: 25 标的 / 407 笔成交下，开启过滤后单笔期望 +0.498R、关闭则 +0.505R
#: （差 -0.007R，z=-0.04），毫无改善，却砍掉 35% 的交易机会、总收益少三分之一。
#:
#: 原因是两层验证测的不是同一件事：P5 测"触及后守不守得住"，
#: 而计划要求出现止跌确认信号才入场——这一步已经把会跌穿的情形过滤掉了，
#: 再叠一层距离过滤属于重复劳动。机制保留，需要时传参开启。
#:
#: ⚠ 上面 P7 的数字含回测口径偏差。2026-10 修正后重跑（25 标的 / 2025-09 至 2026-06）：
#: 开启 +0.33R（289 笔、总 +95R），关闭 +0.15R（437 笔、总 +64R），差 +0.18R、z=1.10——
#: 方向反过来支持过滤，但不显著，默认值不改。P5 修正后 0–0.5 ATR 一档也不再显著差于随机价位
#: （-5.7%，z=-0.80），当初开启过滤的依据已不存在。
MIN_ENTRY_DISTANCE_ATR = 0.0

#: 基准指数非多头时的仓位缩减系数。**当前默认不缩减（1.0）。**
#:
#: index 同向是 P5 八项证据里唯一通过多重比较校正的一项（+11.3%，z=3.61），
#: 据此曾设为 0.5。但计划级回测的方向相反：指数非多头时期望 +0.871R（n=93），
#: 多头时 +0.388R（n=314），差 +0.483R、z=1.48——不显著，
#: 但没有任何证据支持"非多头就该减半"，甚至暗示相反。
#: 2026-10 修正回测口径后重跑：非多头 +0.30R（n=99），多头 +0.07R（n=324），
#: z=-1.13，方向不变。因此不再缩减仓位。方向判断已由下面的 REGIME_GATE 接管。
ADVERSE_INDEX_SCALE = 1.0

#: 大盘状态开关：基准不在 up 状态时，三套计划一律不开新仓。**当前默认开启。**
#:
#: 状态定义见 regime.py（ADX>=20 且 close>SMA200 且 SMA50>SMA200 才算 up），
#: 规则在看回测结果之前定死。25 标的 / 2025-09 至 2026-09。
#: 最初的结果是 up 时 +0.44R、被剔除的 +0.00R、z=2.97，但那次回测有前视（计划日、突破确认日
#: 当天的 4H 参与成交）、4H / 日线复权口径不一致、4H 量能不分时段和 R 放大的问题。
#: 2026-10 全部修正后重跑，589 笔成交：up 时 +0.07R（n=273），被剔除的 +0.08R（n=300），
#: 差 -0.01R、z=-0.07；距离匹配随机入场上同样过滤 -0.09R（z=-0.50）；两段都没有区分度。
#:
#: 仍然默认开启，理由是风险而不是收益：单笔期望没有差异，开关只是把敞口减半；
#: 而它真正防范的下跌市在样本里只有 10 笔、基本没被检验过。只做多的计划在持续下跌里
#: 是什么表现没人知道，在有证据之前宁可少做。这是风控规则，不是被验证过的择时信号。
#: 基准缺失（unknown）时不拦截，只提示。只影响开新仓，不影响 holder_playbook。
REGIME_GATE = True

PLAN_TITLES = {
    "left": "L 左侧支撑直入",
    "aggressive": "A 第一支撑激进试仓",
    "deep": "B 更深支撑低吸",
    "breakout": "C 右侧突破回踩",
}

#: 第一笔占**计划总仓位**的比例，不是账户资产的比例
FIRST_TRANCHE = {"left": (0.20, 0.25), "aggressive": (0.20, 0.25), "deep": (0.25, 0.50), "breakout": (0.30, 0.50)}


@dataclass
class Plan:
    key: str
    title: str
    trigger: str
    entry: float | None
    stop: float | None
    stop_basis: str
    t1: float | None
    t2: float | None
    rr: float | None
    tranche: tuple[float, float]
    cancel_if: list[str]
    executable: bool
    blocked_by: list[str]
    entry_level: str | None = None
    entry_note: str | None = None
    position_scale: float = 1.0
    cautions: list[str] = field(default_factory=list)
    side: str = "left"
    status: str = "no_data"
    status_label: str = ""
    status_reason: str = ""
    #: False 表示尚未经回测确认，不进看板"可执行"分组与 best_rr（L 在回测结论出来前如此）
    validated: bool = True
    support_basis: float | None = None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        lo, hi = self.tranche[0] * self.position_scale, self.tranche[1] * self.position_scale
        d["tranche_text"] = f"计划总仓位的 {lo:.0%}–{hi:.0%}" + (
            f"（已按基准方向缩减至 {self.position_scale:.0%}）" if self.position_scale < 1 else ""
        )
        return d


def _price(level: dict[str, Any] | None) -> float | None:
    if not level:
        return None
    return level.get("display") if level.get("display") is not None else level.get("raw_price")


def _rr(entry: float | None, stop: float | None, target: float | None) -> float | None:
    if entry is None or stop is None or target is None:
        return None
    risk = abs(entry - stop)
    if risk <= 0:
        return None
    return round(abs(target - entry) / risk, 2)


def _split_by_distance(
    levels: list[dict[str, Any]], min_distance_atr: float
) -> tuple[list, list]:
    """按距现价远近把档位分成"可作入场"与"仅作防守"两组。"""
    tradable = [lv for lv in levels if lv.get("distance_atr", 0) >= min_distance_atr]
    too_close = [lv for lv in levels if lv.get("distance_atr", 0) < min_distance_atr]
    return tradable, too_close


def _skip_note(skipped: list[dict[str, Any]], min_distance_atr: float) -> str | None:
    if not skipped:
        return None
    names = "、".join(
        f"{lv['label']}（{lv.get('distance_atr')} ATR）" for lv in skipped
    )
    return (
        f"已跳过 {names}：距现价不足 {min_distance_atr} ATR，只作持仓防守不作入场"
    )


def build_plans(
    supports: list[dict[str, Any]],
    resistances: list[dict[str, Any]],
    *,
    atr: float,
    zone: str,
    current_price: float | None = None,
    event_mode: bool = False,
    event_reason: str | None = None,
    index_bullish: bool | None = None,
    index_symbol: str | None = None,
    min_entry_distance_atr: float | None = None,
    adverse_index_scale: float | None = None,
    regime: dict[str, Any] | None = None,
    regime_gate: bool | None = None,
) -> list[dict[str, Any]]:
    min_dist = (
        MIN_ENTRY_DISTANCE_ATR if min_entry_distance_atr is None else min_entry_distance_atr
    )
    scale_adverse = (
        ADVERSE_INDEX_SCALE if adverse_index_scale is None else adverse_index_scale
    )
    gate = REGIME_GATE if regime_gate is None else regime_gate
    regime_state = (regime or {}).get("state")
    where = f" {index_symbol}" if index_symbol else ""
    sup_tradable, sup_skipped = _split_by_distance(supports, min_dist)
    res_tradable, res_skipped = _split_by_distance(resistances, min_dist)

    # A 用第一个够远的支撑，B 用再下一个；贴现价的档位不参与入场
    s_a = sup_tradable[0] if sup_tradable else None
    s_a_next = sup_tradable[1] if len(sup_tradable) > 1 else None
    s_b = sup_tradable[1] if len(sup_tradable) > 1 else None
    s_b_next = sup_tradable[2] if len(sup_tradable) > 2 else None
    r_c = res_tradable[0] if res_tradable else None

    buf = STOP_BUFFER_ATR * atr
    fallback = FALLBACK_BUFFER_ATR * atr

    def below(level, deeper):
        """止损放在**该档自身**失效之处。

        放到下一档之外是概念错误：那等于把这笔交易的风险定义成"下一档也失守"，
        激进试仓就不再是试仓。只有当下一档近到与本档同属一个反应带时
        （相距不足 0.5 ATR），才一起纳入，否则穿一个就该认错。
        """
        p, d = _price(level), _price(deeper)
        if p is None:
            return None, "无可用入场位"
        if d is not None and (p - d) < FALLBACK_BUFFER_ATR * atr:
            return round(d - buf, 4), f"与下一档 {deeper['label']} 同属一个反应带，放其下 {STOP_BUFFER_ATR} ATR"
        return round(p - fallback, 4), f"{level['label']} 自身失效，{FALLBACK_BUFFER_ATR} ATR 缓冲"

    def first_target(entry: float | None, levels: list, min_gap_atr: float = 0.5):
        """第一目标取距入场足够远的那一档。

        紧贴入场价的压力当 T1，算出来的收益风险比毫无意义——
        那不是目标，是噪声。
        """
        if entry is None:
            return None, None
        for i, lv in enumerate(levels):
            p = _price(lv)
            if p is not None and abs(p - entry) >= min_gap_atr * atr:
                nxt = _price(levels[i + 1]) if i + 1 < len(levels) else None
                return p, nxt
        return None, None

    plans: list[Plan] = []

    # L：到支撑直接限价买入，不等止跌确认；止损 = 支撑原值 × 0.99，盘中触及即离场。
    # 入场档与 A 相同（第一个够远的支撑），两者的差别只在"等不等确认"和止损口径
    s_l = s_a
    raw_l = (s_l or {}).get("raw_price")
    entry_l = _price(s_l)
    basis_l = min(entry_l, raw_l) if entry_l is not None and raw_l is not None else None
    stop_l = (
        float(np.ceil(basis_l * (1 - LEFT_STOP_PCT) * 100) / 100) if basis_l is not None else None
    )
    t1_l, t2_l = first_target(entry_l, [r for r in resistances if r])
    plans.append(
        Plan(
            key="left",
            title=PLAN_TITLES["left"],
            trigger=(
                f"价格回落到 {s_l['label']} {entry_l:g} 直接限价买入，不等止跌/阳线/更高低点确认"
                if s_l else "没有距现价足够远的支撑，计划不成立"
            ),
            entry=entry_l, stop=stop_l,
            stop_basis=(
                f"支撑基准 S={basis_l:g}（{s_l['label']} 原值），P_stop = 0.99×S = {stop_l:g}；"
                "盘中成交价达到或低于即离场，不等收盘、不加缓冲"
                if basis_l is not None else "无可用支撑"
            ),
            entry_level=(s_l or {}).get("label"), entry_note=_skip_note(sup_skipped, min_dist),
            t1=t1_l, t2=t2_l, rr=_rr(entry_l, stop_l, t1_l),
            tranche=FIRST_TRANCHE["left"],
            cancel_if=[
                "挂单成交前价格已跌破 P_stop 或支撑失效：撤单，不在途中接刀",
                "1% 落在该标的正常噪声内、点差或跳空使离场不可靠时降低暴露或不做，不得放宽止损",
                "止损后不摊平；重新进入需要新的有效依据，不因价格再次接近旧 S 就自动买回",
            ],
            executable=True, blocked_by=[], side="left", validated=False,
            support_basis=basis_l,
        )
    )

    entry_a, (stop_a, basis_a) = _price(s_a), below(s_a, s_a_next)
    t1_a, t2_a = first_target(entry_a, [r for r in resistances if r])
    plans.append(
        Plan(
            key="aggressive",
            title=PLAN_TITLES["aggressive"],
            trigger=(s_a or {}).get("confirmation", "没有距现价足够远的支撑，计划不成立"),
            entry=entry_a, stop=stop_a, stop_basis=basis_a,
            entry_level=(s_a or {}).get("label"), entry_note=_skip_note(sup_skipped, min_dist),
            t1=t1_a, t2=t2_a, rr=_rr(entry_a, stop_a, t1_a),
            tranche=FIRST_TRANCHE["aggressive"], side="support_confirm",
            cancel_if=[
                (s_a or {}).get("invalidation", "—"),
                "触发前先跌穿该位，改等下一档，不在途中接刀",
                "止损后不补亏损，重新出现确认信号才重新交易",
            ],
            executable=True, blocked_by=[],
        )
    )

    entry_b, (stop_b, basis_b) = _price(s_b), below(s_b, s_b_next)
    t1_b, t2_b = first_target(entry_b, [lv for lv in ([s_a] + resistances) if lv])
    plans.append(
        Plan(
            key="deep",
            title=PLAN_TITLES["deep"],
            trigger=(
                f"价格进入 {(s_b or {}).get('label','')} 且出现缩量止跌、长下影、"
                "双底、放量反包或更高低点之一"
                if s_b else "没有更深一档的可执行支撑，计划不成立"
            ),
            entry=entry_b, stop=stop_b, stop_basis=basis_b,
            entry_level=(s_b or {}).get("label"), entry_note=None,
            t1=t1_b, t2=t2_b, rr=_rr(entry_b, stop_b, t1_b),
            tranche=FIRST_TRANCHE["deep"], side="support_confirm",
            cancel_if=[
                (s_b or {}).get("invalidation", "—"),
                "日线连续跌破且无法收复时停止加仓，转向下一档",
                "不摊低成本",
            ],
            executable=True, blocked_by=[],
        )
    )

    entry_c = _price(r_c)
    stop_c = round(entry_c - fallback, 4) if entry_c is not None else None
    t1_c, t2_c = first_target(entry_c, [lv for lv in res_tradable[1:] if lv])
    plans.append(
        Plan(
            key="breakout",
            title=PLAN_TITLES["breakout"],
            trigger=(
                f"日线收盘站上 {(r_c or {}).get('label','')}，随后回踩不破"
                if r_c else "没有距现价足够远的压力，计划不成立"
            ),
            entry=entry_c, stop=stop_c,
            stop_basis=f"回踩失败位，{FALLBACK_BUFFER_ATR} ATR 缓冲",
            entry_level=(r_c or {}).get("label"), entry_note=_skip_note(res_skipped, min_dist),
            t1=t1_c, t2=t2_c, rr=_rr(entry_c, stop_c, t1_c),
            tranche=FIRST_TRANCHE["breakout"], side="right",
            cancel_if=[
                (r_c or {}).get("invalidation", "—"),
                "只有影线突破、收盘回落，不算有效突破",
                "回踩直接跌穿原压力则计划取消",
            ],
            executable=True, blocked_by=[],
        )
    )

    adverse_index = index_bullish is False
    for p in plans:
        blocked: list[str] = []
        if p.entry is None or p.stop is None:
            blocked.append(
                f"没有距现价 >= {min_dist:.1f} ATR 的可用档位" if min_dist > 0 else "没有可用档位"
            )
        threshold = MIN_RR[p.key]
        if p.rr is None:
            blocked.append("目标位缺失，收益风险比无法计算")
        elif p.rr < threshold:
            blocked.append(f"收益风险比 {p.rr} 低于门槛 {threshold}")
        if zone == "between":
            blocked.append("现价位于支撑与压力中间，按纪律不交易")
        if event_mode:
            blocked.append(f"事件模式：{event_reason or '窗口内有未定价事件'}")
        if gate and regime_state in ("range", "down"):
            blocked.append(
                f"大盘状态：基准{where} {regime['label']}（{regime['reason']}），"
                "只在向上时开新仓"
            )
        p.blocked_by = blocked
        p.executable = not blocked

        p.cautions = []
        if gate and regime_state == "unknown":
            p.cautions.append(
                f"大盘状态无法判定（{regime.get('reason')}），未按大盘状态拦截，需人工确认"
            )
        if adverse_index and scale_adverse < 1:
            p.position_scale = scale_adverse
            p.cautions.append(
                f"基准{where}未呈多头排列，计划仓按 {scale_adverse:.0%} 缩减"
            )
        else:
            p.position_scale = 1.0

        _set_status(p, current_price, atr)

    return [p.to_dict() for p in plans]


STATUS_LABELS = {
    "waiting_price": "等待到位",
    "waiting_trigger": "等待触发",
    "ready": "到位可执行",
    "ineligible": "不合格",
    "no_data": "数据不足",
}


def _set_status(p: Plan, current_price: float | None, atr: float) -> None:
    """按上游 v2.0.6 第 8 节区分方案状态。等待到位 / 等待触发不等于不合格。"""
    if p.entry is None or p.stop is None or p.t1 is None:
        p.status, p.status_reason = "no_data", "缺少入场、止损或第一目标"
    elif p.blocked_by:
        p.status, p.status_reason = "ineligible", "；".join(p.blocked_by)
    elif current_price is None:
        p.status, p.status_reason = "no_data", "缺少现价"
    else:
        tol = AT_ENTRY_ATR * atr
        if p.key == "breakout":
            if current_price < p.entry - tol:
                p.status, p.status_reason = "waiting_trigger", f"等日线收盘站上 {p.entry:g} 后回踩不破"
            else:
                p.status, p.status_reason = "waiting_trigger", f"已在 {p.entry:g} 附近，等日线收盘确认与回踩"
        elif current_price > p.entry + tol:
            p.status, p.status_reason = "waiting_price", f"现价在入场位 {p.entry:g} 上方，等回落到位"
        elif p.key == "left":
            p.status, p.status_reason = "ready", "已到支撑，可按计划直接尝试（不代表已成交）"
        else:
            p.status, p.status_reason = "waiting_trigger", "已到支撑，等 4H 止跌确认信号"
    p.status_label = STATUS_LABELS[p.status]


def sides(plans: list[dict[str, Any]]) -> dict[str, Any]:
    """上游 v2.0.5 要求每个标的都给左侧与右侧两项建议，不适用也要写原因。

    左侧取 L（支撑直入），右侧取 C（突破回踩）。A/B 是"支撑 + 确认"，介于两者之间，
    仍在 plans 里完整给出，这里不重复。左右侧是入场时机的区分，不是多空。
    """
    by_key = {p["key"]: p for p in plans}

    def brief(key: str) -> dict[str, Any]:
        p = by_key.get(key)
        if p is None:
            return {"plan": None, "status": "no_data", "status_label": STATUS_LABELS["no_data"],
                    "reason": "未生成该方案"}
        return {"plan": key, "title": p["title"], "status": p["status"],
                "status_label": p["status_label"], "reason": p["status_reason"],
                "entry": p["entry"], "stop": p["stop"], "t1": p["t1"], "rr": p["rr"],
                "validated": p.get("validated", True)}

    return {
        "left": brief("left"),
        "right": brief("breakout"),
        "note": "左右侧是入场时机的区分，不是多空；两侧作为备选或分批时共享同一份总风险预算",
    }


def holder_playbook(
    supports: list[dict[str, Any]],
    resistances: list[dict[str, Any]],
    *,
    holding: dict[str, Any] | None,
    current_price: float,
) -> dict[str, Any]:
    """持仓者方案：分批止盈、抬升保护位、破位处理。"""
    s1 = supports[0] if supports else None
    s2 = supports[1] if len(supports) > 1 else None
    r1 = resistances[0] if resistances else None
    r2 = resistances[1] if len(resistances) > 1 else None

    out: dict[str, Any] = {
        "take_profit": [
            {"at": _price(r1), "label": (r1 or {}).get("label"), "action": "减 25%–50%，保留仓位观察能否突破"},
            {"at": _price(r2), "label": (r2 or {}).get("label"), "action": "继续兑现，除非放量突破且回踩成功"},
        ],
        "protective_stop": {
            "at": _price(s1),
            "rule": "价格站上并回踩 R1 成功后，把保护位抬到新形成的更高低点之下",
            "note": "没有突破第一压力前，不预设第二目标一定到达",
        },
        "on_break": [
            {"level": _price(s1), "action": f"{(s1 or {}).get('label','S1')} 失效：减仓，不在途中补仓"},
            {"level": _price(s2), "action": f"{(s2 or {}).get('label','S2')} 失效：退出该逻辑对应的仓位"},
        ],
    }

    if holding and holding.get("cost"):
        cost = float(holding["cost"])
        shares = holding.get("shares")
        pnl = (current_price - cost) / cost
        out["position"] = {
            "cost": cost,
            "shares": shares,
            "unrealised_pct": round(pnl * 100, 2),
            "unrealised_amount": round((current_price - cost) * shares, 2) if shares else None,
            "breakeven_vs_s1": (
                "成本在 S1 之上，破位即亏损，保护位优先"
                if s1 and cost > (_price(s1) or 0)
                else "成本在 S1 之下，尚有缓冲"
            ),
        }
    return out


def size_position(account: float, risk_pct: float, entry: float, stop: float) -> dict[str, Any]:
    distance = abs(entry - stop)
    if distance <= 0:
        return {"error": "入场价与止损价相同"}
    risk_amount = account * risk_pct
    qty = int(risk_amount / distance)
    return {
        "quantity": qty,
        "risk_amount": round(risk_amount, 2),
        "stop_distance": round(distance, 4),
        "notional": round(qty * entry, 2),
        "pct_of_account": round(qty * entry / account * 100, 1) if account else None,
    }
