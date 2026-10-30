"""事件实例的时间解析与中层关系快照绑定。

为什么单独处理时间：
    对话里的“昨天 / last Friday / May 7 / last week”不是同一种精度。
    如果一律把它们当成某个具体日期，或者用会话观察时间冒充事件发生时间，
    后续的“纠正/变化”判断就会建立在错误的时间线上。

本模块只做两件事：
    1. 解析能由当前原文、观察日期或同一 Chunk 的唯一时间 Triple 支持的日期；
    2. 把解析结果、依据和来源 Triple 绑定到 proposal。
    范围/近似表达（last week、this summer、a few weeks ago）只记录区间或
    unresolved 依据，不伪装成某一天；没有足够依据时宁可不绑定。

边界：
    observed_at 始终是“会话被观察的时间”，不是事件发生时间；未给年份的
    月日只在该年日期不晚于观察日时才解析，避免跨年猜测。
"""

from __future__ import annotations

from datetime import date, timedelta
from calendar import monthrange
import re
from typing import Any

from .event_state_layer import triple_ref_key


# 月份/星期映射：只用于解析英文表达，不承担自然语言理解。
_MONTHS = {name.casefold(): index for index, name in enumerate(
    ("January", "February", "March", "April", "May", "June", "July",
     "August", "September", "October", "November", "December"), 1)}
_WEEKDAYS = {name.casefold(): index for index, name in enumerate(
    ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"))}


def _observed_date(value: Any) -> date | None:
    """从任意字符串中提取第一个 YYYY-MM-DD；解析失败返回 None。

    这里只认 ISO 日期，不做本地化/时区转换：图中保存的日期本身就是
    观察日期的字符串表示，解析失败时应保守地视为“时间未知”。
    """
    match = re.search(r"\d{4}-\d{2}-\d{2}", str(value or ""))
    if not match:
        return None
    try:
        return date.fromisoformat(match.group())
    except ValueError:
        return None


def parse_time_expression(raw: str | None, observed_at: str) -> tuple[str | None, str | None]:
    """兼容旧调用方：只返回可确定到单日的事件日期。"""
    detail = parse_time_details(raw, observed_at)
    return detail["date"], detail["basis"] if detail["date"] else None


def parse_time_details(raw: str | None, observed_at: str) -> dict[str, str | None]:
    """解析原文时间；范围或近似值只写区间，不伪装成某一天。

    ``observed_at`` 始终是会话观察时间，不是事件发生时间。未给年份的
    月日仅在该年日期不晚于观察日时才解析，避免跨年猜测。
    """
    value = str(raw or "").strip()
    observed = _observed_date(observed_at)
    result: dict[str, str | None] = {"date": None, "range_start": None,
                                     "range_end": None, "basis": None,
                                     "raw": value or None}
    if not value:
        return result

    def exact_day(day: date, basis: str) -> dict[str, str | None]:
        """记录可确定到单日的解析结果和依据。"""
        result.update(date=day.isoformat(), basis=basis)
        return result

    def interval(start: date, end: date, basis: str) -> dict[str, str | None]:
        """记录只能确定到区间的解析结果；此时 date 保持 None。"""
        result.update(range_start=start.isoformat(), range_end=end.isoformat(),
                      basis=basis)
        return result

    # 1) 原文直接给出 ISO 日期：精度最高，直接采用。
    exact = _observed_date(value)
    if exact:
        return exact_day(exact, "explicit_date")
    # 2) 英文月日（必要时带年份）；没有年份时借用观察年份，但不得跨到未来。
    month_day = re.search(
        r"\b(" + "|".join(_MONTHS) + r")\s+(\d{1,2})(?:st|nd|rd|th)?(?:,?\s+(\d{4}))?\b",
        value, re.I)
    if month_day:
        year = int(month_day.group(3)) if month_day.group(3) else (
            observed.year if observed else None)
        if year is None:
            return result
        try:
            resolved = date(year, _MONTHS[month_day.group(1).casefold()],
                            int(month_day.group(2)))
        except ValueError:
            return result
        # 未标年份的日期若落在观察时间之后，可能属于上一年；不擅自跨年推断。
        if not month_day.group(3) and observed and resolved > observed:
            return result
        return exact_day(resolved, "explicit_month_day_observed_year")
    # 3) 相对表达必须依赖观察日期；没有 observed_at 就无法可靠解析。
    if not observed:
        return result
    lowered = value.casefold()
    # 昨天/今天：相对观察日，精确到单日。
    if lowered in {"yesterday", "last night"}:
        return exact_day(observed - timedelta(days=1), "relative_observed")
    if lowered == "today":
        return exact_day(observed, "relative_observed")
    # “last Friday”这类星期表达：按观察日往前找最近的该星期几。
    weekday_names = "|".join(_WEEKDAYS) + "|mon|tue|tues|wed|thu|thur|thurs|fri|sat|sun"
    weekday = re.fullmatch(r"last\s+(" + weekday_names + r")", lowered)
    if weekday:
        alias = {"mon": "monday", "tue": "tuesday", "tues": "tuesday",
                 "wed": "wednesday", "thu": "thursday", "thur": "thursday",
                 "thurs": "thursday", "fri": "friday", "sat": "saturday", "sun": "sunday"}
        day_name = alias.get(weekday.group(1), weekday.group(1))
        # 若观察日就是该星期几，往前推 7 天，而不是把当天当成“上周”。
        days = (observed.weekday() - _WEEKDAYS[day_name]) % 7 or 7
        return exact_day(observed - timedelta(days=days), "relative_observed")
    # 以下表达只能确定区间，不能确定单日；因此返回 range_start/range_end。
    if lowered == "last week":
        start = observed - timedelta(days=observed.weekday() + 7)
        return interval(start, start + timedelta(days=6), "relative_week")
    if lowered == "last year":
        return interval(date(observed.year - 1, 1, 1),
                        date(observed.year - 1, 12, 31), "relative_year")
    if lowered == "next month":
        year = observed.year + (observed.month == 12)
        month = observed.month % 12 + 1
        return interval(date(year, month, 1),
                        date(year, month, monthrange(year, month)[1]), "relative_month")
    if lowered in {"this summer", "last summer"}:
        # 季节边界受地区影响；未给地理信息时只保留表达，避免擅用北半球日历。
        result["basis"] = "unresolved_season"
        return result
    if lowered in {"this past weekend", "last weekend"}:
        # 周末按周六到周日；若观察日就是周六，往前推 7 天表示“上一个周末”。
        saturday = observed - timedelta(days=(observed.weekday() - 5) % 7 or 7)
        return interval(saturday, saturday + timedelta(days=1), "relative_weekend")
    if lowered == "a few weeks ago":
        # 近似表达没有明确周数，保留原始表达，不编造区间。
        result["basis"] = "unresolved_approximate_weeks"
        return result
    # 无法识别：返回全空结果，由调用方决定是否保留原始表达。
    return result


def bind_event_time(proposal: dict[str, Any], chunk_refs: list[dict[str, str]]) -> None:
    """给新事件绑定日期及来源 Triple；有多个可能事件时不猜测归属。

    绑定顺序：
        1. 先解析 proposal 自带的 event_time 表达；
        2. 在同一 Chunk 中找“同一主体 + 同类活动 + 目标是日期”的 Triple；
        3. 只有当当前活动的客体能被原文支持、且同类活动目标唯一时，才把
           日期 Triple 当作事件时间依据；
        4. 区间/近似表达不精确化成单日；无法确定时保留 raw，不绑定日期。
    本函数原地修改 proposal，不返回值。
    """
    actor = proposal["entity_id"]
    # 用活动谓词的第一个词匹配同类关系，例如 attended_concert -> attended。
    action = proposal["attribute"].casefold().split("_")[0]
    raw = proposal.get("event_time")
    temporal = parse_time_details(raw, proposal["observed_at"])
    parsed, basis = temporal["date"], temporal["basis"]
    activity_refs = proposal["triple_refs"]
    # 候选日期边：同一主体、同类活动、目标本身能被解析成日期。
    time_refs = [ref for ref in chunk_refs
                 if ref["src"] == actor and _observed_date(ref["tgt"])
                 and action in ref["relation"].casefold()]
    # 若同一 Chunk 有两个同类活动，不能把共享日期边随意挂到其中一个。
    targets = {ref["tgt"] for ref in chunk_refs
               if ref["src"] == actor and action in ref["relation"].casefold()
               and not _observed_date(ref["tgt"])}
    # 没有能支持当前客体的活动关系时，日期边不能独自证明事件归属。
    value = str(proposal["value"]).casefold()
    grounded_target = any(ref["src"] == actor and
                          (ref["tgt"].casefold() in value or
                           value in ref["tgt"].casefold())
                          for ref in activity_refs)
    if len(targets) > 1 or not grounded_target:
        time_refs = []
    if temporal["range_start"] or str(basis or "").startswith("unresolved_"):
        # 原文只支持区间时，即使中层恰好有单个日期 Triple，也不能把
        # “上周”等表达悄悄精确化成某一天。
        time_refs = []
    elif parsed:
        # 原文已给出单日：只保留与该日一致的 Triple。
        time_refs = [ref for ref in time_refs if ref["tgt"] == parsed]
    elif len({ref["tgt"] for ref in time_refs}) == 1:
        # 原文没给日期，但同一 Chunk 只有一个明确的日期边：采用它并标注来源。
        parsed = time_refs[0]["tgt"]
        basis = "middle_triple"
    else:
        # 没有日期或存在歧义：不猜。
        time_refs = []
    # 合并活动关系和日期关系，按 triple_ref_key 去重；两类关系可能指向同一 Triple。
    combined = list(activity_refs)
    seen = {triple_ref_key(ref) for ref in combined}
    for ref in time_refs:
        key = triple_ref_key(ref)
        if key not in seen:
            combined.append(ref)
            seen.add(key)
    # 保留原始表达、解析结果、依据和区间；下游据此区分“原文说的”与“程序推的”。
    proposal["event_time_raw"] = raw
    proposal["event_time"] = parsed
    proposal["time_basis"] = basis
    proposal["event_time_range"] = (None if not temporal["range_start"] else {
        "start": temporal["range_start"], "end": temporal["range_end"],
        "basis": basis, "raw": raw})
    proposal["event_triple_refs"] = combined
    proposal["time_triple_refs"] = time_refs


def mentioned_dates(text: str, observed_at: str) -> set[str]:
    """从纠正原文中取明确的旧/新日期，用于校验“不是某日”的证据。

    只返回能解析成 ISO 单日的日期；范围表达（last week / this summer）
    不会进入集合，避免把一段区间误当成某个具体日期去做“纠正”判断。
    """
    # 先收 ISO 日期，再用同一个解析器处理英文月日表达。
    values = set(re.findall(r"\d{4}-\d{2}-\d{2}", text))
    pattern = re.compile(r"\b(?:" + "|".join(_MONTHS) + r")\s+\d{1,2}(?:st|nd|rd|th)?(?:,?\s+\d{4})?\b", re.I)
    for match in pattern.finditer(text):
        parsed, _ = parse_time_expression(match.group(), observed_at)
        if parsed:
            values.add(parsed)
    return values
