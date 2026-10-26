"""事件实例的时间解析与中层关系快照绑定。

只解析能由当前原文、观察日期或同一 Chunk 的唯一时间 Triple 支持的日期。
解析结果附带依据；不会把会话观察时间冒充事件发生时间。
"""

from __future__ import annotations

from datetime import date, timedelta
import re
from typing import Any

from .event_state_layer import triple_ref_key


_MONTHS = {name.casefold(): index for index, name in enumerate(
    ("January", "February", "March", "April", "May", "June", "July",
     "August", "September", "October", "November", "December"), 1)}
_WEEKDAYS = {name.casefold(): index for index, name in enumerate(
    ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"))}


def _observed_date(value: Any) -> date | None:
    match = re.search(r"\d{4}-\d{2}-\d{2}", str(value or ""))
    if not match:
        return None
    try:
        return date.fromisoformat(match.group())
    except ValueError:
        return None


def parse_time_expression(raw: str | None, observed_at: str) -> tuple[str | None, str | None]:
    """返回 ISO 日期和解析依据；无法确定时都返回空值。"""
    value = str(raw or "").strip()
    observed = _observed_date(observed_at)
    if not value:
        return None, None
    exact = _observed_date(value)
    if exact:
        return exact.isoformat(), "explicit_date"
    month_day = re.search(
        r"\b(" + "|".join(_MONTHS) + r")\s+(\d{1,2})(?:,?\s+(\d{4}))?\b",
        value, re.I)
    if month_day:
        year = int(month_day.group(3)) if month_day.group(3) else (
            observed.year if observed else None)
        if year is None:
            return None, None
        try:
            resolved = date(year, _MONTHS[month_day.group(1).casefold()],
                            int(month_day.group(2)))
        except ValueError:
            return None, None
        # 未标年份的日期若落在观察时间之后，可能属于上一年；不擅自跨年推断。
        if not month_day.group(3) and observed and resolved > observed:
            return None, None
        return resolved.isoformat(), "explicit_month_day_observed_year"
    if not observed:
        return None, None
    lowered = value.casefold()
    if lowered == "yesterday":
        return (observed - timedelta(days=1)).isoformat(), "relative_observed"
    if lowered == "today":
        return observed.isoformat(), "relative_observed"
    weekday = re.fullmatch(r"last\s+(" + "|".join(_WEEKDAYS) + r")", lowered)
    if weekday:
        days = (observed.weekday() - _WEEKDAYS[weekday.group(1)]) % 7 or 7
        return (observed - timedelta(days=days)).isoformat(), "relative_observed"
    return None, None


def bind_event_time(proposal: dict[str, Any], chunk_refs: list[dict[str, str]]) -> None:
    """给新事件绑定日期及来源 Triple；有多个可能事件时不猜测归属。"""
    actor = proposal["entity_id"]
    action = proposal["attribute"].casefold().split("_")[0]
    raw = proposal.get("event_time")
    parsed, basis = parse_time_expression(raw, proposal["observed_at"])
    activity_refs = proposal["triple_refs"]
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
    if parsed:
        time_refs = [ref for ref in time_refs if ref["tgt"] == parsed]
    elif len({ref["tgt"] for ref in time_refs}) == 1:
        parsed = time_refs[0]["tgt"]
        basis = "middle_triple"
    else:
        time_refs = []
    combined = list(activity_refs)
    seen = {triple_ref_key(ref) for ref in combined}
    for ref in time_refs:
        key = triple_ref_key(ref)
        if key not in seen:
            combined.append(ref)
            seen.add(key)
    proposal["event_time_raw"] = raw
    proposal["event_time"] = parsed
    proposal["time_basis"] = basis
    proposal["event_triple_refs"] = combined
    proposal["time_triple_refs"] = time_refs


def mentioned_dates(text: str, observed_at: str) -> set[str]:
    """从纠正原文中取明确的旧/新日期，用于校验“不是某日”的证据。"""
    values = set(re.findall(r"\d{4}-\d{2}-\d{2}", text))
    pattern = re.compile(r"\b(?:" + "|".join(_MONTHS) + r")\s+\d{1,2}(?:,?\s+\d{4})?\b", re.I)
    for match in pattern.finditer(text):
        parsed, _ = parse_time_expression(match.group(), observed_at)
        if parsed:
            values.add(parsed)
    return values
