"""让 LLM 判断状态语义，并在写入前进行保守的可执行校验。"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from typing import Any, Mapping

from .evidence import EvidenceCandidate


ATTRIBUTES = frozenset({"employer", "job_title", "residence"})
KINDS = frozenset({"baseline", "change", "correction", "same", "retrospective", "review", "ignore"})


@dataclass(frozen=True)
class Decision:
    """经过校验的结构化结论；review/ignore 不允许更新有效状态。"""

    kind: str
    entity_id: str | None = None
    graph_entity_name: str | None = None
    attribute: str | None = None
    value: str | None = None
    evidence_quote: str | None = None
    triple: tuple[str, str, str] | None = None
    detection: str | None = None
    corrected_fact_id: str | None = None
    event_time: str | None = None
    event_group_id: str | None = None
    reason: str | None = None


def _review(reason: str) -> Decision:
    return Decision(kind="review", reason=reason)


def resolve_entity_id(name: str, candidate: EvidenceCandidate,
                      identity_map: Mapping[str, str] | None) -> str | None:
    """优先使用 Chunk 或 Session 局部映射，处理原图同名节点的歧义。"""
    if identity_map is None:
        return name
    for key in (f"{candidate.chunk_id}|{name}", f"{candidate.session_id}|{name}", name):
        if identity_map.get(key):
            return identity_map[key]
    return None


def _parse(response: str, candidate: EvidenceCandidate, context: dict[str, Any],
           attribute_hint: str | None = None,
           identity_map: Mapping[str, str] | None = None) -> Decision:
    """任何缺字段、越界值或无法从原文核对的输出都不进入有效状态。"""
    try:
        clean = response.strip()
        if clean.startswith("```"):
            clean = clean.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        data = json.loads(clean)
    except (ValueError, IndexError, TypeError):
        return _review("模型没有返回合法 JSON")
    if not isinstance(data, dict) or data.get("kind") not in KINDS:
        return _review("缺少有效判定类别")
    kind = data["kind"]
    if kind in {"ignore", "review"}:
        return Decision(kind=kind, reason=str(data.get("reason") or ""))
    entity = data.get("entity")
    attribute = data.get("attribute")
    value = data.get("value")
    quote = data.get("evidence_quote")
    triple = data.get("triple")
    if (not isinstance(entity, str) or entity not in candidate.entity_names or
            not isinstance(attribute, str) or attribute not in ATTRIBUTES or
            (attribute_hint is not None and attribute != attribute_hint) or
            not isinstance(quote, str) or not quote.strip() or quote not in candidate.text):
        return _review("主体、属性或原文片段无法核对")
    stable_id = resolve_entity_id(entity, candidate, identity_map)
    if stable_id is None:
        return _review("主体未经过稳定身份映射核定")
    if (not isinstance(triple, list) or len(triple) != 3 or
            any(not isinstance(item, str) or not item.strip() for item in triple)):
        return _review("缺少可核对的三元组")
    subject, predicate, object_ = (part.strip() for part in triple)
    # 从合并图得到的边可能属于另一个 Chunk，所以必须检查主语、宾语和关系
    # 在本 Chunk 的证据片段中都能找到。无边 Chunk 也必须满足同样条件。
    quoted = quote.casefold()
    if (subject != entity or subject.casefold() not in quoted or
            predicate.casefold() not in quoted or object_.casefold() not in quoted):
        return _review("三元组不受该 Chunk 原文片段支持")
    # 图边只提供候选提示；合并后的图边可能已被新关系覆盖，不能强迫
    # 模型输出与它完全相同。真正的入库依据始终是当前 Chunk 的原文。
    event_time = data.get("event_time")
    if event_time is not None and (not isinstance(event_time, str) or event_time not in quote):
        return _review("事件时间没有原文依据")
    if kind != "correction" and (not isinstance(value, str) or not value.strip() or
                                 value.casefold() not in quoted):
        return _review("状态值没有原文依据")
    if value is not None and value.casefold() != object_.casefold():
        return _review("状态值与三元组宾语不一致")
    target = data.get("corrected_fact_id")
    if kind == "correction":
        history = context.get(entity, {}).get(attribute, {}).get("history", [])
        if not isinstance(target, str) or target not in {
                item.get("fact_id") for item in history if item.get("status", "active") == "active"}:
            return _review("纠正没有明确的旧事实目标")
        if value is not None and (not isinstance(value, str) or value.casefold() not in quoted):
            return _review("纠正后的值没有原文依据")
    current = context.get(entity, {}).get(attribute, {})
    current_value = current.get("value")
    if kind in {"baseline", "change", "same"}:
        # 模型负责判断语义，程序负责让首次事实、同值与不同值有一致的存储类别。
        kind = "baseline" if current_value is None else ("same" if current_value == value else "change")
    detection = data.get("detection") if kind == "change" else None
    if kind == "change" and detection not in {"explicit", "inferred"}:
        return _review("变化缺少明确或推断的来源标记")
    group_quote = data.get("event_group_quote")
    group_id = None
    if (kind == "change" and isinstance(group_quote, str) and 0 < len(group_quote) <= 160
            and group_quote in quote and value and value.casefold() in group_quote.casefold()):
        # 不信任模型任意生成的 ID；只有同一 Chunk 中同一段具体事件原文
        # 才能得到相同组 ID。不同事件即使同一天，也不会因模型标签碰撞。
        identity = "\x1f".join((candidate.session_id, candidate.chunk_id, group_quote))
        group_id = sha256(identity.encode("utf-8")).hexdigest()[:24]
    return Decision(kind=kind, entity_id=stable_id,
                    graph_entity_name=entity, attribute=attribute, value=value,
                    evidence_quote=quote, triple=(subject, predicate, object_),
                    detection=detection, corrected_fact_id=target if kind == "correction" else None,
                    event_time=event_time,
                    event_group_id=group_id)


async def classify(candidate: EvidenceCandidate, current: dict[str, Any], llm: Any,
                   attribute_hint: str | None = None,
                   identity_map: Mapping[str, str] | None = None) -> Decision:
    """以已有状态和当前证据为唯一上下文，请模型返回单条结构化决定。"""
    names = [name for name in candidate.entity_names if name.casefold() in candidate.text.casefold()]
    if candidate.subject and candidate.subject not in names:
        names.append(candidate.subject)
    # 只发送与这段文本相关的图实体，避免完整图节点表挤占模型上下文。
    relevant = {name: current.get(name, {}) for name in names}
    prompt = (
        "你是事实状态判定器，只根据下方原文和已有状态判断。只能处理当前雇主 employer、"
        "当前职位 job_title、当前常住地 residence。普通活动或无持续状态意义的事件返回 ignore。"
        "首次得知返回 baseline；新旧不同且原文支持当前值时返回 change；明确否认先前事实返回 correction；"
        "回顾过去而非当前值返回 retrospective；不确定返回 review。"
        "不要猜测 old_state 或 event_time。纠正必须给出已有事实的 corrected_fact_id。"
        f"本次只判断属性：{attribute_hint or '任选一个有证据的受控属性'}。没有该属性信息时返回 ignore。"
        "输出单个 JSON 对象，字段 kind, entity, attribute, value, evidence_quote, triple, detection, "
        "corrected_fact_id, event_time, event_group_quote, reason。若同一次事件改变多个属性，"
        "各属性返回相同的短事件原文 event_group_quote；不能仅凭同一会话或日期合并。"
        "evidence_quote 必须是原文连续子串，"
        "triple 为 [主语, 原文关系词组, 宾语]，三者都必须在 evidence_quote 中。\n"
        f"可用实体：{json.dumps(names, ensure_ascii=False)}\n"
        f"已有状态：{json.dumps(relevant, ensure_ascii=False)}\n"
        f"图边提示：{json.dumps([candidate.subject, candidate.predicate, candidate.object], ensure_ascii=False)}\n"
        f"会话观察时间：{candidate.observed_at}\n原文：{candidate.text}"
    )
    response = await llm.generate(prompt)
    if not response:
        return _review("模型返回空内容")
    return _parse(response, candidate, current, attribute_hint, identity_map)
