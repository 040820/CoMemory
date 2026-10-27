"""保持原有 Query→Triple/Chunk 排序，仅补充实体关联的演化证据。"""

from __future__ import annotations

from datetime import date
import re
from typing import Any

from prompt.query_prompt import QUERY_PROMPT
from coregraph.state_slot import comparison_value


def _observed_by_question(observed_at: str, question_time: str) -> bool:
    """可解析日期时不把问题发生之后才观察到的状态透露给回答模型。"""
    if not question_time:
        return True
    values = []
    for text in (observed_at, question_time):
        match = re.search(r"\d{4}-\d{2}-\d{2}", str(text))
        if not match:
            return True
        try:
            values.append(date.fromisoformat(match.group()))
        except ValueError:
            return True
    return values[0] <= values[1]


async def query_with_event_state(question: str, question_time: str, memory: Any,
                                 retriever: Any, answer_llm: Any,
                                 top_k: int = 5) -> dict[str, Any]:
    """检索仍调用已实现的结构化语义版本；事件层不筛选、不改动 Chunk 排名。"""
    retrieved = await retriever.retrieve(question, top_k=top_k)
    evidence = retrieved["evidence"]
    layer = memory.event_state_layer
    if layer.write_policy == 'append_only':
        # 服务于新写入语义：不能使用旧版“最新断言=当前有效值”的展示与覆盖规则。
        # 只适配证据表达，既有结构化语义检索器的 Chunk 排名保持不变。
        from .observation_query import query_observations
        return await query_observations(question, question_time, layer, retrieved,
                                         answer_llm, _observed_by_question)
    # 只在本实验的查询入口应用纠正覆盖层；原始 Chunk 和旧 Triple 仍留在 pkl。
    # 问题发生在纠正之前时，旧日期关系仍可作为当时可见的证据。
    visible_evidence = []
    for hit in evidence:
        visible = dict(hit)
        def visible_triple(triple: dict[str, Any]) -> bool:
            ref = {"src": str(triple["subject"]),
                   "relation": str(triple["predicate"]),
                   "tgt": str(triple["object"]),
                   "chunk_id": str(hit["chunk_id"])}
            return (not layer.is_superseded_event_time_triple(ref, question_time) and
                    not layer.is_superseded_event_attribute_triple(ref, question_time))

        visible["triples"] = [triple for triple in hit.get("triples", [])
                              if visible_triple(triple)]
        visible_evidence.append(visible)
    evidence = visible_evidence
    entity_names = retrieved.get("query_analysis", {}).get("entities", [])
    event_ids: set[str] = set()
    for name in entity_names:
        event_ids.update(layer.by_entity.get(str(name), []))
    for hit in evidence:
        event_ids.update(layer.by_chunk.get(str(hit["chunk_id"]), []))
    if entity_names:
        event_ids = {event_id for event_id in event_ids
                     if layer.events[event_id]["entity_id"] in entity_names}
    # 事件只是额外证据；状态变化按观察时间排列并限制数量，避免压过原始 Chunk。
    events = sorted((layer.events[event_id] for event_id in event_ids
                     if _observed_by_question(layer.events[event_id]["observed_at"], question_time)),
                    key=lambda row: (row["observed_at"], row["event_id"]))[-8:]
    current_facts = [fact for name in entity_names
                     for fact in layer.current_facts_for_entity(str(name))
                     if _observed_by_question(fact["observed_at"], question_time)]

    # 普通历史活动使用 event_instances；旧版 events 只表示持续状态的变化边。
    instance_ids: set[str] = set()
    for hit in evidence:
        instance_ids.update(layer.event_instance_by_chunk.get(str(hit["chunk_id"]), []))
    for name in entity_names:
        for event_id in layer.event_instance_by_actor.get(str(name), []):
            event = layer.event_instances[event_id]
            if str(event["target"]).casefold() in question.casefold():
                instance_ids.add(event_id)
    instances: list[dict[str, Any]] = []
    for event_id in instance_ids:
        event = layer.event_instances[event_id]
        source_facts = [layer.facts[fact_id] for fact_id in event["fact_ids"]
                        if fact_id in layer.facts]
        if not source_facts or not any(_observed_by_question(
                fact["observed_at"], question_time) for fact in source_facts):
            continue
        time_claims = [item for item in layer.event_time_assertions.values()
                       if item["event_id"] == event_id and _observed_by_question(
                           item["observed_at"], question_time)]
        time_claims.sort(key=lambda item: (item["observed_at"], item.get("position", 0)))
        current_claim = time_claims[-1] if time_claims else None
        attribute_claims: dict[str, dict[str, Any]] = {}
        for assertion in layer.event_attribute_assertions.values():
            if (assertion["event_id"] == event_id and _observed_by_question(
                    assertion["observed_at"], question_time)):
                name = assertion["predicate"]
                previous = attribute_claims.get(name)
                if previous is None or (assertion["observed_at"], assertion.get("position", 0)) > (
                        previous["observed_at"], previous.get("position", 0)):
                    attribute_claims[name] = assertion
        instances.append({
            "event_id": event_id, "actor": event["actor"],
            "action": event["action"], "target": event["target"],
            "event_time": current_claim["value"] if current_claim else None,
            "time_basis": current_claim["time_basis"] if current_claim else None,
            "time_assertion_id": current_claim["assertion_id"] if current_claim else None,
            "corrects_assertion_id": current_claim.get("corrects_assertion_id") if current_claim else None,
            "attributes": {name: claim["value"] for name, claim in attribute_claims.items()},
            "attribute_assertion_ids": {name: claim["assertion_id"]
                                        for name, claim in attribute_claims.items()},
            "evidence_chunk_ids": list(event["evidence_chunk_ids"]),
            "anchor_quote": event["anchor_quote"],
        })
    instances.sort(key=lambda item: (item["event_id"] not in {
        event_id for hit in evidence for event_id in layer.event_instance_by_chunk.get(
            str(hit["chunk_id"]), [])}, item["event_id"]))
    instances = instances[:8]

    triples: dict[tuple[str, str, str], None] = {}
    chunks: list[str] = []
    for index, hit in enumerate(evidence, 1):
        chunks.append(f"Chunk {index} ({hit.get('session_time') or ''}): {hit['text'][:2800]}")
        for triple in hit.get("triples", []):
            triples[(str(triple["subject"]), str(triple["predicate"]),
                     str(triple["object"]))] = None
    triple_text = "; ".join(f"({src}, {relation}, {tgt})"
                            for src, relation, tgt in triples) or "None available"
    supported_events = [event for event in events if event["status"] == "supported"]
    extra_context = ""
    if current_facts:
        lines = [f"{fact['entity_id']} / {fact['attribute']} "
                 f"[作用域={fact.get('scope') or {}}] = {comparison_value(fact)} "
                 f"[Chunk {fact['chunk_id']}, {fact['event_time'] or fact['observed_at']}]; "
                 f"原文片段：{fact['quote']}" for fact in current_facts[:12]]
        extra_context += "\n- 经原文支持的当前状态：\n" + "\n".join(lines)
    if supported_events:
        lines = [f"{event['entity_id']} / {event['attribute']}: "
                 f"{event['old_state']} → {event['new_state']} "
                 f"[{event['update_type']}, Chunk {event['chunk_id']}, "
                 f"{event['event_time'] or event['observed_at']}, {event['status']}]; "
                 f"原文片段：{event['quote']}"
                 for event in supported_events]
        extra_context += ("\n- 可追溯的状态变化记录（仅在与问题相关且证据成立时使用；"
                          "原始 Chunk 优先）：\n" + "\n".join(lines))
    if instances:
        lines = [f"{event['event_id']}: {event['actor']} {event['action']} "
                 f"{event['target']}; 有效事件日期={event['event_time'] or '未确定'}; "
                 f"当前事件属性={event['attributes']}; "
                 f"日期依据={event['time_basis'] or '无'}; "
                 f"证据 Chunk={','.join(event['evidence_chunk_ids'])}; "
                 f"原文片段={event['anchor_quote']}"
                 for event in instances]
        extra_context += ("\n- 事件实例及当前有效日期：\n" + "\n".join(lines) +
                          "\n如果事件断言已纠正，以上有效日期和属性优先于旧 Chunk 中的旧值。")
    prompt = QUERY_PROMPT.format(
        question_time=question_time, question=question, triples=triple_text,
        chunks=("\n".join(chunks) or "None available") + extra_context,
    )
    answer = await answer_llm.generate(prompt)
    session_ids = list(dict.fromkeys(str(hit["session_id"]) for hit in evidence
                                          if hit.get("session_id")))
    return {
        "question": question, "answer": answer.strip() if answer else "",
        "top_session_ids": session_ids, "query_analysis": retrieved.get("query_analysis", {}),
        "evidence": evidence, "state_evidence": events,
        "current_state_evidence": current_facts,
        "event_instance_evidence": instances,
    }
