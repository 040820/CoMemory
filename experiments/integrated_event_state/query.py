"""保持原有 Query→Triple/Chunk 排序，仅补充实体关联的演化证据。"""

from __future__ import annotations

from datetime import date
import re
from typing import Any

from prompt.query_prompt import QUERY_PROMPT


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
        lines = [f"{fact['entity_id']} / {fact['attribute']} = {fact['value']} "
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
    }
