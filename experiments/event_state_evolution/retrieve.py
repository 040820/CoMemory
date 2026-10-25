"""在独立状态库之上提供可关闭的检索入口，不修改原检索器。"""

from __future__ import annotations

from typing import Any, Literal

from .decision import ATTRIBUTES
from .store import StateStore


async def retrieve_with_state(query: str, retriever: Any, store: StateStore,
                              top_k: int = 5, mode: Literal["off", "current", "history"] = "off",
                              entity_id: str | None = None,
                              attribute: str | None = None) -> dict[str, Any]:
    """返回原检索证据，并附可溯源的状态证据。

    mode=off 完全透传原检索结果。当前/历史问题需显式指定属性；若查询
    仅匹配一个实体，也可从原检索分析得到实体。生成答案的调用方应使用
    state_evidence 回答当前状态，原始 evidence 中的旧 Chunk 仅供审计。
    """
    if mode not in {"off", "current", "history"}:
        raise ValueError("mode must be off, current or history")
    result = await retriever.retrieve(query, top_k=top_k)
    if mode == "off":
        return result
    if attribute not in ATTRIBUTES:
        raise ValueError("attribute must be employer, job_title or residence")
    if entity_id is None:
        entities = result.get("query_analysis", {}).get("entities", [])
        if len(entities) != 1:
            raise ValueError("entity_id is required when query does not identify one entity")
        entity_id = str(entities[0])
    enriched = dict(result)
    if mode == "current":
        current = store.get_current(entity_id, attribute)
        enriched["state_evidence"] = [current] if current else []
    else:
        history = store.get_history(entity_id, attribute)
        # 保留已被取代的真实历史事实，同时明确标记被纠正的错误事实。
        enriched["state_evidence"] = history["facts"]
        enriched["state_changes"] = history["changes"]
    enriched["state_mode"] = mode
    enriched["state_entity_id"] = entity_id
    enriched["state_attribute"] = attribute
    return enriched
