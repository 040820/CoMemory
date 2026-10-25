"""与 CogniGraph 一同持久化的上层事件与状态索引。

这一层保存演化关系和证据指针；原话仍只保存在 chunk_storage 中。
它不参与实体关系图的节点/边计数，避免事件节点污染原有图检索。
"""

from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import json
from typing import Any


def _id(*parts: str) -> str:
    """相同来源在重跑时获得相同 ID。"""
    return sha256("\x1f".join(parts).encode("utf-8")).hexdigest()[:24]


def _slot(entity_id: str, attribute: str) -> str:
    return json.dumps([entity_id, attribute], ensure_ascii=False, separators=(",", ":"))


def triple_ref_key(ref: dict[str, str]) -> str:
    """关系边可能被合并，因此把当次抽取的关系快照和 Chunk 一起作为引用。"""
    return _id(ref["src"], ref["relation"], ref["tgt"], ref["chunk_id"])


class EventStateLayer:
    """按实体、属性和证据建立可双向追溯的上层索引。"""

    def __init__(self, data: dict[str, Any] | None = None):
        # 写入器对每个 Chunk 先建副本再提交；深拷贝避免纠正旧事实时改到原层。
        source = deepcopy(data) if data else {}
        if source and source.get("version") != 1:
            raise ValueError("不支持的事件状态层版本")
        self.facts: dict[str, dict[str, Any]] = dict(source.get("facts", {}))
        self.events: dict[str, dict[str, Any]] = dict(source.get("events", {}))
        self.current: dict[str, str] = dict(source.get("current", {}))
        self.by_entity: dict[str, list[str]] = {
            key: list(value) for key, value in source.get("by_entity", {}).items()
        }
        self.by_chunk: dict[str, list[str]] = {
            key: list(value) for key, value in source.get("by_chunk", {}).items()
        }
        self.by_triple: dict[str, list[str]] = {
            key: list(value) for key, value in source.get("by_triple", {}).items()
        }
        self.processed_chunks: set[str] = set(source.get("processed_chunks", []))
        self.review_queue: list[dict[str, str]] = list(source.get("review_queue", []))

    def to_dict(self) -> dict[str, Any]:
        """只输出普通 Python 容器，兼容现有 pickle 图文件。"""
        return {
            "version": 1, "facts": self.facts, "events": self.events,
            "current": self.current, "by_entity": self.by_entity,
            "by_chunk": self.by_chunk, "by_triple": self.by_triple,
            "processed_chunks": sorted(self.processed_chunks),
            "review_queue": self.review_queue,
        }

    def current_fact(self, entity_id: str, attribute: str) -> dict[str, Any] | None:
        fact_id = self.current.get(_slot(entity_id, attribute))
        return self.facts.get(fact_id) if fact_id else None

    def current_facts_for_entity(self, entity_id: str) -> list[dict[str, Any]]:
        """仅返回索引指向的有效当前事实，历史/纠正审计不混入。"""
        return [self.facts[fact_id] for fact_id in self.current.values()
                if self.facts[fact_id]["entity_id"] == entity_id]

    def context_for(self, entity_id: str) -> dict[str, dict[str, Any]]:
        """只提供该实体的当前值和可纠正的事实 ID，供 LLM 判断变化。"""
        result: dict[str, dict[str, Any]] = {}
        for fact in self.facts.values():
            if fact["entity_id"] != entity_id or fact["status"] == "corrected":
                continue
            attribute = fact["attribute"]
            result.setdefault(attribute, {"current": None, "history": []})["history"].append({
                "fact_id": fact["fact_id"], "value": fact["value"],
                "chunk_id": fact["chunk_id"],
            })
        for attribute in result:
            current = self.current_fact(entity_id, attribute)
            result[attribute]["current"] = current["value"] if current else None
        return result

    def add_review(self, chunk_id: str, reason: str) -> None:
        item = {"chunk_id": chunk_id, "reason": reason}
        if item not in self.review_queue:
            self.review_queue.append(item)

    def _index_event(self, event: dict[str, Any]) -> None:
        """实体是主入口，同时提供从 Chunk、Triple 返回事件的反向索引。"""
        event_id = event["event_id"]
        self.by_entity.setdefault(event["entity_id"], []).append(event_id)
        self.by_chunk.setdefault(event["chunk_id"], []).append(event_id)
        for ref in event["triple_refs"]:
            self.by_triple.setdefault(triple_ref_key(ref), []).append(event_id)

    def _rebuild_changes(self, entity_id: str, attribute: str) -> None:
        """历史纠正后重新连接仍有效的事实，保留原错误事件供审计。"""
        for event in self.events.values():
            if (event["entity_id"] == entity_id and event["attribute"] == attribute and
                    event["update_type"] == "change"):
                event["status"] = "revoked"
        timeline = [fact for fact in self.facts.values()
                    if fact["entity_id"] == entity_id and fact["attribute"] == attribute
                    and fact["status"] in {"active", "audit"}]
        timeline.sort(key=lambda fact: (fact["position"], fact["fact_id"]))
        previous = None
        for fact in timeline:
            if fact["status"] == "audit":
                # 纠正只否认旧值而未给新值时，历史变化链在此断开。
                previous = None
                continue
            if previous is not None and previous["value"].casefold() != fact["value"].casefold():
                event_id = _id("change", previous["fact_id"], fact["fact_id"])
                if event_id in self.events:
                    self.events[event_id]["status"] = "supported"
                else:
                    event = {
                        "event_id": event_id, "event_group_id": fact.get("event_group_id"),
                        "entity_id": entity_id, "entity_index": fact["entity_index"],
                        "attribute": attribute, "event_type": fact["event_type"],
                        "update_type": "change", "old_fact_id": previous["fact_id"],
                        "new_fact_id": fact["fact_id"], "old_state": previous["value"],
                        "new_state": fact["value"], "event_time": fact["event_time"],
                        "observed_at": fact["observed_at"], "session_id": fact["session_id"],
                        "chunk_id": fact["chunk_id"], "quote": fact["quote"],
                        "triple_refs": fact["triple_refs"], "status": "supported",
                    }
                    self.events[event_id] = event
                    self._index_event(event)
            previous = fact

    def apply(self, proposal: dict[str, Any]) -> str | None:
        """新增基线、变化或纠正；不以普通活动覆盖有效状态。"""
        entity_id = proposal["entity_id"]
        attribute = proposal["attribute"]
        value = proposal.get("value")
        kind = proposal["kind"]
        chunk_id = proposal["chunk_id"]
        slot = _slot(entity_id, attribute)
        current = self.current_fact(entity_id, attribute)
        target_id = proposal.get("target_fact_id")

        if kind == "correction":
            target = self.facts.get(target_id)
            if (not target or target["status"] == "corrected" or
                    target["entity_id"] != entity_id or target["attribute"] != attribute):
                raise ValueError("纠正必须明确指向同一实体、同一属性的有效旧事实")
            old_fact = target
        else:
            if value is None:
                raise ValueError("状态事实必须有新值")
            if current and current["value"].casefold() == value.casefold():
                return None
            old_fact = current
            kind = "change" if current else "baseline"

        fact_id = _id(entity_id, attribute, chunk_id, proposal["quote"], kind)
        if fact_id in self.facts:
            return None
        position = old_fact["position"] if kind == "correction" else len(self.facts) + 1
        fact = {
            "fact_id": fact_id, "entity_id": entity_id,
            "entity_index": proposal["entity_index"], "attribute": attribute,
            "value": value, "kind": kind, "status": "active" if value is not None else "audit",
            "position": position, "event_type": proposal["event_type"],
            "event_group_id": proposal.get("event_group_id"),
            "session_id": proposal["session_id"], "chunk_id": chunk_id,
            "observed_at": proposal["observed_at"], "event_time": proposal.get("event_time"),
            "quote": proposal["quote"], "triple_refs": proposal["triple_refs"],
            "target_fact_id": target_id if kind == "correction" else None,
        }
        self.facts[fact_id] = fact
        if kind == "correction":
            old_fact["status"] = "corrected"
            # 被纠正的历史事实不应使更晚、独立证实的当前值回退。
            if current and current["fact_id"] == target_id:
                if value is None:
                    self.current.pop(slot, None)
                else:
                    self.current[slot] = fact_id
        else:
            self.current[slot] = fact_id

        if kind == "baseline":
            return fact_id

        event_id = _id(kind, old_fact["fact_id"], fact_id)
        event = {
            "event_id": event_id, "event_group_id": proposal.get("event_group_id"),
            "entity_id": entity_id, "entity_index": proposal["entity_index"],
            "attribute": attribute, "event_type": proposal["event_type"],
            "update_type": kind,
            "old_fact_id": old_fact["fact_id"], "new_fact_id": fact_id,
            "old_state": old_fact["value"], "new_state": value,
            "event_time": proposal.get("event_time"),
            "observed_at": proposal["observed_at"],
            "session_id": proposal["session_id"], "chunk_id": chunk_id,
            "quote": proposal["quote"], "triple_refs": proposal["triple_refs"],
            "status": "supported",
        }
        self.events[event_id] = event
        self._index_event(event)
        if kind == "correction":
            self._rebuild_changes(entity_id, attribute)
        return event_id

    def events_for_entity(self, entity_id: str, attribute: str | None = None) -> list[dict[str, Any]]:
        events = [self.events[event_id] for event_id in self.by_entity.get(entity_id, [])]
        if attribute is not None:
            events = [event for event in events if event["attribute"] == attribute]
        return events

    def events_for_triple(self, ref: dict[str, str]) -> list[dict[str, Any]]:
        """从中层关系快照反查引用它的上层变化。"""
        return [self.events[event_id] for event_id in self.by_triple.get(triple_ref_key(ref), [])]
