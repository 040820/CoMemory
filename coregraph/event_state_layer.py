"""与 CogniGraph 一同持久化的上层事件与状态索引。

这一层保存演化关系和证据指针；原话仍只保存在 chunk_storage 中。
它不参与实体关系图的节点/边计数，避免事件节点污染原有图检索。

核心概念：
    fact（事实）：某实体某属性在某个时间点的取值，例如
                  Alice / city = Shanghai，来源是某个 Chunk 的引文。
    current（当前值索引）：(entity_id, attribute) → 最新有效 fact_id。
    event（事件）：两个事实之间的变化或纠正，保存 old/new 状态与证据。
    status：
        active    当前有效的事实
        corrected 已被纠正的旧事实
        audit     只否定旧值但未给出新值的事实（不进入 current）
        supported 已核实的变化事件
        revoked   因纠正而失效、但保留供审计的事件

索引方向（都只为查询时快速反查）：
    by_entity：实体 → 事件
    by_chunk ：Chunk  → 事件
    by_triple：关系快照 → 事件
"""

from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import json
import re
from typing import Any

from .state_slot import cardinality_for, preliminary_operation, slot_key


def _id(*parts: str) -> str:
    """相同来源在重跑时获得相同 ID（确定性哈希，避免重复写入）。"""
    return sha256("\x1f".join(parts).encode("utf-8")).hexdigest()[:24]


def _slot(entity_id: str, attribute: str) -> str:
    """current 索引的键：实体 + 属性。用 JSON 编码避免分隔符冲突。"""
    return json.dumps([entity_id, attribute], ensure_ascii=False, separators=(",", ":"))


def triple_ref_key(ref: dict[str, str]) -> str:
    """关系边可能被合并，因此把当次抽取的关系快照和 Chunk 一起作为引用。"""
    return _id(ref["src"], ref["relation"], ref["tgt"], ref["chunk_id"])


class EventStateLayer:
    """按实体、属性和证据建立可双向追溯的上层索引。"""

    def __init__(self, data: dict[str, Any] | None = None):
        """从 dict 恢复层，或创建空层。

        Args:
            data: to_dict() 的输出；None 表示新建空层。

        深拷贝的原因：写入器对每个 Chunk 先建副本再提交，
        深拷贝避免纠正旧事实时改到原层。
        """
        # 写入器对每个 Chunk 先建副本再提交；深拷贝避免纠正旧事实时改到原层。
        source = deepcopy(data) if data else {}
        if source and source.get("version") != 1:
            raise ValueError("不支持的事件状态层版本")
        self.facts: dict[str, dict[str, Any]] = dict(source.get("facts", {}))
        self.events: dict[str, dict[str, Any]] = dict(source.get("events", {}))
        self.current: dict[str, str] = dict(source.get("current", {}))
        # 新槽位模式：集合属性的有效成员与每次操作的判定记录独立保存。
        self.members: dict[str, list[str]] = {
            key: list(value) for key, value in source.get("members", {}).items()
        }
        self.decisions: list[dict[str, Any]] = list(source.get("decisions", []))
        self.slot_schema_version = source.get(
            "slot_schema_version", 1 if source.get("facts") else 2)
        self.by_entity: dict[str, list[str]] = {
            key: list(value) for key, value in source.get("by_entity", {}).items()
        }
        self.by_chunk: dict[str, list[str]] = {
            key: list(value) for key, value in source.get("by_chunk", {}).items()
        }
        self.by_triple: dict[str, list[str]] = {
            key: list(value) for key, value in source.get("by_triple", {}).items()
        }
        # 已处理 Chunk 用于增量重跑去重；review_queue 保存无法写入的候选与原因。
        self.processed_chunks: set[str] = set(source.get("processed_chunks", []))
        self.review_queue: list[dict[str, str]] = list(source.get("review_queue", []))
        # 事件实例与旧版 events（状态变化边）是两类对象：前者表示一次活动，
        # 后者表示两个持续状态之间的变化或纠正。
        self.event_instances: dict[str, dict[str, Any]] = dict(source.get("event_instances", {}))
        self.event_time_assertions: dict[str, dict[str, Any]] = dict(
            source.get("event_time_assertions", {}))
        self.event_attribute_assertions: dict[str, dict[str, Any]] = dict(
            source.get("event_attribute_assertions", {}))
        self.event_link_decisions: list[dict[str, Any]] = list(
            source.get("event_link_decisions", []))
        self.event_instance_by_actor: dict[str, list[str]] = {
            key: list(value) for key, value in source.get("event_instance_by_actor", {}).items()
        }
        self.event_instance_by_chunk: dict[str, list[str]] = {
            key: list(value) for key, value in source.get("event_instance_by_chunk", {}).items()
        }
        self.event_instance_by_triple: dict[str, list[str]] = {
            key: list(value) for key, value in source.get("event_instance_by_triple", {}).items()
        }
        self.superseded_event_time_triples: set[str] = set(
            source.get("superseded_event_time_triples", []))
        self.superseded_event_attribute_triples: set[str] = set(
            source.get("superseded_event_attribute_triples", []))

    def to_dict(self) -> dict[str, Any]:
        """只输出普通 Python 容器，兼容现有 pickle 图文件。

        processed_chunks 是 set，序列化时转成排序后的 list，
        保证同一状态每次保存的字节表示稳定。
        """
        return {
            "version": 1, "facts": self.facts, "events": self.events,
            "current": self.current, "members": self.members,
            "decisions": self.decisions, "slot_schema_version": self.slot_schema_version,
            "by_entity": self.by_entity,
            "by_chunk": self.by_chunk, "by_triple": self.by_triple,
            "processed_chunks": sorted(self.processed_chunks),
            "review_queue": self.review_queue,
            "event_instances": self.event_instances,
            "event_time_assertions": self.event_time_assertions,
            "event_attribute_assertions": self.event_attribute_assertions,
            "event_link_decisions": self.event_link_decisions,
            "event_instance_by_actor": self.event_instance_by_actor,
            "event_instance_by_chunk": self.event_instance_by_chunk,
            "event_instance_by_triple": self.event_instance_by_triple,
            "superseded_event_time_triples": sorted(self.superseded_event_time_triples),
            "superseded_event_attribute_triples": sorted(
                self.superseded_event_attribute_triples),
        }

    def current_fact(self, entity_id: str, attribute: str,
                     scope: dict[str, str] | None = None,
                     modality: str = "actual") -> dict[str, Any] | None:
        """取某实体某属性的当前有效事实；没有则返回 None。"""
        key = slot_key({"entity_id": entity_id, "attribute": attribute,
                        "scope": scope or {}, "modality": modality})
        fact_id = self.current.get(key)
        if not fact_id and not scope and modality == "actual":
            # 旧版图用二元键；只做读取兼容，不能混入新的写入语义。
            fact_id = self.current.get(_slot(entity_id, attribute))
        return self.facts.get(fact_id) if fact_id else None

    def active_members(self, entity_id: str, attribute: str,
                       scope: dict[str, str] | None = None,
                       modality: str = "actual") -> list[dict[str, Any]]:
        """返回集合槽中所有有效事实；拥有多个对象不会互相覆盖。"""
        key = slot_key({"entity_id": entity_id, "attribute": attribute,
                        "scope": scope or {}, "modality": modality})
        return [self.facts[fact_id] for fact_id in self.members.get(key, [])
                if self.facts[fact_id]["status"] == "active"]

    def slot_context_for(self, entity_id: str) -> list[dict[str, Any]]:
        """只给模型最近的少量参考；完整集合仍留在持久化索引中供程序比对。"""
        single_ids = set(self.current.values())
        set_ids = {fact_id for ids in self.members.values() for fact_id in ids}
        singles = [fact for fact in self.facts.values()
                   if fact["fact_id"] in single_ids and fact["entity_id"] == entity_id
                   and fact["status"] == "active"][-12:]
        remaining = 12 - len(singles)
        members = [fact for fact in self.facts.values()
                   if fact["fact_id"] in set_ids and fact["entity_id"] == entity_id
                   and fact["status"] == "active"][-remaining:] if remaining else []
        return [{"fact_id": fact["fact_id"], "attribute": fact["attribute"],
                 "value": str(fact["value"])[:160], "scope": fact.get("scope") or {},
                 "modality": fact.get("modality", "actual"),
                 "quote": str(fact["quote"])[:160], "chunk_id": fact["chunk_id"]}
                for fact in singles + members]

    def current_facts_for_entity(self, entity_id: str) -> list[dict[str, Any]]:
        """仅返回索引指向的有效当前事实，历史/纠正审计不混入。

        注意：这里依赖 current 索引，因此 corrected/audit 事实不会被返回。
        """
        return [self.facts[fact_id] for fact_id in self.current.values()
                if self.facts[fact_id]["entity_id"] == entity_id]

    def context_for(self, entity_id: str) -> dict[str, dict[str, Any]]:
        """只提供该实体的当前值和可纠正的事实 ID，供 LLM 判断变化。

        返回结构：
            {attribute: {"current": 当前值或 None,
                         "history": [{"fact_id", "value", "chunk_id"}, ...]}}
        history 故意带上 fact_id，因为纠正时需要 target_fact_id。
        """
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
        """把无法自动写入的候选放入待核查队列（相同条目去重）。"""
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

    def _index_event_instance_evidence(self, event_id: str, chunk_id: str,
                                       refs: list[dict[str, str]]) -> None:
        """建立 Chunk/Triple 快照到事件实例的反向索引。"""
        by_chunk = self.event_instance_by_chunk.setdefault(str(chunk_id), [])
        if event_id not in by_chunk:
            by_chunk.append(event_id)
        for ref in refs:
            by_triple = self.event_instance_by_triple.setdefault(triple_ref_key(ref), [])
            if event_id not in by_triple:
                by_triple.append(event_id)

    def _add_time_assertion(self, event_id: str, proposal: dict[str, Any],
                            corrects_id: str | None = None) -> str:
        """时间断言独立保存：日期可以被纠正，原始证据仍完整可查。"""
        assertion_id = _id("event-time", event_id, str(proposal["chunk_id"]),
                           str(proposal["event_time"]), str(proposal["quote"]))
        self.event_time_assertions[assertion_id] = {
            "assertion_id": assertion_id, "event_id": event_id,
            "predicate": "event_time", "value": proposal["event_time"],
            "raw_value": proposal.get("event_time_raw"),
            "time_basis": proposal.get("time_basis"),
            "observed_at": proposal["observed_at"],
            "evidence_chunk_id": str(proposal["chunk_id"]),
            "evidence_quote": proposal["quote"],
            "source_triples": list(proposal.get("time_triple_refs") or []),
            "status": "active", "corrects_assertion_id": corrects_id,
            "position": len(self.event_time_assertions) + 1,
        }
        return assertion_id

    def _add_attribute_assertion(self, event_id: str, predicate: str, value: str,
                                 proposal: dict[str, Any],
                                 corrects_id: str | None = None) -> str:
        """给同一次活动保存一个可独立纠正的属性断言。"""
        assertion_id = _id("event-attribute", event_id, predicate, value,
                           str(proposal["chunk_id"]), proposal["quote"])
        refs = [ref for ref in proposal.get("event_triple_refs") or []
                if value.casefold() in str(ref.get("tgt") or "").casefold() or
                str(ref.get("tgt") or "").casefold() in value.casefold()]
        self.event_attribute_assertions[assertion_id] = {
            "assertion_id": assertion_id, "event_id": event_id,
            "predicate": predicate, "value": value,
            "observed_at": proposal["observed_at"],
            "evidence_chunk_id": str(proposal["chunk_id"]),
            "evidence_quote": proposal["quote"],
            "source_triples": refs,
            "status": "active", "corrects_assertion_id": corrects_id,
            "position": len(self.event_attribute_assertions) + 1,
        }
        return assertion_id

    def add_event_instance(self, fact_id: str, proposal: dict[str, Any]) -> str:
        """将一条历史活动事实提升为独立事件实例，不覆盖其它同类活动。"""
        event_id = _id("event-instance", proposal["entity_id"], proposal["attribute"],
                       str(proposal["value"]), str(proposal.get("event_time") or ""),
                       str(proposal["chunk_id"]), proposal["quote"])
        if event_id in self.event_instances:
            return event_id
        refs = list(proposal.get("event_triple_refs") or proposal["triple_refs"])
        event = {
            "event_id": event_id, "actor": proposal["entity_id"],
            "entity_index": proposal["entity_index"], "action": proposal["attribute"],
            "target": proposal["value"], "fact_ids": [fact_id],
            "anchor_quote": proposal["quote"],
            "scope": None,
            "evidence_chunk_ids": [str(proposal["chunk_id"])],
            "source_triples": refs, "status": "active",
            "event_time": proposal.get("event_time"),
            "current_time_assertion_id": None,
            "current_attributes": {},
        }
        if proposal.get("event_time"):
            event["current_time_assertion_id"] = self._add_time_assertion(event_id, proposal)
        self.event_instances[event_id] = event
        for predicate, value in proposal.get("event_attributes", {}).items():
            event["current_attributes"][predicate] = self._add_attribute_assertion(
                event_id, predicate, value, proposal)
        self.facts[fact_id]["event_id"] = event_id
        self.event_instance_by_actor.setdefault(event["actor"], []).append(event_id)
        self._index_event_instance_evidence(event_id, str(proposal["chunk_id"]), refs)
        return event_id

    def attach_event_evidence(self, event_id: str, proposal: dict[str, Any]) -> None:
        """同一次活动的补充描述只增加证据来源，不创建第二个事件。"""
        event = self.event_instances[event_id]
        chunk_id = str(proposal["chunk_id"])
        if chunk_id not in event["evidence_chunk_ids"]:
            event["evidence_chunk_ids"].append(chunk_id)
        known = {triple_ref_key(ref) for ref in event["source_triples"]}
        refs = list(proposal.get("event_triple_refs") or proposal["triple_refs"])
        for ref in refs:
            if triple_ref_key(ref) not in known:
                event["source_triples"].append(ref)
                known.add(triple_ref_key(ref))
        self._index_event_instance_evidence(event_id, chunk_id, refs)

    def correct_event_time(self, event_id: str, proposal: dict[str, Any],
                           gate: dict[str, Any]) -> str:
        """逻辑覆盖有效日期，并保留原断言、纠正证据与旧 Triple 快照。"""
        event = self.event_instances[event_id]
        old_id = event.get("current_time_assertion_id")
        if not old_id or not proposal.get("event_time"):
            raise ValueError("日期纠正需要已有日期断言和有证据的新日期")
        old = self.event_time_assertions[old_id]
        if old["value"] == proposal["event_time"]:
            raise ValueError("新旧事件日期相同")
        if gate.get("same_event") is not True or gate.get("date_correction") is not True:
            raise ValueError("同一事件和日期纠正未被证实")
        old["status"] = "corrected"
        old["corrected_at"] = proposal["observed_at"]
        for ref in old["source_triples"]:
            self.superseded_event_time_triples.add(triple_ref_key(ref))
        self.attach_event_evidence(event_id, proposal)
        new_id = self._add_time_assertion(event_id, proposal, old_id)
        event["current_time_assertion_id"] = new_id
        event["event_time"] = proposal["event_time"]
        return new_id

    def add_event_time_detail(self, event_id: str, proposal: dict[str, Any]) -> str:
        """同一次事件此前没有日期时，用新证据补充日期。"""
        event = self.event_instances[event_id]
        if event.get("current_time_assertion_id") or not proposal.get("event_time"):
            raise ValueError("补充日期需要尚无日期的事件和明确的新日期")
        self.attach_event_evidence(event_id, proposal)
        assertion_id = self._add_time_assertion(event_id, proposal)
        event["current_time_assertion_id"] = assertion_id
        event["event_time"] = proposal["event_time"]
        return assertion_id

    def update_event_attributes(self, event_id: str, proposal: dict[str, Any],
                                corrected_attributes: set[str]) -> list[dict[str, str]]:
        """只补充或纠正本次原文提及的属性，其他属性和日期保持原值。"""
        event = self.event_instances[event_id]
        current = event.setdefault("current_attributes", {})
        operations: list[dict[str, str]] = []
        for predicate, value in proposal.get("event_attributes", {}).items():
            old_id = current.get(predicate)
            old = self.event_attribute_assertions[old_id] if old_id else None
            if old and old["value"].casefold() == value.casefold():
                continue
            if old and predicate not in corrected_attributes:
                raise ValueError(f"事件属性 {predicate} 的新值缺少纠正证据")
            if old:
                old["status"] = "corrected"
                old["corrected_at"] = proposal["observed_at"]
                for ref in old["source_triples"]:
                    self.superseded_event_attribute_triples.add(triple_ref_key(ref))
            new_id = self._add_attribute_assertion(event_id, predicate, value,
                                                   proposal, old_id)
            current[predicate] = new_id
            operations.append({"predicate": predicate,
                               "operation": "CORRECT" if old else "ADD",
                               "assertion_id": new_id})
        return operations

    def is_superseded_event_time_triple(self, ref: dict[str, str],
                                        question_time: str = "") -> bool:
        """按提问时间屏蔽已被纠正的日期关系；早于纠正的问题仍可看旧断言。"""
        key = triple_ref_key(ref)
        if key not in self.superseded_event_time_triples:
            return False
        query_date = re.search(r"\d{4}-\d{2}-\d{2}", str(question_time or ""))
        if not query_date:
            return True
        for assertion in self.event_time_assertions.values():
            if (assertion.get("status") == "corrected" and
                    any(triple_ref_key(item) == key for item in assertion["source_triples"])):
                corrected_at = str(assertion.get("corrected_at") or "")
                correction_date = re.search(r"\d{4}-\d{2}-\d{2}", corrected_at)
                return (not correction_date or
                        correction_date.group() <= query_date.group())
        return True

    def is_superseded_event_attribute_triple(self, ref: dict[str, str],
                                             question_time: str = "") -> bool:
        """通用事件属性纠正后按提问时间隐藏旧值的 Triple 快照。"""
        key = triple_ref_key(ref)
        if key not in self.superseded_event_attribute_triples:
            return False
        query_date = re.search(r"\d{4}-\d{2}-\d{2}", str(question_time or ""))
        if not query_date:
            return True
        for assertion in self.event_attribute_assertions.values():
            if (assertion.get("status") == "corrected" and
                    any(triple_ref_key(item) == key for item in assertion["source_triples"])):
                corrected_at = re.search(r"\d{4}-\d{2}-\d{2}",
                                         str(assertion.get("corrected_at") or ""))
                return not corrected_at or corrected_at.group() <= query_date.group()
        return True

    def _rebuild_changes(self, entity_id: str, attribute: str) -> None:
        """历史纠正后重新连接仍有效的事实，保留原错误事件供审计。

        场景：A 被记为 x → y → z，后来发现第一条 x 是错的。
        纠正后需要撤销基于错误事实产生的变化事件，并重新计算
        仍有效事实之间的变化链。旧事件不删除，只把 status 改为 revoked。
        """
        for event in self.events.values():
            if (event["entity_id"] == entity_id and event["attribute"] == attribute and
                    event["update_type"] == "change"):
                event["status"] = "revoked"
        # 按写入顺序（position）重放该属性的有效事实。
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
                    # 该变化仍然成立：恢复为 supported。
                    self.events[event_id]["status"] = "supported"
                else:
                    # 重新生成此前不存在的变化事件。
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

    def apply_decision(self, proposal: dict[str, Any], operation: str,
                       old_fact_id: str | None = None,
                       gate: dict[str, Any] | None = None) -> str | None:
        """执行经过槽位和证据门控的操作；只有 REPLACE/CORRECT 生成上层事件。

        旧版 `apply()` 保留供历史图读取和既有实验复现；新写入器只调用本方法。
        v1 非空图没有 scope/modality/操作证据，必须另建图，不能隐式升级。
        """
        if self.slot_schema_version != 2 and self.facts:
            raise ValueError("旧版事实没有状态槽位信息；请在独立目录重新构建")
        if operation not in {"ADD", "REPLACE", "CORRECT", "REFINE", "NOOP", "UNCERTAIN"}:
            raise ValueError("未知状态操作")
        entity_id = str(proposal["entity_id"])
        attribute = str(proposal["attribute"])
        key = slot_key(proposal)
        value = proposal.get("value")
        cardinality = cardinality_for(attribute)
        old = self.facts.get(old_fact_id) if old_fact_id else None
        if old_fact_id and old is None:
            raise ValueError("旧事实 ID 不存在")
        decision = {
            "operation": operation, "entity_id": entity_id, "attribute": attribute,
            "slot_key": key, "old_fact_id": old_fact_id,
            "chunk_id": str(proposal["chunk_id"]), "gate": dict(gate or {}),
            "retrieval_hits": list(proposal.get("retrieval_hits") or []),
        }
        if operation == "UNCERTAIN":
            self.add_review(str(proposal["chunk_id"]), "状态槽位或替换证据不足")
            self.decisions.append(decision)
            return None
        if operation == "NOOP":
            self.decisions.append(decision)
            return old_fact_id
        if value is None and operation != "CORRECT":
            raise ValueError("该操作需要新状态值")
        if value is not None and (not isinstance(value, str) or not value.strip()):
            raise ValueError("状态值必须是非空文本")

        current_id = self.current.get(key)
        member_ids = self.members.get(key, [])
        if operation == "ADD":
            if cardinality == "single" and current_id:
                current = self.facts[current_id]
                if current["value"].casefold() == value.casefold():
                    decision["operation"] = "NOOP"
                    self.decisions.append(decision)
                    return current_id
                raise ValueError("单值槽已有不同状态；必须先完成替换门控")
            if cardinality == "set" and proposal.get("fact_type") != "event":
                for fact_id in member_ids:
                    existing = self.facts[fact_id]
                    if existing["status"] == "active" and existing["value"].casefold() == value.casefold():
                        decision["operation"] = "NOOP"
                        self.decisions.append(decision)
                        return fact_id
        else:
            if old is None or old["entity_id"] != entity_id or slot_key(old) != key:
                raise ValueError("操作必须指向同一实体和状态槽的旧事实")
            if old["status"] in {"corrected", "audit"}:
                raise ValueError("旧事实已失效")
            if operation == "REPLACE":
                if cardinality != "single" or current_id != old_fact_id:
                    raise ValueError("只有当前单值状态可以替换")
                if preliminary_operation(old, proposal) != "GATE":
                    raise ValueError("新旧值并非可门控的互斥变化")
            elif operation == "REFINE":
                if cardinality != "single" or current_id != old_fact_id or (
                        preliminary_operation(old, proposal) != "REFINE"):
                    raise ValueError("细化必须指向当前单值状态且保留旧值语义")
            elif operation == "CORRECT" and value is not None and (
                    str(old["value"]).casefold() == value.casefold()):
                raise ValueError("纠正值与旧值相同")

        if operation in {"REPLACE", "CORRECT"}:
            if not isinstance(gate, dict) or any(gate.get(name) is not True for name in
                                                  ("same_slot", "incompatible", "supported_transition",
                                                   "time_consistent")):
                raise ValueError("替换或纠正未通过状态与时间门控")
            evidence = gate.get("evidence_quote")
            if not isinstance(evidence, str) or not evidence.strip() or evidence not in str(
                    proposal.get("source_text") or proposal["quote"]):
                raise ValueError("门控引文不在当前原文中")

        id_parts = ["slot-v2", operation, entity_id, attribute, key,
                    str(proposal["chunk_id"]), str(proposal["quote"])]
        if proposal.get("fact_type") == "event":
            # 同一引文可能列举两次活动，事件事实 ID 还需区分客体。
            id_parts.extend((str(value), str(proposal.get("event_time") or "")))
        fact_id = _id(*id_parts)
        if fact_id in self.facts:
            return fact_id
        new_fact = {
            "fact_id": fact_id, "entity_id": entity_id,
            "entity_index": proposal["entity_index"], "attribute": attribute,
            "value": value, "kind": operation.casefold(),
            "status": "active" if value is not None else "audit",
            "position": len(self.facts) + 1, "event_type": proposal["event_type"],
            "event_group_id": proposal.get("event_group_id"),
            "session_id": proposal["session_id"], "chunk_id": str(proposal["chunk_id"]),
            "observed_at": proposal["observed_at"], "event_time": proposal.get("event_time"),
            "quote": proposal["quote"], "triple_refs": proposal["triple_refs"],
            "scope": dict(proposal.get("scope") or {}), "modality": proposal["modality"],
            "slot_key": key, "cardinality": cardinality, "target_fact_id": old_fact_id,
            "source_attribute": proposal.get("source_attribute", attribute),
            "fact_type": proposal.get("fact_type", "state"),
            "event_id": None,
            "event_time_raw": proposal.get("event_time_raw"),
            "time_basis": proposal.get("time_basis"),
        }
        self.facts[fact_id] = new_fact
        decision["new_fact_id"] = fact_id
        self.decisions.append(decision)

        if operation == "ADD":
            if cardinality == "single":
                self.current[key] = fact_id
            else:
                self.members.setdefault(key, []).append(fact_id)
            return fact_id
        if operation == "REFINE":
            old["status"] = "refined"
            self.current[key] = fact_id
            return fact_id
        if operation == "REPLACE":
            old["status"] = "superseded"
            self.current[key] = fact_id
        else:
            old["status"] = "corrected"
            if cardinality == "single" and current_id == old_fact_id:
                if value is None:
                    self.current.pop(key, None)
                else:
                    self.current[key] = fact_id
            if cardinality == "set" and old_fact_id in member_ids:
                self.members[key] = [item for item in member_ids if item != old_fact_id]
                if value is not None:
                    self.members[key].append(fact_id)
            # 历史纠正只撤销依赖旧事实的转换，不凭剩余事实时间顺序创建新事件。
            for event in self.events.values():
                if old_fact_id in (event["old_fact_id"], event["new_fact_id"]):
                    event["status"] = "revoked"

        event_id = _id("slot-v2-event", operation, old_fact_id, fact_id)
        event = {
            "event_id": event_id, "event_group_id": proposal.get("event_group_id"),
            "entity_id": entity_id, "entity_index": proposal["entity_index"],
            "attribute": attribute, "scope": dict(proposal.get("scope") or {}),
            "modality": proposal["modality"], "slot_key": key,
            "operation": operation, "gate": dict(gate),
            "event_type": proposal["event_type"],
            "update_type": "change" if operation == "REPLACE" else "correction",
            "old_fact_id": old_fact_id, "new_fact_id": fact_id,
            "old_state": old["value"], "new_state": value,
            "event_time": proposal.get("event_time"), "observed_at": proposal["observed_at"],
            "session_id": proposal["session_id"], "chunk_id": str(proposal["chunk_id"]),
            "quote": proposal["quote"], "triple_refs": proposal["triple_refs"],
            "status": "supported",
        }
        self.events[event_id] = event
        self._index_event(event)
        return fact_id

    def apply(self, proposal: dict[str, Any]) -> str | None:
        """新增基线、变化或纠正；不以普通活动覆盖有效状态。

        Args:
            proposal: _validate 产出的规范化候选，含 kind/entity_id/attribute/
                      value/quote/chunk_id/observed_at 等。

        Returns:
            新建事件的 event_id；只是建立基线（没有变化）或幂等重复时返回 None。

        三种 kind：
            baseline   该属性第一次出现，只记录事实，不产生事件；
            change     与当前值不同，产生一条变化事件；
            correction 否定一条旧事实；可能给新值，也可能只否定（value=None）。
        """
        entity_id = proposal["entity_id"]
        attribute = proposal["attribute"]
        value = proposal.get("value")
        kind = proposal["kind"]
        chunk_id = proposal["chunk_id"]
        slot = _slot(entity_id, attribute)
        current = self.current_fact(entity_id, attribute)
        target_id = proposal.get("target_fact_id")

        if kind == "correction":
            # 纠正必须指向有效旧事实，不能纠正已纠正过的事实。
            target = self.facts.get(target_id)
            if (not target or target["status"] == "corrected" or
                    target["entity_id"] != entity_id or target["attribute"] != attribute):
                raise ValueError("纠正必须明确指向同一实体、同一属性的有效旧事实")
            old_fact = target
        else:
            if value is None:
                raise ValueError("状态事实必须有新值")
            # 与当前值相同视为重复陈述，不产生新事实。
            if current and current["value"].casefold() == value.casefold():
                return None
            old_fact = current
            kind = "change" if current else "baseline"

        # 事实 ID 由来源唯一确定：同一 Chunk 的同一引文重跑不会重复写入。
        fact_id = _id(entity_id, attribute, chunk_id, proposal["quote"], kind)
        if fact_id in self.facts:
            return None
        # 纠正沿用被纠正事实的 position，保证时间线顺序不被新写入打乱。
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
                    # 只否定、未给新值：该属性暂时没有当前值。
                    self.current.pop(slot, None)
                else:
                    self.current[slot] = fact_id
        else:
            self.current[slot] = fact_id

        if kind == "baseline":
            # 首次出现只是建立基线，没有“变化”可记录。
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
            # 纠正会改变历史变化链，需要重算该属性的全部变化事件。
            self._rebuild_changes(entity_id, attribute)
        return event_id

    def events_for_entity(self, entity_id: str, attribute: str | None = None) -> list[dict[str, Any]]:
        """按实体（可选属性）取事件列表。"""
        events = [self.events[event_id] for event_id in self.by_entity.get(entity_id, [])]
        if attribute is not None:
            events = [event for event in events if event["attribute"] == attribute]
        return events

    def events_for_triple(self, ref: dict[str, str]) -> list[dict[str, Any]]:
        """从中层关系快照反查引用它的上层变化。"""
        return [self.events[event_id] for event_id in self.by_triple.get(triple_ref_key(ref), [])]
