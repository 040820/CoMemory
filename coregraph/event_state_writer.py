"""在记忆写入时构建事件状态上层，直接连接实体索引、关系和原话。

event_state 模式的上层记忆构建流程（本文件）：
    1. 按 (session_time, session_id, chunk_id) 排序所有 Chunk；
    2. 以 session 为单位分组：会话之间严格串行，保证时间顺序；
       同一会话内的 Chunk 并发调用 LLM，但按 Chunk 顺序提交结果；
    3. 每个 Chunk 只允许引用“图中已存在的实体”和“本 Chunk 的关系”；
    4. LLM 返回候选事实后，程序逐条验证（实体、原文引文、说话人归属、
       新值是否出现在引文中、纠正是否指向旧事实等），验证通过才写入；
    5. 每个 Chunk 的写入在 EventStateLayer 的副本上完成，全部成功才替换原层，
       避免一条坏数据造成部分更新。

设计原则：
    - LLM 只负责“发现证据”，不负责猜测旧值或时间；
    - 任何写入都必须能同时回溯到实体索引、Chunk 原文和关系三元组；
    - 无法确定或调用失败的内容进入 review_queue，而不是静默丢弃。
"""

from __future__ import annotations

import asyncio
import json
import re
from hashlib import sha256
from itertools import groupby
from typing import Any

from .event_state_layer import EventStateLayer
from .event_candidate_schema import extraction_format, resolve_indices, triple_id
from .event_instance_time import bind_event_time, mentioned_dates
from .state_candidate_retriever import StateCandidateRetriever
from .state_slot import (canonical_attribute, cardinality_for, normalize_modality,
                         normalize_scope, preliminary_operation, slot_key,
                         temporal_gate)
from init.logger import logger


def _chunk_refs(chunk_id: str, relationships: list[dict[str, Any]]) -> list[dict[str, str]]:
    """只取本次抽取明确指向该 Chunk 的关系，不继承合并图的模糊边。

    为什么不能直接用图里的边：
        建图时重复关系会被合并，一条边可能挂着多个 chunk_id；
        若直接使用，会把其它 Chunk 的关系错误地算到当前证据上。
        因此这里只认 relationship 自身携带的 chunk_id/chunk_ids。
    """
    refs: list[dict[str, str]] = []
    for relation in relationships:
        ids = relation.get("chunk_ids") or []
        if not isinstance(ids, (list, tuple, set)):
            ids = [ids]
        ids = {str(value) for value in ids if value is not None}
        if relation.get("chunk_id") is not None:
            ids.add(str(relation["chunk_id"]))
        if chunk_id not in ids:
            continue
        src = str(relation.get("src") or "").strip()
        tgt = str(relation.get("tgt") or "").strip()
        predicate = str(relation.get("relation") or "").strip()
        if src and tgt and predicate:
            ref = {"src": src, "relation": predicate, "tgt": tgt, "chunk_id": chunk_id}
            ref["triple_id"] = triple_id(ref)
            if ref not in refs:
                refs.append(ref)
    return refs


def _speaker_support(text: str, quote: str, name: str, names: list[str]) -> bool:
    """判断一句引文是否确实属于目标实体（主要通过说话人标签）。

    对话场景中“我/我的”这类第一人称引文无法直接看出是谁说的，
    需要借助引文之前最近的说话人标签来归属。

    规则：
        - 引文不能跨越两个说话人标签，否则归属有歧义，直接拒绝；
        - 引文中显式出现实体名，或所有出现位置的前一位说话人都是该实体，才算通过。
    """
    labels: list[tuple[int, str]] = []
    for candidate in names:
        # 匹配 "名字:" / "名字 ：" 形式，忽略大小写，要求前面不是单词字符。
        pattern = rf"(?<!\w){re.escape(candidate)}\s*:"
        labels.extend((match.start(), candidate) for match in re.finditer(pattern, text, re.I))
    labels.sort()
    owners: set[str | None] = set()
    start = 0
    # 同一句引文可能在原文中出现多次，每一处都要检查归属。
    while (start := text.find(quote, start)) >= 0:
        # 引文不能跨越两个说话人，否则可能把后一人的状态记到前一人。
        if any(start < position < start + len(quote) for position, _ in labels):
            return False
        previous = [candidate for position, candidate in labels if position < start]
        owners.add(previous[-1] if previous else None)
        start += len(quote)
    return bool(owners) and (name.casefold() in quote.casefold() or owners == {name})


def _parse_json(response: str) -> list[dict[str, Any]]:
    """解析 LLM 返回的 facts JSON，兼容 markdown 代码块与 {"facts": [...]} 包装。"""
    clean = response.strip()
    if clean.startswith("```"):
        clean = clean.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    data = json.loads(clean)
    if isinstance(data, dict):
        data = data.get("facts")
    if not isinstance(data, list) or not all(isinstance(item, dict) for item in data):
        raise ValueError("模型没有返回 facts 数组")
    return data


def _parse_gate_json(response: str) -> dict[str, Any]:
    """第二次 LLM 调用只接受一个门控 JSON 对象。"""
    clean = response.strip()
    if clean.startswith("```"):
        clean = clean.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    result = json.loads(clean)
    if not isinstance(result, dict):
        raise ValueError("门控结果必须是 JSON 对象")
    return result


class IgnoreCandidate(ValueError):
    """普通对话或附件动作是预期过滤，不应算作待人工审核的事实失败。"""


def _event_targets_overlap(old: dict[str, Any], new: dict[str, Any]) -> bool:
    """同主体的不同活动不能仅靠向量相似进入合并。

    至少需要活动对象的名称或已绑定中层对象有共同锚点；这只是召回条件，
    最终仍由事件身份门控检查。纯粹共享“group/event”等泛称不算共同锚点。
    """
    previous = str(old["target"]).casefold()
    incoming = str(new["value"]).casefold()
    if previous == incoming:
        return True
    stop = {"a", "an", "the", "that", "this", "group", "event", "activity", "meeting"}
    left = set(re.findall(r"[a-z0-9]+", previous)) - stop
    right = set(re.findall(r"[a-z0-9]+", incoming)) - stop
    if left and right and len(left & right) / min(len(left), len(right)) >= 0.5:
        return True
    def targets(refs, actor, text):
        return {str(ref.get("tgt", "")).casefold() for ref in refs
                if ref.get("src") == actor and str(ref.get("tgt", "")).casefold() in text}
    return bool(targets(old.get("source_triples") or [], old["actor"], previous) &
                targets(new.get("event_triple_refs") or new.get("triple_refs") or [],
                        new["entity_id"], incoming))


class EventStateWriter:
    """每个 Chunk 调用一次 LLM；没有持续事实变化时不产生上层事件。"""

    def __init__(self, llm: Any, embedding_manager: Any | None = None,
                 candidate_top_k: int = 5):
        self.llm = llm
        self.candidates = StateCandidateRetriever(embedding_manager, top_k=candidate_top_k)

    async def write_chunks(self, layer: EventStateLayer, chunks: list[dict[str, Any]],
                           relationships: list[dict[str, Any]],
                           entity_index: dict[str, int]) -> EventStateLayer:
        """把一批 Chunk 写入事件状态层。

        Args:
            layer:        已有事件状态层（增量模式下会带入历史事实）。
            chunks:       本批切块结果，含 chunk_id/text/session_id/session_time。
            relationships: 与本批 Chunk 对应的关系抽取结果。
            entity_index: 实体名 → 实体序号，来自中层图。只有能在这里查到的实体
                          才允许成为状态主体。

        Returns:
            新的 EventStateLayer。注意函数采用“返回新层”的风格，
            但内部逐 Chunk 替换局部变量 layer，调用方必须用返回值覆盖旧对象
            （DynamicMemory 中即 self.event_state_layer = await ...）。

        并发模型：
            会话之间顺序执行（事件是有时间先后的，后一会话要看到前一会话的状态）；
            同一会话内的多个 Chunk 并发请求 LLM，以降低等待时间；
            结果按 Chunk 原顺序依次提交，保证同一会话内的状态演化仍是有序的。
        """
        if layer.slot_schema_version != 2 and layer.facts:
            raise ValueError("旧版非空事件层没有状态槽信息；请使用独立目录重建")
        if layer.writer_schema_version != 2 and layer.facts:
            raise ValueError("旧版事件写入协议需要在独立目录重建，不能继续追加")
        def order_key(row: dict[str, Any]) -> tuple[Any, ...]:
            # 时间 → 会话 → Chunk 编号排序；非数字 chunk_id 排在数字之后。
            raw_chunk_id = row.get("chunk_id")
            chunk_id = "" if raw_chunk_id is None else str(raw_chunk_id)
            return (str(row.get("session_time") or ""), str(row.get("session_id") or ""),
                    0 if chunk_id.isdecimal() else 1,
                    int(chunk_id) if chunk_id.isdecimal() else chunk_id)

        ordered = sorted(chunks, key=order_key)
        for _, group_iter in groupby(ordered, key=lambda row: str(row.get("session_id") or "")):
            group = list(group_iter)
            pending: list[tuple[dict[str, Any], list[str], list[dict[str, str]], str]] = []
            # 第一阶段只抽取当前 Chunk 的新事实，不把实体历史塞进提示词。
            for chunk in group:
                raw_chunk_id = chunk.get("chunk_id")
                chunk_id = "" if raw_chunk_id is None else str(raw_chunk_id)
                text = str(chunk.get("text") or "")
                # 已处理过的 Chunk 跳过，使增量写入可以安全重跑。
                if not chunk_id or not text or chunk_id in layer.processed_chunks:
                    continue
                refs = _chunk_refs(chunk_id, relationships)
                # 候选实体 = 原文中出现的实体 ∪ 本 Chunk 关系中涉及的实体。
                names = [name for name in entity_index
                         if re.search(rf"(?<!\w){re.escape(name)}(?!\w)", text, re.I)]
                for ref in refs:
                    for name in (ref["src"], ref["tgt"]):
                        if name in entity_index and name not in names:
                            names.append(name)
                if not names:
                    # 没有可关联实体：记录待核查，不再调用 LLM。
                    layer.add_review(chunk_id, "原图中找不到可关联的实体")
                    layer.processed_chunks.add(chunk_id)
                    continue
                prompt = self._prompt(chunk, names, refs)
                pending.append((chunk, names, refs, prompt))

            # 同一会话内并发调用；return_exceptions=True 保证单个失败不影响其它请求。
            responses = await asyncio.gather(
                *(self.llm.generate(prompt, max_tokens=2048,
                                    response_format=extraction_format(names, refs))
                  for _, names, refs, prompt in pending),
                return_exceptions=True,
            )
            for (chunk, names, refs, _), response in zip(pending, responses):
                chunk_id = str(chunk["chunk_id"])
                # 取消是控制流信号，必须继续向上传播，不能当作普通失败。
                if isinstance(response, asyncio.CancelledError):
                    raise response
                if isinstance(response, BaseException):
                    # 调用失败不标记 processed，留待下次重试。
                    layer.add_review(chunk_id, f"LLM 调用失败，待重试：{type(response).__name__}: {response}")
                    continue
                if not response:
                    layer.add_review(chunk_id, "LLM 返回空内容；保留重试机会")
                    continue
                try:
                    items = _parse_json(response)
                    # 缺引文的模型输出不能直接写入；要求模型仅重做本 Chunk。
                    if any(not isinstance(item.get("quote"), str) or
                           not item["quote"].strip() for item in items):
                        raise ValueError("候选缺少原文 quote")
                except (ValueError, IndexError, TypeError) as first_error:
                    repaired = await self.llm.generate(
                        self._repair_prompt(chunk, names, refs, str(first_error)),
                        max_tokens=2048, response_format=extraction_format(names, refs))
                    try:
                        items = _parse_json(repaired)
                    except (ValueError, IndexError, TypeError) as error:
                        layer.add_review(chunk_id,
                                         f"模型结构无效，重试后仍失败：{error}")
                        layer.processed_chunks.add(chunk_id)
                        continue
                    if any(not isinstance(item.get("quote"), str) or
                           not item["quote"].strip() for item in items):
                        layer.add_review(chunk_id, "重试后仍有候选缺少原文 quote")
                        # 其它带有效 quote 的候选仍可逐条接受；无引文者在下方拒绝。
                except BaseException:
                    raise
                if not isinstance(items, list):
                    # 结构无效说明模型输出无法使用；标记已处理避免反复重试同一坏输出。
                    layer.add_review(chunk_id, "模型结构无效：facts 必须为数组")
                    layer.processed_chunks.add(chunk_id)
                    continue
                # 在副本中写入本 Chunk 的所有属性，避免其中一条失败造成部分更新。
                staged = EventStateLayer(layer.to_dict())
                seen_facts: set[tuple[str, str, str, str]] = set()
                # 同一原文、主体和值已给出具体谓词时，不再重复保存 related_to。
                generic = {"related_to", "is_related_to"}
                concrete = {(str(item.get("entity")), str(item.get("value")).casefold(),
                             str(item.get("quote"))) for item in items
                            if str(item.get("attribute") or "").casefold().replace(" ", "_") not in generic}
                for item in items:
                    try:
                        if (str(item.get("attribute") or "").casefold().replace(" ", "_") in generic and
                                (str(item.get("entity")), str(item.get("value")).casefold(),
                                 str(item.get("quote"))) in concrete):
                            raise IgnoreCandidate("同一证据已保存具体关系，省略重复的泛化关系")
                        proposal = self._validate(item, chunk, names, refs, entity_index, staged)
                        # 集合属性允许同一 Chunk 里有多个对象，只拒绝完全重复的候选。
                        key = (proposal["entity_id"], proposal["attribute"],
                               slot_key(proposal), str(proposal.get("value")))
                        if key in seen_facts:
                            continue
                        seen_facts.add(key)
                        await self._apply_candidate(staged, proposal)
                    except IgnoreCandidate as error:
                        staged.skipped_candidates.append({
                            "chunk_id": chunk_id, "reason": str(error),
                            "attribute": str(item.get("attribute") or "")})
                    except (KeyError, TypeError, ValueError) as error:
                        # 单条候选被拒：写入 review_queue，但该 Chunk 其它候选继续处理。
                        staged.add_review(chunk_id, f"状态候选未写入：{error}")
                staged.processed_chunks.add(chunk_id)
                # 本 Chunk 全部处理完毕才提交副本。
                layer = staged
            logger.info("事件状态层：已处理会话 %s，本批 Chunk %d/%d、全局累计 %d、事实 %d、变化 %d、待核查 %d",
                        str(group[0].get("session_id") or ""),
                        sum(str(row.get("chunk_id")) in layer.processed_chunks for row in group),
                        len(group), len(layer.processed_chunks), len(layer.facts),
                        len(layer.events), len(layer.review_queue))
        return layer

    async def _apply_candidate(self, layer: EventStateLayer,
                               proposal: dict[str, Any]) -> None:
        """先用新事实的三元组召回少量旧证据，再逐条门控。"""
        hits = await self.candidates.search(layer, proposal)
        proposal["retrieval_hits"] = [
            {"fact_id": hit["fact"]["fact_id"], "score": round(hit["score"], 4),
             "triple_refs": hit["source_triples"]}
            for hit in hits
        ]
        if proposal.get("fact_type") == "event" and proposal["kind"] == "state":
            await self._apply_event_candidate(layer, proposal, hits)
            return
        key = slot_key(proposal)
        if proposal["kind"] == "correction":
            old_id = proposal.get("target_fact_id")
            old = layer.facts.get(old_id)
            if not old_id:
                # 初次抽取不带历史，因此不能要求模型凭空生成旧 fact_id。
                # 先用本条三元组召回，随后逐条检查同槽位与纠正证据。
                for hit in hits:
                    candidate = hit["fact"]
                    if slot_key(candidate) != key:
                        continue
                    gate = await self._judge_transition(candidate, proposal, "CORRECT")
                    if self._gate_passes(gate, "CORRECT", proposal):
                        layer.apply_decision(proposal, "CORRECT", candidate["fact_id"], gate)
                        return
                layer.apply_decision(proposal, "UNCERTAIN")
                return
            if not old or slot_key(old) != key:
                layer.apply_decision(proposal, "UNCERTAIN", old_id if old else None)
                return
            consistent, reason = temporal_gate(old, proposal)
            if not consistent:
                layer.apply_decision(proposal, "UNCERTAIN", old_id,
                                     {"time_consistent": False, "time_reason": reason})
                return
            gate = await self._judge_transition(old, proposal, "CORRECT")
            if self._gate_passes(gate, "CORRECT", proposal):
                layer.apply_decision(proposal, "CORRECT", old_id, gate)
            else:
                layer.apply_decision(proposal, "UNCERTAIN", old_id, gate)
            return

        # 多值关系默认 ADD；语义相似的另一个对象也不是替换依据。
        if cardinality_for(proposal["attribute"]) == "set" and not any(
                cardinality_for(hit["fact"]["attribute"]) == "single"
                and hit["score"] >= 0.65 for hit in hits):
            layer.apply_decision(proposal, "ADD")
            return

        exact_old_id = layer.current.get(key)
        ordered: list[dict[str, Any]] = []
        if exact_old_id:
            ordered.append(layer.facts[exact_old_id])
        for hit in hits:
            old = hit["fact"]
            if old["fact_id"] == exact_old_id or hit["score"] < 0.65:
                continue
            if cardinality_for(old["attribute"]) != "single":
                continue
            if layer.current.get(slot_key(old)) != old["fact_id"]:
                continue
            if old.get("scope", {}) != proposal["scope"] or old.get("modality") != proposal["modality"]:
                continue
            ordered.append(old)

        for old in ordered:
            aligned = dict(proposal)
            if old["attribute"] != aligned["attribute"]:
                # 语义匹配只生成候选；最终是否同义仍由原文门控确定。
                aligned["source_attribute"] = aligned["attribute"]
                aligned["attribute"] = old["attribute"]
            preliminary = preliminary_operation(old, aligned)
            if preliminary == "NOOP":
                layer.apply_decision(aligned, "NOOP", old["fact_id"])
                return
            if preliminary == "REFINE":
                layer.apply_decision(aligned, "REFINE", old["fact_id"])
                return
            if preliminary != "GATE":
                continue
            consistent, reason = temporal_gate(old, aligned)
            if not consistent:
                if old["fact_id"] == exact_old_id:
                    layer.apply_decision(aligned, "UNCERTAIN", old["fact_id"],
                                         {"time_consistent": False, "time_reason": reason})
                    return
                continue
            gate = await self._judge_transition(old, aligned, "REPLACE")
            if self._gate_passes(gate, "REPLACE", aligned):
                layer.apply_decision(aligned, "REPLACE", old["fact_id"], gate)
                return
            if old["fact_id"] == exact_old_id:
                layer.apply_decision(aligned, "UNCERTAIN", old["fact_id"], gate)
                return

        # 没有可证实的旧状态：新事实只追加，绝不因相似度或时间排序生成替换。
        layer.apply_decision(proposal, "ADD")

    async def _apply_event_candidate(self, layer: EventStateLayer,
                                     proposal: dict[str, Any],
                                     hits: list[dict[str, Any]]) -> None:
        """同类活动先找事件身份；不同活动独立 ADD，日期纠正指向原实例。"""
        actor = proposal["entity_id"]
        target = str(proposal["value"]).casefold()
        candidate_scores: dict[str, float] = {}
        for hit in hits:
            event_id = hit["fact"].get("event_id")
            if (event_id in layer.event_instances and
                    _event_targets_overlap(layer.event_instances[event_id], proposal)):
                candidate_scores[event_id] = max(
                    candidate_scores.get(event_id, 0.0), hit["score"])
        # 精确目标、包含关系或相同行为不会因 Top-5 被其他属性挤掉。
        for event_id in layer.event_instance_by_actor.get(actor, []):
            event = layer.event_instances[event_id]
            if not _event_targets_overlap(event, proposal):
                continue
            old_target = str(event["target"]).casefold()
            new_terms = set(re.findall(r"[a-z0-9]+", target)) - {"a", "an", "the", "that", "this"}
            old_terms = set(re.findall(r"[a-z0-9]+", old_target)) - {"a", "an", "the", "that", "this"}
            overlap = len(new_terms & old_terms) / max(1, len(new_terms))
            if (target == old_target or target in old_target or old_target in target or
                    overlap >= 0.5):
                candidate_scores[event_id] = max(candidate_scores.get(event_id, 0.0),
                                                 2.0 if target == old_target else
                                                 1.5 if overlap >= 0.5 else 0.8)
        ordered = sorted(candidate_scores, key=lambda event_id: (
            -candidate_scores[event_id], event_id))[:self.candidates.top_k]
        for event_id in ordered:
            old = layer.event_instances[event_id]
            # 两次活动已有不同的精确日期，且新原文没有纠正语气时，
            # 相同 actor/action/target 也不能自动合并。
            if (old.get("event_time") and proposal.get("event_time") and
                    old["event_time"] != proposal["event_time"] and
                    not re.search(r"\b(actually|not|instead|rather|correct|in fact)\b",
                                  proposal["source_text"], re.I)):
                layer.event_link_decisions.append({
                    "candidate_event_id": event_id, "chunk_id": proposal["chunk_id"],
                    "operation": "DISTINCT_BY_DATE", "reason": "不同日期且无纠正证据"})
                continue
            old_context = dict(old)
            old_context["current_attribute_values"] = {
                name: layer.event_attribute_assertions[assertion_id]["value"]
                for name, assertion_id in old.get("current_attributes", {}).items()
            }
            gate = await self._judge_event_identity(old_context, proposal)
            layer.event_link_decisions.append({
                "candidate_event_id": event_id, "chunk_id": proposal["chunk_id"],
                "score": round(candidate_scores[event_id], 4), "gate": gate,
            })
            if gate.get("same_event") is not True:
                if gate.get("same_event") is None:
                    layer.add_review(proposal["chunk_id"], "事件身份判断不确定")
                    return
                continue
            old_time = old.get("event_time")
            new_time = proposal.get("event_time")
            cues = re.search(r"\b(actually|not|instead|rather|correct|in fact)\b",
                             proposal["source_text"], re.I)
            corrected_attributes = set(gate.get("corrected_attributes") or [])
            for name, value in proposal.get("event_attributes", {}).items():
                old_id = old.get("current_attributes", {}).get(name)
                previous = layer.event_attribute_assertions[old_id] if old_id else None
                if (previous and previous["value"].casefold() != value.casefold() and
                        (name not in corrected_attributes or not cues)):
                    layer.add_review(proposal["chunk_id"],
                                     f"事件属性 {name} 的新值缺少明确纠正证据")
                    return
            if old_time and new_time and old_time != new_time:
                # 日期不同不等于纠正；必须能从当前原文指出旧日期和明确纠正词。
                if (gate.get("date_correction") is True and cues and
                        old_time in mentioned_dates(proposal["source_text"],
                                                    proposal["observed_at"])):
                    assertion_id = layer.correct_event_time(event_id, proposal, gate)
                    layer.event_link_decisions[-1]["operation"] = "CORRECT_EVENT_TIME"
                    layer.event_link_decisions[-1]["new_assertion_id"] = assertion_id
                    attribute_operations = layer.update_event_attributes(
                        event_id, proposal, corrected_attributes)
                    if attribute_operations:
                        layer.event_link_decisions[-1]["attribute_operations"] = attribute_operations
                else:
                    layer.add_review(proposal["chunk_id"], "同一事件的日期冲突缺少明确纠正证据")
                return
            if not old_time and new_time:
                assertion_id = layer.add_event_time_detail(event_id, proposal)
                layer.event_link_decisions[-1]["operation"] = "ADD_EVENT_TIME"
                layer.event_link_decisions[-1]["new_assertion_id"] = assertion_id
            else:
                layer.attach_event_evidence(event_id, proposal)
                layer.event_link_decisions[-1]["operation"] = "SAME_EVENT"
            attribute_operations = layer.update_event_attributes(
                event_id, proposal, corrected_attributes)
            if attribute_operations:
                layer.event_link_decisions[-1]["attribute_operations"] = attribute_operations
            return

        fact_id = layer.apply_decision(proposal, "ADD")
        if fact_id:
            event_id = layer.add_event_instance(fact_id, proposal)
            layer.event_link_decisions.append({
                "candidate_event_id": None, "event_id": event_id,
                "chunk_id": proposal["chunk_id"], "operation": "NEW_EVENT",
                "retrieval_hits": list(proposal.get("retrieval_hits") or []),
            })

    async def _judge_event_identity(self, old: dict[str, Any],
                                    new: dict[str, Any]) -> dict[str, Any]:
        """只比较一对活动；相同人物和动词不能自动视为同一次事件。"""
        old_context = {key: old.get(key) for key in
                       ("actor", "action", "target", "event_time", "anchor_quote")}
        old_context["current_attribute_values"] = old.get("current_attribute_values", {})
        old_context["evidence_chunk_ids"] = old.get("evidence_chunk_ids", [])[-8:]
        old_context["source_triples"] = old.get("source_triples", [])[-8:]
        prompt = (
            "EVENT_IDENTITY_GATE\n判断两段证据是否描述现实中同一次活动。"
            "相同人物、相同行为或相似活动名称均不是充分条件。"
            "若当前原文明确纠正该事件日期，另给 date_correction=true。"
            "若明确纠正旧事件的地点或其他属性，给 corrected_attributes 属性名数组；"
            "普通补充信息不要放入该数组。"
            "只输出 JSON：same_event 和 date_correction 为布尔值，"
            "corrected_attributes 为字符串数组，evidence_quote 必须是当前原文的连续子串，"
            "并给简短 reason。\n"
            f"旧事件：{json.dumps(old_context, ensure_ascii=False)}\n"
            f"新候选：{json.dumps({key: new.get(key) for key in ('entity_id', 'attribute', 'value', 'event_time', 'event_time_raw', 'observed_at', 'quote')}, ensure_ascii=False)}\n"
            f"当前完整原文：{new['source_text']}"
        )
        try:
            response = await self.llm.generate(prompt, max_tokens=192)
            result = _parse_gate_json(response)
            quote = result.get("evidence_quote")
            if (type(result.get("same_event")) is not bool or
                    type(result.get("date_correction")) is not bool or
                    not isinstance(result.get("corrected_attributes", []), list) or
                    not all(isinstance(name, str) for name in result.get("corrected_attributes", [])) or
                    not isinstance(quote, str) or not quote.strip() or
                    quote not in new["source_text"]):
                raise ValueError("事件身份判断缺少布尔结论或原文引文")
            return result
        except asyncio.CancelledError:
            raise
        except Exception as error:
            return {"same_event": None, "date_correction": None,
                    "reason": f"事件身份判断无效：{error}"}

    async def _judge_transition(self, old: dict[str, Any], new: dict[str, Any],
                                desired: str) -> dict[str, Any]:
        """让 LLM 对一对事实和完整原文独立作三道门控判断。"""
        prompt = (
            "STATE_SLOT_GATE\n你是事实更新审计员。只判断下面两条事实，不抽取新事实。"
            "依次回答：是否同一现实状态槽位 same_slot；值是否在该作用域和模态下互斥 "
            "incompatible；当前原文是否支持旧状态被替换或明确纠正 supported_transition。"
            "养狗又养猫、参加不同活动、谈不同话题、多个身份都不互斥。"
            "不得仅凭新旧时间排序推断替换；需要指出当前原文中支持判断的连续引文。"
            "若三项任一不成立，operation 必须是 UNCERTAIN；"
            f"只有全部成立时 operation 才可为 {desired}。"
            "仅输出 JSON 对象，字段为 same_slot、incompatible、supported_transition "
            "（均为布尔值）、operation、evidence_quote（当前原文连续子串）、reason。\n"
            f"旧事实及中层关系：{json.dumps({k: old.get(k) for k in ('entity_id', 'attribute', 'value', 'scope', 'modality', 'quote', 'triple_refs', 'observed_at', 'event_time')}, ensure_ascii=False)}\n"
            f"新事实查询及中层关系：{json.dumps({k: new.get(k) for k in ('entity_id', 'attribute', 'source_attribute', 'value', 'scope', 'modality', 'quote', 'triple_refs', 'observed_at', 'event_time')}, ensure_ascii=False)}\n"
            f"当前完整原文：{new['source_text']}"
        )
        try:
            response = await self.llm.generate(prompt, max_tokens=256)
            result = _parse_gate_json(response)
            consistent, reason = temporal_gate(old, new)
            result["time_consistent"] = consistent
            result["time_reason"] = reason
            return result
        except asyncio.CancelledError:
            raise
        except Exception as error:
            return {"operation": "UNCERTAIN", "reason": f"门控结果无效：{error}"}

    @staticmethod
    def _gate_passes(gate: dict[str, Any], desired: str,
                     proposal: dict[str, Any]) -> bool:
        """布尔结论之外，证据必须能在当前 Chunk 原文中逐字定位。"""
        evidence = gate.get("evidence_quote")
        return (gate.get("operation") == desired and
                all(gate.get(name) is True for name in
                    ("same_slot", "incompatible", "supported_transition",
                     "time_consistent")) and
                isinstance(evidence, str) and bool(evidence.strip()) and
                evidence in proposal["source_text"])

    @staticmethod
    def _prompt(chunk: dict[str, Any], names: list[str],
                refs: list[dict[str, str]]) -> str:
        """短提示：限制候选数量，让原文引文和有效 JSON 优先于泛化描述。"""
        return (
            "从对话中抽取最多 4 条有长期价值、可由原文逐字核对的事实。"
            "只输出完整 JSON 对象 {\"facts\":[...]}，没有事实输出空数组。"
            "每条必须填写 entity、attribute、value、kind、fact_type、modality、quote、"
            "scope、event_time、event_attributes、triple_ids、target_fact_id（未知填 null）。"
            "quote 必须是当前原文中连续且完全相同的子串，并包含 value；"
            "无法给出引文的候选不要输出。value 不得是推断或改写。"
            "entity 必须是可用实体；第一人称归属当前说话人。"
            "attribute 用简短英文 snake_case；当前雇主用 employer，当前居所用 residence；"
            "养宠物、参加活动等可并存事实不得作为单值状态。"
            "kind=state；明确否认或纠正旧事实时用 correction，未知旧 ID 的 target_fact_id 填 null。"
            "fact_type=event 仅限现实中一次具体活动；提问、评论、聊天、上传照片"
            "及附件描述都不是现实事件。计划/假设分别标记 modality=planned/hypothetical。"
            "活动的 value 是对象，event_time 只填原文中的时间词，绝不自行推日期；"
            "活动地点仅在同一引文明确说出时写 event_attributes.location。"
            "scope 无具体原文依据填 {}，event_attributes 无依据填 {}。"
            "triple_ids 是字符串 ID 数组，只复制下方本 Chunk 的 triple_id，找不到填 []。"
            "同一次活动只输出一条，不要按中层每条关系复制候选；"
            "feeling、related_to、兴趣、习惯属于 state，不是一次 event。"
            "示例：Anna: I attended the book fair at City Hall yesterday. → "
            "{\"entity\":\"Anna\",\"attribute\":\"attended\",\"value\":\"book fair\","
            "\"kind\":\"state\",\"fact_type\":\"event\",\"modality\":\"actual\","
            "\"quote\":\"I attended the book fair at City Hall yesterday.\","
            "\"scope\":{},\"event_time\":\"yesterday\","
            "\"event_attributes\":{\"location\":\"City Hall\"},\"triple_ids\":[],\"target_fact_id\":null}。\n"
            f"可用中层实体：{json.dumps(names, ensure_ascii=False)}\n"
            f"本 Chunk 的关系：{json.dumps([dict(ref, triple_id=triple_id(ref)) for ref in refs], ensure_ascii=False)}\n"
            f"会话观察时间：{chunk.get('session_time')}\n原文：{chunk['text']}"
        )

    @staticmethod
    def _repair_prompt(chunk: dict[str, Any], names: list[str],
                       refs: list[dict[str, str]], reason: str) -> str:
        """只重试一次；重新依据原文抽取，不让模型修补不存在的引文。"""
        return (
            f"EXTRACTION_REPAIR\n上一次抽取失败：{reason}。请重新从下面原文抽取最多 4 条事实。"
            "只输出完整 JSON：{\"facts\":[...]}。每条必须包含当前原文中连续的 quote，"
            "且 quote 包含 value；做不到就不要输出该条。"
            "字段：entity,attribute,value,kind=state,fact_type=event 或 state,"
            "modality=actual/planned/hypothetical,quote,scope={},event_time=null 或原文时间词,"
            "event_attributes={},triple_ids=来源 ID 字符串数组,target_fact_id=null。"
            "不要输出提问、评论、聊天、附件动作；同次活动只输出一条。\n"
            f"实体：{json.dumps(names, ensure_ascii=False)}\n"
            f"关系：{json.dumps([dict(ref, triple_id=triple_id(ref)) for ref in refs], ensure_ascii=False)}\n"
            f"观察时间：{chunk.get('session_time')}\n原文：{chunk['text']}"
        )

    @staticmethod
    def _validate(item: dict[str, Any], chunk: dict[str, Any], names: list[str],
                  refs: list[dict[str, str]], entity_index: dict[str, int],
                  layer: EventStateLayer) -> dict[str, Any]:
        """程序只接受能定位到现有实体、原文和本次关系的模型结论。

        校验项（任一失败抛 ValueError，由调用方记入 review_queue）：
            1. entity 必须是本 Chunk 的候选实体，且在图中存在；
            2. quote 必须是本 Chunk 的连续原文子串；
            3. 引文归属可确定（见 _speaker_support）；
            4. attribute 必须是短英文 snake_case；
            5. kind 只能是 state / correction；
            6. state 必须有新值，且新值出现在引文中；
            7. correction 必须指向同一实体、同一属性的有效旧事实，
               且引文中提到被否认的旧值；
            8. triple_indices 合法、与主体相连、另一端实体也出现在引文中；
            9. event_time 必须出现在引文中，否则置空。

        返回的 proposal 是 EventStateLayer.apply 的输入，
        所有字段都已规范化（类型、截断长度、空值处理）。
        """
        name = item.get("entity")
        quote = item.get("quote")
        if not isinstance(name, str) or name not in names or name not in entity_index:
            raise ValueError("主体无法映射到现有实体索引")
        text = str(chunk["text"])
        if (not isinstance(quote, str) or not quote.strip()) and item.get("fact_type") == "event":
            # 模型偶尔漏填 quote。仅当原文中该说话人恰好有一行时，
            # 才从这一行逐字截取引文；多轮同名说话人不猜选哪一句。
            speaker_lines = [line.partition(":")[2].strip() for line in text.splitlines()
                             if line.casefold().startswith(name.casefold() + ":")]
            if len(speaker_lines) == 1:
                quote = speaker_lines[0]
        if not isinstance(quote, str) or not quote.strip() or quote not in text:
            raise ValueError("证据不是当前 Chunk 的连续原文")
        # 照片中“有人撑伞”等描述不等于说话人亲历/执行该动作。
        # 引文即使省去了 attached 前缀，只要仍位于附件说明内部也应识别。
        attachment_spans = [(match.start(), match.end()) for match in re.finditer(
            r"\(attached\b[^)]*\)", text, re.I | re.S)]
        positions = [match.start() for match in re.finditer(re.escape(quote), text)]
        if positions and all(any(start <= pos and pos + len(quote) <= end
                                 for start, end in attachment_spans) for pos in positions):
            raise IgnoreCandidate("附件说明不能直接归属说话人的现实事实")
        if not _speaker_support(text, quote, name, names):
            raise ValueError("原文无法确定状态属于该实体")
        raw_attribute = item.get("attribute")
        attribute = canonical_attribute(raw_attribute)
        # 对话中的交流动作和附件说明不属于用户现实经历；预期过滤单独计数。
        conversational = {"ask", "asks", "asked", "comment", "comments",
                          "commented", "mention", "mentions", "mentioned",
                          "talk_to", "talks_to", "talked_to", "talks_about",
                          "talked_about", "said", "responded", "replied",
                          "discussed", "question", "questions", "expressed_gratitude_to",
                          "thanks", "thanked", "congratulated", "attached", "attached_in_photo",
                          "attached_a_photo_of", "attached_photo_of",
                          "shared_a_photo_of", "shared_photo_of"}
        reported_conversation = (attribute in {"talk_to", "talked_to", "talked_about",
                                               "asked", "discussed"} and
                                 bool(re.search(r"\b(?:I|we)\s+(?:talked|spoke|asked|discussed)\b",
                                                quote, re.I)))
        conversation_family = bool(re.match(
            r"^(?:ask(?:s|ed)?|comment(?:s|ed)?|mention(?:s|ed)?|question(?:s)?|thank(?:s|ed)?)(?:_|$)",
            attribute))
        if ((attribute in conversational or conversation_family) and not reported_conversation) or attribute.startswith("attached_"):
            raise IgnoreCandidate("对话行为或附件描述不是现实事件")
        if attribute in {"related_to", "is_related_to"} and re.match(
                r"\s*(?:wow\b|thanks\b|that's\b|that is\b)", quote, re.I):
            raise IgnoreCandidate("一般性赞叹没有可持续状态或现实事件")
        modality = normalize_modality(item.get("modality"))
        if modality == "actual" and re.match(
                r"\s*(?:(?:I\s+(?:am\s+)?|I'm\s+))?"
                r"(?:gonna\b|going to\b|planning to\b|plan to\b|intend to\b|will\b)",
                quote, re.I):
            modality = "planned"
        kind = item.get("kind")
        if kind in {"baseline", "change"}:
            # 兼容模型沿用旧摘要原型的类别，真正的基线/变化由当前值决定。
            kind = "state"
        if kind == "correction" and item.get("fact_type") == "event":
            # 事件日期纠正要先匹配事件实例；不能要求模型凭空填写旧 fact_id。
            kind = "state"
        if kind not in {"state", "correction"}:
            raise ValueError("只接受持续事实或明确纠正")
        # 模型有开放属性命名空间；事件分类既接纳模型标注，也给常见活动谓词兜底。
        # 持续状态及计划不因属性词形相似就强制变成历史事件。
        event_actions = {"attended", "attended_event", "participated_in", "joined",
                         "visited", "went_to", "traveled_to", "hosted", "completed"}
        fact_type = ("event" if (kind == "state" and modality == "actual" and
                     cardinality_for(attribute) == "set" and
                     (item.get("fact_type") == "event" or attribute in event_actions))
                     else "state")
        if attribute in {"feels", "feeling", "given_courage", "expressing",
                         "related_to", "is_related_to", "interested_in",
                         "likes", "loves", "prefers", "owns", "has",
                         "member_of", "is_member_of", "is_part_of", "part_of", "belongs_to",
                         "supports", "support", "supported_by"}:
            fact_type = "state"
        if re.match(r"^(?:feel(?:s|ing)?|felt)(?:_|$)", attribute):
            fact_type = "state"
            if not re.search(r"\b(?:(?:I|we)\s+(?:feel|felt|was|were|am|are)|(?:me|us)\s+feel)\b",
                             quote, re.I):
                raise ValueError("感受状态没有主体自身感受的原文依据")
        if fact_type == "event" and re.search(
                r"\b(?:regular(?:ly)?|usually|often|every\s+(?:day|week|month|year)|weekly)\b",
                quote, re.I) and not item.get("event_time"):
            # 常态活动频率不是一次有独立身份的历史事件。
            fact_type = "state"
        raw_event_attributes = item.get("event_attributes") or {}
        if not isinstance(raw_event_attributes, dict) or len(raw_event_attributes) > 8:
            raise ValueError("事件属性必须是最多八项的对象")
        event_attributes: dict[str, str] = {}
        if fact_type == "event":
            for raw_name, raw_value in raw_event_attributes.items():
                name_key = canonical_attribute(raw_name)
                if name_key in {"event_time", "time", "date", "event", "participant",
                                "actor", "action", "target"}:
                    layer.add_review(str(chunk["chunk_id"]), "事件属性未写入：事件身份和日期不能重复写入普通属性")
                    continue
                if (not isinstance(raw_value, str) or not raw_value.strip() or
                        raw_value.casefold() not in quote.casefold()):
                    layer.add_review(str(chunk["chunk_id"]), f"事件属性 {name_key} 未写入：没有对应事件引文依据")
                    continue
                if raw_value.casefold() == str(item.get("value") or "").casefold():
                    # 把活动对象再次标成“地点”等属性没有增加信息，且会误触发纠正。
                    continue
                if name_key == "location":
                    location = raw_value.strip()
                    if (re.search(r"\b(?:today|tomorrow|yesterday|last|next|attached)\b",
                                  location, re.I) or
                            location.casefold() in {"community", "support", "attached",
                                                    "photo", "image", "unknown"}):
                        layer.add_review(str(chunk["chunk_id"]), "事件属性 location 未写入：不是地点名称")
                        continue
                    # 值必须作为当前活动引文中的地点短语出现，而不是同 Chunk
                    # 另一句话的普通名词、日期或照片描述。
                    pattern = (r"\b(?:at|in|near|inside|outside|on|from|to)\s+"
                               r"(?:the\s+)?" + re.escape(location) + r"\b")
                    if not re.search(pattern, quote, re.I):
                        layer.add_review(str(chunk["chunk_id"]), "事件属性 location 未写入：缺少同一活动的地点表达")
                        continue
                    if re.search(re.escape(location) + r"\s+[A-Z][a-z]+", quote):
                        layer.add_review(str(chunk["chunk_id"]), "事件属性 location 未写入：截断了完整地点名称")
                        continue
                event_attributes[name_key] = raw_value.strip()
        # 活动实例由 actor/action/target/time 定位，模型随意填写的 project/topic
        # 不应把同一次活动分裂成多个状态槽；事件作用域由专门字段另行组织。
        if fact_type == "event":
            scope = {}
        else:
            # 持续状态的作用域可以由本 Chunk 另一句话证明，但不可凭空创造。
            try:
                scope = normalize_scope(item.get("scope"), text)
            except ValueError as error:
                if cardinality_for(attribute) == "single" or kind == "correction":
                    raise
                scope = {}
                layer.add_review(str(chunk["chunk_id"]), f"忽略无证据的集合事实作用域：{error}")
        value = item.get("value")
        if value is not None and (not isinstance(value, str) or not value.strip() or
                                  value.casefold() not in quote.casefold()):
            raise ValueError("新状态值没有原文依据")
        if kind == "state" and value is None:
            raise ValueError("状态事实缺少新值")
        if attribute == "employer" and not re.search(
                r"\b(?:work(?:ed|ing|s)?\s+(?:at|for)|employ(?:ed|er|ment|s)?|job|hired)\b",
                quote, re.I):
            raise ValueError("雇主槽缺少就业关系的原文依据，团体成员身份不能作为雇主")
        if attribute in {"residence", "current_residence", "primary_residence"} and not re.search(
                r"\b(?:live[sd]?|living|resid\w*|home|moved?)\b", quote, re.I):
            raise ValueError("居所槽缺少居住关系的原文依据")
        if fact_type == "event" and (
                str(value).casefold() == str(item.get("event_time") or "").casefold() or
                re.fullmatch(r"(?:today|yesterday|last\s+\w+|\d{4}-\d{2}-\d{2}|(?:\w+\s+){1,3}ago)",
                             str(value), re.I)):
            raise ValueError("事件客体不能是时间表达")
        target_id = item.get("target_fact_id") if kind == "correction" else None
        if kind == "correction":
            # 纠正必须精确指向同一实体同一属性的旧事实，防止模型张冠李戴。
            target = layer.facts.get(target_id)
            if target_id and (not target or target["entity_id"] != name or
                              canonical_attribute(target["attribute"]) != attribute):
                raise ValueError("纠正没有指向同一实体属性的旧事实")
            candidate_slot = {"entity_id": name, "attribute": attribute,
                              "scope": scope, "modality": modality}
            if target and slot_key(target) != slot_key(candidate_slot):
                raise ValueError("纠正目标不属于同一作用域和模态")
        indices = resolve_indices(item, refs)
        selected_refs = []
        for index in dict.fromkeys(indices):
            ref = refs[index]
            if name not in (ref["src"], ref["tgt"]):
                raise ValueError("Triple 与主体不相连")
            other = ref["tgt"] if ref["src"] == name else ref["src"]
            if other.casefold() not in quote.casefold():
                if fact_type == "event" and re.fullmatch(r"\d{4}-\d{2}-\d{2}", other):
                    # 日期关系允许由事件实例绑定器在同一 Chunk 中单独核验。
                    continue
                raise ValueError("Triple 的另一实体未出现在本条证据中")
            selected_refs.append(ref)
        if not selected_refs:
            # 模型未列索引时，只自动挂接本 Chunk 且另一端实体出现在证据中的关系。
            for ref in refs:
                if name not in (ref["src"], ref["tgt"]):
                    continue
                other = ref["tgt"] if ref["src"] == name else ref["src"]
                if other.casefold() in quote.casefold():
                    selected_refs.append(ref)
        if fact_type == "event":
            # 中层若明确给出活动客体，事件 value 必须对应该客体；
            # 不能把 yesterday 等时间词误当参加的对象。
            actor_targets = [ref["tgt"] for ref in refs
                             if ref["src"] == name and
                             canonical_attribute(ref["relation"]) == attribute and not re.fullmatch(
                                 r"\d{4}-\d{2}-\d{2}", ref["tgt"])]
            if actor_targets and not any(
                    str(value).casefold() == target.casefold() or
                    str(value).casefold() in target.casefold() or
                    target.casefold() in str(value).casefold()
                    for target in actor_targets):
                raise ValueError("事件客体与本 Chunk 的关系对象不一致")
        event_time = item.get("event_time")
        # 事件发生时间必须能在原文中直接找到；模型推断出的时间一律丢弃。
        if not isinstance(event_time, str) or event_time not in quote:
            event_time = None
        group_quote = item.get("event_group_quote")
        group_id = None
        # 同一次原文描述多个属性变化时，用 group_id 把它们关联起来。
        if (isinstance(group_quote, str) and 0 < len(group_quote) <= 160 and
                group_quote in text and name.casefold() in group_quote.casefold() and
                value is not None and value.casefold() in group_quote.casefold()):
            group_id = sha256((str(chunk.get("session_id")) + "\x1f" +
                               str(chunk["chunk_id"]) + "\x1f" + group_quote).encode("utf-8")).hexdigest()[:24]
        event_type = item.get("event_type")
        if not isinstance(event_type, str) or not event_type.strip():
            # 没给事件类型时用“属性_change”兜底，保证事件总有可展示的类别。
            event_type = attribute + "_change"
        proposal = {
            "kind": kind, "entity_id": name, "entity_index": entity_index[name],
            "attribute": attribute, "value": value, "quote": quote,
            "source_text": text, "scope": scope, "modality": modality,
            "chunk_id": str(chunk["chunk_id"]),
            "session_id": str(chunk.get("session_id") or ""),
            "observed_at": str(chunk.get("session_time") or ""),
            "event_time": event_time, "event_group_id": group_id,
            "event_type": event_type.strip()[:80], "target_fact_id": target_id,
            "triple_refs": selected_refs,
            "fact_type": fact_type,
            "event_attributes": event_attributes,
        }
        if fact_type == "event":
            bind_event_time(proposal, refs)
        return proposal
