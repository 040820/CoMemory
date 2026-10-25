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
from .state_slot import (canonical_attribute, cardinality_for, normalize_modality,
                         normalize_scope, preliminary_operation, slot_key)
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


class EventStateWriter:
    """每个 Chunk 调用一次 LLM；没有持续事实变化时不产生上层事件。"""

    def __init__(self, llm: Any):
        self.llm = llm

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
            # 提示词均使用会话开始前的状态快照。真正的首次/变化分类仍在顺序提交时决定。
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
                # 只把“该实体的当前值 + 可纠正的历史事实 ID”喂给模型，
                # 不提供别人的状态，减少指代混淆。
                context = {name: layer.slot_context_for(name) for name in names}
                prompt = self._prompt(chunk, context, refs)
                pending.append((chunk, names, refs, prompt))

            # 同一会话内并发调用；return_exceptions=True 保证单个失败不影响其它请求。
            responses = await asyncio.gather(
                *(self.llm.generate(prompt, max_tokens=768) for _, _, _, prompt in pending),
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
                except (ValueError, IndexError, TypeError) as error:
                    # 结构无效说明模型输出无法使用；标记已处理避免反复重试同一坏输出。
                    layer.add_review(chunk_id, f"模型结构无效：{error}")
                    layer.processed_chunks.add(chunk_id)
                    continue
                # 在副本中写入本 Chunk 的所有属性，避免其中一条失败造成部分更新。
                staged = EventStateLayer(layer.to_dict())
                seen_facts: set[tuple[str, str, str, str]] = set()
                for item in items:
                    try:
                        proposal = self._validate(item, chunk, names, refs, entity_index, staged)
                        # 集合属性允许同一 Chunk 里有多个对象，只拒绝完全重复的候选。
                        key = (proposal["entity_id"], proposal["attribute"],
                               slot_key(proposal), str(proposal.get("value")))
                        if key in seen_facts:
                            continue
                        seen_facts.add(key)
                        await self._apply_candidate(staged, proposal)
                    except (KeyError, TypeError, ValueError) as error:
                        # 单条候选被拒：写入 review_queue，但该 Chunk 其它候选继续处理。
                        staged.add_review(chunk_id, f"状态候选未写入：{error}")
                staged.processed_chunks.add(chunk_id)
                # 本 Chunk 全部处理完毕才提交副本。
                layer = staged
            logger.info("事件状态层：已处理会话 %s，累计 Chunk %d/%d、事实 %d、变化 %d、待核查 %d",
                        str(group[0].get("session_id") or ""), len(layer.processed_chunks),
                        len(ordered), len(layer.facts), len(layer.events), len(layer.review_queue))
        return layer

    async def _apply_candidate(self, layer: EventStateLayer,
                               proposal: dict[str, Any]) -> None:
        """先按槽位排除并存事实；只有可能替换/纠正时才调用独立证据门控。"""
        key = slot_key(proposal)
        if proposal["kind"] == "correction":
            old_id = proposal.get("target_fact_id")
            old = layer.facts.get(old_id)
            if not old or slot_key(old) != key:
                layer.apply_decision(proposal, "UNCERTAIN", old_id)
                return
            gate = await self._judge_transition(old, proposal, "CORRECT")
            if self._gate_passes(gate, "CORRECT", proposal):
                layer.apply_decision(proposal, "CORRECT", old_id, gate)
            else:
                layer.apply_decision(proposal, "UNCERTAIN", old_id, gate)
            return

        if cardinality_for(proposal["attribute"]) == "set":
            layer.apply_decision(proposal, "ADD")
            return
        old_id = layer.current.get(key)
        if old_id is None:
            layer.apply_decision(proposal, "ADD")
            return
        old = layer.facts[old_id]
        preliminary = preliminary_operation(old, proposal)
        if preliminary in {"NOOP", "REFINE"}:
            layer.apply_decision(proposal, preliminary, old_id)
        elif preliminary == "GATE":
            gate = await self._judge_transition(old, proposal, "REPLACE")
            if self._gate_passes(gate, "REPLACE", proposal):
                layer.apply_decision(proposal, "REPLACE", old_id, gate)
            else:
                layer.apply_decision(proposal, "UNCERTAIN", old_id, gate)
        else:
            layer.apply_decision(proposal, "UNCERTAIN", old_id)

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
            f"旧事实：{json.dumps({k: old.get(k) for k in ('entity_id', 'attribute', 'value', 'scope', 'modality', 'quote')}, ensure_ascii=False)}\n"
            f"新事实：{json.dumps({k: new.get(k) for k in ('entity_id', 'attribute', 'value', 'scope', 'modality', 'quote')}, ensure_ascii=False)}\n"
            f"当前完整原文：{new['source_text']}"
        )
        try:
            response = await self.llm.generate(prompt, max_tokens=256)
            return _parse_gate_json(response)
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
                    ("same_slot", "incompatible", "supported_transition")) and
                isinstance(evidence, str) and bool(evidence.strip()) and
                evidence in proposal["source_text"])

    @staticmethod
    def _prompt(chunk: dict[str, Any], context: dict[str, Any],
                refs: list[dict[str, str]]) -> str:
        """一条 Chunk 的开放属性判定提示；只要求模型发现证据，不让它猜旧值。

        提示词约束了输出字段：
            entity/attribute/value/kind/event_type/quote/event_time/
            target_fact_id/event_group_quote/triple_indices/scope/modality。
        这里只抽取候选；是否替换由独立证据门控判断。
        """
        return (
            "你在构建长期记忆的事实候选。抽取原文能直接支持的实体属性值，包括持续状态和可并存事实；"
            "是否替换旧状态由后续独立门控判断，你不能直接决定 REPLACE。"
            "属性不设固定白名单，但同一属性应沿用已有属性名。"
            "过去的回忆不得自动当成当前状态；假设和计划必须标清模态。"
            "每条结果给出 entity（必须在可用实体中）、attribute（短英文 snake_case）、"
            "value（原文出现的状态短语；否认旧事实而未知新值时为 null）、"
            "kind（state 或明确纠正 correction）、scope（项目、地点、对象、时间范围等有原文依据的作用域对象；无则 {}）、"
            "modality（actual、planned、hypothetical 或 negated）、"
            "event_type（具体事件名称，例如 job_change）、"
            "quote（连续原文子串）、event_time（仅原文明确日期，否则 null）、"
            "target_fact_id（纠正时必填）、event_group_quote（同次多属性变化共用的短原文）、"
            "triple_indices（相关关系的从 0 开始索引；无关系可为空数组）。"
            "不要复制已有相同事实，不要猜测作用域、旧值、身份或时间。"
            "只输出 JSON 对象 {\"facts\": [...]}; 没有事实时输出 {\"facts\": []}。\n"
            f"可用实体及历史事实：{json.dumps(context, ensure_ascii=False)}\n"
            f"本 Chunk 的关系：{json.dumps(refs, ensure_ascii=False)}\n"
            f"会话观察时间：{chunk.get('session_time')}\n原文：{chunk['text']}"
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
        if not isinstance(quote, str) or not quote.strip() or quote not in text:
            raise ValueError("证据不是当前 Chunk 的连续原文")
        if not _speaker_support(text, quote, name, names):
            raise ValueError("原文无法确定状态属于该实体")
        raw_attribute = item.get("attribute")
        attribute = canonical_attribute(raw_attribute)
        scope = normalize_scope(item.get("scope"), quote)
        modality = normalize_modality(item.get("modality"))
        kind = item.get("kind")
        if kind in {"baseline", "change"}:
            # 兼容模型沿用旧摘要原型的类别，真正的基线/变化由当前值决定。
            kind = "state"
        if kind not in {"state", "correction"}:
            raise ValueError("只接受持续事实或明确纠正")
        value = item.get("value")
        if value is not None and (not isinstance(value, str) or not value.strip() or
                                  value.casefold() not in quote.casefold()):
            raise ValueError("新状态值没有原文依据")
        if kind == "state" and value is None:
            raise ValueError("状态事实缺少新值")
        target_id = item.get("target_fact_id") if kind == "correction" else None
        if kind == "correction":
            # 纠正必须精确指向同一实体同一属性的旧事实，防止模型张冠李戴。
            target = layer.facts.get(target_id)
            if not target or target["entity_id"] != name or target["attribute"] != attribute:
                raise ValueError("纠正没有指向同一实体属性的旧事实")
            candidate_slot = {"entity_id": name, "attribute": attribute,
                              "scope": scope, "modality": modality}
            if slot_key(target) != slot_key(candidate_slot):
                raise ValueError("纠正目标不属于同一作用域和模态")
        indices = item.get("triple_indices", [])
        if not isinstance(indices, list) or any(type(index) is not int or
                                                index < 0 or index >= len(refs) for index in indices):
            raise ValueError("Triple 引用越界")
        selected_refs = []
        for index in dict.fromkeys(indices):
            ref = refs[index]
            if name not in (ref["src"], ref["tgt"]):
                raise ValueError("Triple 与主体不相连")
            other = ref["tgt"] if ref["src"] == name else ref["src"]
            if other.casefold() not in quote.casefold():
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
        return {
            "kind": kind, "entity_id": name, "entity_index": entity_index[name],
            "attribute": attribute, "value": value, "quote": quote,
            "source_text": text, "scope": scope, "modality": modality,
            "chunk_id": str(chunk["chunk_id"]),
            "session_id": str(chunk.get("session_id") or ""),
            "observed_at": str(chunk.get("session_time") or ""),
            "event_time": event_time, "event_group_id": group_id,
            "event_type": event_type.strip()[:80], "target_fact_id": target_id,
            "triple_refs": selected_refs,
        }
