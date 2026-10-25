"""在记忆写入时构建事件状态上层，直接连接实体索引、关系和原话。"""

from __future__ import annotations

import json
import re
from hashlib import sha256
from typing import Any

from .event_state_layer import EventStateLayer


def _chunk_refs(chunk_id: str, relationships: list[dict[str, Any]]) -> list[dict[str, str]]:
    """只取本次抽取明确指向该 Chunk 的关系，不继承合并图的模糊边。"""
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
    """第一人称引文通过最近的说话人标签归属，重复归属不一致时拒绝。"""
    labels: list[tuple[int, str]] = []
    for candidate in names:
        pattern = rf"(?<!\w){re.escape(candidate)}\s*:"
        labels.extend((match.start(), candidate) for match in re.finditer(pattern, text, re.I))
    labels.sort()
    owners: set[str | None] = set()
    start = 0
    while (start := text.find(quote, start)) >= 0:
        # 引文不能跨越两个说话人，否则可能把后一人的状态记到前一人。
        if any(start < position < start + len(quote) for position, _ in labels):
            return False
        previous = [candidate for position, candidate in labels if position < start]
        owners.add(previous[-1] if previous else None)
        start += len(quote)
    return bool(owners) and (name.casefold() in quote.casefold() or owners == {name})


def _parse_json(response: str) -> list[dict[str, Any]]:
    clean = response.strip()
    if clean.startswith("```"):
        clean = clean.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    data = json.loads(clean)
    if isinstance(data, dict):
        data = data.get("facts")
    if not isinstance(data, list) or not all(isinstance(item, dict) for item in data):
        raise ValueError("模型没有返回 facts 数组")
    return data


class EventStateWriter:
    """每个 Chunk 调用一次 LLM；没有持续事实变化时不产生上层事件。"""

    def __init__(self, llm: Any):
        self.llm = llm

    async def write_chunks(self, layer: EventStateLayer, chunks: list[dict[str, Any]],
                           relationships: list[dict[str, Any]],
                           entity_index: dict[str, int]) -> EventStateLayer:
        """按会话观察时间顺序增量处理，异常 Chunk 不破坏已有状态。"""
        for chunk in sorted(chunks, key=lambda row: (str(row.get("session_time") or ""),
                                                      str(row.get("chunk_id") or ""))):
            chunk_id = str(chunk.get("chunk_id") or "")
            text = str(chunk.get("text") or "")
            if not chunk_id or not text or chunk_id in layer.processed_chunks:
                continue
            refs = _chunk_refs(chunk_id, relationships)
            names = [name for name in entity_index
                     if re.search(rf"(?<!\w){re.escape(name)}(?!\w)", text, re.I)]
            for ref in refs:
                for name in (ref["src"], ref["tgt"]):
                    if name in entity_index and name not in names:
                        names.append(name)
            if not names:
                layer.add_review(chunk_id, "原图中找不到可关联的实体")
                layer.processed_chunks.add(chunk_id)
                continue
            context = {name: layer.context_for(name) for name in names}
            prompt = (
                "你在构建长期记忆的事件与状态上层。只记录某实体可持续追踪的事实，"
                "例如参与关系、职位、住处、计划、健康状态；普通一次性活动、猜测、"
                "无状态意义的事件不记录。属性不设固定白名单，但同一属性应沿用已有属性名。"
                "从当前原话中找新的当前状态或对旧事实的明确纠正；过去的回忆不得覆盖当前。"
                "每条结果给出 entity（必须在可用实体中）、attribute（短英文 snake_case）、"
                "value（原文出现的状态短语；否认旧事实而未知新值时为 null）、"
                "kind（state 或 correction）、event_type（具体事件名称，例如 joined_group）、"
                "quote（连续原文子串）、event_time（仅原文明确日期，否则 null）、"
                "target_fact_id（纠正时必填）、event_group_quote（同次多属性变化共用的短原文）、"
                "triple_indices（相关关系的从 0 开始索引；无关系可为空数组）。"
                "不要复制已有相同状态，不要猜测旧值、身份或时间。"
                "只输出 JSON 对象 {\"facts\": [...]}; 没有事实时输出 {\"facts\": []}。\n"
                f"可用实体及现有属性：{json.dumps(context, ensure_ascii=False)}\n"
                f"本 Chunk 的关系：{json.dumps(refs, ensure_ascii=False)}\n"
                f"会话观察时间：{chunk.get('session_time')}\n原文：{text}"
            )
            try:
                response = await self.llm.generate(prompt)
            except Exception as error:
                # 单个 Chunk 的服务异常不应中止整张图的构建；不标记已处理以便下次重试。
                layer.add_review(chunk_id, f"LLM 调用失败，待重试：{type(error).__name__}: {error}")
                continue
            if not response:
                layer.add_review(chunk_id, "LLM 返回空内容；保留重试机会")
                continue
            try:
                items = _parse_json(response)
            except (ValueError, IndexError, TypeError) as error:
                layer.add_review(chunk_id, f"模型结构无效：{error}")
                layer.processed_chunks.add(chunk_id)
                continue
            # 在副本中写入本 Chunk 的所有属性，避免其中一条失败造成部分更新。
            staged = EventStateLayer(layer.to_dict())
            seen_slots: set[tuple[str, str]] = set()
            for item in items:
                try:
                    proposal = self._validate(item, chunk, names, refs, entity_index, staged)
                    slot = (proposal["entity_id"], proposal["attribute"])
                    if slot in seen_slots:
                        raise ValueError("同一 Chunk 对同一实体属性给出多个状态")
                    seen_slots.add(slot)
                    staged.apply(proposal)
                except (KeyError, TypeError, ValueError) as error:
                    staged.add_review(chunk_id, f"状态候选未写入：{error}")
            staged.processed_chunks.add(chunk_id)
            layer = staged
        return layer

    @staticmethod
    def _validate(item: dict[str, Any], chunk: dict[str, Any], names: list[str],
                  refs: list[dict[str, str]], entity_index: dict[str, int],
                  layer: EventStateLayer) -> dict[str, Any]:
        """程序只接受能定位到现有实体、原文和本次关系的模型结论。"""
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
        if not isinstance(raw_attribute, str):
            raise ValueError("缺少状态属性")
        attribute = re.sub(r"[^a-z0-9]+", "_", raw_attribute.casefold()).strip("_")
        if not re.fullmatch(r"[a-z][a-z0-9_]{1,63}", attribute):
            raise ValueError("状态属性必须是短英文标识")
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
            target = layer.facts.get(target_id)
            if not target or target["entity_id"] != name or target["attribute"] != attribute:
                raise ValueError("纠正没有指向同一实体属性的旧事实")
            if str(target["value"]).casefold() not in quote.casefold():
                raise ValueError("纠正原文没有指明被否认的旧值")
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
        if not isinstance(event_time, str) or event_time not in quote:
            event_time = None
        group_quote = item.get("event_group_quote")
        group_id = None
        if (isinstance(group_quote, str) and 0 < len(group_quote) <= 160 and
                group_quote in text and name.casefold() in group_quote.casefold() and
                value is not None and value.casefold() in group_quote.casefold()):
            group_id = sha256((str(chunk.get("session_id")) + "\x1f" +
                               str(chunk["chunk_id"]) + "\x1f" + group_quote).encode("utf-8")).hexdigest()[:24]
        event_type = item.get("event_type")
        if not isinstance(event_type, str) or not event_type.strip():
            event_type = attribute + "_change"
        return {
            "kind": kind, "entity_id": name, "entity_index": entity_index[name],
            "attribute": attribute, "value": value, "quote": quote,
            "chunk_id": str(chunk["chunk_id"]),
            "session_id": str(chunk.get("session_id") or ""),
            "observed_at": str(chunk.get("session_time") or ""),
            "event_time": event_time, "event_group_id": group_id,
            "event_type": event_type.strip()[:80], "target_fact_id": target_id,
            "triple_refs": selected_refs,
        }
