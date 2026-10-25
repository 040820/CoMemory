"""基于已有 LiCoMemory 图的查询驱动结构化语义检索。

本模块只读取现有图和 Chunk 原文，不构建记忆、不调用大模型，也不修改原有检索流程。
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Mapping
import re

import numpy as np


@dataclass(frozen=True)
class RetrievalConfig:
    """检索参数。

    基础分由 Chunk 语义、最相关三元组、实体关系词面匹配构成；
    图扩展分在选出种子 Chunk 后加入。权重是实验初值，可独立调整。"""

    # top_k 为最终证据数；seed_k 为图扩展种子数；批大小限制单次嵌入请求；原文长度限制只影响嵌入输入。
    top_k: int = 15
    seed_k: int = 20
    embedding_batch_size: int = 32
    semantic_weight: float = 0.65
    fact_weight: float = 0.20
    lexical_weight: float = 0.10
    graph_weight: float = 0.05
    max_context_characters: int = 4000

    def __post_init__(self) -> None:
        """检查数量与权重，避免空批次或所有候选分数恒为零。"""
        if min(self.top_k, self.seed_k, self.embedding_batch_size, self.max_context_characters) < 1:
            raise ValueError("top_k, seed_k, embedding_batch_size and max_context_characters must be positive")
        weights = (self.semantic_weight, self.fact_weight, self.lexical_weight, self.graph_weight)
        if any(weight < 0 for weight in weights) or sum(weights) <= 0:
            raise ValueError("retrieval weights must be nonnegative and at least one must be positive")


def _ids(data: Mapping[str, Any], singular: str, plural: str) -> set[str]:
    """兼容初次构图的单个 ID 和增量合并后的 ID 列表，并统一为字符串。"""
    values = [data.get(singular), data.get(plural)]
    found: set[str] = set()
    for value in values:
        if isinstance(value, (list, tuple, set)):
            found.update(str(item) for item in value if item is not None and str(item).strip())
        elif value is not None and str(value).strip():
            found.add(str(value))
    return found


def _match_spans(query: str, phrase: str) -> list[tuple[int, int]]:
    """返回名称在原问题中的位置；英文按词边界匹配，中文允许连续文本匹配。"""
    if not phrase or not phrase.strip():
        return []
    pattern = re.escape(phrase)
    if not any("\u4e00" <= char <= "\u9fff" for char in phrase):
        pattern = rf"(?<!\w){pattern}(?!\w)"
    return [match.span() for match in re.finditer(pattern, query, flags=re.IGNORECASE)]


def _normalize(vector: Any) -> np.ndarray:
    """将嵌入转为一维单位向量，使后续点积等于余弦相似度。"""
    result = np.asarray(vector, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(result))
    if not np.isfinite(norm) or norm == 0:
        raise ValueError("Embedding backend returned a zero or non-finite vector")
    return result / norm


class StructuralSemanticRetriever:
    """从已有 CogniGraph 检索原始 Chunk。

    dynamic_memory 必须已有 graph_builder.graph 与 chunk_storage；
    embedding_manager 必须提供异步 get_embeddings(texts) 接口。
    类只保存进程内向量缓存，不会改变图、原文或原有检索器。"""

    def __init__(self, dynamic_memory: Any, embedding_manager: Any,
                 config: RetrievalConfig | None = None) -> None:
        self.memory = dynamic_memory
        self.embeddings = embedding_manager
        self.config = config or RetrievalConfig()
        # 仅在当前进程缓存文本向量，不写索引文件；文本改变后会重新计算。
        self._embedding_cache: dict[str, np.ndarray] = {}

    def clear_cache(self) -> None:
        """更换嵌入模型后清空向量缓存，避免新旧模型混用。"""
        self._embedding_cache.clear()

    def _collect(self) -> tuple[dict[str, dict[str, Any]], dict[str, set[str]], set[str], set[str]]:
        """只读收集本次检索所需的临时映射。

        返回值依次为：以 Chunk ID 为键的原文与三元组、实体到 Chunk ID 集合、
        图中实体名称集合、图中关系名称集合。缺少原文的 Chunk 不会返回。"""
        graph = self.memory.graph_builder.graph
        storage = self.memory.chunk_storage
        if graph is None or storage is None:
            raise ValueError("A loaded CogniGraph and chunk_storage are required")

        chunks: dict[str, dict[str, Any]] = {}
        entity_chunks: dict[str, set[str]] = defaultdict(set)
        entity_names: set[str] = set()
        relation_names: set[str] = set()

        # 以存在原文的 chunk_storage 记录为候选全集。
        for key, raw in storage.items():
            if not isinstance(raw, Mapping) or not raw.get("text"):
                continue
            chunk_id = str(key)
            chunks[chunk_id] = {"raw": raw, "triples": [], "entities": set()}

        for node in graph.nodes:
            entity_names.add(str(node))

        # 按照边的 chunk_id 与 chunk_ids，将同一 Chunk 的多条三元组聚合。
        for src, tgt, edge in graph.edges(data=True):
            if not isinstance(edge, Mapping):
                continue
            relation = str(edge.get("relation_name") or "").strip()
            if not relation:
                continue
            relation_names.add(relation)
            src_name, tgt_name = str(src), str(tgt)
            fact = {
                "subject": src_name,
                "predicate": relation,
                "object": tgt_name,
                "description": str(edge.get("description") or ""),
                "session_time": str(edge.get("session_time") or ""),
            }
            for chunk_id in _ids(edge, "chunk_id", "chunk_ids"):
                if chunk_id not in chunks:
                    continue
                chunks[chunk_id]["triples"].append(fact)
                chunks[chunk_id]["entities"].update((src_name, tgt_name))
                entity_chunks[src_name].add(chunk_id)
                entity_chunks[tgt_name].add(chunk_id)

        # 节点来源仅辅助图扩展，不会据此虚构三元组。
        for node, attrs in graph.nodes(data=True):
            if not isinstance(attrs, Mapping):
                continue
            for chunk_id in _ids(attrs, "chunk_id", "chunk_ids"):
                if chunk_id in chunks:
                    entity_chunks[str(node)].add(chunk_id)

        return chunks, entity_chunks, entity_names, relation_names

    @staticmethod
    def _analyze(query: str, entities: set[str], relations: set[str]) -> dict[str, Any]:
        """利用图中名称做无 LLM 解析；target 为移除已匹配实体后的剩余问题，排序仍使用完整 Query。"""
        # 优先保留较长名称，避免同一位置的短名称被重复计入实体匹配分数。
        def select(names: set[str]) -> tuple[list[str], list[tuple[int, int]]]:
            found: list[str] = []
            occupied: list[tuple[int, int]] = []
            for name in sorted(names, key=lambda item: (-len(item), item)):
                spans = [span for span in _match_spans(query, name)
                         if all(span[1] <= start or span[0] >= end for start, end in occupied)]
                if spans:
                    found.append(name)
                    occupied.extend(spans)
            return found, occupied

        found_entities, entity_spans = select(entities)
        found_relations, _ = select(relations)
        # 按原问题位置移除实体，避免重复替换时意外删掉查询目标的词。
        target = "".join(" " if any(start <= index < end for start, end in entity_spans) else char
                         for index, char in enumerate(query))
        target = re.sub(r"\s+", " ", target).strip(" ?\uFF1F,\uFF0C.\u3002") or query
        return {"entities": found_entities, "relations": found_relations, "target": target}

    @staticmethod
    def _fact_text(fact: Mapping[str, str]) -> str:
        """拼接主语、关系、宾语和已有描述、时间，供单条关系嵌入。"""
        text = f"{fact['subject']} | {fact['predicate']} | {fact['object']}"
        if fact.get("description"):
            text += f" | {fact['description']}"
        if fact.get("session_time"):
            text += f" | {fact['session_time']}"
        return text

    def _unit_text(self, chunk: dict[str, Any]) -> str:
        """将同一 Chunk 的全部去重关系与原始上下文组成语义单元。"""
        facts = list(dict.fromkeys(self._fact_text(fact) for fact in chunk["triples"]))
        # 保留全部关系文本，只截取参与嵌入的原文；最终证据返回完整原文。
        return "\n".join(["Relationships:", *facts, "Original context:",
                          str(chunk["raw"]["text"])[:self.config.max_context_characters]])

    async def _vectors(self, texts: list[str]) -> list[np.ndarray]:
        """按批次计算缺失向量并复用缓存；向量数量与输入不一致时报错。"""
        missing = list(dict.fromkeys(text for text in texts if text not in self._embedding_cache))
        size = self.config.embedding_batch_size
        for offset in range(0, len(missing), size):
            batch = missing[offset:offset + size]
            vectors = await self.embeddings.get_embeddings(batch)
            if vectors is None or len(vectors) != len(batch):
                raise RuntimeError("Embedding backend did not return one vector per input text")
            for text, vector in zip(batch, vectors):
                self._embedding_cache[text] = _normalize(vector)
        return [self._embedding_cache[text] for text in texts]

    @staticmethod
    def _cosine(query_vector: np.ndarray, vector: np.ndarray) -> float:
        """单位向量点积即余弦相似度；负相关得分截为零。"""
        if query_vector.shape != vector.shape:
            raise ValueError("Query and memory embeddings have different dimensions")
        return max(0.0, float(np.dot(query_vector, vector)))

    def _graph_support(self, seeds: list[str], chunks: dict[str, dict[str, Any]],
                       entity_chunks: dict[str, set[str]],
                       query_entities: set[str]) -> dict[str, float]:
        """从高分 Chunk 出发做有界图扩展。

        先找共享实体的其他 Chunk，再找图上与该实体相邻的实体所关联的 Chunk。
        这是一次邻接扩展，不迭代传播；每个候选仅记录最大支持值。"""
        graph = self.memory.graph_builder.graph
        support: dict[str, float] = defaultdict(float)
        for seed_id in seeds:
            for entity in chunks[seed_id]["entities"]:
                if entity not in graph:
                    continue
                # 显式命中查询实体时权重更高；共享实体高于相邻实体。
                weight = 1.0 if entity in query_entities else 0.5
                for chunk_id in entity_chunks.get(entity, ()):
                    if chunk_id != seed_id:
                        support[chunk_id] = max(support[chunk_id], weight)
                neighbors = set(graph.predecessors(entity)) | set(graph.successors(entity))
                for neighbor in neighbors:
                    for chunk_id in entity_chunks.get(str(neighbor), ()):
                        if chunk_id != seed_id:
                            support[chunk_id] = max(support[chunk_id], 0.25 * weight)
        return support

    async def retrieve(self, query: str, top_k: int | None = None) -> dict[str, Any]:
        """返回排序后的原始 Chunk 证据。

        输出包含 query、规则解析结果 query_analysis、证据列表 evidence。
        每条证据带 Chunk ID、Session 元数据、完整原文、图中三元组、总分和分项分。
        Session 元数据只用于追溯来源，不作为候选筛选门槛。"""
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a nonempty string")
        limit = self.config.top_k if top_k is None else top_k
        if limit < 1:
            raise ValueError("top_k must be positive")

        chunks, entity_chunks, entity_names, relation_names = self._collect()
        analysis = self._analyze(query, entity_names, relation_names)
        if not chunks:
            return {"query": query, "query_analysis": analysis, "evidence": []}

        # 分别嵌入完整 Query、各 Chunk 语义单元和去重的单条三元组。
        chunk_ids = sorted(chunks)
        unit_texts = [self._unit_text(chunks[chunk_id]) for chunk_id in chunk_ids]
        fact_texts = list(dict.fromkeys(self._fact_text(fact)
                         for chunk_id in chunk_ids for fact in chunks[chunk_id]["triples"]))
        vectors = await self._vectors([query, *unit_texts, *fact_texts])
        query_vector = vectors[0]
        unit_vectors = vectors[1:1 + len(unit_texts)]
        fact_vectors = dict(zip(fact_texts, vectors[1 + len(unit_texts):]))

        # 基础分为语义单元相似度、最佳三元组相似度、实体关系匹配度的加权和。
        initial: dict[str, dict[str, float]] = {}
        query_entities = set(analysis["entities"])
        query_relations = set(analysis["relations"])
        for chunk_id, vector in zip(chunk_ids, unit_vectors):
            facts = chunks[chunk_id]["triples"]
            semantic = self._cosine(query_vector, vector)
            fact_score = max((self._cosine(query_vector, fact_vectors[self._fact_text(fact)])
                              for fact in facts), default=0.0)
            entity_score = len(query_entities & chunks[chunk_id]["entities"]) / max(1, len(query_entities))
            relation_score = (len(query_relations & {fact["predicate"] for fact in facts})
                              / max(1, len(query_relations)))
            lexical = 0.7 * entity_score + 0.3 * relation_score
            base = (self.config.semantic_weight * semantic +
                    self.config.fact_weight * fact_score +
                    self.config.lexical_weight * lexical)
            initial[chunk_id] = {"semantic": semantic, "fact": fact_score,
                                 "lexical": lexical, "base": base}

        # 先选基础分最高的种子，再加入图扩展分；同分按 Chunk ID 稳定排序。
        seeds = sorted(chunk_ids, key=lambda item: (-initial[item]["base"], item))[:self.config.seed_k]
        support = self._graph_support(seeds, chunks, entity_chunks, query_entities)
        ranked = sorted(chunk_ids, key=lambda item: (-(initial[item]["base"] +
                            self.config.graph_weight * support.get(item, 0.0)), item))

        # 每个 Chunk 仅输出一次，并附原文、三元组和各项得分，便于追溯。
        evidence = []
        for chunk_id in ranked[:limit]:
            chunk = chunks[chunk_id]
            raw = chunk["raw"]
            scores = initial[chunk_id]
            evidence.append({
                "chunk_id": chunk_id,
                "session_id": raw.get("session_id"),
                "session_time": raw.get("session_time"),
                "text": raw["text"],
                "triples": chunk["triples"],
                "score": scores["base"] + self.config.graph_weight * support.get(chunk_id, 0.0),
                "scores": {"semantic": scores["semantic"], "query_to_triple": scores["fact"],
                           "entity_relation": scores["lexical"],
                           "graph": support.get(chunk_id, 0.0)},
            })
        return {"query": query, "query_analysis": analysis, "evidence": evidence}
