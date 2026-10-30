"""以新事实的三元组表示查询已连接中层 Triple 的旧事实证据。

这是 legacy 召回路径：先用主体/实体索引缩小范围，再用“语义相似度 + 槽位/
属性/关系重叠”的加法分数选少量旧事实。它的分数是启发式排序分，不是证据
置信度；新的置信度模型在 `evidence_confidence.py` 中独立计算。
"""

from __future__ import annotations

from math import sqrt
from typing import Any

from .state_slot import slot_key, comparison_value


def _triple_text(refs: list[dict[str, str]]) -> str:
    """保留主体、关系和客体；不用只有实体名的宽泛历史上下文。"""
    return "; ".join(
        f"({ref.get('src', '')}, {ref.get('relation', '')}, {ref.get('tgt', '')})"
        for ref in refs[:8]
    )


def _semantic_text(fact: dict[str, Any]) -> str:
    """供 BGE-M3 匹配的短文本，同时包含结构关系与原文片段。

    文本里同时放原始字段和 comparison_value，是为了让“归一化后的语义值”
    与“原文表达”都能参与相似度；引文截断到 240 字符只是编码预算。
    """
    return (f"主体：{fact['entity_id']}；属性：{fact['attribute']}；"
            f"原始值：{fact.get('value') or ''}；语义值：{comparison_value(fact)}；"
            f"作用域：{fact.get('scope') or {}}；原始属性：{fact.get('source_attribute') or fact['attribute']}；"
            f"三元组：{_triple_text(fact.get('triple_refs') or [])}；"
            f"原文：{str(fact.get('quote') or '')[:240]}")


def _unit(vector: list[float]) -> list[float]:
    """L2 归一化；零向量返回空列表，明确表示“不可用”，不伪造相似度。"""
    norm = sqrt(sum(float(value) ** 2 for value in vector))
    return [float(value) / norm for value in vector] if norm else []


class StateCandidateRetriever:
    """先以中层实体索引约束主体，再用三元组语义只选少量旧证据。"""

    def __init__(self, embedding_manager: Any | None, top_k: int = 5):
        """注入 Embedding 管理器；为 None 时退化为结构/属性排序。"""
        if top_k < 1:
            raise ValueError("top_k 必须为正数")
        self.embeddings = embedding_manager
        self.top_k = top_k
        # 缓存只在当前进程内使用；图和事实数据本身不被修改。
        self._fact_vectors: dict[str, list[float]] = {}

    async def search(self, layer: Any, proposal: dict[str, Any]) -> list[dict[str, Any]]:
        """返回 Top-k 旧事实；每条都有事实 ID、相似度及中层 Triple 引用。"""
        # 先按主体/实体索引过滤，避免把全库事实都做一次向量比较。
        # observed 是“有原文支持但现实有效性待推导”的断言，可以参与候选召回；
        # 事件候选只和事件事实比较，状态候选只和状态事实比较。
        candidates = [fact for fact in layer.facts.values()
                      if fact.get("status") in {"active", "superseded", "refined", "observed"}
                      and (proposal.get('fact_type') == 'event' or fact.get('fact_type') != 'event')
                      and fact.get("entity_id") == proposal["entity_id"]
                      and fact.get("entity_index") == proposal["entity_index"]
                      and fact.get("fact_id") != proposal.get("fact_id")]
        if not candidates:
            return []

        # 只为尚未缓存的历史事实编码；查询本身每次都编码一次。
        query_text = _semantic_text(proposal)
        missing = [fact for fact in candidates
                   if fact["fact_id"] not in self._fact_vectors]
        query_vector: list[float] = []
        if self.embeddings is not None:
            texts = [query_text] + [_semantic_text(fact) for fact in missing]
            vectors = await self.embeddings.get_embeddings(texts)
            # 数量不一致时不做部分对齐，避免把向量错配到事实；语义分保持 0。
            if len(vectors) == len(texts):
                query_vector = _unit(vectors[0])
                for fact, vector in zip(missing, vectors[1:]):
                    self._fact_vectors[fact["fact_id"]] = _unit(vector)

        # 精确槽位优先于泛化的属性/关系重叠；这些是排序启发式，不是置信度。
        exact_key = slot_key(proposal)
        query_relations = {str(ref.get("relation") or "").casefold()
                           for ref in proposal.get("triple_refs") or []}
        hits: list[dict[str, Any]] = []
        for fact in candidates:
            vector = self._fact_vectors.get(fact["fact_id"], [])
            semantic = sum(a * b for a, b in zip(query_vector, vector)) if (
                query_vector and vector) else 0.0
            same_slot = slot_key(fact) == exact_key
            same_attribute = fact["attribute"] == proposal["attribute"]
            old_relations = {str(ref.get("relation") or "").casefold()
                             for ref in fact.get("triple_refs") or []}
            relation_overlap = bool(query_relations & old_relations)
            # 权重是手工设定的排序偏好：同槽 +2.0，同属性 +0.3，关系重叠 +0.1。
            # 语义相似度保留原值；这里不做概率解释。
            score = (semantic + (2.0 if same_slot else 0.0) +
                     (0.3 if same_attribute else 0.0) +
                     (0.1 if relation_overlap else 0.0))
            hits.append({"fact": fact, "score": score,
                         "source_triples": fact.get("triple_refs") or [],
                         "same_slot": same_slot})
        # 分数相同按写入位置（越新越靠前）再按事实 ID 稳定排序。
        hits.sort(key=lambda hit: (-hit["score"],
                                   -hit["fact"].get("position", 0),
                                   hit["fact"]["fact_id"]))
        return hits[:self.top_k]
