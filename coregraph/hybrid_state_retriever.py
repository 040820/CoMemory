"""Event–State 写入模块：StateSlot 精确召回与原始事实语义召回。

只返回候选与路径审计信息，不修改事实或决定替换。路径 A 使用上层历史槽位
索引；路径 B 使用 BGE-M3（由项目 EmbeddingManager 提供）比较原始关系和
引文。二者独立取候选，归一化后仅重算 A，避免错误槽位同时污染两条路径。
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

from .state_candidate_retriever import _unit
from .state_slot import slot_key


def subject_key(fact: dict[str, Any]) -> str:
    """服务于历史事实索引：同时限定实体名称与中层实体 ID，避免同名串联。"""
    return json.dumps([fact.get('entity_id'), fact.get('entity_index')],
                      ensure_ascii=False, separators=(',', ':'))


def raw_semantic_text(fact: dict[str, Any]) -> str:
    """服务于路径 B：只用原始表达，不加入归一化维度、值或槽位加分。

    历史 Fact 保留 source_* 时优先使用；新候选尚未归一化则回退到抽取字段。
    限制引文和 Triple 数量是 Embedding 输入预算，不截断实际存储证据。
    """
    refs = fact.get('triple_refs') or []
    triples = '; '.join(f"({r.get('src')}, {r.get('relation')}, {r.get('tgt')})"
                        for r in refs[:8])
    return json.dumps({
        'subject': fact.get('entity_id'),
        'predicate': fact.get('source_attribute', fact.get('attribute')),
        'object': fact.get('source_value', fact.get('value')),
        'scope': fact.get('source_scope', fact.get('scope')),
        'modality': fact.get('source_modality', fact.get('modality')),
        'triples': triples, 'quote': str(fact.get('quote') or '')[:600],
    }, ensure_ascii=False, sort_keys=True)


class HybridStateRetriever:
    """写入历史召回器；模式可独立启用 A/B，便于之后开展召回消融实验。"""

    def __init__(self, embeddings: Any, top_k: int = 5,
                 structured_k: int = 5, semantic_k: int = 5, mode: str = 'hybrid'):
        """各路先取各自 Top-k，再融合到 top_k；双路至少保留两个最终席位。"""
        if mode not in {'hybrid', 'structured', 'semantic'}:
            raise ValueError('不支持的状态召回模式')
        if min(top_k, structured_k, semantic_k) < 1 or (mode == 'hybrid' and top_k < 2):
            raise ValueError('召回预算必须为正数，双路最终预算至少为 2')
        self.embeddings, self.top_k = embeddings, top_k
        self.structured_k, self.semantic_k, self.mode = structured_k, semantic_k, mode
        # 仅在当前进程缓存原始文本与向量；文本改变会重新嵌入，不写回原图文件。
        self._vectors: dict[str, tuple[str, list[float]]] = {}

    @staticmethod
    def _eligible(fact: dict[str, Any], proposal: dict[str, Any]) -> bool:
        """服务于候选边界：状态匹配不混入活动，也不把已纠正断言当有效旧状态。

        全历史索引仍保留 corrected/audit 以供追溯，召回比较只使用可比较版本。
        """
        return (fact.get('status') in {'active', 'superseded', 'refined'}
                and fact.get('fact_type') != 'event'
                and subject_key(fact) == subject_key(proposal)
                and fact.get('fact_id') != proposal.get('fact_id'))

    async def _structured(self, layer: Any, proposal: dict[str, Any]) -> list[dict[str, Any]]:
        """路径 A：精确槽位含作用域/模态；不要求状态取值与新值相同。

        优先显式纠正目标、当前值，再按观察时间及写入位置排序历史。时间只定位
        候选，不能据此推断替换；实际时间相容性仍由 Writer 的门控负责。
        """
        key = slot_key(proposal)
        facts = [layer.facts[fid] for fid in layer.facts_by_slot.get(key, [])
                 if fid in layer.facts and self._eligible(layer.facts[fid], proposal)]
        facts.sort(key=lambda fact: (
            fact['fact_id'] == proposal.get('target_fact_id'),
            fact['fact_id'] == layer.current.get(key),
            str(fact.get('observed_at') or '') <= str(proposal.get('observed_at') or ''),
            str(fact.get('observed_at') or ''), fact.get('position', 0), fact['fact_id']),
            reverse=True)
        return [{'fact': fact} for fact in facts[:self.structured_k]]

    async def _semantic(self, layer: Any, proposal: dict[str, Any]) -> tuple[list, str]:
        """路径 B：同主体下的原始 Triple/引文余弦相似度，不依赖 StateSlot。

        Embedding 失败时明确记录降级；禁止用零向量或槽位分数伪装语义结果。
        仅缺失向量分批编码，防止历史积累后单次请求无限增长。
        """
        if self.embeddings is None:
            return [], 'unavailable:no_embedding_manager'
        facts = [layer.facts[fid] for fid in layer.facts_by_subject.get(subject_key(proposal), [])
                 if fid in layer.facts and self._eligible(layer.facts[fid], proposal)]
        if not facts:
            return [], 'empty_history'
        try:
            query = await self.embeddings.get_embeddings([raw_semantic_text(proposal)])
            vector = _unit(query[0])
            if not vector:
                raise ValueError('查询向量为空')
            missing = [f for f in facts if self._vectors.get(f['fact_id'], (None,))[0]
                       != raw_semantic_text(f)]
            for start in range(0, len(missing), 64):
                batch = missing[start:start + 64]
                texts = [raw_semantic_text(f) for f in batch]
                vectors = await self.embeddings.get_embeddings(texts)
                if len(vectors) != len(texts):
                    raise ValueError('Embedding 数量与文本数量不一致')
                for fact, source, values in zip(batch, texts, vectors):
                    unit = _unit(values)
                    if len(unit) != len(vector):
                        raise ValueError('历史向量维度不一致或为空')
                    self._vectors[fact['fact_id']] = (source, unit)
            hits = []
            for fact in facts:
                old_vector = self._vectors[fact['fact_id']][1]
                if len(old_vector) != len(vector):
                    raise ValueError('缓存向量维度与查询不一致')
                hits.append({'fact': fact, 'semantic_score': sum(
                    a * b for a, b in zip(vector, old_vector))})
            hits.sort(key=lambda h: (-h['semantic_score'], h['fact']['fact_id']))
            return hits[:self.semantic_k], 'ok'
        except asyncio.CancelledError:
            raise
        except Exception as error:
            return [], f'unavailable:{type(error).__name__}:{error}'

    async def search(self, layer: Any, proposal: dict[str, Any],
                     semantic_result: tuple[list, str] | None = None) -> dict[str, Any]:
        """服务于 Writer：返回 hits、trace 及可复用的原始语义路径结果。

        RRF 只融合各路排名，不把余弦值与精确匹配分数直接相加，也不解释为
        证据置信度。归一化后调用方复用 semantic_result，保证 B 与归一化独立。
        """
        async def route_a():
            return await self._structured(layer, proposal) if self.mode != 'semantic' else []

        async def route_b():
            if self.mode == 'structured':
                return [], 'disabled'
            return semantic_result if semantic_result is not None else await self._semantic(layer, proposal)

        a, b_result = await asyncio.gather(route_a(), route_b())
        b, status = b_result
        merged: dict[str, dict[str, Any]] = {}
        for route, hits in (('A', a), ('B', b)):
            for rank, hit in enumerate(hits, 1):
                fact = hit['fact']
                item = merged.setdefault(fact['fact_id'], {
                    'fact': fact, 'score': 0.0, 'routes': [], 'route_ranks': {},
                    'same_slot': slot_key(fact) == slot_key(proposal),
                    'source_triples': fact.get('triple_refs') or []})
                item['routes'].append(route)
                item['route_ranks'][route] = rank
                item['score'] += 1.0 / (60 + rank)
                if 'semantic_score' in hit:
                    item['semantic_score'] = hit['semantic_score']
        ranked = sorted(merged, key=lambda fid: (-merged[fid]['score'], fid))
        # 服务于融合配额：先保护两路首位；A 的首位已优先当前状态或纠正目标。
        # 这样高频语义邻居不会挤走精确旧值，精确路径也不会吞掉所有语义席位。
        protected = list(dict.fromkeys(hits[0]['fact']['fact_id'] for hits in (a, b) if hits))
        # 当前状态可能排在显式历史纠正目标之后；预算允许时也保护当前值，
        # 避免只召回历史而漏掉仍在生效的互斥状态。
        current_id = layer.current.get(slot_key(proposal))
        if current_id in merged and current_id not in protected:
            protected.append(current_id)
        selected = (protected + [fid for fid in ranked if fid not in protected])[:self.top_k]
        trace = {
            'mode': self.mode, 'slot_key': slot_key(proposal),
            'structured_ids': [h['fact']['fact_id'] for h in a],
            'semantic_ids': [h['fact']['fact_id'] for h in b],
            'semantic_scores': {h['fact']['fact_id']: h['semantic_score'] for h in b},
            'semantic_status': status, 'union_ids': ranked, 'protected_ids': protected,
            'selected_ids': selected, 'dropped_ids': [fid for fid in ranked if fid not in selected],
            'protected_dropped_ids': [fid for fid in protected if fid not in selected],
            'budgets': {'A': self.structured_k, 'B': self.semantic_k, 'final': self.top_k},
            'score_type': 'reciprocal_rank_fusion_not_confidence',
        }
        return {'hits': [merged[fid] for fid in selected], 'trace': trace,
                'semantic_result': b_result}
