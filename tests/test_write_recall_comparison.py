"""独立召回实验的编排检查：固定历史、阶段去重与非 gold 统计口径。

使用确定性伪向量只检查实验隔离和产物协议，不据此评价 BGE-M3 检索质量。
真实模型和 sample0 效果由服务器实验单独产生。
"""

import asyncio
import json
from copy import deepcopy

from coregraph.event_state_layer import EventStateLayer
from coregraph.hybrid_state_retriever import HybridStateRetriever
from experiments.integrated_event_state.compare_write_recall import (
    ComparisonRetriever, METHODS, append_json, final_queries, summarize)


class FakeEmbeddings:
    async def get_embeddings(self, texts):
        return [[1.0, (sum(map(ord, text)) % 17 + 1) / 17] for text in texts]


def proposal(chunk_id, value):
    quote = f'I work for {value}.'
    return {'entity_id': 'User', 'entity_index': 0, 'attribute': 'employer',
            'value': value, 'scope': {}, 'modality': 'actual', 'kind': 'state',
            'fact_type': 'state', 'event_type': 'employment', 'chunk_id': str(chunk_id),
            'session_id': f'D{chunk_id}', 'observed_at': '2023-05-08',
            'source_text': quote, 'quote': quote,
            'triple_refs': [{'src': 'User', 'relation': 'works_for',
                             'tgt': value, 'chunk_id': str(chunk_id)}]}


def test_shadow_recall_preserves_history_and_phase_identity(tmp_path):
    """真实四路接口共用不可变历史；归一前后不计成两个新事实样本。"""
    layer = EventStateLayer()
    layer.write_policy = 'append_only'
    layer.writer_schema_version = 4
    old_id = layer.append_observation(proposal(1, 'Company A'))
    before = deepcopy(layer.to_dict())
    embeddings = FakeEmbeddings()
    recorder = ComparisonRetriever(HybridStateRetriever(embeddings), embeddings,
                                    tmp_path / 'write_retrieval.jsonl')
    new = proposal(2, 'Company B')

    async def exercise():
        first = await recorder.search(layer, new)
        await recorder.search(layer, dict(new, source_attribute='employer',
                                           source_value='Company B'), first['semantic_result'])
    asyncio.run(exercise())
    assert layer.to_dict() == before
    rows = [json.loads(line) for line in recorder.path.read_text(encoding='utf-8').splitlines()]
    assert [row['phase'] for row in rows] == ['bootstrap', 'normalized']
    assert rows[0]['query_id'] == rows[1]['query_id']
    assert len(final_queries(recorder.path)) == 1
    for name in METHODS:
        assert rows[-1]['methods'][name]['hits'][0]['fact']['fact_id'] == old_id


def test_report_does_not_call_pooled_labels_gold(tmp_path):
    """只给一个方法相关候选、另一个空候选，验证辅助池覆盖率与空结果分开。"""
    new, old = proposal(2, 'Company B'), dict(proposal(1, 'Company A'), fact_id='F1')
    append_json(tmp_path / 'write_retrieval.jsonl', {
        'query_id': 'Q1', 'phase': 'normalized', 'new_fact': new,
        'subject_history_count': 1, 'methods': {name: {'hits': (
            [{'fact': old, 'rank': 1, 'score': 0.5}] if name == 'hybrid' else [])}
            for name in METHODS}})
    append_json(tmp_path / 'pooled_relation_audit.jsonl', {
        'query_id': 'Q1', 'old_fact_id': 'F1', 'judgment': {'relation': 'conflict'}})
    metrics = summarize(tmp_path)
    assert metrics['human_gold_available'] is False
    assert metrics['methods']['hybrid']['pooled_llm_related_coverage'] == 1
    assert metrics['methods']['structured']['empty_rate'] == 1
    assert metrics['methods']['structured']['pooled_llm_related_coverage'] == 0
