"""证据支撑模块的固定输入回归：检验检索边界，不代表真实数据集效果。"""

import importlib
import json
from copy import deepcopy
from pathlib import Path
import sys
import types
import unittest


# 服务于轻量测试：执行真实模块，避免 coregraph.__init__ 提前加载 GPU 依赖。
PACKAGE = '_evidence_support_test_core'
package = types.ModuleType(PACKAGE)
package.__path__ = [str(Path(__file__).resolve().parents[1] / 'coregraph')]
sys.modules[PACKAGE] = package
api = importlib.import_module(PACKAGE + '.evidence_support')


def graph():
    """独立事实对：A/B 工作说法、兼职说明、反证和一个未来来源。"""
    texts = {'C1': 'I work at A.', 'C2': 'I work at B.',
             'C3': 'A is full time; B is part time.', 'C4': 'B was only a plan.',
             'C5': 'Future evidence.', 'C6': 'Another related detail.'}
    facts = {
        'F1': dict(entity_id='User', attribute='employer', value='A', chunk_id='C1'),
        'F2': dict(entity_id='User', attribute='employer', value='B', chunk_id='C2'),
        'F3': dict(entity_id='User', attribute='employer', value='A and B', chunk_id='C3'),
        'F4': dict(entity_id='Manager', attribute='clarifies', value='B was planned', chunk_id='C4'),
        'F5': dict(entity_id='User', attribute='employer', value='Future', chunk_id='C5'),
        'F6': dict(entity_id='Other', attribute='detail', value='detail', chunk_id='C6')}
    for fid, row in facts.items():
        row.update(fact_id=fid, quote=texts[row['chunk_id']], modality='actual', scope={})
    return dict(chunk_storage={cid: dict(chunk_id=cid, session_id='D1', text=text,
        session_time='2024-02-01' if cid == 'C5' else '2024-01-01') for cid, text in texts.items()},
        fact_storage=dict(facts={}, observations=facts, time_assertions={}, attribute_assertions={}),
        event_state_layer=dict(schema='event_index_v1', groups={}, links=[
            dict(link_id='L1', from_fact_id='F1', to_fact_id='F4', relation='candidate'),
            dict(link_id='L2', from_fact_id='F4', to_fact_id='F6', relation='candidate'),
            dict(link_id='L3', from_fact_id='F5', to_fact_id='F3', relation='candidate')]))


class Embeddings:
    """固定向量只检验余弦算法；不模拟或宣称 BGE-M3 的语义识别质量。"""
    async def get_embeddings(self, texts):
        return [[1.0, 0.0] if 'B was only a plan' in text or 'counterevidence' in text
                else [0.0, 1.0] for text in texts]


class RetrievalTests(unittest.IsolatedAsyncioTestCase):
    async def test_sources_are_mandatory_and_input_is_unchanged(self):
        source = graph(); before = deepcopy(source)
        result = await api.EvidenceRetriever(source).retrieve('Compare jobs', ['F1', 'F2'])
        self.assertTrue({'C1', 'C2'} <= {r['chunk_id'] for r in result['evidence']})
        self.assertEqual(source, before)
        self.assertTrue(result['diagnostics']['semantic_status'].startswith('unavailable'))

    async def test_structured_route_finds_same_slot_without_requiring_equal_values(self):
        result = await api.EvidenceRetriever(graph()).retrieve('Compare', ['F1', 'F2'])
        candidate = next(r for r in result['candidates'] if r['fact_id'] == 'F3')
        self.assertTrue(any(route.startswith('structured:') for route in candidate['route_ranks']))

    async def test_semantic_route_can_find_counterevidence_from_another_subject(self):
        result = await api.EvidenceRetriever(graph(), Embeddings()).retrieve('counterevidence', ['F1', 'F2'])
        candidate = next(r for r in result['candidates'] if r['fact_id'] == 'F4')
        self.assertIn('semantic:task', candidate['route_ranks'])

    async def test_bfs_depth_and_paths_are_explicit_not_proof(self):
        one = await api.EvidenceRetriever(graph(), config=api.EvidenceSupportConfig(max_hops=1)).retrieve('Compare', ['F1'])
        self.assertFalse(any(r['fact_id'] == 'F6' for r in one['candidates']))
        two = await api.EvidenceRetriever(graph()).retrieve('Compare', ['F1'])
        hit = next(r for r in two['candidates'] if r['fact_id'] == 'F6')
        self.assertEqual(hit['paths'][0]['fact_ids'], ['F1', 'F4', 'F6'])
        self.assertEqual(hit['support_status'], 'not_assessed')

    async def test_cutoff_excludes_future_from_all_routes_and_raw_text(self):
        source = graph()
        # 同一 Chunk 中即使断言自称较早，也不能绕过来源会话时间。
        source['fact_storage']['observations']['F5']['observed_at'] = '2023-01-01'
        result = await api.EvidenceRetriever(source, Embeddings()).retrieve('Compare', ['F1', 'F2'], as_of='2024-01-15')
        self.assertNotIn('F5', {r['fact_id'] for r in result['candidates']})
        self.assertNotIn('C5', {r['chunk_id'] for r in result['evidence']})
        with self.assertRaises(ValueError):
            await api.EvidenceRetriever(source).retrieve('Compare', ['F5'], as_of='2024-01-15')

    async def test_undated_sources_are_not_assumed_historical(self):
        source = graph(); source['chunk_storage']['C4'].pop('session_time')
        result = await api.EvidenceRetriever(source).retrieve('Compare', ['F1'], as_of='2024-01-15')
        self.assertNotIn('F4', {r['fact_id'] for r in result['candidates']})
        self.assertNotIn('F6', {r['fact_id'] for r in result['candidates']})

    async def test_same_chunk_endpoints_are_kept_together(self):
        source = graph()
        source['fact_storage']['observations']['F2'].update(chunk_id='C1', quote='I work at A.')
        result = await api.EvidenceRetriever(source).retrieve('Compare', ['F1', 'F2'])
        record = next(r for r in result['evidence'] if r['chunk_id'] == 'C1')
        self.assertTrue({'F1', 'F2'} <= set(record['fact_ids']))
        self.assertEqual(len(result['evidence']), len({r['chunk_id'] for r in result['evidence']}))

    async def test_invalid_vectors_are_audited_not_given_fake_scores(self):
        class BadEmbedding:
            async def get_embeddings(self, texts): return [[float('nan')]] * len(texts)
        result = await api.EvidenceRetriever(graph(), BadEmbedding()).retrieve('Compare', ['F1'])
        self.assertTrue(result['diagnostics']['semantic_status'].startswith('unavailable'))
        self.assertFalse(any(any(k.startswith('semantic:') for k in r['route_ranks']) for r in result['candidates']))

    async def test_zero_similarity_is_not_treated_as_support_confidence(self):
        result = await api.EvidenceRetriever(graph(), Embeddings()).retrieve('counterevidence', ['F1'])
        for row in result['candidates']:
            self.assertNotIn('confidence', row)
            self.assertEqual(row['support_status'], 'not_assessed')

    async def test_graph_node_budget_stops_expansion(self):
        config = api.EvidenceSupportConfig(max_graph_nodes=1)
        result = await api.EvidenceRetriever(graph(), config=config).retrieve('Compare', ['F1'])
        self.assertLessEqual(result['diagnostics']['graph_nodes_visited'], 1)
        self.assertTrue(result['diagnostics']['graph_truncated'])

    async def test_question_without_anchors_uses_semantic_seeds_for_graph_expansion(self):
        result = await api.EvidenceRetriever(graph(), Embeddings()).retrieve('counterevidence')
        self.assertTrue(result['diagnostics']['graph_seed_fact_ids'])
        self.assertTrue(any(name.startswith('graph:') for name in result['routes']))

    async def test_seen_sources_do_not_consume_followup_top_k(self):
        config = api.EvidenceSupportConfig(max_hops=0, semantic_k=1)
        result = await api.EvidenceRetriever(graph(), Embeddings(), config).retrieve(
            'counterevidence', exclude_chunk_ids=['C4'])
        self.assertTrue(result['evidence'])
        self.assertNotIn('C4', {r['chunk_id'] for r in result['evidence']})

    async def test_multiple_edge_types_do_not_consume_neighbor_quota_twice(self):
        source = graph()
        source['event_state_layer']['groups']['G'] = dict(fact_ids=['F1', 'F4'])
        source['event_state_layer']['links'].append(dict(from_fact_id='F1', to_fact_id='F6'))
        config = api.EvidenceSupportConfig(max_hops=1, max_neighbors=2)
        result = await api.EvidenceRetriever(source, config=config).retrieve('Compare', ['F1'])
        self.assertTrue({'F4', 'F6'} <= set(result['routes']['graph:F1']))

    async def test_time_assertions_without_subject_do_not_share_a_state_slot(self):
        source = graph()
        source['fact_storage']['time_assertions'] = {
            'A1': dict(event_id='E1', predicate='event_time', value='2023-01-01', evidence_chunk_id='C1'),
            'A2': dict(event_id='E2', predicate='event_time', value='2023-01-02', evidence_chunk_id='C2')}
        result = await api.EvidenceRetriever(source, config=api.EvidenceSupportConfig(max_hops=0)).retrieve('When?', ['A1'])
        self.assertNotIn('A2', result['routes']['structured:A1'])

    async def test_rrf_uses_two_independent_anchor_rankings(self):
        config = api.EvidenceSupportConfig(max_hops=0)
        result = await api.EvidenceRetriever(graph(), config=config).retrieve('Compare', ['F1', 'F2'])
        fact = next(row for row in result['candidates'] if row['fact_id'] == 'F3')
        self.assertEqual(fact['route_ranks'], {'structured:F1': 1, 'structured:F2': 1})
        self.assertAlmostEqual(fact['rrf_score'], 2 / 61)

    async def test_seen_graph_nodes_remain_available_as_transit(self):
        result = await api.EvidenceRetriever(graph()).retrieve('Compare', ['F1'], exclude_chunk_ids=['C4'])
        self.assertIn('F6', result['routes']['graph:F1'])
        self.assertNotIn('C4', {r['chunk_id'] for r in result['evidence']})

    def test_invalid_budgets_are_rejected(self):
        for values in ({'max_hops': -1}, {'rrf_k': 0}, {'max_chunks': 1},
                       {'semantic_k': True}, {'min_cosine': float('nan')}):
            with self.assertRaises(ValueError): api.EvidenceSupportConfig(**values)


class ReviewTests(unittest.IsolatedAsyncioTestCase):
    def response(self, *, relation=False, status='SUFFICIENT'):
        """固定 Judge 输出，所有引用均可回查；不作为人工关系标注。"""
        citations = [dict(fact_id='F1', chunk_id='C1', quote='I work at A.'),
                     dict(fact_id='F2', chunk_id='C2', quote='I work at B.')]
        value = dict(status=status, reason='固定协议响应', missing_evidence=[], followup_queries=[],
            checks={key: 'pass' for key in ('identity', 'scope', 'modality', 'time', 'granularity')},
            citations=citations, evidence_roles=[dict(**citation, target_fact_id=citation['fact_id'],
                claim='Source reports this employment.', role='supports') for citation in citations])
        if relation:
            value.update(relation='coexists', old_fact_id='F1', new_fact_id='F2')
        return value

    async def test_relation_report_does_not_apply_updates(self):
        value = self.response(relation=True)
        class LLM:
            async def generate(self, prompt, **kwargs): return json.dumps(value)
        source = graph(); before = deepcopy(source)
        result = await api.EvidenceSupport(source, Embeddings(), LLM()).review_pair('F1', 'F2')
        self.assertEqual(result['status'], 'SUFFICIENT')
        self.assertEqual(result['decision_proposal']['proposed_action'], 'RETAIN')
        self.assertFalse(result['update_applied'])
        self.assertEqual(source, before)
        self.assertTrue(result['evidence_roles'])

    async def test_retrieval_only_is_not_called_sufficient(self):
        result = await api.EvidenceSupport(graph()).review_pair('F1', 'F2')
        self.assertEqual(result['status'], 'NOT_EVALUATED')
        self.assertEqual(result['rounds'], [])

    async def test_forged_role_quote_cannot_pass_review(self):
        value = self.response(); value['evidence_roles'][0]['quote'] = 'Invented quote'
        class LLM:
            async def generate(self, prompt, **kwargs): return json.dumps(value)
        result = await api.EvidenceSupport(graph(), Embeddings(), LLM()).review_question('Where?')
        self.assertEqual(result['status'], 'DEFER')
        self.assertIn('evidence_role', result['rounds'][-1].get('stop_reason', ''))

    async def test_repeated_gap_query_stops_when_no_new_sources(self):
        value = self.response(status='NEED_MORE')
        value.update(missing_evidence=['Need context'], followup_queries=['counterevidence'])
        class LLM:
            calls = 0
            async def generate(self, prompt, **kwargs): self.calls += 1; return json.dumps(value)
        llm = LLM()
        result = await api.EvidenceSupport(graph(), Embeddings(), llm).review_question('Where?')
        self.assertEqual(result['status'], 'DEFER')
        self.assertLessEqual(llm.calls, 3)
        self.assertIn(result['stop_reason'], ('no_new_evidence', 'judge_budget_exhausted'))

    async def test_question_and_relation_share_retrieval_report_format(self):
        support = api.EvidenceSupport(graph(), Embeddings())
        question = await support.review_question('Where?', claim_ids=['F1'])
        relation = await support.review_pair('F1', 'F2')
        self.assertEqual(question['schema'], relation['schema'])
        self.assertEqual(question['retrieval_rounds'][0]['schema'], 'evidence_retrieval_v1')

    async def test_answer_uses_only_reviewed_sources_and_validates_citations(self):
        value = self.response()
        class LLM:
            async def generate(self, prompt, **kwargs): return json.dumps(value)
        class Answer:
            async def generate(self, prompt, **kwargs):
                return json.dumps(dict(answer='At A.', citations=[dict(fact_id='F1', chunk_id='C1', quote='I work at A.')]))
        result = await api.EvidenceSupport(graph(), Embeddings(), LLM()).answer_question('Where?', Answer())
        self.assertEqual(result['answer'], 'At A.')
        self.assertEqual(result['answer_status'], 'source_grounded')

    async def test_unsupported_answer_is_rejected(self):
        value = self.response()
        class LLM:
            async def generate(self, prompt, **kwargs): return json.dumps(value)
        class Answer:
            async def generate(self, prompt, **kwargs):
                return json.dumps(dict(answer='Future!', citations=[dict(fact_id='F5', chunk_id='C5', quote='Invented')]))
        result = await api.EvidenceSupport(graph(), Embeddings(), LLM()).answer_question('Where?', Answer())
        self.assertIsNone(result['answer'])
        self.assertEqual(result['answer_status'], 'rejected')

    async def test_gap_search_adds_new_counterevidence_before_next_judge(self):
        first = self.response(relation=True, status='NEED_MORE')
        first.update(missing_evidence=['Need context'], followup_queries=['counterevidence'])
        final = self.response(relation=True)
        class LLM:
            calls = 0
            async def generate(self, prompt, **kwargs):
                self.calls += 1
                return json.dumps(first if self.calls == 1 else final)
        config = api.EvidenceSupportConfig(max_hops=0, semantic_k=1, structured_k=1)
        result = await api.EvidenceSupport(graph(), Embeddings(), LLM(), config).review_pair('F1', 'F2')
        self.assertEqual(len(result['rounds']), 2)
        self.assertNotIn('C4', {r['chunk_id'] for r in result['retrieval_rounds'][0]['evidence']})
        self.assertIn('C4', {r['chunk_id'] for r in result['retrieval_rounds'][1]['evidence']})

    async def test_answer_backend_failure_keeps_review_audit(self):
        value = self.response()
        class LLM:
            async def generate(self, prompt, **kwargs): return json.dumps(value)
        class Answer:
            async def generate(self, prompt, **kwargs): raise RuntimeError('offline')
        result = await api.EvidenceSupport(graph(), Embeddings(), LLM()).answer_question('Where?', Answer())
        self.assertEqual(result['answer_status'], 'failed')
        self.assertTrue(result['rounds'])

    async def test_wrong_answer_json_types_are_rejected_with_audit(self):
        value = self.response()
        class LLM:
            async def generate(self, prompt, **kwargs): return json.dumps(value)
        for bad in ([], dict(answer='A', citations=[dict(fact_id='F1', chunk_id='C1', quote=7)])):
            class Answer:
                async def generate(self, prompt, **kwargs): return json.dumps(bad)
            result = await api.EvidenceSupport(graph(), Embeddings(), LLM()).answer_question('Where?', Answer())
            self.assertEqual(result['answer_status'], 'rejected')
            self.assertTrue(result['rounds'])


if __name__ == '__main__':
    unittest.main()
