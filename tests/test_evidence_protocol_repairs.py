"""证据协议修复回归：真实模块、固定响应；不将协议通过率当成关系准确率。"""
import importlib
import json
import unittest
import ast
from copy import deepcopy
from pathlib import Path

import test_evidence_support as fixtures
from test_evidence_support import api, graph, Embeddings

judge_api = importlib.import_module(api.__package__.rsplit('.', 1)[0] + '.evidence_judge')


class ProtocolRepairTests(unittest.IsolatedAsyncioTestCase):
    async def test_question_seed_skips_future_source_before_judge(self):
        class SeedRetriever:
            async def retrieve(self, question, top_k=None):
                return dict(evidence=[dict(chunk_id='C5'), dict(chunk_id='C1')])
        source = graph();before = deepcopy(source)
        support = api.EvidenceSupport(source, Embeddings());support.query_seed_retriever = SeedRetriever()
        report = await support.review_question('Where?', as_of='2024-01-15')
        self.assertEqual(report['task']['anchor_fact_ids'], ['F1'])
        self.assertNotIn('C5', {r['chunk_id'] for r in report['evidence']})
        self.assertEqual(source, before)

    async def test_question_reuses_existing_chunk_retriever_to_seed_grounded_fact(self):
        class SeedRetriever:
            async def retrieve(self, question, top_k=None):
                return dict(evidence=[dict(chunk_id='C1', score=0.9)])
        support = api.EvidenceSupport(graph(), Embeddings())
        support.query_seed_retriever = SeedRetriever()
        report = await support.review_question('Where?', as_of='2024-01-15')
        self.assertEqual(report['task']['anchor_fact_ids'], ['F1'])
        self.assertEqual(report['query_seed_trace']['method'], 'structural_semantic_chunk_then_fact')

    async def test_answer_selects_existing_quote_id_without_copying_source_text(self):
        class Judge:
            async def generate(self, prompt, **kwargs): return json.dumps(fixtures.ReviewTests().response())
        class Answer:
            async def generate(self, prompt, **kwargs):
                data = json.loads(prompt.split('\n')[-1])
                q = next(q for e in data['evidence'] for q in e['quote_options'] if q['fact_id'] == 'F1')
                return json.dumps(dict(answer='At A.', citations=[dict(quote_id=q['quote_id'])]))
        result = await api.EvidenceSupport(graph(), Embeddings(), Judge()).answer_question('Where?', Answer())
        self.assertEqual(result['answer'], 'At A.')
        self.assertEqual(result['answer_citations'][0]['quote'], 'I work at A.')
        self.assertEqual(result['answer_status'], 'source_grounded')

    async def test_question_protocol_repair_budget_is_shared_across_rounds(self):
        service = importlib.import_module(api.__package__ + '.service')
        class LLM:
            calls = 0
            async def generate(self, prompt, **kwargs):
                self.calls += 1
                return 'broken JSON'
        llm = LLM();judge = service.QuestionEvidenceJudge(llm)
        evidence = (await api.EvidenceRetriever(graph()).retrieve('Where?', ['F1']))['evidence']
        for _ in range(2):
            await judge.assess(dict(query='Where?', mode='question'), evidence)
        self.assertEqual(llm.calls, 3)

    def test_question_windows_prioritize_query_rank_before_graph_only_insertion(self):
        service = importlib.import_module(api.__package__ + '.service')
        sources = [dict(chunk_id='background', text='Background only.', fact_ids=['F1'], facts=[]),
                   dict(chunk_id='query', text='Direct query material.', fact_ids=['F2'], facts=[], task_semantic_rank=1)]
        _, shown, _ = service.QuestionEvidenceJudge(None)._prompt(dict(query='Question?', mode='question'), sources)
        self.assertEqual(shown[0]['chunk_id'], 'query')

    async def test_question_protocol_failure_is_repaired_once_without_changing_evidence(self):
        class LLM:
            calls = 0
            async def generate(self, prompt, **kwargs):
                self.calls += 1
                if self.calls == 1:
                    value = fixtures.ReviewTests().response()
                    value.update(citations=[], evidence_roles=[])
                    return json.dumps(value)
                return json.dumps(fixtures.ReviewTests().response())
        llm = LLM()
        report = await api.EvidenceSupport(graph(), Embeddings(), llm).review_question('Where?')
        self.assertEqual(report['status'], 'SUFFICIENT')
        self.assertEqual(llm.calls, 2)
        self.assertEqual(report['rounds'][0]['response_repairs_used'], 1)
        self.assertEqual(len(report['rounds']), 1)
        self.assertEqual(report['judge_calls'], 2)

    def test_two_required_sources_fit_without_repeating_text_in_prompt(self):
        service = importlib.import_module(api.__package__ + '.service')
        conflict = importlib.import_module(api.__package__.rsplit('.', 1)[0] + '.conflict_review')
        pair = conflict.explicit_fact_pair(graph(), 'F1', 'F2')
        sources = []
        for cid, fid in [('C2', 'F2'), ('C1', 'F1')]:
            quotes = [f'Original evidence {i}: ' + chr(65+i) * 130 + '.' for i in range(4)]
            ids = [fid] + [fid + str(i) for i in range(1, 4)]
            sources.append(dict(chunk_id=cid, text='\n'.join(quotes), session_id='D1',
                session_time='2024-01-01', fact_ids=ids,
                facts=[dict(fact_id=f, entity_id='User', attribute='detail', value='v' * 160) for f in ids],
                fact_quotes=dict(zip(ids, quotes)), mandatory_fact_ids=[fid]))
        judge = service.RelationEvidenceJudge(None, pair)
        prompt, shown, context = judge._prompt(dict(query='Compare', mode='memory_relation'), sources)
        self.assertEqual(context['missing_mandatory_fact_ids'], [])
        self.assertEqual(len(shown), 2)
        self.assertLessEqual(len(prompt.encode('utf-8')), 12000)

    async def test_endpoint_selection_cannot_drop_old_side_when_extra_citations_only_mention_new(self):
        class LLM:
            async def generate(self, prompt, **kwargs):
                self.schema = kwargs['response_format']['json_schema']['schema']
                data = json.loads(prompt.split('\n')[-1])
                selected = {q['fact_id']: q for r in data['evidence'] for q in r['quote_options']}
                value = fixtures.ReviewTests().response(relation=True)
                value['endpoint_citations'] = dict(old_quote_id=selected['F1']['quote_id'], new_quote_id=selected['F2']['quote_id'])
                value['citations'] = [dict(quote_id=selected['F2']['quote_id'])]
                value['evidence_roles'] = [dict(quote_id=selected['F2']['quote_id'], target_fact_id='F2',
                    claim='New employment report.', role='supports')]
                return json.dumps(value)
        llm = LLM()
        report = await api.EvidenceSupport(graph(), Embeddings(), llm).review_pair('F1', 'F2')
        self.assertEqual(report['status'], 'SUFFICIENT')
        self.assertIn('endpoint_citations', llm.schema['required'])
        self.assertEqual({c['fact_id'] for c in report['rounds'][0]['citations']}, {'F1', 'F2'})

    def test_candidate_patch_copy_keeps_original_audit_and_does_not_update_facts(self):
        from experiments.evidence_support.replay_relations import build_candidate_graph
        source = graph(); before = deepcopy(source)
        original = dict(old_fact_id='F1', new_fact_id='F2', relation='uncertain', relation_id='old',
                        classification=dict(validation_error='bad quote'), quote='I work at B.')
        audit = dict(fact_relations=[original])
        result = dict(relation='replacement', evaluation_status='model_assessed', evidence_quote='I work at B.')
        merged, updated_audit = build_candidate_graph(source, audit, [dict(row_index=0,
            old_fact_id='F1', new_fact_id='F2', classification=result)])
        self.assertEqual(source, before)
        self.assertEqual(merged['fact_storage'], before['fact_storage'])
        self.assertEqual(len(updated_audit['fact_relations']), 2)
        self.assertEqual(updated_audit['fact_relations'][0], original)
        self.assertTrue(any(l['relation'] == 'replacement' for l in merged['event_state_layer']['links']))
        self.assertEqual(merged['event_state_layer']['links'][-1]['status'], 'candidate')

    async def test_longmemeval_question_and_session_dates_are_comparable(self):
        source = graph()
        source['chunk_storage']['C1']['session_time'] = '2023/05/20'
        source['fact_storage']['observations']['F1']['observed_at'] = '2023/05/20 (Sat) 02:21'
        report = await api.EvidenceRetriever(source).retrieve('Compare', ['F1'], as_of='2023/05/30 (Tue) 23:40')
        self.assertIn('F1', {r['fact_id'] for r in report['candidates']})

    def test_failed_relation_replay_preserves_missing_and_ungrounded_endpoints(self):
        from experiments.evidence_support.replay_relations import select_failed_pairs
        source = graph()
        row = dict(old_fact_id='F1', new_fact_id='F2', classification=dict(validation_error='bad quote'))
        selected, skipped = select_failed_pairs(source, [row])
        self.assertEqual(len(selected), 1)
        source['fact_storage']['observations']['F2']['quote'] = 'Not in the source'
        selected, skipped = select_failed_pairs(source, [row])
        self.assertEqual(selected, [])
        self.assertEqual(skipped[0]['reason'], 'source_quote_not_grounded')

    async def test_new_graph_query_returns_shared_review_and_cited_sessions(self):
        module = importlib.import_module(api.__package__ + '.query')
        class Support:
            async def answer_question(self, question, llm, **kwargs):
                return dict(status='SUFFICIENT', answer='At A.', answer_status='source_grounded',
                    answer_citations=[dict(chunk_id='C1')], evidence=[dict(chunk_id='C1', session_id='D1')])
        result = await module.query_with_evidence_support(graph(), 'Where?', '', None, None, support=Support())
        self.assertEqual(result['answer'], 'At A.')
        self.assertEqual(result['top_session_ids'], ['D1'])
        self.assertEqual(result['evidence_report']['status'], 'SUFFICIENT')

    def test_longmem_module_is_valid_python(self):
        path = Path(__file__).resolve().parents[1] / 'dataset/longmem.py'
        ast.parse(path.read_text(encoding='utf-8'))

    async def test_numbered_citation_is_resolved_to_source_span_and_shadow_scores(self):
        class LLM:
            async def generate(self, prompt, **kwargs):
                data = json.loads(prompt.split('\n')[-1])
                value = fixtures.ReviewTests().response(relation=True)
                selected = [next(q for row in data['evidence'] for q in row['quote_options']
                                 if q['fact_id'] == fid) for fid in ('F1', 'F2')]
                value['citations'] = [dict(quote_id=q['quote_id']) for q in selected]
                value['endpoint_citations'] = dict(old_quote_id=selected[0]['quote_id'], new_quote_id=selected[1]['quote_id'])
                value['evidence_roles'] = [dict(quote_id=q['quote_id'], target_fact_id=q['fact_id'],
                    claim='Reported employer.', role='supports') for q in selected]
                return json.dumps(value)
        report = await api.EvidenceSupport(graph(), Embeddings(), LLM()).review_pair('F1', 'F2')
        self.assertEqual(report['status'], 'SUFFICIENT')
        self.assertEqual(report['confidence']['mode'], 'shadow')
        self.assertEqual(set(report['confidence']['targets']), {'F1', 'F2'})
        # 固定端点没有召回排名，也必须保留已绑定原文的支持度。
        self.assertGreater(report['confidence']['targets']['F1']['belief'], 0)
        self.assertGreater(report['confidence']['targets']['F2']['belief'], 0)
        self.assertIsNone(report['confidence']['relation_confidence'])
        self.assertFalse(report['confidence']['affects_decision'])
        self.assertEqual(report['rounds'][0]['citations'][0]['quote'], 'I work at A.')

    async def test_technical_failure_is_not_semantic_uncertainty_or_scored(self):
        class LLM:
            async def generate(self, prompt, **kwargs): return 'broken JSON'
        report = await api.EvidenceSupport(graph(), Embeddings(), LLM()).review_pair('F1', 'F2')
        self.assertEqual(report['evaluation_status'], 'technical_error')
        self.assertEqual(report['confidence']['status'], 'NOT_EVALUATED')
        self.assertIsNone(report['confidence']['uncertainty'])

    def test_two_distant_quotes_in_one_chunk_have_original_spans(self):
        source = dict(chunk_id='C1', text='first sentence.' + 'x' * 4000 + 'last sentence.',
            fact_ids=['F1', 'F2'], facts=[],
            fact_quotes={'F1': 'first sentence.', 'F2': 'last sentence.'}, mandatory_fact_ids=['F1', 'F2'])
        _, shown, _ = judge_api.EvidenceJudge(None)._prompt(dict(mode='question', query='Compare'), [source])
        options = shown[0]['quote_options']
        self.assertTrue({'F1', 'F2'} <= {q['fact_id'] for q in options})
        for quote in options:
            self.assertEqual(source['text'][slice(*quote['span'])], quote['quote'])

    async def test_forged_quote_id_is_rejected_without_semantic_scoring(self):
        value = fixtures.ReviewTests().response(relation=True)
        value['citations'] = [dict(quote_id='not-in-source')]
        class LLM:
            async def generate(self, prompt, **kwargs): return json.dumps(value)
        report = await api.EvidenceSupport(graph(), Embeddings(), LLM()).review_pair('F1', 'F2')
        self.assertEqual(report['status'], 'DEFER')
        self.assertEqual(report['evaluation_status'], 'technical_error')


if __name__ == '__main__':
    unittest.main()
