"""服务于置信度模块：固定输入检查，不需要网络、LLM 或 GPU 依赖。"""

import asyncio
from copy import deepcopy
import json
from pathlib import Path
import runpy
import unittest

# 独立模块只依赖标准库；直接加载避免包 __init__ 连带导入 torch 等无关依赖。
api = runpy.run_path(str(Path(__file__).resolve().parents[1] / 'coregraph/memory_confidence.py'))
register = api['register_confidence']
Evaluator = api['MemoryConfidenceEvaluator']


def graph():
    return dict(chunk_storage={'1': {'text': 'Caroline: I went on May 7.'},
                               '2': {'text': 'Caroline: I remembered wrong; it was May 6.'}},
        fact_storage=dict(facts={}, observations={
            'F1': dict(fact_id='F1', entity_id='Caroline', attribute='date', value='May 7',
                       chunk_id='1', quote='I went on May 7.'),
            'F2': dict(fact_id='F2', entity_id='Caroline', attribute='date', value='May 6',
                       chunk_id='2', quote='I remembered wrong; it was May 6.')},
            time_assertions={}, attribute_assertions={}),
        event_state_layer={'schema': 'event_index_v1', 'groups': {}, 'links': [dict(
            link_id='R1', from_fact_id='F1', to_fact_id='F2', relation='correction', status='candidate')]})


class FakeLLM:
    model = 'fixed-test'
    def __init__(self, response):
        self.response = response
        self.calls = 0
    async def generate(self, *args, **kwargs):
        self.calls += 1
        return json.dumps(self.response)


class MemoryConfidenceTests(unittest.IsolatedAsyncioTestCase):
    def test_register_is_idempotent_and_never_makes_truth_decisions(self):
        data = graph()
        records = register(data)
        self.assertIsNone(data['fact_storage']['observations']['F1']['extraction_confidence'])
        self.assertEqual(len(records['evidence_links']), 2)
        self.assertEqual(len(records['relation_evaluations']), 1)
        self.assertEqual(data['event_state_layer']['links'][0]['status'], 'candidate')
        before = deepcopy(data)
        self.assertEqual(register(data), records)
        self.assertEqual(data, before)

    async def test_fact_score_is_uncalibrated_and_support_direction_is_separate(self):
        data = graph()
        register(data)
        llm = FakeLLM(dict(extraction_score=0.8, evidence_score=0.9, evidence_role='supports',
            evidence_quote='I went on May 7.', reason='日期由原文直接给出'))
        await Evaluator(llm).assess_fact(data, 'F1')
        fact = data['fact_storage']['observations']['F1']
        assessment = data['confidence_storage']['fact_evaluations'][fact['extraction_evaluation_id']]
        self.assertEqual(fact['extraction_confidence'], 0.8)
        self.assertEqual(assessment['calibration'], 'uncalibrated')
        self.assertNotIn('truth_probability', assessment)
        evidence = next(r for r in data['confidence_storage']['evidence_links'].values() if r['fact_id'] == 'F1')
        self.assertEqual(evidence['role'], 'supports')
        self.assertEqual(evidence['score'], 0.9)

    async def test_invalid_scores_or_invented_quote_leave_explicit_failure(self):
        for change in ({'extraction_score': True}, {'extraction_score': 1.5},
                       {'extraction_score': float('nan')}, {'evidence_quote': 'not in source'}):
            data = graph()
            register(data)
            response = dict(extraction_score=0.8, evidence_score=0.9, evidence_role='supports',
                            evidence_quote='I went on May 7.', reason='test')
            response.update(change)
            await Evaluator(FakeLLM(response)).assess_fact(data, 'F1')
            fid = data['fact_storage']['observations']['F1']['extraction_evaluation_id']
            result = data['confidence_storage']['fact_evaluations'][fid]
            self.assertEqual(result['status'], 'failed')
            self.assertIsNone(result['score'])

    async def test_relation_local_support_never_confirms_or_invalidates_history(self):
        data = graph()
        register(data)
        response = dict(score=0.95, verdict='supported_candidate', reason='明确日期纠正，但仅检查局部证据',
            checks={key: 'pass' for key in api['RELATION_CHECKS']}, evidence_quotes=[
                dict(chunk_id='2', quote='I remembered wrong; it was May 6.')])
        await Evaluator(FakeLLM(response)).assess_relation(data, 'R1')
        link = data['event_state_layer']['links'][0]
        record = data['confidence_storage']['relation_evaluations'][link['evaluation_id']]
        self.assertEqual(record['scope'], 'pair_local')
        self.assertEqual(record['validation_status'], 'not_verified')
        self.assertEqual(link['status'], 'candidate')
        self.assertNotIn('current', data)
        self.assertEqual(data['fact_storage']['observations']['F1']['value'], 'May 7')

    async def test_unknown_key_condition_defers_despite_high_score(self):
        data = graph()
        register(data)
        checks = {key: 'pass' for key in api['RELATION_CHECKS']}
        checks['same_event_or_slot'] = 'unknown'
        response = dict(score=0.99, verdict='supported_candidate', reason='事件身份不明', checks=checks,
                        evidence_quotes=[dict(chunk_id='2', quote='it was May 6.')])
        await Evaluator(FakeLLM(response)).assess_relation(data, 'R1')
        result = next(iter(data['confidence_storage']['relation_evaluations'].values()))
        self.assertEqual(result['verdict'], 'uncertain')
        self.assertEqual(result['model_verdict'], 'supported_candidate')

    def test_changed_input_invalidates_old_assessment(self):
        data = graph()
        records = register(data)
        fid = data['fact_storage']['observations']['F1']['extraction_evaluation_id']
        records['fact_evaluations'][fid].update(score=0.9, status='assessed')
        data['fact_storage']['observations']['F1']['value'] = 'May 8'
        register(data)
        self.assertIsNone(data['fact_storage']['observations']['F1']['extraction_confidence'])
        self.assertEqual(records['fact_evaluations'][fid]['status'], 'stale')

    async def test_retry_failure_clears_previous_support_but_preserves_history(self):
        data = graph()
        good = dict(extraction_score=0.8, evidence_score=0.9, evidence_role='supports',
                    evidence_quote='I went on May 7.', reason='test')
        await Evaluator(FakeLLM(good)).assess_fact(data, 'F1')
        await Evaluator(FakeLLM({})).assess_fact(data, 'F1')
        evidence = next(r for r in data['confidence_storage']['evidence_links'].values() if r['fact_id'] == 'F1')
        self.assertEqual(evidence['status'], 'failed')
        self.assertEqual(evidence['role'], 'unassessed')
        self.assertIsNone(evidence['span'])
        self.assertTrue(evidence['history'])
        self.assertIn('raw_response', evidence['trace'])

    async def test_cancel_propagates_and_dependent_evidence_is_not_independent(self):
        data = graph()
        data['fact_storage']['observations']['F3'] = dict(data['fact_storage']['observations']['F1'], fact_id='F3')
        records = register(data)
        same_chunk = [r for r in records['evidence_links'].values() if r['chunk_id'] == '1']
        self.assertEqual(len({r['independence_group'] for r in same_chunk}), 1)
        class CancelLLM:
            async def generate(self, *args, **kwargs):
                raise asyncio.CancelledError()
        with self.assertRaises(asyncio.CancelledError):
            await Evaluator(CancelLLM()).assess_fact(data, 'F1')

    async def test_changed_source_does_not_reuse_old_chunk_support(self):
        data = graph()
        response = dict(extraction_score=0.8, evidence_score=0.9, evidence_role='supports',
                        evidence_quote='I went on May 7.', reason='test')
        await Evaluator(FakeLLM(response)).assess_fact(data, 'F1')
        data['chunk_storage']['3'] = {'text': 'Caroline: I went on May 8.'}
        data['fact_storage']['observations']['F1'].update(chunk_id='3', quote='I went on May 8.', value='May 8')
        register(data)
        response['evidence_quote'] = 'I went on May 8.'
        await Evaluator(FakeLLM(response)).assess_fact(data, 'F1')
        links = {r['chunk_id']: r for r in data['confidence_storage']['evidence_links'].values() if r['fact_id'] == 'F1'}
        self.assertIsNone(links['1']['score'])
        self.assertEqual(links['1']['status'], 'stale')
        self.assertFalse(links['1']['is_current_source'])
        self.assertEqual(links['3']['status'], 'assessed')
        self.assertEqual(links['3']['quote'], response['evidence_quote'])
        self.assertTrue(links['3']['is_current_source'])

    async def test_correction_with_unknown_or_failed_time_is_uncertain(self):
        for state in ('unknown', 'fail'):
            data = graph()
            checks = {key: 'pass' for key in api['RELATION_CHECKS']}
            checks['temporal_compatible'] = state
            response = dict(score=0.99, verdict='supported_candidate', reason='time unclear', checks=checks,
                            evidence_quotes=[dict(chunk_id='2', quote='it was May 6.')])
            await Evaluator(FakeLLM(response)).assess_relation(data, 'R1')
            result = next(iter(data['confidence_storage']['relation_evaluations'].values()))
            self.assertEqual(result['verdict'], 'uncertain')
            self.assertIsNone(result['score'])

    async def test_truncated_response_is_failure_even_if_json_looks_complete(self):
        """服务于真实接口回归：finish_reason=length 时不能把貌似完整输出当作成功。"""
        class TruncatedLLM(FakeLLM):
            async def generate(self, *args, **kwargs):
                kwargs['metadata_sink']['finish_reason'] = 'length'
                return await super().generate(*args, **kwargs)
        data = graph()
        response = dict(extraction_score=1.0, evidence_score=1.0, evidence_role='supports',
                        evidence_quote='I went on May 7.', reason='direct quote')
        await Evaluator(TruncatedLLM(response)).assess_fact(data, 'F1')
        result = next(r for r in data['confidence_storage']['fact_evaluations'].values() if r['fact_id'] == 'F1')
        self.assertEqual(result['status'], 'failed')
        self.assertIsNone(result['score'])
        self.assertEqual(result['trace']['metadata']['finish_reason'], 'length')


if __name__ == '__main__':
    unittest.main()
