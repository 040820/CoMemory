"""冲突审查模块回归测试：固定响应检验双端绑定、待审查队列与图优先调度。"""
import importlib
import json
from pathlib import Path
import sys
import types
import unittest

# 只加载本模块需要的标准库组件，避免单元测试触发 coregraph 的 GPU 初始化。
package = '_conflict_review_test_core'
if package not in sys.modules:
    module = types.ModuleType(package)
    module.__path__ = [str(Path(__file__).resolve().parents[1] / 'coregraph')]
    sys.modules[package] = module
api = importlib.import_module(package + '.conflict_review')
judge_api = importlib.import_module(package + '.evidence_judge')


def graph():
    return dict(event_state_layer=dict(schema='event_index_v1', groups={}, links=[
        dict(link_id='L1', from_fact_id='F1', to_fact_id='F2', relation='conflict', status='candidate'),
        dict(link_id='L2', from_fact_id='F1', to_fact_id='F3', relation='coexists', status='candidate')]),
        fact_storage=dict(observations={
            'F1': dict(fact_id='F1', entity_id='User', attribute='residence', value='A', chunk_id='1', quote='I live in A.'),
            'F2': dict(fact_id='F2', entity_id='User', attribute='residence', value='B', chunk_id='2', quote='I moved to B.'),
            'F3': dict(fact_id='F3', entity_id='User', attribute='owns', value='Cat', chunk_id='3', quote='I have a cat.')},
            facts={}, time_assertions={}, attribute_assertions={}),
        chunk_storage={str(i): dict(text=text, session_id='D1', chunk_id=i) for i, text in
                       [(1, 'I live in A.'), (2, 'I moved to B.'), (3, 'I have a cat.')]})


def response(both=True, relation='replacement'):
    citations = [dict(fact_id='F2', chunk_id='2', quote='I moved to B.')]
    if both:
        citations.append(dict(fact_id='F1', chunk_id='1', quote='I live in A.'))
    return dict(status='SUFFICIENT', reason='fixed response', missing_evidence=[], followup_queries=[],
        checks={key: 'pass' for key in judge_api.CHECKS}, citations=citations,
        relation=relation, old_fact_id='F1', new_fact_id='F2')


class LLM:
    def __init__(self, value):
        self.value = value
    async def generate(self, prompt, **kwargs):
        return json.dumps(self.value)


class ConflictReviewTests(unittest.IsolatedAsyncioTestCase):
    def test_conflict_marker_survives_save_load_without_changing_facts(self):
        store_api = importlib.import_module(package + '.triple_fact_store')
        lean = importlib.import_module(package + '.lean_event_index')
        store = store_api.TripleFactStore()
        tid = store.upsert(dict(src='User', relation='lives_in', tgt='A', chunk_id='1'))
        new_tid = store.upsert(dict(src='User', relation='lives_in', tgt='B', chunk_id='2'))
        fact = dict(fact_id='F1', entity_id='User', entity_index=0, attribute='residence',
            value='A', chunk_id='1', quote='I live in A.', scope={}, modality='actual',
            position=1, fact_type='state', status='observed', triple_refs=[store.facts[tid]])
        new = dict(fact, fact_id='F2', value='B', position=2, chunk_id='2',
                   quote='I moved to B.', triple_refs=[store.facts[new_tid]])
        layer_api = importlib.import_module(package + '.event_state_layer')
        layer = layer_api.EventStateLayer(dict(version=1, write_policy='append_only',
            facts={'F1': fact, 'F2': new}))
        layer.record_fact_relation(new, 'F1', dict(relation='conflict',
            evidence_quote='I moved to B.'), 'F2', 'recorded_only')
        self.assertEqual(layer.fact_relations[0]['review_status'], 'pending')
        upper, checkpoint, audit = lean.pack_event_index(layer.to_dict(), store)
        self.assertEqual(upper['links'][0]['review_status'], 'pending')
        self.assertEqual(upper['links'][0]['review_kind'], 'potential_conflict')
        restored = lean.unpack_event_index(upper, store, checkpoint)
        self.assertEqual(restored['fact_relations'][0]['review_status'], 'pending')
        self.assertEqual(restored['facts']['F1']['status'], 'observed')
        self.assertEqual(api.conflict_marker('coexists'), {})

    def test_pending_index_reuses_upper_endpoints_only(self):
        pairs = api.ConflictPairIndex(graph()).pending('F2')
        self.assertEqual(len(pairs), 1)
        self.assertEqual((pairs[0]['old_fact_id'], pairs[0]['new_fact_id']), ('F1', 'F2'))
        self.assertEqual(api.ConflictPairIndex(graph()).pending('F3'), [])

    async def evidence(self):
        g = graph()
        async def unused(*args):
            raise AssertionError('已知两端不应重新进行初始语义召回')
        pair = api.ConflictPairIndex(g).pending('F2')[0]
        provider = api.PairEvidenceProvider(g, unused, pair)
        records = await provider.initial({}, 5)
        self.assertEqual([r['chunk_id'] for r in records], ['2', '1'])
        return pair, records

    async def test_source_only_sufficient_cannot_resolve_pair(self):
        pair, records = await self.evidence()
        result = await api.ConflictPairJudge(LLM(response(False)), pair).assess(
            dict(mode='memory_relation', query='Review pair', target_fact=pair['new_fact']), records)
        self.assertIn(result['status'], ('NEED_MORE', 'DEFER'))
        self.assertFalse(result['decision']['update_applied'])
        self.assertEqual(result['decision']['proposed_action'], 'DEFER')

    async def test_both_sources_produce_proposal_without_overwrite(self):
        pair, records = await self.evidence()
        result = await api.ConflictPairJudge(LLM(response()), pair).assess(
            dict(mode='memory_relation', query='Review pair', target_fact=pair['new_fact']), records)
        self.assertEqual(result['status'], 'SUFFICIENT')
        self.assertEqual(result['decision']['proposed_action'], 'REPLACE')
        self.assertFalse(result['decision']['update_applied'])

    async def test_model_cannot_switch_to_another_pair(self):
        pair, records = await self.evidence()
        output = response(); output['old_fact_id'] = 'F3'
        result = await api.ConflictPairJudge(LLM(output), pair).assess(
            dict(mode='memory_relation', query='Review pair', target_fact=pair['new_fact']), records)
        self.assertNotEqual(result['status'], 'SUFFICIENT')

    async def test_candidate_can_be_downgraded_to_coexistence(self):
        pair, records = await self.evidence()
        result = await api.ConflictPairJudge(LLM(response(relation='coexists')), pair).assess(
            dict(mode='memory_relation', query='Review pair', target_fact=pair['new_fact']), records)
        self.assertEqual(result['decision']['proposed_action'], 'RETAIN')
        self.assertFalse(result['decision']['update_applied'])

    async def test_unresolved_conflict_cannot_become_replace(self):
        pair, records = await self.evidence()
        value = response(relation='conflict'); value['status'] = 'CONFLICT'
        result = await api.ConflictPairJudge(LLM(value), pair).assess(
            dict(mode='memory_relation', query='Review pair', target_fact=pair['new_fact']), records)
        self.assertEqual(result['status'], 'CONFLICT')
        self.assertEqual(result['decision']['proposed_action'], 'DEFER')

    async def test_graph_expansion_precedes_semantic_fallback(self):
        pair, records = await self.evidence()
        calls = []
        class Provider:
            async def initial(self, *args): return records
            async def search(self, *args): calls.append('semantic'); return []
            async def expand(self, evidence, stage, k):
                calls.append(stage)
                return [dict(chunk_id='3', text='More evidence.')] if stage == 'linked' else []
        class Judge:
            async def assess(self, task, evidence):
                return dict(status='NEED_MORE' if len(evidence) == 2 else 'DEFER',
                            reason='fixed', followup_queries=['clarify move'])
        await judge_api.EvidenceJudgeController(Judge(), Provider(), graph_first=True).run(
            'Review pair', target_fact=pair['new_fact'])
        self.assertEqual(calls, ['linked'])

    async def test_new_evidence_can_anchor_next_graph_hop(self):
        pair, records = await self.evidence()
        calls = []
        class Provider:
            async def initial(self, *args): return records
            async def search(self, *args): raise AssertionError('不应跳过可用图路径')
            async def expand(self, evidence, stage, k):
                calls.append(stage)
                return [dict(chunk_id=str(len(evidence) + 1), text='Next supporting context.')]
        class Judge:
            async def assess(self, task, evidence):
                return dict(status='NEED_MORE' if len(evidence) < 4 else 'DEFER',
                            reason='fixed', followup_queries=[])
        result = await judge_api.EvidenceJudgeController(Judge(), Provider(), graph_first=True).run(
            'Review pair', target_fact=pair['new_fact'])
        self.assertEqual(calls, ['linked', 'linked'])
        self.assertEqual(len(result['evidence']), 4)


if __name__ == '__main__':
    unittest.main()
