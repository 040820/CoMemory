"""服务于 AMA 启发的调度模块：固定输入验证，不需要 LLM、GPU 或联网。"""

import asyncio
from copy import deepcopy
import json
from pathlib import Path
import runpy
import unittest

api = runpy.run_path(str(Path(__file__).resolve().parents[1] / 'coregraph/evidence_judge.py'))
Judge, Controller = api['EvidenceJudge'], api['EvidenceJudgeController']
GraphProvider = runpy.run_path(str(Path(__file__).resolve().parents[1] / 'coregraph/graph_evidence_provider.py'))['GraphEvidenceProvider']


def item(cid='1', text='We have a cat.', fact='F1'):
    return dict(chunk_id=cid, text=text, session_id='D1', session_time='2023-05-01',
                fact_ids=[fact], facts=[dict(fact_id=fact, entity_id='User', attribute='owns', value='cat')])


def result(status, query=None, quote='We have a cat.', cid='1'):
    return dict(status=status, reason='fixed test', citations=[dict(chunk_id=cid, quote=quote, fact_id='F'+cid)],
                missing_evidence=['need identity'] if status == 'NEED_MORE' else [],
                followup_queries=[query] if query else [],
                checks={k: 'pass' for k in api['CHECKS']})


class LLM:
    model='fixed-test'
    def __init__(self, values):
        self.values=list(values); self.calls=[]
    async def generate(self, prompt, **kwargs):
        self.calls.append(prompt)
        return json.dumps(self.values.pop(0))


class Provider:
    def __init__(self, extra=None):
        self.calls=[]; self.extra=extra or []
    async def initial(self, task, top_k):
        self.calls.append('initial'); return [item()]
    async def search(self, query, top_k):
        self.calls.append(query); return deepcopy(self.extra)
    async def expand(self, evidence, stage, top_k):
        self.calls.append(stage); return []


class EvidenceJudgeTests(unittest.IsolatedAsyncioTestCase):
    async def test_sufficient_stops_and_has_no_update_side_effect(self):
        provider=Provider(); llm=LLM([result('SUFFICIENT')])
        output=await Controller(Judge(llm), provider).run('What pets do I own?', mode='question')
        self.assertEqual(output['status'], 'SUFFICIENT')
        self.assertEqual(provider.calls, ['initial'])
        self.assertFalse(output['update_applied'])

    async def test_need_more_uses_gap_query_and_preserves_both_sources(self):
        extra=item('2', 'We got another cat named Bailey.', 'F2')
        provider=Provider([extra]); llm=LLM([result('NEED_MORE', query='another cat'),
            result('SUFFICIENT', quote=extra['text'], cid='2')])
        output=await Controller(Judge(llm), provider).run('Pets?', mode='question')
        self.assertEqual(output['status'], 'SUFFICIENT')
        self.assertEqual({i['chunk_id'] for i in output['evidence']}, {'1','2'})
        self.assertEqual(len(llm.calls), 2)
        self.assertTrue(any('another cat' in q for q in provider.calls))

    async def test_exhausted_budget_defers_instead_of_passing(self):
        provider=Provider([item('2', 'More context.', 'F2')])
        llm=LLM([result('NEED_MORE', query='more'), result('NEED_MORE', query='more again')])
        output=await Controller(Judge(llm), provider, max_rounds=2).run('Pets?', mode='question')
        self.assertEqual(output['status'], 'DEFER')
        self.assertEqual(output['stop_reason'], 'judge_budget_exhausted')
        self.assertEqual(len(llm.calls),2)

    async def test_no_new_evidence_stops_without_repeating_identical_judgment(self):
        llm=LLM([result('NEED_MORE')])
        output=await Controller(Judge(llm), Provider()).run('Pets?', mode='question')
        self.assertEqual(output['status'], 'DEFER')
        self.assertEqual(output['stop_reason'], 'no_new_evidence')
        self.assertEqual(len(llm.calls),1)

    async def test_invented_citation_and_invalid_json_cannot_pass(self):
        bad=result('SUFFICIENT', quote='invented quote')
        output=await Controller(Judge(LLM([bad])), Provider()).run('Pets?', mode='question')
        self.assertEqual(output['status'], 'DEFER')
        self.assertEqual(output['stop_reason'], 'judge_failed')

    async def test_conflict_is_not_an_update_and_requires_comparable_identity(self):
        response=result('CONFLICT')
        response['checks']['identity']='unknown'
        judge=Judge(LLM([response]))
        output=await judge.assess(dict(query='new claim', mode='memory_relation', target_fact={'fact_id':'F2'}), [item()])
        self.assertEqual(output['status'], 'DEFER')
        self.assertEqual(output['model_status'], 'CONFLICT')

    async def test_prompt_is_bounded_and_cut_is_reported(self):
        llm=LLM([result('SUFFICIENT', quote='猫')])
        evidence=[item(text='猫'*10000)]
        output=await Judge(llm, max_prompt_bytes=9000).assess(dict(query='Pets?',mode='question'),evidence)
        self.assertLessEqual(len(llm.calls[0].encode('utf-8')),9000)
        self.assertTrue(output['context']['truncated_chunk_ids'])
        self.assertEqual(evidence[0]['text'], '猫'*10000)

    async def test_cancel_propagates(self):
        class Cancel:
            async def generate(self,*args,**kwargs): raise asyncio.CancelledError()
        with self.assertRaises(asyncio.CancelledError):
            await Controller(Judge(Cancel()),Provider()).run('Pets?', mode='question')

    async def test_valid_conflict_only_reports_conflict(self):
        class TwoSides(Provider):
            async def initial(self,*args):
                return [item(),item('2','We no longer have a cat.','F2')]
        response=result('CONFLICT')
        response['citations'].append(dict(chunk_id='2',quote='We no longer have a cat.',fact_id='F2'))
        output=await Controller(Judge(LLM([response])),TwoSides()).run('Check new fact',
            mode='memory_relation',target_fact={'fact_id':'F2'})
        self.assertEqual(output['status'],'CONFLICT')
        self.assertFalse(output['update_applied'])
        self.assertEqual(output['validation_status'],'not_verified')

    async def test_invalid_json_and_length_are_explicit_defer(self):
        class Broken:
            async def generate(self,*args,**kwargs): return 'not JSON'
        class Truncated:
            async def generate(self,*args,**kwargs):
                kwargs['metadata_sink']['finish_reason']='length'
                return json.dumps(result('SUFFICIENT'))
        for llm in (Broken(),Truncated()):
            output=await Controller(Judge(llm),Provider()).run('Pets?',mode='question')
            self.assertEqual(output['stop_reason'],'judge_failed')

    async def test_retrieval_failure_is_not_silent_success(self):
        class Failing(Provider):
            async def initial(self,*args): raise RuntimeError('retrieval down')
        llm=LLM([])
        output=await Controller(Judge(llm),Failing()).run('Pets?',mode='question')
        self.assertEqual(output['stop_reason'],'retrieval_failed')
        self.assertFalse(llm.calls)

    async def test_anchor_window_citation_span_matches_original(self):
        original='x'*3000+'We have a cat.'
        source=item(text=original); source['anchor_quote']='We have a cat.'
        output=await Judge(LLM([result('SUFFICIENT')])).assess(dict(query='Pets?',mode='question'),[source])
        start,end=output['citations'][0]['span']
        self.assertEqual(original[start:end],'We have a cat.')
        self.assertEqual(start,3000)

    async def test_applicable_failed_or_unknown_checks_cannot_pass_any_mode(self):
        for mode in ('question','memory_relation'):
            for check in api['CHECKS']:
                for state in ('fail','unknown'):
                    response=result('SUFFICIENT'); response['checks'][check]=state
                    output=await Judge(LLM([response])).assess(dict(query='Check',mode=mode,
                        target_fact={'fact_id':'F2'}),[item()])
                    self.assertNotEqual(output['status'],'SUFFICIENT',(mode,check,state))

    async def test_conflict_needs_both_fact_sources_and_cannot_invent_fact_binding(self):
        response=result('CONFLICT')
        output=await Judge(LLM([response])).assess(dict(query='Check',mode='memory_relation',
            target_fact={'fact_id':'F1'}),[item()])
        self.assertEqual(output['status'],'DEFER')
        forged=result('CONFLICT')
        forged['citations'].append(dict(chunk_id='1',quote='We have a cat.',fact_id='F2'))
        output=await Judge(LLM([forged])).assess(dict(query='Check',mode='memory_relation',
            target_fact={'fact_id':'F1'}),[item()])
        self.assertEqual(output['stop_reason'],'judge_failed')

    async def test_not_applicable_cannot_skip_required_identity_or_precision(self):
        for key in ('identity','granularity'):
            response=result('SUFFICIENT'); response['checks'][key]='not_applicable'
            output=await Judge(LLM([response])).assess(dict(query='Check',mode='memory_relation',
                target_fact={'fact_id':'F2'}),[item()])
            self.assertEqual(output['status'],'DEFER')


class GraphProviderTests(unittest.IsolatedAsyncioTestCase):
    def fixture(self):
        return dict(chunk_storage={
            '1':dict(chunk_id=1,session_id='D1',text='Original cat.'),
            '2':dict(chunk_id=2,session_id='D2',text='Other session.'),
            'hashed':dict(chunk_id='hashed',source_chunk_id=2,session_id='D1',text='Another cat.')},
            fact_storage=dict(facts={},observations={
                'F1':dict(fact_id='F1',chunk_id='1',quote='Original cat.'),
                'F2':dict(fact_id='F2',chunk_id='2'),
                'F3':dict(fact_id='F3',chunk_id='hashed',quote='Another cat.')},
                time_assertions={},attribute_assertions={}),
            event_state_layer=dict(schema='event_index_v1',groups={'E1':dict(fact_ids=['F1','F3'])},links=[]))

    async def test_neighbors_stay_in_session_even_with_hashed_ids(self):
        graph=self.fixture()
        async def search(*args): return []
        provider=GraphProvider(graph,search)
        neighbors=await provider.expand([provider.record('1','test')],'neighbors',5)
        self.assertEqual([r['chunk_id'] for r in neighbors],['hashed'])

    async def test_linked_backtrace_and_source_remain_unchanged(self):
        graph=self.fixture(); before=deepcopy(graph)
        async def search(*args): return []
        provider=GraphProvider(graph,search)
        linked=await provider.expand([provider.record('1','test')],'linked',5)
        self.assertEqual(linked[0]['chunk_id'],'hashed')
        self.assertEqual(graph,before)

    async def test_retrieved_text_must_come_from_canonical_chunk(self):
        async def search(*args):
            return {'evidence':[dict(chunk_id='1',text='forged'),dict(chunk_id='missing',text='forged')]}
        provider=GraphProvider(self.fixture(),search)
        records=await provider.search('cat',5)
        self.assertEqual(len(records),1)
        self.assertEqual(records[0]['text'],'Original cat.')

    async def test_initial_always_contains_new_fact_source(self):
        async def search(*args): return []
        async def seeds(*args): return ['F1']
        graph=self.fixture()
        provider=GraphProvider(graph,search,seeds)
        initial=await provider.initial(dict(query='new',target_fact={'fact_id':'F3'}),5)
        self.assertEqual([r['chunk_id'] for r in initial],['hashed','1'])


if __name__=='__main__': unittest.main()
