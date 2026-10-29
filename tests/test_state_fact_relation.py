"""关系判断引文校验回归：格式容错、严格拒绝改写、失败保留原始响应。

本测试只加载 state_fact_relation 单模块，避免触发 coregraph 的模型依赖。
"""

import asyncio
import importlib.util
import json
from pathlib import Path
import unittest

MODULE_PATH = Path(__file__).resolve().parents[1] / 'coregraph' / 'state_fact_relation.py'
_spec = importlib.util.spec_from_file_location('_state_fact_relation_under_test', MODULE_PATH)
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)
StateFactRelationJudge = _module.StateFactRelationJudge


def fact(quote, fact_id='F2', value='poetry reading'):
    return {'fact_id': fact_id, 'entity_id': 'Caroline', 'attribute': 'attended',
            'value': value, 'source_attribute': 'attended', 'source_value': value,
            'scope': {}, 'modality': 'actual', 'semantic_state': None,
            'observed_at': '2023-10-13', 'event_time': '2023-10-06',
            'status': 'observed', 'quote': quote,
            'triple_refs': [{'src': 'Caroline', 'relation': 'attended', 'tgt': value}]}


def response(quote, relation='unrelated', same_slot=False, reason='fixed response'):
    return {'relation': relation, 'same_slot': same_slot,
            'evidence_quote': quote, 'reason': reason}


class LLM:
    def __init__(self, payload):
        self.payload = payload
        self.prompts = []

    async def generate(self, prompt, **kwargs):
        self.prompts.append(prompt)
        return self.payload if isinstance(self.payload, str) else json.dumps(self.payload)


class StateFactRelationTests(unittest.IsolatedAsyncioTestCase):
    async def test_numbered_sources_bind_both_sides_without_model_copying_quotes(self):
        payload = dict(relation='coexists', same_slot=False, evidence_refs=['old', 'new'], reason='two events')
        llm = LLM(payload)
        result = await StateFactRelationJudge(llm).judge(fact('Old original.', 'F1'), fact('New original.'))
        self.assertEqual(result['relation'], 'coexists')
        self.assertEqual(result['evidence_quote'], 'New original.')
        self.assertEqual({r['side'] for r in result['evidence']}, {'old', 'new'})
        self.assertEqual(result['evaluation_status'], 'model_assessed')

    async def test_no_output_cap_and_truncation_is_technical_failure(self):
        class Truncated:
            async def generate(self, prompt, **kwargs):
                self.options = kwargs
                kwargs.get('metadata_sink', {}).update(finish_reason='length')
                return '{"relation":'
        llm = Truncated()
        result = await StateFactRelationJudge(llm).judge(fact('old', 'F1'), fact('new'))
        self.assertIsNone(llm.options['max_tokens'])
        self.assertEqual(result['evaluation_status'], 'technical_error')
        self.assertEqual(result['failure_kind'], 'truncated_output')

    async def test_source_selection_cannot_omit_new_fact_for_non_uncertain_relation(self):
        result = await self.judge('New original.', dict(relation='replacement', same_slot=True,
            evidence_refs=['old'], reason='unsupported transition'))
        self.assertEqual(result['relation'], 'uncertain')
        self.assertEqual(result['evaluation_status'], 'technical_error')

    async def judge(self, quote, payload):
        old = fact('We had a blast last year at the Pride fest.', fact_id='F1', value='Pride fest')
        new = fact(quote)
        return await StateFactRelationJudge(LLM(payload)).judge(old, new)

    async def test_wrapped_single_quote_is_canonicalized_to_source_span(self):
        source = ('Gonna continue my edu and check out career options, '
                  'which is pretty exciting!')
        result = await self.judge(source, response("'" + source + "'"))
        self.assertEqual(result['relation'], 'unrelated')
        self.assertEqual(result['evidence_quote'], source)
        self.assertEqual(result['quote_match'], 'stripped_quotes')
        self.assertEqual(result['model_evidence_quote'], "'" + source + "'")

    async def test_typographic_quotes_and_whitespace_are_accepted(self):
        source = 'I ran a charity race last Saturday – it was really rewarding.'
        wrapped = '“I ran a charity race last Saturday –\n it was really rewarding.”'
        result = await self.judge(source, response(wrapped, relation='complement'))
        self.assertEqual(result['relation'], 'complement')
        self.assertEqual(result['evidence_quote'], source)
        self.assertIn('stripped_quotes', result['quote_match'])
        self.assertIn('whitespace', result['quote_match'])

    async def test_leading_colon_is_accepted(self):
        source = "Researching adoption agencies — it's been a dream to have a family."
        result = await self.judge(source, response(':' + source))
        self.assertEqual(result['evidence_quote'], source)
        self.assertEqual(result['quote_match'], 'stripped_prefix')

    async def test_paraphrase_is_rejected_and_raw_output_is_kept(self):
        source = "I'm starting to realize that self-care is really important."
        paraphrase = 'I ran a charity race for mental health last Saturday – it was really rewarding.'
        result = await self.judge(source, response(paraphrase, relation='complement'))
        self.assertEqual(result['relation'], 'uncertain')
        self.assertIn('缺少当前事实原文依据', result['validation_error'])
        self.assertEqual(result['model_relation'], 'complement')
        self.assertEqual(result['model_evidence_quote'], paraphrase)
        self.assertIn('charity race', result['raw_response'])
        self.assertIsNone(result['quote_match'])
        self.assertFalse(result['raw_response_truncated'])

    async def test_quote_from_old_fact_is_rejected(self):
        source = 'I went to a poetry reading last Fri - it was really powerful!'
        old_quote = 'We had a blast last year at the Pride fest.'
        result = await self.judge(source, response(old_quote))
        self.assertEqual(result['relation'], 'uncertain')
        self.assertEqual(result['model_evidence_quote'], old_quote)

    async def test_word_boundary_prevents_partial_token_match(self):
        source = 'The said hello to everyone.'
        result = await self.judge(source, response('he said'))
        self.assertEqual(result['relation'], 'uncertain')

    async def test_empty_quote_is_rejected(self):
        source = 'I went to a poetry reading last Fri - it was really powerful!'
        result = await self.judge(source, response(''))
        self.assertEqual(result['relation'], 'uncertain')
        self.assertIn('缺少当前事实原文依据', result['validation_error'])

    async def test_non_boolean_same_slot_is_rejected(self):
        source = 'I went to a poetry reading last Fri - it was really powerful!'
        payload = response(source)
        payload['same_slot'] = 'false'
        result = await self.judge(source, payload)
        self.assertEqual(result['relation'], 'uncertain')
        self.assertIn('same_slot 不是布尔值', result['validation_error'])
        self.assertIn('poetry reading', result['raw_response'])

    async def test_invalid_json_is_rejected_with_raw_response(self):
        source = 'I went to a poetry reading last Fri - it was really powerful!'
        result = await self.judge(source, 'not json')
        self.assertEqual(result['relation'], 'uncertain')
        self.assertEqual(result['raw_response'], 'not json')
        self.assertIsNone(result['model_relation'])

    async def test_prompt_requests_verbatim_quote_without_wrapping(self):
        source = 'I went to a poetry reading last Fri - it was really powerful!'
        llm = LLM(response(source))
        await StateFactRelationJudge(llm).judge(
            fact('We had a blast last year at the Pride fest.', fact_id='F1'), fact(source))
        self.assertIn('不要加引号', llm.prompts[0])
        self.assertIn('直接逐字复制', llm.prompts[0])

    async def test_cancelled_error_propagates(self):
        class CancelledLLM:
            async def generate(self, prompt, **kwargs):
                raise asyncio.CancelledError()

        with self.assertRaises(asyncio.CancelledError):
            await StateFactRelationJudge(CancelledLLM()).judge(
                fact('old', fact_id='F1'), fact('new'))


if __name__ == '__main__':
    unittest.main()
