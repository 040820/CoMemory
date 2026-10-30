"""双数据集 pilot 的纯函数边界回归；不加载 GPU，不调用真实模型。"""
import importlib
import unittest


class PilotTests(unittest.TestCase):
    def test_selection_is_reproducible_and_does_not_depend_on_answers(self):
        api = importlib.import_module('experiments.cross_dataset_pilot.selection')
        qa = [dict(question=f'q{i}', answer='secret', category=category, evidence=['D1:1'])
              for category in (1, 2, 4, 5) for i in range(category * 10, category * 10 + 6)]
        native = [dict(sample_id='sample', qa=qa, conversation={'session_1': [dict(dia_id='D1:1', text='Text')]})]
        a = api.select_locomo(native)
        for q in qa:q['answer'] = 'changed secret'
        b = api.select_locomo(native)
        self.assertEqual([x['case_id'] for x in a], [x['case_id'] for x in b])
        self.assertEqual(len(a), 14)

    def test_longmem_conversion_keeps_all_sessions_and_full_time_precision(self):
        api = importlib.import_module('experiments.cross_dataset_pilot.selection')
        row = dict(haystack_dates=['2023/05/20 (Sat) 02:57', '2023/05/20 (Sat) 02:21'],
            haystack_session_ids=['later', 'earlier'], haystack_sessions=[
                [dict(role='user', content='later message')], [dict(role='user', content='earlier message')]])
        corpus = api.longmem_corpus(row)
        self.assertEqual([x['session_id'] for x in corpus], ['earlier', 'later'])
        self.assertTrue(corpus[0]['session_time'].endswith('02:21:00'))
        self.assertIn('later message', corpus[1]['context'])

    def test_unequal_parallel_arrays_are_rejected_instead_of_silently_dropping_history(self):
        api = importlib.import_module('experiments.cross_dataset_pilot.selection')
        with self.assertRaises(ValueError):
            api.longmem_corpus(dict(haystack_dates=['2024/01/01'], haystack_session_ids=[], haystack_sessions=[]))

    def test_recall_counts_only_the_actual_stage_and_deduplicates_sessions(self):
        api = importlib.import_module('experiments.cross_dataset_pilot.metrics')
        source = dict(chunk_storage={'C1': dict(session_id='D1'), 'C2': dict(session_id='D2')})
        report = dict(query_seed_trace={'chunk_candidates': ['C1']},
            evidence=[dict(chunk_id='C1', session_id='D1'), dict(chunk_id='C2', session_id='D2')],
            rounds=[dict(shown_evidence=[dict(chunk_id='C1', session_id='D1')])],
            answer_citations=[dict(chunk_id='C1'), dict(chunk_id='C1')])
        value = api.source_coverage(report, ['D1', 'D2'], source)
        self.assertEqual(value['candidate']['recall'], 1.0)
        self.assertEqual(value['shown']['recall'], 0.5)
        self.assertEqual(value['cited']['recall'], 0.5)
        self.assertIsNone(api.source_coverage(report, [], source)['shown']['recall'])

    def test_technical_failure_is_not_successful_abstention(self):
        api = importlib.import_module('experiments.cross_dataset_pilot.metrics')
        result = api.abstention(dict(status='DEFER', evaluation_status='technical_error', answer=None))
        self.assertTrue(result['technical_failure'])
        self.assertFalse(result['semantic_abstention'])


if __name__ == '__main__':
    unittest.main()
