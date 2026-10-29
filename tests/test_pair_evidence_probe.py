"""实验评估回归检查：解释对照固定两端，评分不能把检索到误当作引用到。"""
from pathlib import Path
import runpy
import unittest

api = runpy.run_path(str(Path(__file__).resolve().parents[1] /
    'experiments/integrated_event_state/pair_evidence_probe.py'))


class ProbeScoringTests(unittest.TestCase):
    def test_retrieval_failure_is_not_correct_abstention(self):
        case = next(c for c in api['cases']() if c['case_id'] == 'appointment_unexplained')
        result = dict(status='DEFER', stop_reason='retrieval_failed', evidence=[], rounds=[])
        score = api['score_case'](case, result)
        self.assertTrue(score['technical_failure'])
        self.assertFalse(score['joint_pass'])

    def test_appointment_controls_have_identical_endpoints(self):
        controls = [c for c in api['cases']() if c['case_id'].startswith('appointment_')]
        self.assertEqual(len(controls), 3)
        for fid in ('F1', 'F2'):
            self.assertTrue(all(c['observations'][fid] == controls[0]['observations'][fid] for c in controls))
        self.assertEqual({c['reference']['relations'][0] for c in controls},
                         {'replacement', 'correction', 'uncertain'})

    def test_right_relation_without_explanation_citation_is_not_success(self):
        case = next(c for c in api['cases']() if c['case_id'] == 'appointment_rescheduled')
        shown = [dict(chunk_id=cid, text=c['text'], fact_ids=[fid])
                 for fid, cid in [('F1', 'C1'), ('F2', 'C2')]
                 for c in [case['chunks'][cid]]]
        result = dict(status='SUFFICIENT', stop_reason='judge_sufficient',
            evidence=[dict(chunk_id=cid) for cid in ('C1', 'C2', 'CB')],
            rounds=[dict(relation='replacement', shown_evidence=shown, citations=[
                dict(chunk_id=cid, fact_id=fid, quote=case['chunks'][cid]['text'])
                for fid, cid in [('F1', 'C1'), ('F2', 'C2')]])])
        score = api['score_case'](case, result)
        self.assertTrue(score['relation_correct'])
        self.assertTrue(score['explanation_retrieved'])
        self.assertFalse(score['explanation_shown'])
        self.assertFalse(score['explanation_cited'])
        self.assertFalse(score['joint_pass'])


if __name__ == '__main__':
    unittest.main()
