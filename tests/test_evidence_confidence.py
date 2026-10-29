"""证据置信度纯函数回归：Noisy-OR、来源可信度、独立性、带符号信念与阈值。"""

import importlib.util
from pathlib import Path
import sys
import unittest

MODULE_PATH = Path(__file__).resolve().parents[1] / 'coregraph' / 'evidence_confidence.py'
_spec = importlib.util.spec_from_file_location('_evidence_confidence_under_test', MODULE_PATH)
_module = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _module
_spec.loader.exec_module(_module)

EvidenceConfidenceConfig = _module.EvidenceConfidenceConfig
rank_strength = _module.rank_strength
route_confidence = _module.route_confidence
route_confidence_detail = _module.route_confidence_detail
source_credibility = _module.source_credibility
source_credibility_detail = _module.source_credibility_detail
combine_independent = _module.combine_independent
signed_belief = _module.signed_belief
bilateral_coverage = _module.bilateral_coverage
quote_credibility_match = _module.quote_credibility_match
evaluate_candidate = _module.evaluate_candidate


def credibility(**overrides):
    values = dict(has_source_trace=1.0, quote_match='exact', field_completeness=1.0,
                  source_type=1.0, alignment_confidence=1.0)
    values.update(overrides)
    return values


def item(role='supports', side='old', target='F1', routes=None, cred=None, group=None):
    return {
        'fact_id': f'{side}-fact', 'chunk_id': f'{side}-chunk', 'target_fact_id': target,
        'side': side, 'role': role, 'independence_group': group or f'{side}-chunk',
        'route_strengths': routes or {'structured': 1.0, 'bm25': 1.0},
        'credibility_features': cred or credibility(),
    }


class RouteConfidenceTests(unittest.TestCase):
    def test_single_and_multi_route_noisy_or(self):
        self.assertAlmostEqual(route_confidence({'structured': 1.0}), 0.35)
        self.assertAlmostEqual(
            route_confidence({'structured': 1.0, 'semantic': 1.0}), 0.48)
        self.assertAlmostEqual(
            route_confidence({'structured': 1.0, 'bm25': 1.0, 'semantic': 1.0}), 0.636)

    def test_unknown_route_is_ignored_and_strength_is_clipped(self):
        detail = route_confidence_detail({'unknown': 1.0, 'structured': 5.0})
        self.assertEqual(detail['ignored_routes'], ['unknown'])
        self.assertAlmostEqual(detail['score'], 0.35)
        self.assertAlmostEqual(route_confidence({'structured': 0.0}), 0.0)
        self.assertAlmostEqual(route_confidence({}), 0.0)

    def test_rank_strength(self):
        self.assertAlmostEqual(rank_strength(1, 5), 1.0)
        self.assertAlmostEqual(rank_strength(5, 5), 0.2)
        self.assertAlmostEqual(rank_strength(6, 5), 0.0)
        self.assertAlmostEqual(rank_strength(0, 5), 0.0)
        self.assertAlmostEqual(rank_strength(1, 0), 0.0)


class CredibilityTests(unittest.TestCase):
    def test_exact_inexact_and_missing_quote(self):
        self.assertAlmostEqual(source_credibility(credibility()), 0.95)
        self.assertAlmostEqual(
            source_credibility(credibility(quote_match='inexact')), 0.83)
        self.assertAlmostEqual(
            source_credibility(credibility(quote_match='none')), 0.75)

    def test_missing_features_clip_to_floor(self):
        self.assertAlmostEqual(source_credibility({}), 0.05)
        self.assertAlmostEqual(source_credibility(None), 0.05)

    def test_llm_self_confidence_is_zero_weight_by_default(self):
        base = source_credibility(credibility())
        with_llm = source_credibility(credibility(llm_self_confidence=1.0))
        self.assertAlmostEqual(with_llm, base)
        custom = dict(_module.DEFAULT_CREDIBILITY_WEIGHTS)
        custom['llm_self_confidence'] = 0.05
        self.assertAlmostEqual(source_credibility(credibility(llm_self_confidence=1.0),
                                                  custom), 0.98)

    def test_credibility_ceiling_is_applied(self):
        self.assertAlmostEqual(
            source_credibility(credibility(has_time=True, has_quantity=True)), 0.98)

    def test_quote_match_mapping(self):
        self.assertEqual(quote_credibility_match('exact'), 'exact')
        self.assertEqual(quote_credibility_match('stripped_quotes'), 'inexact')
        self.assertEqual(quote_credibility_match('stripped_quotes+whitespace'), 'inexact')
        self.assertEqual(quote_credibility_match(None), 'none')


class BeliefTests(unittest.TestCase):
    def test_same_independence_group_is_counted_once(self):
        self.assertAlmostEqual(combine_independent({'g1': [0.8, 0.9], 'g2': [0.5]}), 0.95)
        self.assertAlmostEqual(combine_independent({'g1': [0.8, 0.9]}), 0.9)
        self.assertAlmostEqual(combine_independent({}), 0.0)

    def test_signed_belief(self):
        result = signed_belief({'g1': [0.8], 'g2': [0.5]}, {'g3': [0.4]})
        self.assertAlmostEqual(result['belief'], 0.9)
        self.assertAlmostEqual(result['disbelief'], 0.4)
        self.assertAlmostEqual(result['conflict_mass'], 0.36)
        self.assertAlmostEqual(result['uncertainty'], 0.06)

    def test_bilateral_coverage_takes_minimum(self):
        result = bilateral_coverage({'old': {'g1': [0.9]}, 'new': {'g2': [0.5]}})
        self.assertAlmostEqual(result['coverage']['old'], 0.9)
        self.assertAlmostEqual(result['coverage']['new'], 0.5)
        self.assertAlmostEqual(result['bilateral'], 0.5)
        self.assertAlmostEqual(bilateral_coverage({'old': {'g1': [0.9]}})['bilateral'], 0.0)


class EvaluateCandidateTests(unittest.TestCase):
    def test_both_sides_supported_does_not_establish_relation(self):
        result = evaluate_candidate(
            [item(side='old', target='F1'), item(side='new', target='F2')],
            old_fact_ids={'F1'}, new_fact_ids={'F2'})
        self.assertEqual(result['status'], 'NEED_MORE')
        self.assertIsNone(result['belief'])
        self.assertAlmostEqual(result['targets']['F1']['belief'], 0.51775, places=6)
        self.assertAlmostEqual(result['bilateral'], 0.51775, places=6)
        self.assertFalse(result['update_applied'])

    def test_only_one_side_is_need_more(self):
        result = evaluate_candidate(
            [item(side='new', target='F2')],
            old_fact_ids={'F1'}, new_fact_ids={'F2'})
        self.assertEqual(result['status'], 'NEED_MORE')
        self.assertEqual(result['bilateral'], 0)
        self.assertIn('relation_not_evaluated', result['reasons'])

    def test_support_and_refute_of_different_targets_are_not_a_conflict(self):
        routes = {'structured': 1.0, 'bm25': 1.0, 'semantic': 1.0}
        result = evaluate_candidate(
            [item(side='new', target='F2', routes=routes, cred=credibility()),
             item(role='refutes', side='old', target='F1', routes=routes,
                  cred=credibility())],
            old_fact_ids={'F1'}, new_fact_ids={'F2'})
        self.assertEqual(result['status'], 'NEED_MORE')
        self.assertIsNone(result['conflict_mass'])
        self.assertEqual(result['targets']['F1']['conflict_mass'], 0)
        self.assertEqual(result['targets']['F2']['conflict_mass'], 0)

    def test_background_and_uncertain_do_not_count(self):
        result = evaluate_candidate(
            [item(side='old', target='F1'), item(side='new', target='F2'),
             item(role='background', side='old', target='F1', routes={'structured': 1.0}),
             item(role='uncertain', side='new', target='F2', routes={'structured': 1.0})],
            old_fact_ids={'F1'}, new_fact_ids={'F2'})
        self.assertEqual(result['status'], 'NEED_MORE')
        self.assertAlmostEqual(result['targets']['F1']['belief'], 0.51775, places=6)

    def test_question_mode_does_not_require_bilateral(self):
        routes = {'structured': 1.0, 'bm25': 1.0, 'semantic': 1.0}
        result = evaluate_candidate(
            [item(side='question', target=None, routes=routes, group='q1'),
             item(side='question', target=None, routes=routes, group='q2')],
            mode='question')
        self.assertEqual(result['status'], 'SUFFICIENT')
        self.assertIsNone(result['bilateral'])

    def test_no_evidence(self):
        result = evaluate_candidate([], old_fact_ids={'F1'}, new_fact_ids={'F2'})
        self.assertEqual(result['status'], 'NOT_EVALUATED')
        self.assertIn('no_evidence', result['reasons'])

    def test_custom_thresholds_are_used(self):
        config = EvidenceConfidenceConfig(thresholds={'sufficient_belief': 0.40,
                                                      'refute_ceiling': 0.25,
                                                      'uncertainty_ceiling': 0.50,
                                                      'conflict_mass': 0.35,
                                                      'bilateral_floor': 0.50,
                                                      'uncertainty_defer': 0.50})
        result = evaluate_candidate(
            [item(side='old', target='F1'), item(side='new', target='F2'),
             item(target='relation', side='question')],
            config, old_fact_ids={'F1'}, new_fact_ids={'F2'})
        self.assertEqual(result['status'], 'SUFFICIENT')

    def test_unknown_role_and_missing_sides_raise(self):
        with self.assertRaises(ValueError):
            evaluate_candidate([item(role='invented')], old_fact_ids={'F1'},
                               new_fact_ids={'F2'})
        with self.assertRaises(ValueError):
            evaluate_candidate([item()], old_fact_ids={'F1'})

    def test_invalid_config_is_rejected(self):
        with self.assertRaises(ValueError):
            EvidenceConfidenceConfig(route_weights={'structured': 1.5})
        with self.assertRaises(ValueError):
            EvidenceConfidenceConfig(role_weights={'supports': 1.0})


if __name__ == '__main__':
    unittest.main()
