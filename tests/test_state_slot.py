"""状态槽位的基数、作用域和模态边界。"""

import unittest

from coregraph.state_slot import (cardinality_for, normalize_scope, slot_key,
                                  preliminary_operation, temporal_gate)


class StateSlotTests(unittest.TestCase):
    def test_owns_pet_is_a_set(self):
        self.assertEqual(cardinality_for("owns_pet"), "set")
        old = {"entity_id": "User", "attribute": "owns_pet", "value": "Dog",
               "scope": {}, "modality": "actual"}
        new = {**old, "value": "Cat"}
        self.assertEqual(preliminary_operation(old, new), "ADD")

    def test_different_projects_are_different_slots(self):
        old = {"entity_id": "User", "attribute": "current_role", "value": "Researcher",
               "scope": {"project": "A"}, "modality": "actual"}
        new = {**old, "value": "Reviewer", "scope": {"project": "B"}}
        self.assertNotEqual(slot_key(old), slot_key(new))
        self.assertEqual(preliminary_operation(old, new), "ADD")

    def test_plan_and_actual_are_not_replacements(self):
        old = {"entity_id": "User", "attribute": "current_employer", "value": "B Co",
               "scope": {}, "modality": "planned"}
        new = {**old, "value": "C Co", "modality": "actual"}
        self.assertEqual(preliminary_operation(old, new), "ADD")

    def test_single_value_change_requires_gate(self):
        old = {"entity_id": "User", "attribute": "current_employer", "value": "B Co",
               "scope": {}, "modality": "actual"}
        new = {**old, "value": "C Co"}
        self.assertEqual(cardinality_for("current_employer"), "single")
        self.assertEqual(preliminary_operation(old, new), "GATE")

    def test_more_specific_value_is_refinement(self):
        old = {"entity_id": "User", "attribute": "current_role", "value": "Engineer",
               "scope": {"project": "A"}, "modality": "actual"}
        new = {**old, "value": "Software Engineer"}
        self.assertEqual(preliminary_operation(old, new), "REFINE")

    def test_unknown_attribute_defaults_to_set(self):
        self.assertEqual(cardinality_for("associated_with"), "set")

    def test_scope_value_must_come_from_evidence(self):
        self.assertEqual(normalize_scope({"project": "Apollo"}, "I work on Apollo"),
                         {"project": "Apollo"})
        with self.assertRaises(ValueError):
            normalize_scope({"project": "Apollo"}, "I work on Gemini")

    def test_time_gate_rejects_past_recollection_as_current_replacement(self):
        old = {"modality": "actual", "observed_at": "2025-04-01",
               "event_time": "2025-03-01"}
        new = {"modality": "actual", "observed_at": "2025-05-01",
               "event_time": "2024-09-01"}
        self.assertFalse(temporal_gate(old, new)[0])

    def test_time_gate_does_not_require_an_explicit_date(self):
        old = {"modality": "actual", "observed_at": "2025-04-01", "event_time": None}
        new = {"modality": "actual", "observed_at": "2025-05-01", "event_time": None}
        self.assertTrue(temporal_gate(old, new)[0])

    def test_time_gate_uses_old_event_time_before_observation_time(self):
        old = {"modality": "actual", "observed_at": "2025-03-01",
               "event_time": "2025-01-01"}
        new = {"modality": "actual", "observed_at": "2025-04-01",
               "event_time": "2025-02-01"}
        self.assertTrue(temporal_gate(old, new)[0])

    def test_time_gate_separates_planned_and_actual(self):
        old = {"modality": "planned", "observed_at": "2025-04-01", "event_time": None}
        new = {"modality": "actual", "observed_at": "2025-05-01", "event_time": None}
        self.assertFalse(temporal_gate(old, new)[0])


if __name__ == "__main__":
    unittest.main()
