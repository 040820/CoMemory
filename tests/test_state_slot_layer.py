"""状态槽位写入操作与证据门控。"""

import unittest

from coregraph.event_state_layer import EventStateLayer


def fact(chunk: str, attribute: str, value: str | None,
         scope: dict[str, str] | None = None, modality: str = "actual") -> dict:
    quote = f"Alice: {attribute} is {value}." if value else f"Alice: never had {attribute}."
    return {"entity_id": "Alice", "entity_index": 0, "attribute": attribute,
            "value": value, "scope": scope or {}, "modality": modality,
            "quote": quote, "source_text": quote, "chunk_id": chunk,
            "session_id": "D" + chunk, "observed_at": f"2023-01-{int(chunk):02}",
            "event_time": None, "event_type": attribute + "_change",
            "event_group_id": None, "triple_refs": []}


def gate(evidence: str, **overrides) -> dict:
    result = {"same_slot": True, "incompatible": True,
              "supported_transition": True, "evidence_quote": evidence,
              "reason": "原文明确说明状态转换"}
    result.update(overrides)
    return result


class StateSlotLayerTests(unittest.TestCase):
    def test_multiple_pets_are_additions(self):
        layer = EventStateLayer()
        layer.apply_decision(fact("1", "owns_pet", "Dog"), "ADD")
        layer.apply_decision(fact("2", "owns_pet", "Cat"), "ADD")
        self.assertEqual(len(layer.active_members("Alice", "owns_pet")), 2)
        self.assertEqual(layer.events, {})

    def test_project_roles_and_modalities_do_not_collide(self):
        layer = EventStateLayer()
        layer.apply_decision(fact("1", "current_role", "Researcher", {"project": "A"}), "ADD")
        layer.apply_decision(fact("2", "current_role", "Reviewer", {"project": "B"}), "ADD")
        layer.apply_decision(fact("3", "current_employer", "B Co", modality="planned"), "ADD")
        layer.apply_decision(fact("4", "current_employer", "C Co"), "ADD")
        self.assertEqual(layer.current_fact("Alice", "current_role", {"project": "A"})["value"], "Researcher")
        self.assertEqual(layer.current_fact("Alice", "current_role", {"project": "B"})["value"], "Reviewer")
        self.assertEqual(layer.current_fact("Alice", "current_employer", modality="planned")["value"], "B Co")
        self.assertEqual(layer.current_fact("Alice", "current_employer")["value"], "C Co")
        self.assertEqual(layer.events, {})

    def test_replace_requires_three_gates_and_real_evidence(self):
        layer = EventStateLayer()
        old_id = layer.apply_decision(fact("1", "current_employer", "B Co"), "ADD")
        new = fact("2", "current_employer", "C Co")
        with self.assertRaises(ValueError):
            layer.apply_decision(new, "REPLACE", old_id, gate(new["quote"], incompatible=False))
        with self.assertRaises(ValueError):
            layer.apply_decision(new, "REPLACE", old_id, gate("invented quotation"))
        self.assertEqual(layer.current_fact("Alice", "current_employer")["value"], "B Co")
        layer.apply_decision(new, "REPLACE", old_id, gate(new["quote"]))
        self.assertEqual(layer.current_fact("Alice", "current_employer")["value"], "C Co")
        self.assertEqual(len(layer.events), 1)
        event = next(iter(layer.events.values()))
        self.assertEqual(event["operation"], "REPLACE")
        self.assertEqual(event["old_fact_id"], old_id)

    def test_single_slot_cannot_add_different_value_without_gate(self):
        layer = EventStateLayer()
        layer.apply_decision(fact("1", "current_employer", "B Co"), "ADD")
        with self.assertRaises(ValueError):
            layer.apply_decision(fact("2", "current_employer", "C Co"), "ADD")

    def test_refine_and_uncertain_do_not_create_change_event(self):
        layer = EventStateLayer()
        old = layer.apply_decision(fact("1", "current_role", "Engineer"), "ADD")
        layer.apply_decision(fact("2", "current_role", "Software Engineer"), "REFINE", old)
        layer.apply_decision(fact("3", "current_role", "Reviewer"), "UNCERTAIN", old)
        self.assertEqual(layer.current_fact("Alice", "current_role")["value"], "Software Engineer")
        self.assertEqual(layer.events, {})
        self.assertEqual(layer.decisions[-1]["operation"], "UNCERTAIN")

    def test_correction_revokes_dependent_change_without_inventing_chain(self):
        layer = EventStateLayer()
        old = layer.apply_decision(fact("1", "current_employer", "B Co"), "ADD")
        second = fact("2", "current_employer", "C Co")
        layer.apply_decision(second, "REPLACE", old, gate(second["quote"]))
        correction = fact("3", "current_employer", "A Co")
        layer.apply_decision(correction, "CORRECT", old, gate(correction["quote"]))
        self.assertEqual(layer.current_fact("Alice", "current_employer")["value"], "C Co")
        changes = [event for event in layer.events.values() if event["operation"] == "REPLACE"]
        self.assertEqual(len(changes), 1)
        self.assertEqual(changes[0]["status"], "revoked")
        self.assertEqual(len(layer.events), 2)

    def test_same_value_is_not_a_correction(self):
        layer = EventStateLayer()
        old = layer.apply_decision(fact("1", "current_employer", "B Co"), "ADD")
        repeated = fact("2", "current_employer", "B Co")
        with self.assertRaises(ValueError):
            layer.apply_decision(repeated, "CORRECT", old, gate(repeated["quote"]))
        self.assertEqual(layer.events, {})

    def test_serialization_keeps_slot_members_and_decisions(self):
        layer = EventStateLayer()
        layer.apply_decision(fact("1", "owns_pet", "Dog"), "ADD")
        restored = EventStateLayer(layer.to_dict())
        self.assertEqual(restored.active_members("Alice", "owns_pet")[0]["value"], "Dog")
        self.assertEqual(restored.decisions[0]["operation"], "ADD")

    def test_legacy_facts_are_readable_but_cannot_mix_with_new_slots(self):
        old = EventStateLayer({"version": 1, "facts": {"legacy": {"fact_id": "legacy"}},
                               "current": {}, "events": {}})
        self.assertEqual(old.facts["legacy"]["fact_id"], "legacy")
        with self.assertRaises(ValueError):
            old.apply_decision(fact("1", "owns_pet", "Dog"), "ADD")


if __name__ == "__main__":
    unittest.main()
