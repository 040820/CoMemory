"""真实对话暴露的证据、时间、事件身份和槽位回归测试。"""

import json
import unittest

from coregraph.event_instance_time import bind_event_time, parse_time_details
from coregraph.event_state_layer import EventStateLayer
from coregraph.event_state_writer import EventStateWriter, _event_targets_overlap
from coregraph.state_slot import canonical_attribute, cardinality_for
from coregraph.event_candidate_schema import resolve_indices, triple_id


class RepairingLLM:
    def __init__(self):
        self.calls = 0

    async def generate(self, prompt, **kwargs):
        self.calls += 1
        if self.calls == 1:
            return '{"facts": [{"entity": "Caroline", "attribute": "attended", "value": "LGBTQ support group", "kind": "state", "fact_type": "event", "modality": "actual"}]}'
        return json.dumps({"facts": [{"entity": "Caroline", "attribute": "attended",
            "value": "LGBTQ support group", "kind": "state", "fact_type": "event",
            "modality": "actual", "scope": {}, "quote": "I went to a LGBTQ support group yesterday.",
            "event_time": "yesterday", "event_attributes": {}, "triple_indices": [0]}]})


class LoCoMoUpperRepairTests(unittest.IsolatedAsyncioTestCase):
    async def test_old_writer_protocol_cannot_mix_with_new_alias_slots(self):
        layer = EventStateLayer({"version": 1, "facts": {"old": {}}, "slot_schema_version": 2})
        with self.assertRaisesRegex(ValueError, "独立目录重建"):
            await EventStateWriter(RepairingLLM()).write_chunks(layer, [], [], {})

    def test_reference_id_is_stable_and_rejects_foreign_chunk(self):
        ref = {"src": "Caroline", "relation": "attended", "tgt": "group", "chunk_id": "1"}
        self.assertEqual(resolve_indices({"triple_ids": [triple_id(ref)]}, [ref]), [0])
        foreign = dict(ref, chunk_id="2")
        with self.assertRaisesRegex(ValueError, "当前 Chunk"):
            resolve_indices({"triple_ids": [triple_id(foreign)]}, [ref])

    def test_legacy_object_reference_requires_exact_source_match(self):
        ref = {"src": "Caroline", "relation": "attended", "tgt": "group", "chunk_id": "1"}
        self.assertEqual(resolve_indices({"triple_indices": [dict(ref)]}, [ref]), [0])
        with self.assertRaisesRegex(ValueError, "原始来源"):
            resolve_indices({"triple_indices": [dict(ref, relation="invented")]}, [ref])

    async def test_missing_quote_gets_one_evidence_preserving_retry(self):
        model = RepairingLLM()
        layer = await EventStateWriter(model).write_chunks(EventStateLayer(), [{
            "chunk_id": "1", "session_id": "D1", "session_time": "2023-05-08",
            "text": "Caroline: I went to a LGBTQ support group yesterday."}], [{
            "src": "Caroline", "relation": "attended", "tgt": "LGBTQ support group",
            "chunk_id": "1"}], {"Caroline": 0, "LGBTQ support group": 1})
        self.assertEqual(model.calls, 2)
        self.assertEqual(len(layer.event_instances), 1)
        self.assertFalse(layer.review_queue)

    async def test_correction_can_retrieve_old_fact_without_inventing_fact_id(self):
        class Model:
            async def generate(self, prompt, **kwargs):
                quote = "Actually I work at C Co, not B Co."
                if "STATE_SLOT_GATE" in prompt:
                    return json.dumps({"same_slot": True, "incompatible": True,
                        "supported_transition": True, "operation": "CORRECT",
                        "evidence_quote": quote, "reason": "明确纠正"})
                correction = "Actually" in prompt.split("原文：", 1)[-1]
                return json.dumps({"facts": [{"entity": "Alice", "attribute": "employer",
                    "value": "C Co" if correction else "B Co",
                    "kind": "correction" if correction else "state",
                    "fact_type": "state", "modality": "actual", "scope": {},
                    "quote": quote if correction else "I work at B Co.",
                    "target_fact_id": None, "triple_indices": []}]})
        chunks = [{"chunk_id": "1", "session_id": "D1", "session_time": "2023-05-08",
                   "text": "Alice: I work at B Co."},
                  {"chunk_id": "2", "session_id": "D2", "session_time": "2023-05-09",
                   "text": "Alice: Actually I work at C Co, not B Co."}]
        layer = await EventStateWriter(Model()).write_chunks(
            EventStateLayer(), chunks, [], {"Alice": 0})
        self.assertEqual(layer.current_fact("Alice", "employer")["value"], "C Co")
        self.assertEqual({row["status"] for row in layer.facts.values()}, {"active", "corrected"})
        self.assertTrue(any(row["operation"] == "CORRECT" for row in layer.decisions))

    def test_dialogue_action_does_not_become_real_event(self):
        chunk = {"chunk_id": "2", "session_id": "D1", "session_time": "2023-05-08",
                 "text": "Melanie: What now?"}
        item = {"entity": "Melanie", "attribute": "asks", "value": "What now?",
                "kind": "state", "fact_type": "event", "modality": "actual",
                "quote": "What now?", "scope": {}, "triple_indices": []}
        with self.assertRaisesRegex(ValueError, "对话行为"):
            EventStateWriter._validate(item, chunk, ["Melanie"], [],
                                        {"Melanie": 0}, EventStateLayer())

    def test_location_requires_same_event_quote_and_location_phrase(self):
        chunk = {"chunk_id": "3", "session_id": "D1", "session_time": "2023-05-08",
                 "text": "Caroline: I attended the pride parade.\nMelanie: Next month is busy."}
        item = {"entity": "Caroline", "attribute": "attended", "value": "pride parade",
                "kind": "state", "fact_type": "event", "modality": "actual",
                "quote": "I attended the pride parade.", "scope": {},
                "event_attributes": {"location": "Next month"}, "triple_indices": []}
        layer = EventStateLayer()
        proposal = EventStateWriter._validate(item, chunk, ["Caroline", "Melanie"], [],
                                              {"Caroline": 0, "Melanie": 1}, layer)
        self.assertEqual(proposal["event_attributes"], {})
        self.assertEqual(proposal["value"], "pride parade")
        self.assertTrue(any("事件属性" in row["reason"] for row in layer.review_queue))

    def test_known_single_value_aliases_stay_in_one_slot(self):
        self.assertEqual(canonical_attribute("works_at"), "employer")
        self.assertEqual(canonical_attribute("current employer"), "employer")
        self.assertEqual(cardinality_for("works_at"), "single")
        self.assertEqual(cardinality_for("owns_pet"), "set")

    def test_group_membership_is_not_employment(self):
        quote = "Our group, Connected LGBTQ Activists, helps people."
        item = {"entity": "Caroline", "attribute": "employer", "value": "Connected LGBTQ Activists",
                "kind": "state", "fact_type": "state", "modality": "actual",
                "quote": quote, "scope": {}, "triple_ids": []}
        with self.assertRaisesRegex(ValueError, "就业关系"):
            EventStateWriter._validate(item, {"chunk_id": "100", "text": "Caroline: " + quote},
                ["Caroline"], [], {"Caroline": 0}, EventStateLayer())

    def test_question_predicate_variant_is_filtered(self):
        item = {"entity": "Melanie", "attribute": "asked_question", "value": "Did you join in?",
                "kind": "state", "fact_type": "event", "modality": "actual",
                "quote": "Did you join in?", "scope": {}, "triple_ids": []}
        with self.assertRaisesRegex(ValueError, "对话行为"):
            EventStateWriter._validate(item, {"chunk_id": "77", "text": "Melanie: Did you join in?"},
                ["Melanie"], [], {"Melanie": 0}, EventStateLayer())

    def test_photo_subject_is_not_automatically_the_speaker(self):
        item = {"entity": "Caroline", "attribute": "held", "value": "rainbow umbrella",
                "kind": "state", "fact_type": "event", "modality": "actual",
                "quote": "a woman holding a rainbow umbrella in the air", "scope": {}, "triple_ids": []}
        text = "Caroline: It was awesome. (attached is a photo of a woman holding a rainbow umbrella in the air)"
        with self.assertRaisesRegex(ValueError, "附件说明"):
            EventStateWriter._validate(item, {"chunk_id": "92", "text": text},
                ["Caroline"], [], {"Caroline": 0}, EventStateLayer())

    def test_event_evaluation_does_not_imply_the_speaker_felt_powerful(self):
        quote = "I went to a support group yesterday and it was so powerful."
        item = {"entity": "Caroline", "attribute": "felt", "value": "powerful",
                "kind": "state", "fact_type": "event", "modality": "actual",
                "quote": quote, "scope": {}, "triple_ids": []}
        with self.assertRaisesRegex(ValueError, "自身感受"):
            EventStateWriter._validate(item, {"chunk_id": "1", "text": "Caroline: " + quote},
                ["Caroline"], [], {"Caroline": 0}, EventStateLayer())

    def test_disjoint_event_targets_cannot_merge_through_shared_actor(self):
        self.assertFalse(_event_targets_overlap(
            {"actor": "Caroline", "target": "Mental Health", "source_triples": []},
            {"entity_id": "Caroline", "value": "LGBTQ activist group", "triple_refs": []}))

    def test_regular_meetings_are_a_habit_not_one_historical_event(self):
        quote = "We have regular meetings and campaigns."
        item = {"entity": "Caroline", "attribute": "attended", "value": "meetings",
                "kind": "state", "fact_type": "event", "modality": "actual",
                "quote": quote, "scope": {}, "triple_ids": []}
        proposal = EventStateWriter._validate(item, {"chunk_id": "100", "text": "Caroline: " + quote},
            ["Caroline"], [], {"Caroline": 0}, EventStateLayer())
        self.assertEqual(proposal["fact_type"], "state")

    def test_explicit_future_statement_cannot_become_actual_state(self):
        chunk = {"chunk_id": "4", "session_time": "2023-05-08",
                 "text": "Caroline: Gonna continue my edu."}
        item = {"entity": "Caroline", "attribute": "education", "value": "edu",
                "kind": "state", "fact_type": "state", "modality": "actual",
                "quote": "Gonna continue my edu.", "scope": {}, "triple_ids": []}
        proposal = EventStateWriter._validate(item, chunk, ["Caroline"], [],
                                              {"Caroline": 0}, EventStateLayer())
        self.assertEqual(proposal["modality"], "planned")

    def test_reported_external_conversation_is_not_filtered_as_current_chat(self):
        chunk = {"chunk_id": "8", "session_time": "2023-05-08",
                 "text": "Caroline: I talked to my doctor yesterday."}
        item = {"entity": "Caroline", "attribute": "talked_to", "value": "doctor",
                "kind": "state", "fact_type": "event", "modality": "actual",
                "quote": "I talked to my doctor yesterday.", "scope": {}, "triple_ids": [],
                "event_time": "yesterday"}
        proposal = EventStateWriter._validate(item, chunk, ["Caroline"], [],
                                              {"Caroline": 0}, EventStateLayer())
        self.assertEqual(proposal["fact_type"], "event")

    def test_temporal_ranges_do_not_become_single_dates(self):
        relative = parse_time_details("last night", "2023-05-08")
        self.assertEqual(relative["date"], "2023-05-07")
        week = parse_time_details("last week", "2023-05-08")
        self.assertIsNone(week["date"])
        self.assertEqual(week["range_start"], "2023-05-01")
        self.assertEqual(week["range_end"], "2023-05-07")
        summer = parse_time_details("this summer", "2023-05-08")
        self.assertIsNone(summer["date"])
        self.assertIsNone(summer["range_start"])
        self.assertEqual(summer["basis"], "unresolved_season")
        month_day = parse_time_details("July 3rd", "2023-08-01")
        self.assertEqual(month_day["date"], "2023-07-03")

    def test_range_is_not_overridden_by_middle_date_triple(self):
        activity = {"src": "Caroline", "relation": "attended",
                    "tgt": "support group", "chunk_id": "1"}
        dated = {"src": "Caroline", "relation": "attended on",
                 "tgt": "2023-05-03", "chunk_id": "1"}
        proposal = {"entity_id": "Caroline", "attribute": "attended",
                    "value": "support group", "event_time": "last week",
                    "observed_at": "2023-05-08", "triple_refs": [activity]}
        bind_event_time(proposal, [activity, dated])
        self.assertIsNone(proposal["event_time"])
        self.assertEqual(proposal["event_time_range"]["start"], "2023-05-01")
        self.assertEqual(proposal["time_triple_refs"], [])


if __name__ == "__main__":
    unittest.main()
