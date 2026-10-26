"""用可控模型输出验证事件身份、独立事件和局部断言纠正。"""

import json
import unittest
from types import SimpleNamespace

from coregraph.event_state_layer import EventStateLayer
from coregraph.event_state_writer import EventStateWriter
from experiments.integrated_event_state.query import query_with_event_state


class EventLLM:
    """固定抽取和身份判断，只测试程序是否正确连接与保存证据。"""

    def __init__(self):
        self.identity_calls = 0

    async def generate(self, prompt, **kwargs):
        if "EVENT_IDENTITY_GATE" in prompt:
            self.identity_calls += 1
            current = prompt.split("当前完整原文：", 1)[1]
            school = "school event" in current
            repeated_visit = "again on May 14" in current
            corrected_time = "May 6" in current
            corrected_place = "at the Library" in current
            return json.dumps({
                "same_event": not (school or repeated_visit),
                "date_correction": corrected_time,
                "corrected_attributes": ["location"] if corrected_place else [],
                "evidence_quote": current.split("\n", 1)[0],
                "reason": "test-controlled same-event judgment",
            })
        current = prompt.split("原文：", 1)[1]
        if "again on May 14" in current:
            value, time, attributes = "LGBTQ support group", "May 14", {}
            quote = "I attended the LGBTQ support group again on May 14."
        elif "school event" in current:
            value, time, attributes = "school event", "last Friday", {}
            quote = "I attended a school event last Friday."
        elif "May 6" in current:
            value, time, attributes = "LGBTQ support group", "May 6", {}
            quote = "I actually went to that LGBTQ support group on May 6, not May 7."
        elif "at the Library" in current:
            value, time, attributes = "LGBTQ support group", None, {"location": "Library"}
            quote = "It was at the Library, not the Community Center, for that LGBTQ support group."
        elif "That LGBTQ support group" in current:
            value, time, attributes = "LGBTQ support group", None, {}
            quote = "That LGBTQ support group I attended was helpful."
        else:
            value, time, attributes = "LGBTQ support group", "yesterday", {"location": "Community Center"}
            quote = "I attended the LGBTQ support group at the Community Center yesterday."
        return json.dumps({"facts": [{
            "kind": "state", "fact_type": "event", "entity": "Caroline",
            "attribute": "attended", "value": value, "quote": quote,
            "scope": {}, "modality": "actual", "event_time": time,
            "event_attributes": attributes, "triple_indices": [],
        }]})


def chunk(number, day, text):
    return {"chunk_id": str(number), "session_id": f"D{number}",
            "session_time": day, "text": "Caroline: " + text}


def relations():
    return [
        {"src": "Caroline", "relation": "attended", "tgt": "LGBTQ support group", "chunk_id": "1"},
        {"src": "Caroline", "relation": "attended on", "tgt": "2023-05-07", "chunk_id": "1"},
        {"src": "Caroline", "relation": "attended", "tgt": "LGBTQ support group", "chunk_id": "2"},
        {"src": "Caroline", "relation": "attended", "tgt": "school event", "chunk_id": "3"},
        {"src": "Caroline", "relation": "attended", "tgt": "LGBTQ support group", "chunk_id": "4"},
        {"src": "Caroline", "relation": "attended on", "tgt": "2023-05-06", "chunk_id": "4"},
        {"src": "Caroline", "relation": "attended", "tgt": "LGBTQ support group", "chunk_id": "5"},
        {"src": "Caroline", "relation": "attended", "tgt": "LGBTQ support group", "chunk_id": "6"},
        {"src": "Caroline", "relation": "attended on", "tgt": "2023-05-14", "chunk_id": "6"},
    ]


class EventInstanceTests(unittest.IsolatedAsyncioTestCase):
    async def build(self, chunks):
        model = EventLLM()
        layer = await EventStateWriter(model).write_chunks(
            EventStateLayer(), chunks, relations(),
            {"Caroline": 0, "LGBTQ support group": 1, "school event": 2})
        return layer, model

    async def test_same_event_in_two_chunks_keeps_one_anchor(self):
        layer, model = await self.build([
            chunk(1, "2023-05-08", "I attended the LGBTQ support group at the Community Center yesterday."),
            chunk(2, "2023-05-09", "That LGBTQ support group I attended was helpful."),
        ])
        self.assertEqual(len(layer.event_instances), 1)
        event = next(iter(layer.event_instances.values()))
        self.assertEqual(event["evidence_chunk_ids"], ["1", "2"])
        self.assertEqual(event["event_time"], "2023-05-07")
        self.assertGreaterEqual(model.identity_calls, 1)

    async def test_two_activities_stay_separate(self):
        layer, _ = await self.build([
            chunk(1, "2023-05-08", "I attended the LGBTQ support group at the Community Center yesterday."),
            chunk(3, "2023-05-15", "I attended a school event last Friday."),
        ])
        self.assertEqual(len(layer.event_instances), 2)
        self.assertEqual({event["target"] for event in layer.event_instances.values()},
                         {"LGBTQ support group", "school event"})
        self.assertEqual(layer.events, {})

    async def test_repeated_same_activity_still_gets_distinct_event(self):
        layer, _ = await self.build([
            chunk(1, "2023-05-08", "I attended the LGBTQ support group at the Community Center yesterday."),
            chunk(6, "2023-05-15", "I attended the LGBTQ support group again on May 14."),
        ])
        self.assertEqual(len(layer.event_instances), 2)
        self.assertEqual({event["event_time"] for event in layer.event_instances.values()},
                         {"2023-05-07", "2023-05-14"})

    async def test_time_correction_changes_one_assertion_and_keeps_history(self):
        layer, _ = await self.build([
            chunk(1, "2023-05-08", "I attended the LGBTQ support group at the Community Center yesterday."),
            chunk(4, "2023-05-20", "I actually went to that LGBTQ support group on May 6, not May 7."),
        ])
        self.assertEqual(len(layer.event_instances), 1)
        event = next(iter(layer.event_instances.values()))
        self.assertEqual(event["event_time"], "2023-05-06")
        assertions = list(layer.event_time_assertions.values())
        self.assertEqual({row["status"] for row in assertions}, {"active", "corrected"})
        old = next(row for row in assertions if row["status"] == "corrected")
        new = next(row for row in assertions if row["status"] == "active")
        self.assertEqual(new["corrects_assertion_id"], old["assertion_id"])
        self.assertFalse(layer.is_superseded_event_time_triple(old["source_triples"][0], "2023-05-09"))
        self.assertTrue(layer.is_superseded_event_time_triple(old["source_triples"][0], "2023-05-21"))

    async def test_location_correction_changes_only_location_and_keeps_history(self):
        layer, _ = await self.build([
            chunk(1, "2023-05-08", "I attended the LGBTQ support group at the Community Center yesterday."),
            chunk(5, "2023-05-21", "It was at the Library, not the Community Center, for that LGBTQ support group."),
        ])
        self.assertEqual(len(layer.event_instances), 1)
        event = next(iter(layer.event_instances.values()))
        self.assertEqual(event["event_time"], "2023-05-07")
        current = layer.event_attribute_assertions[event["current_attributes"]["location"]]
        self.assertEqual(current["value"], "Library")
        old = next(row for row in layer.event_attribute_assertions.values()
                   if row["value"] == "Community Center")
        self.assertEqual(old["status"], "corrected")
        self.assertEqual(current["corrects_assertion_id"], old["assertion_id"])

    async def test_query_before_and_after_correction_uses_visible_date(self):
        layer, _ = await self.build([
            chunk(1, "2023-05-08", "I attended the LGBTQ support group at the Community Center yesterday."),
            chunk(4, "2023-05-20", "I actually went to that LGBTQ support group on May 6, not May 7."),
            chunk(5, "2023-05-21", "It was at the Library, not the Community Center, for that LGBTQ support group."),
        ])

        class Retriever:
            async def retrieve(self, query, top_k=None):
                return {"query_analysis": {"entities": ["Caroline"]}, "evidence": [{
                    "chunk_id": "1", "session_id": "D1", "session_time": "2023-05-08",
                    "text": "Caroline: I attended the LGBTQ support group yesterday.",
                    "triples": [{"subject": "Caroline", "predicate": "attended on",
                                 "object": "2023-05-07"}],
                }]}

        class AnswerLLM:
            async def generate(self, prompt):
                return "checked"

        memory = SimpleNamespace(event_state_layer=layer)
        before = await query_with_event_state("When did Caroline attend the group?",
                                              "2023-05-09", memory, Retriever(), AnswerLLM())
        after = await query_with_event_state("When did Caroline attend the group?",
                                             "2023-05-22", memory, Retriever(), AnswerLLM())
        self.assertEqual(len(before["evidence"][0]["triples"]), 1)
        self.assertEqual(before["event_instance_evidence"][0]["event_time"], "2023-05-07")
        self.assertEqual(before["event_instance_evidence"][0]["attributes"]["location"],
                         "Community Center")
        self.assertEqual(after["evidence"][0]["triples"], [])
        self.assertEqual(after["event_instance_evidence"][0]["event_time"], "2023-05-06")
        self.assertEqual(after["event_instance_evidence"][0]["attributes"]["location"],
                         "Library")

    async def test_time_word_cannot_become_event_target(self):
        source = chunk(1, "2023-05-08", "I attended the LGBTQ support group yesterday.")
        item = {"entity": "Caroline", "attribute": "attended", "value": "yesterday",
                "kind": "state", "fact_type": "event", "modality": "actual",
                "quote": "I attended the LGBTQ support group yesterday.",
                "event_time": "yesterday", "triple_indices": []}
        with self.assertRaisesRegex(ValueError, "事件客体不能是时间表达"):
            EventStateWriter._validate(item, source, ["Caroline"],
                                        [{"src": "Caroline", "relation": "attended",
                                          "tgt": "LGBTQ support group", "chunk_id": "1"}],
                                        {"Caroline": 0}, EventStateLayer())

    async def test_single_speaker_event_recovers_missing_quote_from_source(self):
        source = chunk(4, "2023-05-20",
                       "I actually went to that LGBTQ support group on May 6, not May 7.")
        item = {"entity": "Caroline", "attribute": "attended",
                "value": "LGBTQ support group", "kind": "correction",
                "fact_type": "event", "modality": "actual", "event_time": "May 6",
                "event_attributes": {"location": "LGBTQ support group"},
                "triple_indices": []}
        proposal = EventStateWriter._validate(
            item, source, ["Caroline"],
            [{"src": "Caroline", "relation": "attended",
              "tgt": "LGBTQ support group", "chunk_id": "4"}],
            {"Caroline": 0}, EventStateLayer())
        self.assertEqual(proposal["event_time"], "2023-05-06")
        self.assertEqual(proposal["event_attributes"], {})
        self.assertIn("not May 7", proposal["quote"])

    async def test_other_event_attribute_corrects_without_changing_location(self):
        layer = EventStateLayer()
        first = {"entity_id": "Caroline", "entity_index": 0,
                 "attribute": "attended", "value": "LGBTQ support group",
                 "fact_type": "event", "kind": "state", "scope": {},
                 "modality": "actual", "event_type": "attendance",
                 "event_group_id": None, "session_id": "D1", "chunk_id": "1",
                 "observed_at": "2023-05-08", "event_time": "2023-05-07",
                 "quote": "Pat organized the LGBTQ support group at the Community Center.",
                 "triple_refs": [], "event_triple_refs": [], "time_triple_refs": [],
                 "event_attributes": {"organizer": "Pat", "location": "Community Center"}}
        fact_id = layer.apply_decision(first, "ADD")
        event_id = layer.add_event_instance(fact_id, first)
        correction = {**first, "chunk_id": "2", "session_id": "D2",
                      "observed_at": "2023-05-21", "event_time": None,
                      "quote": "Actually Alex organized the LGBTQ support group, not Pat.",
                      "event_attributes": {"organizer": "Alex"}}
        changed = layer.update_event_attributes(event_id, correction, {"organizer"})
        event = layer.event_instances[event_id]
        location = layer.event_attribute_assertions[event["current_attributes"]["location"]]
        organizer = layer.event_attribute_assertions[event["current_attributes"]["organizer"]]
        self.assertEqual(location["value"], "Community Center")
        self.assertEqual(organizer["value"], "Alex")
        self.assertEqual(changed[0]["operation"], "CORRECT")
        self.assertEqual(layer.event_attribute_assertions[
            organizer["corrects_assertion_id"]]["status"], "corrected")


if __name__ == "__main__":
    unittest.main()
