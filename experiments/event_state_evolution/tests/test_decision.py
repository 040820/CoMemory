import json
import unittest

from experiments.event_state_evolution.decision import classify
from experiments.event_state_evolution.evidence import EvidenceCandidate


class FakeLLM:
    def __init__(self, payload):
        self.payload = payload

    async def generate(self, prompt):
        return json.dumps(self.payload) if isinstance(self.payload, dict) else self.payload


def candidate(text="Alice now works at C Co.", subject="Alice", object_="C Co"):
    return EvidenceCandidate("key", "2", "D2", "2023-02-01", text, subject,
                             "works at" if subject else None, object_, ("Alice", "B Co", "C Co"))


class DecisionTests(unittest.IsolatedAsyncioTestCase):
    async def test_accepts_supported_implicit_change(self):
        row = candidate()
        llm = FakeLLM({"kind": "change", "entity": "Alice", "attribute": "employer",
                       "value": "C Co", "evidence_quote": "Alice now works at C Co.",
                       "triple": ["Alice", "works at", "C Co"], "detection": "inferred",
                       "event_time": None})
        decision = await classify(row, {"Alice": {"employer": {"value": "B Co", "fact_id": "f1"}}}, llm)
        self.assertEqual((decision.kind, decision.entity_id, decision.value, decision.detection),
                         ("change", "Alice", "C Co", "inferred"))

    async def test_rejects_graph_edge_not_supported_by_chunk(self):
        row = candidate(text="Alice joined a book club.")
        llm = FakeLLM({"kind": "change", "entity": "Alice", "attribute": "employer",
                       "value": "C Co", "evidence_quote": "Alice joined a book club.",
                       "triple": ["Alice", "works at", "C Co"]})
        self.assertEqual((await classify(row, {}, llm)).kind, "review")

    async def test_drops_fabricated_date_and_rejects_unsupported_entity(self):
        row = candidate()
        payload = {"kind": "baseline", "entity": "Alice", "attribute": "employer",
                   "value": "C Co", "evidence_quote": row.text,
                   "triple": ["Alice", "works at", "C Co"], "event_time": "2022-10-10"}
        first = await classify(row, {}, FakeLLM(payload))
        self.assertEqual((first.kind, first.event_time), ("baseline", None))
        payload["event_time"] = None
        payload["entity"] = "Bob"
        self.assertEqual((await classify(row, {}, FakeLLM(payload))).kind, "review")

    async def test_empty_response_and_retrospective_do_not_update(self):
        self.assertEqual((await classify(candidate(), {}, FakeLLM(""))).kind, "review")
        row = candidate(text="Alice previously worked at B Co.", object_="B Co")
        llm = FakeLLM({"kind": "retrospective", "entity": "Alice", "attribute": "employer",
                       "value": "B Co", "evidence_quote": row.text,
                       "triple": ["Alice", "worked at", "B Co"]})
        self.assertEqual((await classify(row, {}, llm)).kind, "retrospective")

    async def test_correction_requires_target_fact(self):
        row = candidate(text="Alice: I never worked at B Co.", subject=None, object_=None)
        payload = {"kind": "correction", "entity": "Alice", "attribute": "employer",
                   "value": None, "evidence_quote": "Alice: I never worked at B Co.",
                   "triple": ["Alice", "never worked at", "B Co"], "corrected_fact_id": "f1"}
        decision = await classify(row, {"Alice": {"employer": {"history": [
            {"fact_id": "f1", "value": "B Co"}]}}}, FakeLLM(payload))
        self.assertEqual((decision.kind, decision.corrected_fact_id), ("correction", "f1"))

    async def test_value_must_match_triple_and_attribute_hint(self):
        row = candidate()
        payload = {"kind": "baseline", "entity": "Alice", "attribute": "employer",
                   "value": "B Co", "evidence_quote": row.text,
                   "triple": ["Alice", "works at", "C Co"]}
        self.assertEqual((await classify(row, {}, FakeLLM(payload))).kind, "review")
        payload["value"] = "C Co"
        self.assertEqual((await classify(row, {}, FakeLLM(payload), "residence")).kind, "review")

    async def test_correction_cannot_target_already_corrected_fact(self):
        row = candidate(text="Alice: I never worked at B Co.", subject=None, object_=None)
        payload = {"kind": "correction", "entity": "Alice", "attribute": "employer",
                   "value": None, "evidence_quote": row.text,
                   "triple": ["Alice", "never worked at", "B Co"], "corrected_fact_id": "f1"}
        context = {"Alice": {"employer": {"history": [
            {"fact_id": "f1", "value": "B Co", "status": "corrected"}]}}}
        self.assertEqual((await classify(row, context, FakeLLM(payload))).kind, "review")

    async def test_event_group_uses_shared_original_event_phrase(self):
        row = candidate(text="Alice joined C Co and became Manager.")
        context = {"Alice": {"employer": {"value": "B Co"},
                             "job_title": {"value": "Clerk"}}}
        employer = {"kind": "change", "entity": "Alice", "attribute": "employer",
                    "value": "C Co", "evidence_quote": row.text,
                    "triple": ["Alice", "joined", "C Co"], "detection": "explicit",
                    "event_group_quote": row.text, "event_group_id": "arbitrary-1"}
        title = {"kind": "change", "entity": "Alice", "attribute": "job_title",
                 "value": "Manager", "evidence_quote": row.text,
                 "triple": ["Alice", "became", "Manager"], "detection": "explicit",
                 "event_group_quote": row.text, "event_group_id": "arbitrary-2"}
        first = await classify(row, context, FakeLLM(employer), "employer")
        second = await classify(row, context, FakeLLM(title), "job_title")
        self.assertEqual(first.event_group_id, second.event_group_id)
        self.assertNotEqual(first.event_group_id, "arbitrary-1")
        employer["event_group_quote"] = "unseen event"
        self.assertIsNone((await classify(row, context, FakeLLM(employer))).event_group_id)

    async def test_event_group_can_span_two_attribute_evidence_clauses(self):
        row = candidate(text="Alice joined C Co and became Manager.")
        payload = {"kind": "change", "entity": "Alice", "attribute": "employer",
                   "value": "C Co", "evidence_quote": "Alice joined C Co",
                   "triple": ["Alice", "joined", "C Co"], "detection": "explicit",
                   "event_group_quote": row.text}
        context = {"Alice": {"employer": {"value": "B Co"}}}
        result = await classify(row, context, FakeLLM(payload), "employer")
        self.assertIsNotNone(result.event_group_id)

    async def test_curated_identity_mapping_rejects_unknown_name(self):
        row = candidate()
        payload = {"kind": "baseline", "entity": "Alice", "attribute": "employer",
                   "value": "C Co", "evidence_quote": row.text,
                   "triple": ["Alice", "works at", "C Co"]}
        self.assertEqual((await classify(row, {}, FakeLLM(payload), identity_map={})).kind, "review")
        accepted = await classify(row, {}, FakeLLM(payload),
                                  identity_map={"Alice": "sample:participant_a"})
        self.assertEqual((accepted.entity_id, accepted.graph_entity_name),
                         ("sample:participant_a", "Alice"))
        scoped = await classify(row, {}, FakeLLM(payload),
                                identity_map={"2|Alice": "sample:second_alice"})
        self.assertEqual(scoped.entity_id, "sample:second_alice")

    async def test_first_person_quote_uses_verified_speaker_and_drops_guessed_date(self):
        row = candidate(text="Alice: I now work as a teacher at C Co. Bob: Nice!",
                        object_="teacher")
        payload = {"kind": "change", "entity": "Alice", "attribute": "job_title",
                   "value": "teacher", "evidence_quote": "I now work as a teacher at C Co.",
                   "triple": ["Alice", "work as", "teacher"],
                   "event_time": "2023-02-01", "detection": "explicit"}
        decision = await classify(row, {}, FakeLLM(payload), "job_title")
        self.assertEqual((decision.kind, decision.entity_id, decision.event_time),
                         ("baseline", "Alice", None))

    async def test_first_person_quote_cannot_be_attributed_to_other_speaker(self):
        row = candidate(text="Bob: I now work as a teacher at C Co. Alice: Congrats!",
                        object_="teacher")
        payload = {"kind": "baseline", "entity": "Alice", "attribute": "job_title",
                   "value": "teacher", "evidence_quote": "I now work as a teacher at C Co.",
                   "triple": ["Alice", "work as", "teacher"]}
        self.assertEqual((await classify(row, {}, FakeLLM(payload), "job_title")).kind,
                         "review")

    async def test_change_with_freeform_detection_is_conservatively_inferred(self):
        row = candidate(text="Alice: I now work as a teacher at C Co.", object_="teacher")
        payload = {"kind": "change", "entity": "Alice", "attribute": "job_title",
                   "value": "teacher", "evidence_quote": "I now work as a teacher at C Co.",
                   "triple": ["Alice", "work as", "teacher"],
                   "detection": "This appears to be an explicit change."}
        context = {"Alice": {"job_title": {"value": "artist"}}}
        decision = await classify(row, context, FakeLLM(payload), "job_title")
        self.assertEqual((decision.kind, decision.detection), ("change", "inferred"))


if __name__ == "__main__":
    unittest.main()
