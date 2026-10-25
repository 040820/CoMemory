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

    async def test_rejects_fabricated_date_and_unsupported_entity(self):
        row = candidate()
        payload = {"kind": "baseline", "entity": "Alice", "attribute": "employer",
                   "value": "C Co", "evidence_quote": row.text,
                   "triple": ["Alice", "works at", "C Co"], "event_time": "2022-10-10"}
        self.assertEqual((await classify(row, {}, FakeLLM(payload))).kind, "review")
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


if __name__ == "__main__":
    unittest.main()
