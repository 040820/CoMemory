"""并发抽取后仍按 Chunk 顺序更新同一属性，且历史纠正重建变化链。"""

import asyncio
import json
import unittest

from coregraph.event_state_layer import EventStateLayer
from coregraph.event_state_writer import EventStateWriter


class FakeLLM:
    async def generate(self, prompt, **kwargs):
        if "STATE_SLOT_GATE" in prompt:
            return json.dumps({"same_slot": True, "incompatible": True,
                               "supported_transition": True, "operation": "REPLACE",
                               "evidence_quote": "Alice: I now work at C Co.",
                               "reason": "now 表示当前雇主发生变化"})
        if "I work at B Co." in prompt:
            await asyncio.sleep(0.02)
            value = "B Co"
            quote = "Alice: I work at B Co."
        else:
            value = "C Co"
            quote = "Alice: I now work at C Co."
        return json.dumps({"facts": [{"kind": "state", "entity": "Alice",
                                      "attribute": "employer", "value": value,
                                      "quote": quote, "scope": {}, "modality": "actual",
                                      "triple_indices": [0]}]})


class AdditiveLLM:
    async def generate(self, prompt, **kwargs):
        return json.dumps({"facts": [{"kind": "state", "entity": "Alice",
                                      "attribute": "attended", "value": "Book club",
                                      "quote": "Alice: I attended Book club.",
                                      "scope": {}, "modality": "actual",
                                      "triple_indices": []}]})


class FalseGateLLM(FakeLLM):
    async def generate(self, prompt, **kwargs):
        if "STATE_SLOT_GATE" in prompt:
            return json.dumps({"same_slot": True, "incompatible": False,
                               "supported_transition": False, "operation": "UNCERTAIN",
                               "evidence_quote": "Alice: I now work at C Co.",
                               "reason": "缺少替换依据"})
        return await super().generate(prompt, **kwargs)


def proposal(chunk, value, kind="state", target=None):
    return {"kind": kind, "entity_id": "Alice", "entity_index": 0,
            "attribute": "employer", "value": value,
            "quote": f"Alice works at {value}." if value else "Alice never worked at B Co.",
            "chunk_id": chunk, "session_id": "D" + chunk,
            "observed_at": "2023-01-" + chunk.zfill(2), "event_time": None,
            "event_type": "job_change", "event_group_id": None,
            "target_fact_id": target, "triple_refs": []}


class IntegratedEventWriterTests(unittest.IsolatedAsyncioTestCase):
    async def test_legacy_nonempty_layer_requires_independent_rebuild(self):
        legacy = EventStateLayer({"version": 1, "facts": {"legacy": {"fact_id": "legacy"}},
                                  "events": {}, "current": {}})
        chunks = [{"chunk_id": "1", "session_id": "D1", "session_time": "2023-01-01",
                   "text": "Alice: I attended Book club."}]
        with self.assertRaises(ValueError):
            await EventStateWriter(AdditiveLLM()).write_chunks(
                legacy, chunks, [], {"Alice": 0})

    async def test_chunk_zero_is_processed(self):
        class EmptyLLM:
            async def generate(self, prompt, **kwargs):
                return '{"facts": []}'

        chunks = [{"chunk_id": 0, "session_id": "D1", "session_time": "2023-01-01",
                   "text": "Alice: Hello."}]
        layer = await EventStateWriter(EmptyLLM()).write_chunks(
            EventStateLayer(), chunks, [], {"Alice": 0})
        self.assertIn("0", layer.processed_chunks)

    async def test_additive_activity_is_recorded_without_change_event(self):
        chunks = [{"chunk_id": "1", "session_id": "D1", "session_time": "2023-01-01",
                   "text": "Alice: I attended Book club."}]
        layer = await EventStateWriter(AdditiveLLM()).write_chunks(
            EventStateLayer(), chunks, [], {"Alice": 0})
        self.assertEqual(len(layer.active_members("Alice", "attended")), 1)
        self.assertEqual(layer.events, {})

    async def test_same_session_responses_commit_in_chunk_order(self):
        chunks = [{"chunk_id": "2", "session_id": "D1", "session_time": "2023-01-01",
                   "text": "Alice: I now work at C Co."},
                  {"chunk_id": "1", "session_id": "D1", "session_time": "2023-01-01",
                   "text": "Alice: I work at B Co."}]
        relations = [{"src": "Alice", "tgt": "B Co", "relation": "work at", "chunk_id": "1"},
                     {"src": "Alice", "tgt": "C Co", "relation": "work at", "chunk_id": "2"}]
        layer = await EventStateWriter(FakeLLM()).write_chunks(
            EventStateLayer(), chunks, relations, {"Alice": 0, "B Co": 1, "C Co": 2})
        self.assertEqual(layer.current_fact("Alice", "employer")["value"], "C Co")
        self.assertEqual([(e["old_state"], e["new_state"]) for e in layer.events.values()],
                         [("B Co", "C Co")])
        self.assertEqual(len(layer.processed_chunks), 2)

    async def test_gate_rejection_does_not_replace_current_state(self):
        chunks = [{"chunk_id": "1", "session_id": "D1", "session_time": "2023-01-01",
                   "text": "Alice: I work at B Co."},
                  {"chunk_id": "2", "session_id": "D1", "session_time": "2023-01-01",
                   "text": "Alice: I now work at C Co."}]
        relations = [{"src": "Alice", "tgt": "B Co", "relation": "work at", "chunk_id": "1"},
                     {"src": "Alice", "tgt": "C Co", "relation": "work at", "chunk_id": "2"}]
        layer = await EventStateWriter(FalseGateLLM()).write_chunks(
            EventStateLayer(), chunks, relations, {"Alice": 0, "B Co": 1, "C Co": 2})
        self.assertEqual(layer.current_fact("Alice", "employer")["value"], "B Co")
        self.assertEqual(layer.events, {})
        self.assertEqual(layer.decisions[-1]["operation"], "UNCERTAIN")

    async def test_historical_correction_rebuilds_change_chain(self):
        layer = EventStateLayer()
        old_id = layer.apply(proposal("1", "B Co"))
        layer.apply(proposal("2", "C Co"))
        layer.apply(proposal("3", "D Co"))
        layer.apply(proposal("4", "A Co", "correction", old_id))
        self.assertEqual(layer.current_fact("Alice", "employer")["value"], "D Co")
        changes = {(e["old_state"], e["new_state"], e["status"])
                   for e in layer.events.values() if e["update_type"] == "change"}
        self.assertIn(("B Co", "C Co", "revoked"), changes)
        self.assertIn(("A Co", "C Co", "supported"), changes)
        self.assertIn(("C Co", "D Co", "supported"), changes)


if __name__ == "__main__":
    unittest.main()
