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
                               "supported_transition": True, "time_consistent": True,
                               "operation": "REPLACE",
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


class CaptureLLM(FakeLLM):
    def __init__(self):
        self.extraction_prompts = []
        self.gate_prompts = []

    async def generate(self, prompt, **kwargs):
        if "STATE_SLOT_GATE" in prompt:
            self.gate_prompts.append(prompt)
        else:
            self.extraction_prompts.append(prompt)
        return await super().generate(prompt, **kwargs)


class FakeEmbeddings:
    def __init__(self):
        self.calls = []

    async def get_embeddings(self, texts):
        self.calls.extend(texts)
        return [[1.0, 0.0] if any(word in text.casefold() for word in
                               ("employer", "works_at", "work at")) else [0.0, 1.0]
                for text in texts]


class AliasLLM:
    async def generate(self, prompt, **kwargs):
        if "STATE_SLOT_GATE" in prompt:
            return json.dumps({"same_slot": True, "incompatible": True,
                               "supported_transition": True, "operation": "REPLACE",
                               "evidence_quote": "Alice: I now work at C Co.",
                               "reason": "same current job"})
        current = prompt.split("原文：", 1)[-1]
        if "B Co" in current:
            attribute, value, quote = "employer", "B Co", "Alice: I work at B Co."
        else:
            attribute, value, quote = "works_at", "C Co", "Alice: I now work at C Co."
        return json.dumps({"facts": [{"kind": "state", "entity": "Alice",
                                      "attribute": attribute, "value": value,
                                      "quote": quote, "scope": {}, "modality": "actual",
                                      "triple_indices": []}]})


class PastRecollectionLLM:
    def __init__(self):
        self.gate_calls = 0

    async def generate(self, prompt, **kwargs):
        if "STATE_SLOT_GATE" in prompt:
            self.gate_calls += 1
            raise AssertionError("时间倒置应在 LLM 门控前被程序拦截")
        current = prompt.split("原文：", 1)[-1]
        value, event_time = (("B Co", "2025-03-01") if "B Co" in current
                             else ("C Co", "2024-09-01"))
        quote = f"Alice: I worked at {value} on {event_time}."
        return json.dumps({"facts": [{"kind": "state", "entity": "Alice",
                                      "attribute": "employer", "value": value,
                                      "quote": quote, "scope": {}, "modality": "actual",
                                      "event_time": event_time, "triple_indices": []}]})


def proposal(chunk, value, kind="state", target=None):
    return {"kind": kind, "entity_id": "Alice", "entity_index": 0,
            "attribute": "employer", "value": value,
            "quote": f"Alice works at {value}." if value else "Alice never worked at B Co.",
            "chunk_id": chunk, "session_id": "D" + chunk,
            "observed_at": "2023-01-" + chunk.zfill(2), "event_time": None,
            "event_type": "job_change", "event_group_id": None,
            "target_fact_id": target, "triple_refs": []}


class IntegratedEventWriterTests(unittest.IsolatedAsyncioTestCase):
    async def test_unsupported_scope_is_dropped_only_for_set_facts(self):
        class ScopeLLM:
            async def generate(self, prompt, **kwargs):
                attribute = "associated_with" if "associated" in prompt else "employer"
                value = "Book club" if attribute == "associated_with" else "B Co"
                return json.dumps({"facts": [{"kind": "state", "entity": "Alice",
                    "attribute": attribute, "value": value,
                    "quote": f"Alice: I {('associated with' if attribute == 'associated_with' else 'work at')} {value}.",
                    "scope": "time_range", "modality": "actual", "triple_indices": []}]})

        chunks = [{"chunk_id": "1", "session_id": "D1", "session_time": "2025-01-01",
                   "text": "Alice: I associated with Book club."},
                  {"chunk_id": "2", "session_id": "D2", "session_time": "2025-02-01",
                   "text": "Alice: I work at B Co."}]
        layer = await EventStateWriter(ScopeLLM()).write_chunks(
            EventStateLayer(), chunks, [], {"Alice": 0})
        self.assertEqual(len(layer.active_members("Alice", "associated_with")), 1)
        self.assertEqual(layer.active_members("Alice", "associated_with")[0]["scope"], {})
        self.assertIsNone(layer.current_fact("Alice", "employer"))
        self.assertTrue(any("忽略无证据" in row["reason"] for row in layer.review_queue))

    async def test_relation_alias_can_find_same_middle_index_slot(self):
        chunks = [{"chunk_id": "1", "session_id": "D1", "session_time": "2025-01-01",
                   "text": "Alice: I work at B Co."},
                  {"chunk_id": "2", "session_id": "D2", "session_time": "2025-02-01",
                   "text": "Alice: I now work at C Co."}]
        layer = await EventStateWriter(AliasLLM(), FakeEmbeddings()).write_chunks(
            EventStateLayer(), chunks, [], {"Alice": 0})
        self.assertEqual(layer.current_fact("Alice", "employer")["value"], "C Co")
        self.assertEqual(len(layer.events), 1)
        self.assertEqual(next(iter(layer.events.values()))["operation"], "REPLACE")

    async def test_past_recollection_cannot_replace_current_state(self):
        llm = PastRecollectionLLM()
        chunks = [{"chunk_id": "1", "session_id": "D1", "session_time": "2025-04-01",
                   "text": "Alice: I worked at B Co on 2025-03-01."},
                  {"chunk_id": "2", "session_id": "D2", "session_time": "2025-05-01",
                   "text": "Alice: I worked at C Co on 2024-09-01."}]
        layer = await EventStateWriter(llm, FakeEmbeddings()).write_chunks(
            EventStateLayer(), chunks, [], {"Alice": 0})
        self.assertEqual(llm.gate_calls, 0)
        self.assertEqual(layer.current_fact("Alice", "employer")["value"], "B Co")
        self.assertEqual(layer.events, {})

    async def test_extraction_does_not_receive_the_whole_entity_history(self):
        llm = CaptureLLM()
        embeddings = FakeEmbeddings()
        chunks = [{"chunk_id": "1", "session_id": "D1", "session_time": "2023-01-01",
                   "text": "Alice: I work at B Co."},
                  {"chunk_id": "2", "session_id": "D2", "session_time": "2023-01-02",
                   "text": "Alice: I now work at C Co."}]
        relations = [{"src": "Alice", "tgt": "B Co", "relation": "work at", "chunk_id": "1"},
                     {"src": "Alice", "tgt": "C Co", "relation": "work at", "chunk_id": "2"}]
        await EventStateWriter(llm, embeddings).write_chunks(
            EventStateLayer(), chunks, relations, {"Alice": 0, "B Co": 1, "C Co": 2})
        self.assertEqual(len(llm.extraction_prompts), 2)
        self.assertNotIn("I work at B Co.", llm.extraction_prompts[1])
        self.assertTrue(embeddings.calls)
        self.assertIn("I work at B Co.", llm.gate_prompts[0])

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

    async def test_hashed_chunk_sources_commit_in_original_order(self):
        """写入回归：并发响应完成后，事实位置仍按来源 0、7 提交，保留哈希证据 ID。"""
        chunks = [{"chunk_id": "chunk-a", "source_chunk_id": 7,
                   "session_id": "D2", "session_time": "2023-01-01",
                   "text": "Alice: I now work at C Co."},
                  {"chunk_id": "chunk-z", "source_chunk_id": 0,
                   "session_id": "D2", "session_time": "2023-01-01",
                   "text": "Alice: I work at B Co."}]
        relations = [{"src": "Alice", "tgt": "B Co", "relation": "work at", "chunk_id": "chunk-z"},
                     {"src": "Alice", "tgt": "C Co", "relation": "work at", "chunk_id": "chunk-a"}]
        layer = await EventStateWriter(FakeLLM(), state_write_policy='append_only').write_chunks(
            EventStateLayer(), chunks, relations, {"Alice": 0, "B Co": 1, "C Co": 2})
        facts = sorted(layer.facts.values(), key=lambda fact: fact['position'])
        self.assertEqual([(fact['chunk_id'], fact['value']) for fact in facts],
                         [('chunk-z', 'B Co'), ('chunk-a', 'C Co')])
        self.assertEqual(layer.processed_chunks, {'chunk-z', 'chunk-a'})

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
