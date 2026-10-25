"""用新事实查询定位少量与中层 Triple 相连的旧证据。"""

import unittest

from coregraph.event_state_layer import EventStateLayer
from coregraph.state_candidate_retriever import StateCandidateRetriever


class FakeEmbeddings:
    def __init__(self):
        self.calls = []

    async def get_embeddings(self, texts):
        self.calls.extend(texts)
        vectors = []
        for text in texts:
            lower = text.casefold()
            vectors.append([1.0 if any(term in lower for term in
                                        ("employer", "works_at", "work at")) else 0.0,
                            1.0 if "residence" in lower else 0.0])
        return vectors


def record(layer, entity, index, attribute, value, chunk):
    proposal = {"entity_id": entity, "entity_index": index,
                "attribute": attribute, "value": value, "scope": {},
                "modality": "actual", "quote": f"{entity}: {value}",
                "source_text": f"{entity}: {value}", "chunk_id": chunk,
                "session_id": "D1", "observed_at": "2025-01-01",
                "event_time": None, "event_type": "state",
                "event_group_id": None,
                "triple_refs": [{"src": entity, "relation": attribute,
                                 "tgt": value, "chunk_id": chunk}]}
    return layer.apply_decision(proposal, "ADD")


class StateCandidateRetrieverTests(unittest.IsolatedAsyncioTestCase):
    async def test_semantic_relation_search_is_bounded_and_entity_constrained(self):
        layer = EventStateLayer()
        employer_id = record(layer, "Alice", 1, "current_employer", "B Co", "1")
        record(layer, "Alice", 1, "current_residence", "Paris", "2")
        record(layer, "Bob", 2, "current_employer", "D Co", "3")
        embeddings = FakeEmbeddings()
        retriever = StateCandidateRetriever(embeddings, top_k=1)
        query = {"entity_id": "Alice", "entity_index": 1,
                 "attribute": "works_at", "value": "C Co", "scope": {},
                 "modality": "actual", "quote": "Alice: I now work at C Co.",
                 "triple_refs": [{"src": "Alice", "relation": "work at",
                                  "tgt": "C Co", "chunk_id": "4"}]}
        hits = await retriever.search(layer, query)
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0]["fact"]["fact_id"], employer_id)
        self.assertTrue(any("works_at" in text for text in embeddings.calls))

    async def test_current_exact_slot_is_always_in_candidate_pool(self):
        layer = EventStateLayer()
        old_id = record(layer, "Alice", 1, "current_employer", "B Co", "1")
        embeddings = FakeEmbeddings()
        retriever = StateCandidateRetriever(embeddings, top_k=1)
        query = {"entity_id": "Alice", "entity_index": 1,
                 "attribute": "current_employer", "value": "C Co",
                 "scope": {}, "modality": "actual", "quote": "C Co",
                 "triple_refs": []}
        hits = await retriever.search(layer, query)
        self.assertEqual(hits[0]["fact"]["fact_id"], old_id)


if __name__ == "__main__":
    unittest.main()
