import unittest
from types import SimpleNamespace

import networkx as nx

from experiments.event_state_evolution.evidence import collect_candidates


class EvidenceTests(unittest.TestCase):
    def test_collects_only_existing_chunks_and_keeps_unlinked_text(self):
        graph = nx.DiGraph()
        graph.add_edge("Alice", "B Co", relation_name="works at", chunk_ids=["1", "2", "3"])
        storage = {
            "1": {"text": "Alice works at B Co.", "session_id": "D1", "session_time": "2023-01-01"},
            "2": {"text": "Alice works at C Co.", "session_id": "D2", "session_time": "2023-02-01"},
            "4": {"text": "Alice: I never worked at B Co.", "session_id": "D3", "session_time": "2023-03-01"},
        }
        memory = SimpleNamespace(graph_builder=SimpleNamespace(graph=graph), chunk_storage=storage)
        rows = collect_candidates(memory, "sample")
        self.assertEqual([(r.chunk_id, r.subject, r.object) for r in rows],
                         [("1", None, None), ("1", "Alice", "B Co"),
                          ("2", None, None), ("2", "Alice", "B Co"),
                          ("4", None, None)])
        self.assertEqual(rows[0].source_key, collect_candidates(memory, "sample")[0].source_key)
        self.assertNotEqual(rows[0].source_key, rows[1].source_key)
        self.assertEqual(rows[4].entity_names, ("Alice", "B Co"))


if __name__ == "__main__":
    unittest.main()
