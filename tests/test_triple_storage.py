"""原始 Triple 出现记录的持久化与旧图兼容测试。

覆盖：
1. 同一对实体被不同关系覆盖时，原始出现记录仍按 Chunk 保留；
2. triple_storage 随 pkl 保存/加载；
3. 旧 pkl 没有 triple_storage 时，从图边和上层 fact 快照兜底恢复。
"""

import os
import tempfile
import unittest

from coregraph.dynamic_memory import DynamicMemory
from init.config import Config


class TripleStorageTests(unittest.TestCase):
    def setUp(self):
        self.memory = DynamicMemory(Config())

    def test_record_keeps_overwritten_relations(self):
        chunks = [
            {"chunk_id": "1", "session_id": "D1", "session_time": "2023-05-01"},
            {"chunk_id": "2", "session_id": "D2", "session_time": "2023-06-01"},
        ]
        relationships = [
            {"src": "Alice", "relation": "works_at", "tgt": "B Co", "chunk_id": "1"},
            {"src": "Alice", "relation": "left", "tgt": "B Co", "chunk_id": "2"},
            {"src": "Alice", "relation": "works_at", "tgt": "C Co", "chunk_id": "2"},
        ]

        self.memory._record_triple_occurrences(relationships, chunks)

        self.assertEqual(len(self.memory.triple_storage["1"]), 1)
        self.assertEqual(len(self.memory.triple_storage["2"]), 2)
        self.assertEqual(
            {(row["relation"], row["tgt"]) for row in self.memory.triple_storage["2"]},
            {("left", "B Co"), ("works_at", "C Co")},
        )
        # 图边可能只保留最后一条，但原始记录仍带正确的 Chunk 会话信息。
        self.assertEqual(self.memory.triple_storage["2"][0]["session_id"], "D2")
        self.assertEqual(self.memory.triple_storage["2"][0]["session_time"], "2023-06-01")
        self.assertEqual(len(self.memory._relationships_for_chunks(chunks)), 3)

    def test_event_time_does_not_replace_observation_time(self):
        chunks = [{"chunk_id": "1", "session_id": "D1",
                   "session_time": "2023-05-08"}]
        self.memory._record_triple_occurrences([{
            "src": "Alice", "relation": "attended", "tgt": "support group",
            "chunk_id": "1", "create_time": "2023-05-07"}], chunks)
        row = self.memory.triple_storage["1"][0]
        self.assertEqual(row["session_time"], "2023-05-08")
        self.assertEqual(row["event_time"], "2023-05-07")

    def test_save_and_load_roundtrip(self):
        self.memory.chunk_storage = {"1": {"chunk_id": 1, "text": "Alice works at B Co."}}
        self.memory.triple_storage = {
            "1": [{"src": "Alice", "relation": "works_at", "tgt": "B Co",
                   "chunk_id": "1", "session_id": "D1", "session_time": "2023-05-01"}]
        }
        graph = self.memory.graph_builder.graph
        graph.add_node("Alice")
        graph.add_node("B Co")
        graph.add_edge("Alice", "B Co", relation_name="works_at", chunk_id="1")

        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "graph.pkl")
            self.memory.save_graph(path)
            loaded = DynamicMemory(Config())
            loaded.load_graph(path)

        self.assertEqual(loaded.triple_storage, self.memory.triple_storage)

    def test_backfill_from_graph_when_storage_missing(self):
        graph = self.memory.graph_builder.graph
        graph.add_node("Alice")
        graph.add_node("B Co")
        graph.add_edge(
            "Alice", "B Co",
            relation_name="works_at",
            chunk_id="1",
            chunk_ids=["1", "2"],
            session_id="D1",
            session_time="2023-05-01",
        )

        self.memory._backfill_triple_storage()

        self.assertEqual(set(self.memory.triple_storage), {"1", "2"})
        for chunk_id in ("1", "2"):
            self.assertEqual(self.memory.triple_storage[chunk_id][0]["relation"], "works_at")
            self.assertEqual(self.memory.triple_storage[chunk_id][0]["session_id"], "D1")

    def test_load_legacy_pickle_without_triple_storage(self):
        graph = self.memory.graph_builder.graph
        graph.add_node("Alice")
        graph.add_node("B Co")
        graph.add_edge("Alice", "B Co", relation_name="works_at", chunk_id="1")

        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "legacy.pkl")
            import pickle
            with open(path, "wb") as stream:
                pickle.dump({
                    "graph": graph,
                    "chunk_storage": {"1": {"chunk_id": 1, "text": "Alice works at B Co."}},
                    "entity_name_to_index": {"Alice": 0, "B Co": 1},
                }, stream)
            loaded = DynamicMemory(Config())
            loaded.load_graph(path)

        self.assertEqual(loaded.triple_storage["1"][0]["relation"], "works_at")


if __name__ == "__main__":
    unittest.main()
