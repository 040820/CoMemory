"""用同一主体对牛奶的相反关系验证多关系和逐来源证据保全。"""

import pickle
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import networkx as nx

from coregraph.dynamic_memory import DynamicMemory
from coregraph.graph_builder import GraphBuilder
from experiments.structural_semantic_retrieval.retriever import StructuralSemanticRetriever
from init.config import Config


def config():
    result = Config()
    result.upper_memory_mode = "event_state"
    return result


def relations():
    return [
        {"src": "User", "relation": "allergic_to", "tgt": "Milk", "chunk_id": "1",
         "session_id": "D1", "session_time": "2025-01-01", "event_time": "2024-12-31"},
        {"src": "User", "relation": "not_allergic_to", "tgt": "Milk", "chunk_id": "2",
         "session_id": "D2", "session_time": "2025-02-01"},
    ]


ENTITIES = [{"entity": "User"}, {"entity": "Milk"}]
CHUNKS = {"1": {"text": "User: I am allergic to milk.", "session_id": "D1"},
          "2": {"text": "User: I am no longer allergic to milk.", "session_id": "D2"}}


class MultiRelationGraphTests(unittest.IsolatedAsyncioTestCase):
    def test_same_entities_keep_two_relations_with_separate_evidence(self):
        builder = GraphBuilder(config())
        builder.build_from_entities_and_relationships(ENTITIES, relations())
        self.assertTrue(builder.graph.is_multigraph())
        self.assertEqual(builder.graph.number_of_nodes(), 2)
        self.assertEqual(builder.graph.number_of_edges(), 2)
        rows = {row["relation_name"]: row for _, _, row in builder.graph.edges(data=True)}
        self.assertEqual(rows["allergic_to"]["chunk_id"], "1")
        self.assertEqual(rows["not_allergic_to"]["chunk_id"], "2")
        self.assertEqual(rows["allergic_to"]["session_time"], "2025-01-01")
        self.assertEqual(rows["allergic_to"]["event_time"], "2024-12-31")

    def test_repeated_chunk_is_idempotent_but_new_source_is_independent(self):
        builder = GraphBuilder(config())
        for row in relations():
            builder.add_entities_and_relationships_incrementally(ENTITIES, [row])
        builder.add_entities_and_relationships_incrementally(ENTITIES, [relations()[0]])
        builder.add_entities_and_relationships_incrementally(ENTITIES,
            [dict(relations()[0], chunk_id="3", session_id="D3", session_time="2025-03-01")])
        self.assertEqual(builder.graph.number_of_edges(), 3)
        self.assertEqual({r["chunk_id"] for _, _, r in builder.graph.edges(data=True)}, {"1", "2", "3"})

    def test_baseline_keeps_existing_single_edge_behavior(self):
        builder = GraphBuilder(Config())
        builder.build_from_entities_and_relationships(ENTITIES, relations())
        self.assertFalse(builder.graph.is_multigraph())
        self.assertEqual(builder.graph.number_of_edges(), 1)

    async def test_relation_embeddings_do_not_overwrite_other_predicates(self):
        class Embeddings:
            async def get_embeddings(self, texts):
                return [[float(i), 1.0] for i in range(len(texts))]
        memory = DynamicMemory(config())
        memory.embedding_manager = Embeddings()
        memory.graph_builder.build_from_entities_and_relationships(ENTITIES, relations())
        await memory._precompute_relationship_embeddings(relations())
        vectors = {r["relation_name"]: r["embedding"]
                   for _, _, r in memory.graph_builder.graph.edges(data=True)}
        self.assertEqual(vectors["allergic_to"], [0.0, 1.0])
        self.assertEqual(vectors["not_allergic_to"], [1.0, 1.0])

    def test_pickle_roundtrip_preserves_both_edges(self):
        memory = DynamicMemory(config())
        memory.chunk_storage = CHUNKS
        memory._record_triple_occurrences(relations())
        memory.graph_builder.build_from_entities_and_relationships(ENTITIES, relations())
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "graph.pkl")
            memory.save_graph(path)
            loaded = DynamicMemory(config())
            loaded.load_graph(path)
        self.assertTrue(loaded.graph_builder.graph.is_multigraph())
        self.assertEqual(loaded.graph_builder.graph.number_of_edges(), 2)

    def test_legacy_graph_can_rebuild_from_extracted_occurrences_without_rewriting_source(self):
        graph = nx.DiGraph()
        graph.add_edge("User", "Milk", relation_name="allergic_to", chunk_id="1")
        stored = {row["chunk_id"]: [row] for row in relations()}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "old.pkl"
            path.write_bytes(pickle.dumps({"graph": graph, "chunk_storage": CHUNKS,
                "triple_storage": stored, "triple_storage_origin": "extracted"}))
            original = path.read_bytes()
            loaded = DynamicMemory(config())
            loaded.load_graph(str(path))
            self.assertEqual(path.read_bytes(), original)
        self.assertTrue(loaded.graph_builder.graph.is_multigraph())
        self.assertEqual(loaded.graph_builder.graph.number_of_edges(), 2)

    def test_retrieval_prefers_exact_sources_over_collapsed_legacy_edges(self):
        graph = nx.DiGraph()
        graph.add_edge("User", "Milk", relation_name="allergic_to", chunk_ids=["1", "2"])
        memory = SimpleNamespace(graph_builder=SimpleNamespace(graph=graph), chunk_storage=CHUNKS,
            triple_storage={row["chunk_id"]: [row] for row in relations()},
            triple_storage_origin="extracted")
        chunks, _, _, _ = StructuralSemanticRetriever(memory, None)._collect()
        self.assertEqual([t["predicate"] for t in chunks["1"]["triples"]], ["allergic_to"])
        self.assertEqual([t["predicate"] for t in chunks["2"]["triples"]], ["not_allergic_to"])

    def test_multigraph_retrieval_does_not_mix_old_and_new_chunk_relations(self):
        builder = GraphBuilder(config())
        builder.build_from_entities_and_relationships(ENTITIES, relations())
        memory = SimpleNamespace(graph_builder=builder, chunk_storage=CHUNKS)
        chunks, _, _, _ = StructuralSemanticRetriever(memory, None)._collect()
        self.assertEqual([t["predicate"] for t in chunks["1"]["triples"]], ["allergic_to"])
        self.assertEqual([t["predicate"] for t in chunks["2"]["triples"]], ["not_allergic_to"])


if __name__ == "__main__":
    unittest.main()
