"""三层简化结构的存储/查询检查；只使用固定输入，不调用 LLM 或真实模型。

验证重点是证据不丢失：不同谓词、方向、Chunk 来源独立保存；图边只索引事实；
增量 ID、持久化引用和旧图迁移不能破坏原始数据。此处不评估模型检索准确率。
"""

from copy import deepcopy
from pathlib import Path
import pickle
import tempfile
from types import SimpleNamespace
import unittest

import networkx as nx

from coregraph.dynamic_memory import DynamicMemory
from coregraph.graph_builder import GraphBuilder
from coregraph.triple_fact_store import TripleFactStore
from experiments.structural_semantic_retrieval.retriever import StructuralSemanticRetriever
from init.config import Config


def configuration():
    config = Config()
    config.upper_memory_mode = 'event_state'
    config.middle_memory_mode = 'entity_fact_index'
    return config


def relationships():
    return [dict(src='User', relation='allergic_to', tgt='Milk', chunk_id='1',
                 session_id='D1', session_time='2023-01-01', event_time='2022-12-31'),
            dict(src='User', relation='not_allergic_to', tgt='Milk', chunk_id='2',
                 session_id='D2', session_time='2023-02-01')]


ENTITIES = [{'entity': 'User'}, {'entity': 'Milk'}]
CHUNKS = {'1': {'chunk_id': '1', 'session_id': 'D1', 'text': 'User: I am allergic to Milk.'},
          '2': {'chunk_id': '2', 'session_id': 'D2', 'text': 'User: I am not allergic to Milk.'}}


class EntityFactIndexTests(unittest.IsolatedAsyncioTestCase):
    def memory(self):
        memory = DynamicMemory(configuration())
        memory.chunk_storage = deepcopy(CHUNKS)
        memory.entity_name_to_index = {'User': 0, 'Milk': 1}
        memory._record_triple_occurrences(relationships(), list(CHUNKS.values()))
        memory.graph_builder.build_from_entities_and_relationships(ENTITIES, relationships())
        return memory

    def test_different_predicates_and_directions_share_only_index_edge(self):
        builder = GraphBuilder(configuration())
        refs = relationships() + [dict(src='Milk', relation='affects', tgt='User', chunk_id='3')]
        builder.build_from_entities_and_relationships(ENTITIES, refs)
        self.assertFalse(builder.graph.is_directed())
        self.assertEqual(builder.graph.number_of_edges(), 1)
        self.assertEqual(len(builder.fact_store.facts), 3)
        edge = builder.graph.edges['User', 'Milk']
        self.assertEqual(set(edge), {'edge_type', 'fact_ids', 'weight'})
        self.assertEqual(len(edge['fact_ids']), 3)
        self.assertIn(('Milk', 'affects', 'User'), {(f['src'], f['relation'], f['tgt'])
                                                  for f in builder.fact_store.facts.values()})

    def test_duplicate_source_is_idempotent_new_source_is_not_collapsed(self):
        memory = self.memory()
        memory.graph_builder.add_entities_and_relationships_incrementally([], relationships())
        memory.graph_builder.add_entities_and_relationships_incrementally([], [dict(relationships()[0], chunk_id='3')])
        self.assertEqual(len(memory.fact_store.facts), 3)
        self.assertEqual(memory.graph_builder.graph.number_of_edges(), 1)
        self.assertEqual(memory.graph_builder.graph.edges['User', 'Milk']['weight'], 3)

    def test_fact_store_duplicate_can_preserve_array_embeddings(self):
        """向量可能来自 NumPy；元数据补充不能把数组与空字符串作布尔比较。"""
        import numpy as np
        store = TripleFactStore()
        record = dict(relationships()[0], embedding=np.array([1.0, 0.0]))
        fid = store.upsert(record)
        self.assertEqual(store.upsert(record), fid)
        self.assertEqual(len(store.facts), 1)
        self.assertEqual(store.facts[fid]['embedding'].tolist(), [1.0, 0.0])

    def test_chunk_id_collision_keeps_both_sessions_and_repeats_stably(self):
        memory = self.memory()
        source = dict(chunk_id='1', session_id='D3', text='User: Another memory.')
        first = memory._store_chunk_batch([source])[0]
        second = memory._store_chunk_batch([source])[0]
        self.assertNotEqual(first['chunk_id'], '1')
        self.assertEqual(first['chunk_id'], second['chunk_id'])
        self.assertEqual(memory.chunk_storage['1']['text'], CHUNKS['1']['text'])
        self.assertEqual(source['chunk_id'], '1')
        self.assertEqual(len(memory.chunk_storage), 3)

    async def test_fact_vectors_are_separate_and_do_not_live_on_shared_edge(self):
        memory = self.memory()
        class Embeddings:
            async def get_embeddings(self, texts):
                return [[1.0, 0.0], [0.0, 1.0]]
        memory.embedding_manager = Embeddings()
        await memory._precompute_relationship_embeddings(relationships())
        vectors = {fact['relation']: fact['embedding'] for fact in memory.fact_store.facts.values()}
        self.assertNotEqual(vectors['allergic_to'], vectors['not_allergic_to'])
        self.assertNotIn('embedding', memory.graph_builder.graph.edges['User', 'Milk'])

    def test_query_reads_complete_facts_and_expands_undirected_index(self):
        memory = self.memory()
        retriever = StructuralSemanticRetriever(memory, None)
        chunks, entities, _, _ = retriever._collect()
        self.assertEqual(chunks['1']['triples'][0]['predicate'], 'allergic_to')
        self.assertEqual(chunks['2']['triples'][0]['predicate'], 'not_allergic_to')
        self.assertTrue(chunks['1']['triples'][0]['fact_id'])
        self.assertIn('2', retriever._graph_support(['1'], chunks, entities, {'User'}))

    def test_persistence_only_stores_one_middle_fact_table_and_upper_ids(self):
        memory = self.memory()
        memory.event_state_layer.write_policy = 'append_only'
        memory.event_state_layer.writer_schema_version = 4
        quote = CHUNKS['1']['text']
        proposal = dict(entity_id='User', entity_index=0, attribute='allergy', value='Milk',
            kind='state', fact_type='state', scope={}, modality='actual', chunk_id='1',
            session_id='D1', observed_at='2023-01-01', event_type='allergy', quote=quote,
            source_text=quote, triple_refs=[relationships()[0]])
        observation_id = memory.event_state_layer.append_observation(proposal)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'new.pkl'
            memory.save_graph(str(path))
            data = pickle.loads(path.read_bytes())
            self.assertNotIn('triple_storage', data)
            self.assertEqual(len(data['fact_storage']['facts']), 2)
            upper = data['event_state_layer']['facts'][observation_id]
            self.assertNotIn('triple_refs', upper)
            self.assertEqual(len(upper['middle_fact_ids']), 1)
            loaded = DynamicMemory(configuration())
            loaded.load_graph(str(path))
        restored = loaded.event_state_layer.facts[observation_id]['triple_refs'][0]
        self.assertEqual(restored['relation'], 'allergic_to')
        self.assertEqual(restored['chunk_id'], '1')
        self.assertEqual(loaded.graph_builder.graph.number_of_edges(), 1)
        self.assertFalse(loaded.event_state_layer.current)

    def test_old_modes_still_use_original_graph_types(self):
        old = Config()
        self.assertIsInstance(GraphBuilder(old).graph, nx.DiGraph)
        old.upper_memory_mode = 'event_state'
        self.assertIsInstance(GraphBuilder(old).graph, nx.MultiDiGraph)

    def test_empty_fact_table_roundtrip_keeps_extracted_origin(self):
        """空的新图也有明确主表，不能误触旧图恢复逻辑而降为 partial。"""
        memory = DynamicMemory(configuration())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'empty.pkl'
            memory.save_graph(str(path))
            loaded = DynamicMemory(configuration())
            loaded.load_graph(str(path))
        self.assertEqual(loaded.fact_store.origin, 'extracted')

    def test_association_file_without_fact_table_is_rejected(self):
        """图边没有事实语义，缺失主表时必须报错，不能给调用方返回虚假 Triple。"""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'broken.pkl'
            path.write_bytes(pickle.dumps({'graph': GraphBuilder(configuration()).graph}))
            with self.assertRaises(ValueError):
                DynamicMemory(configuration()).load_graph(str(path))

    async def test_real_session_entrypoint_preserves_reused_local_chunk_ids(self):
        """走正式逐会话入口，确认 ID 修正同时传给抽取器、事实表和上层。"""
        memory = DynamicMemory(configuration())
        memory.config.chunk.dialogue_input = True
        memory.embedding_manager = None
        memory.chunk_processor = SimpleNamespace(process_corpus=lambda corpus: [dict(corpus[0], chunk_id='0')])
        costs = SimpleNamespace(total_prompt_tokens=0, total_completion_tokens=0)
        memory.llm_manager = SimpleNamespace(model='fake', cost_manager=SimpleNamespace(get_costs=lambda: costs))
        memory.cost_manager = SimpleNamespace(update_entity_extraction_cost=lambda *a: None,
                                               update_relationship_extraction_cost=lambda *a: None)
        class Extractor:
            async def extract_from_chunks(self, chunks):
                return ENTITIES, [dict(src='User', relation='reports', tgt='Milk',
                                       chunk_id=chunks[0]['chunk_id'])]
            def deduplicate_entities(self, entities, threshold):
                return entities, {}
            def _update_relationship_entity_references(self, rows, mapping):
                return rows
            def deduplicate_relationships(self, rows, mapping, threshold):
                return rows
        seen = []
        class Writer:
            async def write_chunks(self, layer, chunks, refs, entity_index):
                seen.append((chunks[0]['chunk_id'], refs[0]['chunk_id']))
                return layer
        memory.dialogue_extractor, memory.event_state_writer = Extractor(), Writer()
        await memory.add_single_session([dict(session_id='D1', session_time='2023-01-01', text='User: Milk first.')])
        await memory.add_single_session([dict(session_id='D2', session_time='2023-02-01', text='User: Milk later.')])
        self.assertEqual(len(memory.chunk_storage), 2)
        self.assertEqual(len(memory.fact_store.facts), 2)
        self.assertEqual(memory.graph_builder.graph.number_of_edges(), 1)
        self.assertTrue(all(chunk == ref for chunk, ref in seen))
        self.assertNotEqual(seen[0][0], seen[1][0])

    def test_migration_is_explicit_preserves_source_and_marks_partial(self):
        from experiments.integrated_event_state.migrate_entity_fact_index import migrate_graph
        graph = nx.DiGraph()
        graph.add_edge('User', 'Milk', relation_name='not_allergic_to', chunk_id='2')
        source = dict(graph=graph, chunk_storage=CHUNKS, entity_name_to_index={'User': 0, 'Milk': 1})
        with tempfile.TemporaryDirectory() as directory:
            path, output = Path(directory) / 'old.pkl', Path(directory) / 'new.pkl'
            path.write_bytes(pickle.dumps(source))
            original = path.read_bytes()
            report = migrate_graph(path, output)
            self.assertEqual(path.read_bytes(), original)
            result = pickle.loads(output.read_bytes())
            self.assertEqual(len(result['fact_storage']['facts']), 1)
            self.assertEqual(report['origin'], 'recovered_partial')
            self.assertEqual(result['graph'].number_of_edges(), 1)
            with self.assertRaises(FileExistsError):
                migrate_graph(path, output)

    def test_migration_uses_full_source_table_instead_of_collapsed_edge(self):
        """旧图只有一边，但完整原始记录有两种关系时，两条事实都要保留。"""
        from experiments.integrated_event_state.migrate_entity_fact_index import migrate_graph
        graph = nx.DiGraph()
        graph.add_edge('User', 'Milk', relation_name='not_allergic_to', chunk_id='2')
        source = dict(graph=graph, chunk_storage=CHUNKS, triple_storage_origin='extracted',
                      triple_storage={ref['chunk_id']: [ref] for ref in relationships()})
        with tempfile.TemporaryDirectory() as directory:
            path, output = Path(directory) / 'old.pkl', Path(directory) / 'new.pkl'
            path.write_bytes(pickle.dumps(source))
            report = migrate_graph(path, output)
            data = pickle.loads(output.read_bytes())
        self.assertEqual(report['origin'], 'extracted')
        self.assertEqual(len(data['fact_storage']['facts']), 2)
        self.assertEqual(data['graph'].number_of_edges(), 1)


if __name__ == '__main__':
    unittest.main()
