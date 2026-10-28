"""服务于精简三层持久化：检查事实归属、引用完整性和追加写入语义，不调用模型。"""

from copy import deepcopy
from pathlib import Path
import pickle
import tempfile
import unittest

from coregraph.lean_event_index import pack_event_index, unpack_event_index
from coregraph.triple_fact_store import TripleFactStore, encode_upper_references, decode_upper_references


class LeanEventIndexTests(unittest.TestCase):
    def fixture(self):
        store = TripleFactStore()
        tid = store.upsert(dict(src='Caroline', relation='attended', tgt='group', chunk_id='1'))
        ref = store.facts[tid]
        fact = dict(fact_id='F1', entity_id='Caroline', entity_index=0,
                    attribute='attended', value='group', chunk_id='1', quote='went yesterday',
                    fact_type='event', scope={}, modality='actual', position=1,
                    event_time='2023-05-07', event_time_raw='yesterday', event_id='E1',
                    source_attribute='attended', source_value='group', source_scope={},
                    source_modality='actual', triple_refs=[ref], event_triple_refs=[ref],
                    kind='observation', status='observed', operation='APPEND',
                    validity_status='unresolved', semantic_normalization={'reason': 'diagnostic'})
        relation = dict(relation_id='R1', old_fact_id='F1', new_fact_id='F2',
                        relation='correction', classification={'evidence_quote': 'went yesterday'})
        layer = dict(version=1, write_policy='append_only', writer_schema_version=4,
                     slot_schema_version=2, facts={'F1': fact, 'F2': dict(fact, fact_id='F2')},
                     event_instances={'E1': dict(event_id='E1', actor='Caroline', action='attended',
                         target='group', fact_ids=['F1', 'F2'], observed_time_assertion_ids=['A1'],
                         observed_attribute_assertion_ids=[])},
                     event_time_assertions={'A1': dict(assertion_id='A1', event_id='E1', value='2023-05-07',
                         evidence_chunk_id='1', evidence_quote='went yesterday', source_triples=[ref])},
                     event_attribute_assertions={}, current={}, members={}, events={},
                     fact_relations=[relation, dict(relation, relation_id='R2', relation='unrelated'),
                                     dict(relation, relation_id='R3', relation='uncertain')],
                     processed_chunks=['1'], review_queue=[{'reason': 'quote missing'}],
                     decisions=[{'new_fact_id': 'F1', 'retrieval_trace': {'A': [tid]}}])
        return store, layer

    def test_upper_has_only_groups_and_useful_candidate_links(self):
        store, layer = self.fixture()
        upper, checkpoint, audit = pack_event_index(layer, store)
        self.assertEqual(set(upper), {'schema', 'groups', 'links'})
        self.assertEqual(upper['groups']['E1'], {'event_id': 'E1', 'fact_ids': ['F1', 'F2', 'A1']})
        self.assertEqual(len(upper['links']), 1)
        self.assertEqual(upper['links'][0]['status'], 'candidate')
        self.assertEqual(len(audit['fact_relations']), 3)
        self.assertEqual(checkpoint['processed_chunks'], ['1'])
        self.assertNotIn('quote', str(upper))
        self.assertEqual(len(store.observations), 2)
        self.assertNotIn('semantic_normalization', store.observations['F1'])
        self.assertNotIn('source_value', store.observations['F1'])

    def test_save_load_restores_middle_history_time_and_evidence(self):
        store, layer = self.fixture()
        upper, checkpoint, _ = pack_event_index(layer, store)
        loaded = TripleFactStore(deepcopy(store.to_dict()))
        restored = unpack_event_index(upper, loaded, checkpoint)
        self.assertEqual(restored['facts']['F1']['event_time'], '2023-05-07')
        self.assertEqual(restored['facts']['F1']['quote'], 'went yesterday')
        self.assertEqual(restored['facts']['F1']['source_value'], 'group')
        self.assertEqual(restored['event_time_assertions']['A1']['value'], '2023-05-07')
        self.assertEqual(restored['event_instances']['E1']['actor'], 'Caroline')
        self.assertEqual(restored['facts']['F1']['status'], 'observed')
        self.assertFalse(restored['current'])
        self.assertEqual(len(loaded.chunk_view()['1']), 1)
        self.assertEqual(len(loaded.facts), 1)
        self.assertEqual(decode_upper_references(upper, loaded)['facts']['F1']['value'], 'group')

    def test_missing_group_endpoint_and_gated_policy_are_rejected(self):
        store, layer = self.fixture()
        layer['event_instances']['E1']['fact_ids'].append('missing')
        with self.assertRaises(ValueError):
            pack_event_index(layer, store)

    def test_dynamic_memory_atomic_round_trip_and_audit_separation(self):
        """服务于真实保存入口：恢复后中层历史、处理检查点和完整写入轨迹都不丢。"""
        from coregraph.dynamic_memory import DynamicMemory
        from coregraph.event_state_layer import EventStateLayer
        from init.config import Config
        config = Config()
        config.upper_memory_mode = 'event_state'
        config.middle_memory_mode = 'entity_fact_index'
        config.upper_memory_layout = 'event_index'
        config.state_write_policy = 'append_only'
        config.memory_confidence_enabled = True
        store, source = self.fixture()
        memory = DynamicMemory(config)
        memory.graph_builder.fact_store = store
        memory.graph_builder.graph.add_nodes_from(['Caroline', 'group'])
        tid = next(iter(store.facts))
        memory.graph_builder.graph.add_edge('Caroline', 'group', edge_type='entity_association', fact_ids=[tid], weight=1)
        memory.chunk_storage = {'1': {'text': 'went yesterday'}}
        memory.entity_name_to_index = {'Caroline': 0, 'group': 1}
        memory.event_state_layer = EventStateLayer(source)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'memory.pkl'
            memory.save_graph(str(path))
            saved = pickle.loads(path.read_bytes())
            self.assertEqual(set(saved['event_state_layer']), {'schema', 'groups', 'links'})
            self.assertNotIn('review_queue', saved['event_state_layer'])
            self.assertTrue((path.parent / saved['ingestion_state']['audit_file']).is_file())
            self.assertIsNone(saved['fact_storage']['observations']['F1']['extraction_confidence'])
            evaluation_id = saved['event_state_layer']['links'][0]['evaluation_id']
            self.assertIn(evaluation_id, saved['confidence_storage']['relation_evaluations'])
            restored = DynamicMemory(config)
            restored.load_graph(str(path))
            self.assertEqual(restored.event_state_layer.processed_chunks, {'1'})
            self.assertEqual(restored.event_state_layer.review_queue, source['review_queue'])
            self.assertEqual(restored.event_state_layer.decisions, source['decisions'])
            self.assertEqual(len(restored.fact_store.observations), 2)
            self.assertFalse(restored.event_state_layer.current)
            self.assertEqual(restored.confidence_records, saved['confidence_storage'])
            self.assertIn('extraction_evaluation_id', restored.event_state_layer.facts['F1'])
            # 已有评价在再次保存时不能被 null 初始化覆盖。
            fact_eval_id = restored.event_state_layer.facts['F1']['extraction_evaluation_id']
            restored.confidence_records['fact_evaluations'][fact_eval_id].update(score=0.85, status='assessed')
            restored.save_graph(str(path))
            re_saved = pickle.loads(path.read_bytes())
            self.assertEqual(re_saved['event_state_layer'], saved['event_state_layer'])
            self.assertEqual(re_saved['fact_storage']['observations']['F1']['extraction_confidence'], 0.85)
            again = DynamicMemory(config)
            again.load_graph(str(path))
            self.assertEqual(again.event_state_layer.facts['F1']['extraction_confidence'], 0.85)

    def test_normalized_semantics_and_distinct_source_are_retained(self):
        """服务于泛化：归一化视图不能吞掉原始属性/值/作用域或不精确的时间范围。"""
        store, layer = self.fixture()
        fact = layer['facts']['F1']
        fact.update(attribute='health_condition', value='absent', source_attribute='not_allergic_to',
                    source_value='Milk', scope={'target': 'Milk'}, source_scope={},
                    event_time=None, event_time_range={'start': '2023-05-01', 'end': '2023-05-07'},
                    semantic_state={'dimension': 'allergy', 'polarity': 'negative'})
        upper, checkpoint, _ = pack_event_index(layer, store)
        restored = unpack_event_index(upper, TripleFactStore(store.to_dict()), checkpoint)['facts']['F1']
        for key in ('attribute', 'value', 'scope', 'source_attribute', 'source_value', 'source_scope',
                    'event_time_range', 'semantic_state'):
            self.assertEqual(restored[key], fact[key])
        self.assertIsNone(restored['event_time'])

    def test_relative_time_range_basis_round_trip_does_not_duplicate_keyword(self):
        """真实 sample0 回归：时间区间自带 basis，加载时必须保留而不能重复传参。"""
        store, layer = self.fixture()
        layer['facts']['F1'].update(event_time=None, event_time_raw='last week',
            time_basis='relative_interval', event_time_range=dict(start='2023-05-01',
                end='2023-05-07', basis='relative_week'))
        upper, checkpoint, _ = pack_event_index(layer, store)
        restored = unpack_event_index(upper, store, checkpoint)
        mention = restored['event_instances']['E1']['time_mentions'][0]
        self.assertEqual(mention['basis'], 'relative_week')
        self.assertEqual(mention['start'], '2023-05-01')

    def test_bad_audit_is_rejected_and_ordinary_facts_need_no_group(self):
        """服务于可靠续跑：审计损坏明确报错；普通事实不会为了凑三层而制造事件。"""
        from coregraph.lean_event_index import write_audit_snapshot, read_audit_snapshot
        store, layer = self.fixture()
        layer['event_instances'] = {}
        layer['event_time_assertions'] = {}
        for f in layer['facts'].values():
            f.update(event_id=None, fact_type='state')
        upper, checkpoint, audit = pack_event_index(layer, store)
        self.assertFalse(upper['groups'])
        with tempfile.TemporaryDirectory() as directory:
            graph = str(Path(directory) / 'graph.pkl')
            checkpoint['audit_file'] = write_audit_snapshot(graph, audit)
            self.assertEqual(read_audit_snapshot(graph, checkpoint), audit)
            (Path(directory) / checkpoint['audit_file']).write_text('{}', encoding='utf-8')
            with self.assertRaises(ValueError):
                read_audit_snapshot(graph, checkpoint)
        store, layer = self.fixture()
        layer['write_policy'] = 'gated'
        with self.assertRaises(ValueError):
            pack_event_index(layer, store)


if __name__ == '__main__':
    unittest.main()
