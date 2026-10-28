"""实验回归检查：哈希 Chunk 的存储顺序不能冒充 Writer 的实际处理顺序。

只加载实验脚本的纯函数 AST，避免单纯测试范围过滤时加载 GPU/LLM 依赖。
"""
import ast
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import unittest


class EmptyGraph:
    """空关联图替身：本测试聚焦历史范围，不测试 NetworkX 实现。"""
    def copy(self):
        return self

    def edges(self, **kwargs):
        return []

    def __iter__(self):
        return iter(())

    def remove_nodes_from(self, nodes):
        pass


class HistoryOrderTest(unittest.TestCase):
    def load_functions(self):
        path = Path(__file__).resolve().parents[1] / 'experiments/integrated_event_state/test_recall_sufficiency_suite.py'
        tree = ast.parse(path.read_text(encoding='utf-8'))
        tree.body = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                     and node.name in ('restrict_graph', 'writer_chunk_order')]
        namespace = dict(deepcopy=deepcopy, GraphEvidenceProvider=SimpleNamespace(
            source_chunk=lambda row: str(row.get('chunk_id', row.get('evidence_chunk_id', '')))))
        exec(compile(tree, str(path), 'exec'), namespace)
        return namespace

    def test_hashed_chunks_follow_writer_not_dictionary_order(self):
        # 字典按原始对话写入 z、a、m；现有 Writer 在会话内按哈希 ID 排 a、m、z。
        # 当前测试必须复现现有 Writer 的行为，不能在补查时偷偷换成另一种顺序。
        source = dict(chunk_storage={cid: dict(chunk_id=cid, session_id='D2',
            session_time='2023-05-10', source_chunk_id=i) for i, cid in enumerate(
                ['chunk-z', 'chunk-a', 'chunk-m'])},
            fact_storage=dict(facts={}, observations={}, time_assertions={}, attribute_assertions={}),
            event_state_layer=dict(groups={}, links=[]), graph=EmptyGraph())
        actual = self.load_functions()['restrict_graph'](source, 'chunk-m')
        self.assertEqual(list(actual['chunk_storage']), ['chunk-a', 'chunk-m'])
        self.assertEqual(list(source['chunk_storage']), ['chunk-z', 'chunk-a', 'chunk-m'])


if __name__ == '__main__':
    unittest.main()
