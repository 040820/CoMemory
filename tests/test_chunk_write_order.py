"""写入顺序回归：证据的唯一身份不能决定会话内的先后顺序。

直接加载 Writer 中的实际排序函数，避免本地顺序检查加载模型或调用 vLLM。
测试覆盖哈希 ID、数字 0、混合 ID 及缺少原始编号的兼容输入。
"""

import ast
from copy import deepcopy
from pathlib import Path
from typing import Any
import unittest


def writer_order(rows):
    """服务于写入回归测试：执行生产函数，而不是在测试中复制排序规则。"""
    path = Path(__file__).resolve().parents[1] / 'coregraph/event_state_writer.py'
    tree = ast.parse(path.read_text(encoding='utf-8-sig'))
    key = next(node for node in ast.walk(tree)
               if isinstance(node, ast.FunctionDef) and node.name == 'order_key')
    namespace = {'Any': Any}
    exec(compile(ast.Module(body=[key], type_ignores=[]), str(path), 'exec'), namespace)
    return sorted(rows, key=namespace['order_key'])


def chunk(cid, source=None, session='D2', time='2023-05-10'):
    row = dict(chunk_id=cid, session_id=session, session_time=time)
    if source is not None:
        row['source_chunk_id'] = source
    return row


class ChunkWriteOrderTests(unittest.TestCase):
    def test_hash_order_does_not_replace_source_order(self):
        rows = [chunk('chunk-a', 7), chunk('chunk-z', 0)]
        self.assertEqual([r['source_chunk_id'] for r in writer_order(rows)], [0, 7])

    def test_original_zero_and_mixed_ids_share_numeric_sequence(self):
        rows = [chunk('2'), chunk('chunk-a', '1'), chunk('chunk-z', 0)]
        self.assertEqual([r['chunk_id'] for r in writer_order(rows)],
                         ['chunk-z', 'chunk-a', '2'])

    def test_numeric_ids_without_source_metadata_keep_numeric_order(self):
        self.assertEqual([r['chunk_id'] for r in writer_order([chunk('10'), chunk('2')])],
                         ['2', '10'])

    def test_opaque_ids_without_order_keep_input_order(self):
        rows = [chunk('chunk-z'), chunk('chunk-a'), chunk('chunk-m')]
        self.assertEqual(writer_order(rows), rows)

    def test_session_time_still_precedes_local_order(self):
        rows = [chunk('chunk-a', 0), chunk('chunk-z', 7, 'D1', '2023-05-08')]
        self.assertEqual([r['session_id'] for r in writer_order(rows)], ['D1', 'D2'])

    def test_evidence_identity_and_input_are_not_modified(self):
        rows = [chunk('chunk-a', 7), chunk('chunk-z', 0)]
        original = deepcopy(rows)
        writer_order(rows)
        self.assertEqual(rows, original)


if __name__ == '__main__':
    unittest.main()
