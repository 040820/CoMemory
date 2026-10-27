"""将已有记忆另存为实体关联索引＋独立事实表；不调用模型、不覆盖来源文件。

所属模块：独立结构迁移工具。只转换已有事实的表示，不能补回旧 DiGraph 已
覆盖的关系，不能修复旧事实抽取错误，也不重新裁决任何状态或事件身份。
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from coregraph.dynamic_memory import DynamicMemory
from init.config import Config


def migrate_graph(source: Path, output: Path) -> dict:
    """服务于显式迁移：读取旧文件，在新文件中保存新索引，返回结构统计。

    输入和输出必须不同且输出不能已存在。源哈希、来源完整性和新中层数量
    写入旁路 JSON，便于区分“格式迁移成功”与“原始事实完整重建”。
    """
    source, output = source.resolve(), output.resolve()
    if source == output:
        raise ValueError('不能覆盖来源图，请指定新的输出路径')
    if output.exists() or output.with_suffix('.migration.json').exists():
        raise FileExistsError('迁移输出已存在，拒绝覆盖')
    before = sha256(source.read_bytes()).hexdigest()
    config = Config()
    config.upper_memory_mode = 'event_state'
    config.middle_memory_mode = 'entity_fact_index'
    memory = DynamicMemory(config)
    memory.load_graph(str(source))
    output.parent.mkdir(parents=True, exist_ok=True)
    memory.save_graph(str(output))
    if sha256(source.read_bytes()).hexdigest() != before:
        raise RuntimeError('迁移期间来源图发生变化，请检查并重新选择快照')
    report = {
        'source': str(source), 'source_sha256': before, 'output': str(output),
        'schema': 'entity_fact_index_v1', 'origin': memory.fact_store.origin,
        'entities': memory.graph_builder.graph.number_of_nodes(),
        'association_edges': memory.graph_builder.graph.number_of_edges(),
        'middle_facts': len(memory.fact_store.facts), 'chunks': len(memory.chunk_storage),
        'upper_observations': len(memory.event_state_layer.facts),
        'candidate_relations': len(memory.event_state_layer.fact_relations),
        'llm_called': False, 'lost_legacy_facts_recovered': False,
    }
    output.with_suffix('.migration.json').write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(migrate_graph(args.source, args.output), ensure_ascii=False, indent=2))
