"""中层事实存储模块：独立保存完整 Triple，实体图只索引事实 ID。

借鉴 HippoRAG 的事实存储/图传播职责分离。事实按来源 Chunk 保留出现记录，
同一实体对的不同谓词、相反方向和历史来源均不会覆盖；精确重复写入幂等。
本模块不判断事实有效性，不推导替换，也不把边权当作证据可信度。
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from .event_candidate_schema import triple_id


class TripleFactStore:
    """服务于中层：事实主表为唯一持久化内容，其余反向索引可从主表重建。"""

    def __init__(self, data: dict[str, Any] | None = None):
        self.facts: dict[str, dict[str, Any]] = {}
        self.by_chunk: dict[str, list[str]] = {}
        self.by_entity: dict[str, list[str]] = {}
        self.origin = 'extracted'
        if data is not None:
            if data.get('version') != 1:
                raise ValueError('不支持的中层事实表版本')
            self.origin = data.get('origin', 'unknown')
            for fact_id, record in data.get('facts', {}).items():
                if triple_id(record) != fact_id:
                    raise ValueError('事实 ID 与主体、谓词、客体、Chunk 来源不一致')
                self.upsert(record)

    def upsert(self, record: dict[str, Any]) -> str:
        """保存一个来源出现记录，返回中层 fact_id；不会合并不同 Chunk。

        输入使用既有 src/relation/tgt/chunk_id 字段；fact_id 与 triple_id 相同，
        便于旧上层引用迁移。重复记录只补充非空元数据，不能重复累积支持次数。
        """
        required = ('src', 'relation', 'tgt', 'chunk_id')
        if any(record.get(key) is None or not str(record[key]).strip() for key in required):
            raise ValueError('完整事实需要 src/relation/tgt/chunk_id')
        row = deepcopy(record)
        for key in required:
            row[key] = str(row[key]).strip()
        fact_id = triple_id(row)
        for key in ('fact_id', 'triple_id'):
            if row.get(key) and row[key] != fact_id:
                raise ValueError('提供的事实 ID 不匹配原始 Triple 来源')
        row.update(fact_id=fact_id, triple_id=fact_id)
        if fact_id in self.facts:
            self.facts[fact_id].update({key: value for key, value in row.items()
                if value is not None and not (isinstance(value, str) and value == '')})
            return fact_id
        self.facts[fact_id] = row
        self.by_chunk.setdefault(row['chunk_id'], []).append(fact_id)
        for entity in dict.fromkeys((row['src'], row['tgt'])):
            self.by_entity.setdefault(entity, []).append(fact_id)
        return fact_id

    def chunk_view(self) -> dict[str, list[dict[str, Any]]]:
        """服务于旧写入器兼容：返回引用主表记录的按 Chunk 视图，不另存一套事实。"""
        return {cid: [self.facts[fid] for fid in ids] for cid, ids in self.by_chunk.items()}

    def to_dict(self) -> dict[str, Any]:
        """持久化主表与来源完整性；反向索引不重复落盘。"""
        return {'version': 1, 'origin': self.origin, 'facts': self.facts}


# 服务于上层序列化：只替换明确的 Triple 引用字段，不改普通文本和模型解释。
REFERENCE_FIELDS = {
    'triple_refs': 'middle_fact_ids', 'source_triples': 'source_middle_fact_ids',
    'event_triple_refs': 'event_middle_fact_ids', 'time_triple_refs': 'time_middle_fact_ids',
}


def encode_upper_references(layer: dict[str, Any], store: TripleFactStore) -> dict[str, Any]:
    """服务于三层持久化：上层用中层事实 ID 引用证据，避免复制 Triple 快照。

    任何悬空引用均显式失败，不静默丢掉来源。原始 Chunk 引文、断言值和候选
    关系判断仍由上层保留，它们不是中层 Triple 的重复索引。
    """
    def visit(value):
        if isinstance(value, list):
            return [visit(item) for item in value]
        if not isinstance(value, dict):
            return deepcopy(value)
        result = {}
        for key, item in value.items():
            if key in REFERENCE_FIELDS and isinstance(item, list):
                ids = []
                for ref in item:
                    fid = triple_id(ref)
                    if fid not in store.facts:
                        raise ValueError(f'上层引用缺少中层事实：{fid}')
                    ids.append(fid)
                result[REFERENCE_FIELDS[key]] = ids
            else:
                result[key] = visit(item)
        return result
    result = visit(layer)
    result['middle_reference_schema'] = 1
    return result


def decode_upper_references(layer: dict[str, Any] | None,
                             store: TripleFactStore) -> dict[str, Any] | None:
    """服务于读取兼容：解析中层 ID 为运行时引用视图，不对来源文件执行升级写入。"""
    if not layer or 'middle_reference_schema' not in layer:
        return layer
    if layer['middle_reference_schema'] != 1:
        raise ValueError('不支持的上层中层引用版本')
    reverse = {target: source for source, target in REFERENCE_FIELDS.items()}

    def visit(value):
        if isinstance(value, list):
            return [visit(item) for item in value]
        if not isinstance(value, dict):
            return deepcopy(value)
        result = {}
        for key, item in value.items():
            if key in reverse:
                try:
                    # 上层运行时只恢复轻量来源引用，向量和描述仍只属于事实主表。
                    # 否则同一事实被引用多次会在内存反复复制大向量。
                    result[reverse[key]] = [{name: store.facts[fid][name] for name in
                        ('src', 'relation', 'tgt', 'chunk_id', 'triple_id', 'fact_id')}
                        for fid in item]
                except KeyError as error:
                    raise ValueError(f'上层证据引用不存在：{error}') from error
            elif key != 'middle_reference_schema':
                result[key] = visit(item)
        return result
    return visit(layer)
