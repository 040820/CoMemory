"""精简三层存储适配模块：中层拥有事实，上层只拥有事件分组和候选关联。

服务于 entity_fact_index + append_only 实验模式。现有抽取器、历史召回器和
问答接口使用兼容视图；落盘时不重复保存完整 Observation、模型解释或核查队列。
本模块只组织来源断言，不确认现实真值，不合并事件身份，也不建立因果结论。
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any
from hashlib import sha256
import json
import os
from pathlib import Path
import tempfile

from .triple_fact_store import encode_upper_references, decode_upper_references


# 服务于检查点恢复：这些是运行元数据，不是知识图的高层节点。
CHECKPOINT_FIELDS = ('writer_schema_version', 'slot_schema_version', 'write_policy', 'processed_chunks')
AUDIT_FIELDS = ('decisions', 'review_queue', 'skipped_candidates', 'event_link_decisions', 'fact_relations')
USEFUL_RELATIONS = {'duplicate', 'complement', 'coexists', 'conflict', 'correction', 'replacement'}
FACT_DEFAULTS = dict(kind='observation', operation='APPEND', status='observed',
                     assertion_status='source_grounded', validity_status='unresolved',
                     target_fact_id=None, scope={}, modality='actual', semantic_state=None,
                     semantic_normalization=None, event_id=None, event_attributes={},
                     event_time=None, event_time_raw=None, event_time_range=None,
                     event_triple_refs=[], time_triple_refs=[])


def write_audit_snapshot(graph_path: str, audit: dict[str, Any]) -> str:
    """服务于可复现实验：审计独立存储，按内容哈希命名，不覆盖历史检查点。

    每次会话保存一份审计快照；当前主图只保存相对路径。采用同目录临时文件和
    原子替换，旧图/旧审计在新图提交失败时仍可恢复。这里不执行任何模型判断。
    """
    content = json.dumps(audit, ensure_ascii=False, sort_keys=True, default=str).encode('utf-8')
    root = Path(graph_path).resolve().parent
    relative = Path(Path(graph_path).stem + '_audit') / (sha256(content).hexdigest() + '.json')
    target = root / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as stream:
            temporary = stream.name
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.replace(temporary, target)
        finally:
            if os.path.exists(temporary):
                os.remove(temporary)
    return relative.as_posix()


def read_audit_snapshot(graph_path: str, checkpoint: dict[str, Any]) -> dict[str, Any]:
    """服务于增量恢复：严格读取本检查点对应的审计，缺失/损坏时明确失败。

    不把其他运行的审计混入当前历史；相对路径必须处于图文件所在目录中。
    """
    root = Path(graph_path).resolve().parent
    relative = checkpoint.get('audit_file')
    if not relative:
        raise ValueError('精简图缺少审计检查点路径')
    target = (root / relative).resolve()
    if root not in target.parents:
        raise ValueError('审计路径越出图目录')
    content = target.read_bytes()
    if sha256(content).hexdigest() != target.stem:
        raise ValueError('审计内容与检查点哈希不一致')
    return json.loads(content)


def pack_event_index(layer: dict[str, Any], store: Any):
    """服务于保存模块：返回精简上层、检查点和独立审计，并更新中层语义主表。

    原始 Triple 主表及 ID 完全不变。来源断言保留时间、作用域、模态和证据；
    删除可恢复的常量、空字段和相同的 source_* 副本。模型处理轨迹只进入审计。
    旧 gated 协议可能包含已裁决历史，拒绝把它静默降格为追加写入。
    """
    if layer.get('write_policy') != 'append_only' or any(layer.get(k) for k in ('current', 'members', 'events')):
        raise ValueError('精简事件索引仅支持未裁决的 append_only 记忆')
    encoded = encode_upper_references(layer, store)
    observations = {}
    for fid, source in encoded.get('facts', {}).items():
        row = deepcopy(source)
        if row.get('fact_id') != fid or not row.get('middle_fact_ids'):
            raise ValueError(f'中层来源断言缺少稳定 ID 或原始 Triple：{fid}')
        for key, value in FACT_DEFAULTS.items():
            if row.get(key) == value:
                row.pop(key, None)
        for key in ('attribute', 'value', 'scope', 'modality'):
            if row.get('source_' + key) == source.get(key):
                row.pop('source_' + key, None)
        # slot_key/基数可由规范字段重新计算；归一化诊断存在 decisions 审计中。
        for key in ('slot_key', 'cardinality', 'semantic_normalization'):
            row.pop(key, None)
        observations[fid] = {k: v for k, v in row.items() if v is not None}
    time_assertions = deepcopy(encoded.get('event_time_assertions', {}))
    attribute_assertions = deepcopy(encoded.get('event_attribute_assertions', {}))
    all_ids = set(observations) | set(time_assertions) | set(attribute_assertions)
    if len(all_ids) != len(observations) + len(time_assertions) + len(attribute_assertions):
        raise ValueError('来源断言与属性断言 ID 冲突')
    groups = {}
    for eid, event in encoded.get('event_instances', {}).items():
        ids = list(dict.fromkeys(event.get('fact_ids', []) + event.get('observed_time_assertion_ids', [])
                                 + event.get('observed_attribute_assertion_ids', [])))
        if not ids or any(fid not in all_ids for fid in ids):
            raise ValueError(f'事件分组引用不存在的中层事实：{eid}')
        groups[eid] = {'event_id': eid, 'fact_ids': ids}
    links = []
    for relation in layer.get('fact_relations', []):
        old, new, label = (relation.get(k) for k in ('old_fact_id', 'new_fact_id', 'relation'))
        # 无关、无法判断和模型格式失败仅留审计，不污染知识关联。
        # 可用引文只证明该建议有来源，不证明“纠正/变化”已成立。
        quote = relation.get('classification', {}).get('evidence_quote')
        if label not in USEFUL_RELATIONS or not quote or new not in observations or old not in observations:
            continue
        if quote not in str(layer['facts'][new].get('quote') or ''):
            continue
        links.append({'link_id': relation['relation_id'], 'from_fact_id': old, 'to_fact_id': new,
                      'relation': label, 'status': 'candidate'})
    # 完整验证后才替换主表，失败不会部分提交中层。
    store.observations, store.time_assertions, store.attribute_assertions = (
        observations, time_assertions, attribute_assertions)
    checkpoint = {k: deepcopy(layer.get(k)) for k in CHECKPOINT_FIELDS}
    checkpoint['audit_counts'] = {k: len(layer.get(k, [])) for k in AUDIT_FIELDS}
    audit = {k: deepcopy(layer.get(k, [])) for k in AUDIT_FIELDS}
    return {'schema': 'event_index_v1', 'groups': groups, 'links': links}, checkpoint, audit


def unpack_event_index(upper: dict[str, Any], store: Any,
                       checkpoint: dict[str, Any] | None = None) -> dict[str, Any]:
    """服务于加载/检索模块：从中层投影旧接口，证据只按 ID 展开，不改动文件。

    此投影是运行时兼容层，避免同时重写抽取、双路召回和问答逻辑。分组是来源
    锚点，不代表跨 Chunk 的事件已确认相同；所有关联仍是候选，当前状态保持空。
    """
    if upper.get('schema') != 'event_index_v1':
        raise ValueError('不支持的精简事件索引版本')
    data = dict(version=1, write_policy='append_only', writer_schema_version=4,
                slot_schema_version=2, facts=deepcopy(store.observations),
                event_time_assertions=deepcopy(store.time_assertions),
                event_attribute_assertions=deepcopy(store.attribute_assertions),
                middle_reference_schema=1, events={}, current={}, members={},
                processed_chunks=[], event_instances={}, fact_relations=[])
    data.update({k: deepcopy(v) for k, v in (checkpoint or {}).items() if k in CHECKPOINT_FIELDS})
    data = decode_upper_references(data, store)
    from .state_slot import slot_key, cardinality_for
    from .event_state_layer import triple_ref_key
    for fact in data['facts'].values():
        for key, default in FACT_DEFAULTS.items():
            fact.setdefault(key, deepcopy(default))
        for key in ('attribute', 'value', 'scope', 'modality'):
            fact.setdefault('source_' + key, deepcopy(fact.get(key)))
        fact['slot_key'] = slot_key(fact)
        fact['cardinality'] = cardinality_for(fact['attribute'], fact)
    for key in ('by_entity', 'by_chunk', 'by_triple', 'event_instance_by_actor',
                'event_instance_by_chunk', 'event_instance_by_triple'):
        data[key] = {}
    for eid, group in upper['groups'].items():
        ids = group['fact_ids']
        known = set(data['facts']) | set(data['event_time_assertions']) | set(data['event_attribute_assertions'])
        if group.get('event_id') != eid or any(fid not in known for fid in ids):
            raise ValueError(f'精简事件分组存在悬空引用：{eid}')
        facts = [data['facts'][fid] for fid in ids if fid in data['facts']]
        if not facts:
            raise ValueError(f'事件分组没有来源断言：{eid}')
        anchor = facts[0]
        refs = [r for f in facts for r in (f.get('event_triple_refs') or f['triple_refs'])]
        chunks = list(dict.fromkeys(str(f['chunk_id']) for f in facts))
        data['event_instances'][eid] = dict(event_id=eid, actor=anchor['entity_id'],
            entity_index=anchor['entity_index'], action=anchor['attribute'], target=anchor['value'],
            fact_ids=[f['fact_id'] for f in facts], anchor_quote=anchor['quote'], scope=None,
            evidence_chunk_ids=chunks, source_triples=refs, status='observed', identity_status='unresolved',
            event_time=None, reported_event_time=anchor.get('event_time'), current_time_assertion_id=None,
            current_attributes={}, time_mentions=[dict(chunk_id=f['chunk_id'], quote=f['quote'],
                raw=f.get('event_time_raw'), basis=f.get('time_basis'), **(f.get('event_time_range') or {}))
                for f in facts if f.get('event_time_raw') and not f.get('event_time')],
            observed_time_assertion_ids=[fid for fid in ids if fid in data['event_time_assertions']],
            observed_attribute_assertion_ids=[fid for fid in ids if fid in data['event_attribute_assertions']])
        data['event_instance_by_actor'].setdefault(anchor['entity_id'], []).append(eid)
        for cid in chunks:
            data['event_instance_by_chunk'].setdefault(cid, []).append(eid)
        for ref in refs:
            target = data['event_instance_by_triple'].setdefault(triple_ref_key(ref), [])
            if eid not in target:
                target.append(eid)
    for link in upper['links']:
        if link['from_fact_id'] not in data['facts'] or link['to_fact_id'] not in data['facts']:
            raise ValueError('精简事件关联存在悬空端点')
        data['fact_relations'].append(dict(relation_id=link['link_id'], old_fact_id=link['from_fact_id'],
            new_fact_id=link['to_fact_id'], relation=link['relation'], status='candidate',
            outcome='recorded_only', applied_to_old=False, inference_status='not_evaluated'))
        if link.get('evaluation_id'):
            # 服务于解释入口：只投影独立评价引用，不把评价详情复制回上层。
            data['fact_relations'][-1]['evaluation_id'] = link['evaluation_id']
    return data
