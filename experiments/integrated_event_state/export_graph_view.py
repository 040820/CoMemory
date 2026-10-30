"""只读图快照导出器：服务于实体关联与三层证据可视化，不调用模型或写回原图。

只导出实际持久化字段，排除向量和重复审计，供离线 HTML 演示读取。实体关联
边不伪装成语义关系；具体方向和谓词取自独立 Triple 主表。没有评分不补造分数。
"""

import argparse
from hashlib import sha256
import json
from pathlib import Path
import pickle


def build_view(data, source_path='', source_sha256=''):
    """把实体、事实、事件分组和原始 Chunk 转成轻量 JSON，保留来源与完整原文。"""
    if not isinstance(data, dict) or data.get('event_state_layer', {}).get('schema') != 'event_index_v1':
        raise ValueError('本导出器要求 event_index_v1 图快照')
    graph, table, upper = data['graph'], data['fact_storage'], data['event_state_layer']
    triples = {fid: {k: row.get(k) for k in ('src', 'relation', 'tgt', 'chunk_id')}
               for fid, row in table.get('facts', {}).items()}
    fields = ('entity_id', 'attribute', 'predicate', 'value', 'event_id', 'chunk_id',
              'observed_at', 'event_time', 'event_time_raw', 'event_time_range', 'modality',
              'quote', 'evidence_chunk_id', 'evidence_quote', 'raw_value', 'time_basis',
              'middle_fact_ids', 'source_middle_fact_ids', 'time_middle_fact_ids',
              'extraction_confidence', 'extraction_evaluation_id')
    observations = {fid: {k: row[k] for k in fields if k in row}
                    for fid, row in table.get('observations', {}).items()}
    assertions = {fid: dict({k: row[k] for k in fields if k in row}, record_type=kind)
                  for kind, name in (('time', 'time_assertions'), ('attribute', 'attribute_assertions'))
                  for fid, row in table.get(name, {}).items()}
    chunks = {str(cid): {k: row.get(k) for k in ('text', 'session_id', 'session_time')}
              for cid, row in data['chunk_storage'].items()}
    associations = [dict(source=str(s), target=str(t), fact_ids=list(edge.get('fact_ids', [])))
                    for s, t, edge in graph.edges(data=True)]
    sessions = {str(row.get('session_id')) for row in chunks.values()}
    return dict(meta=dict(source_path=source_path, source_sha256=source_sha256,
        schema='event_index_v1', source_origin=table.get('origin'), sessions=len(sessions),
        entities=graph.number_of_nodes(), associations=len(associations), triples=len(triples),
        chunks=len(chunks), observations=len(observations), groups=len(upper['groups']),
        candidate_links=len(upper['links']), confidence_enabled='confidence_storage' in data),
        entities=sorted(str(node) for node in graph.nodes), associations=associations,
        triples=triples, observations=observations, assertions=assertions,
        groups=upper['groups'], links=upper['links'], chunks=chunks)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument('--source', type=Path)
    inputs.add_argument('--from-json', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--html-output', type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('输出文件已存在，拒绝覆盖历史演示')
    if args.html_output and args.html_output.exists():
        raise ValueError('HTML 已存在，拒绝覆盖历史演示')
    if args.source:
        content = args.source.read_bytes()
        result = build_view(pickle.loads(content), str(args.source.resolve()), sha256(content).hexdigest())
    else:
        result = json.loads(args.from_json.read_text(encoding='utf-8'))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, separators=(',', ':'), default=str), encoding='utf-8')
    if args.html_output:
        # 原始对话只是数据，转义 HTML 脚本闭合符，不能让对话内容变成页面代码。
        payload = json.dumps(result, ensure_ascii=False, separators=(',', ':'), default=str).replace('<', '\\u003c').replace('&', '\\u0026')
        template = Path(__file__).with_name('graph_viewer_fragment.html').read_text(encoding='utf-8')
        args.html_output.parent.mkdir(parents=True, exist_ok=True)
        args.html_output.write_text(template.replace('__GRAPH_DATA__', payload), encoding='utf-8')
    print(json.dumps(result['meta'], ensure_ascii=False))
    print('bytes', args.output.stat().st_size)


if __name__ == '__main__':
    main()
