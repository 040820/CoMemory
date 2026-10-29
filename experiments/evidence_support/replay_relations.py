"""关系失败重放入口：复用已保存的两端 Fact，只重跑写入期关系判断。

服务于引用协议修复验证；不重新抽取 Chunk/Triple，不修改源图。输出关系审计和
独立 links_patch.json，补丁仅包含候选关联，不表示已确认演化或更新许可。
"""
import argparse
import asyncio
from collections import Counter
from copy import deepcopy
from hashlib import sha256
import json
import os
from pathlib import Path
import pickle
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def build_candidate_graph(graph, audit, replayed):
    """仅在副本补充候选链接及审计，不改中层事实、旧审计或有效性。

    追加新审计而不是覆盖失败行，保证重新 load/save 时仍能恢复新候选，同时
    能追溯“旧协议失败→新协议结果”。不同关系标签仍只表示待核查假设。
    """
    result, history = deepcopy(graph), deepcopy(audit)
    links = result['event_state_layer']['links']
    known = {(l['from_fact_id'], l['to_fact_id'], l['relation']) for l in links}
    observations = result['fact_storage']['observations']
    for row in replayed:
        classification = row['classification']
        label = classification['relation']
        old, new = row['old_fact_id'], row['new_fact_id']
        if classification.get('evaluation_status') != 'model_assessed' or label in ('uncertain', 'unrelated'):
            continue
        if old not in observations or new not in observations:
            raise ValueError('候选补丁引用缺失 Fact')
        quote = classification.get('evidence_quote')
        if not quote or quote not in observations[new].get('quote', ''):
            raise ValueError('候选补丁缺少新事实连续原文')
        original = history['fact_relations'][row['row_index']]
        if (original['old_fact_id'], original['new_fact_id']) != (old, new):
            raise ValueError('审计行号与补丁端点不一致')
        key = (old, new, label)
        if key in known:
            continue
        known.add(key)
        lid = sha256(json.dumps([old, new, label]).encode()).hexdigest()[:24]
        links.append(dict(link_id=lid, from_fact_id=old, to_fact_id=new, relation=label,
            status='candidate', review_status='pending', inference_status='not_evaluated'))
        record = dict(deepcopy(original), classification=deepcopy(classification), relation=label,
            relation_id=lid, outcome='recorded_only', applied_to_old=False,
            replay_of_row_index=row['row_index'], replay_of_relation_id=original.get('relation_id'))
        history['fact_relations'].append(record)
    checkpoint = result.setdefault('ingestion_state', {})
    checkpoint['audit_counts'] = {key: len(value) for key, value in history.items() if isinstance(value, list)}
    return result, history


def select_failed_pairs(graph, rows, *, sensitive_only=False, limit=None):
    """重放选择模块：只选技术失败且双端仍存在的历史关系，保留原审计行号。

    真正 uncertain、未曾报错的语义判断不纳入“协议修复收益”。缺少端点显式
    记录 skipped，不以空事实继续请求模型。敏感筛选只用于预检，不代表全量。
    """
    facts = graph['fact_storage']['observations']
    selected, skipped = [], []
    for index, row in enumerate(rows):
        classification = row.get('classification') or {}
        failed = classification.get('validation_error') or classification.get('evaluation_status') == 'technical_error'
        # 兼容没有 validation_error 字段的旧技术失败记录。
        failed = failed or str(classification.get('reason', '')).startswith('关系判断失败：')
        if not failed:
            continue
        if sensitive_only and classification.get('model_relation') not in ('conflict', 'correction', 'replacement'):
            continue
        old, new = row.get('old_fact_id'), row.get('new_fact_id')
        if old == new or old not in facts or new not in facts:
            skipped.append(dict(row_index=index, reason='missing_or_identical_endpoints', old_fact_id=old, new_fact_id=new))
            continue
        if any(not str(facts[fid].get('quote') or '').strip() or
               str(facts[fid]['quote']) not in str(graph['chunk_storage'].get(
                   str(facts[fid].get('chunk_id')), {}).get('text', '')) for fid in (old, new)):
            skipped.append(dict(row_index=index, reason='source_quote_not_grounded',
                                old_fact_id=old, new_fact_id=new))
            continue
        selected.append(dict(row_index=index, original=row, old_fact=dict(facts[old], fact_id=old),
                             new_fact=dict(facts[new], fact_id=new)))
    if limit is not None:
        if limit < 1:
            raise ValueError('limit 必须为正数')
        selected = selected[:limit]
    return selected, skipped


async def run(args):
    """模型装配、逐对重放、落盘及源图指纹检查；每条结果及时 flush。"""
    from coregraph.state_fact_relation import StateFactRelationJudge
    from coregraph.evidence_support.transport import AuditedLLM
    from init.config import Config
    from base.llm import LLMManager

    data = args.source.read_bytes()
    source_hash = sha256(data).hexdigest()
    graph = pickle.loads(data)
    rows = [json.loads(line) for line in args.relations.read_text(encoding='utf-8').splitlines() if line.strip()]
    selected, skipped = select_failed_pairs(graph, rows, sensitive_only=args.sensitive_only, limit=args.limit)
    source_audit = None
    if args.write_graph_copy:
        # 在耗时模型请求前核对原审计行号，避免任务结束才发现输入审计来自另一张图。
        from coregraph.lean_event_index import read_audit_snapshot
        source_audit = read_audit_snapshot(str(args.source), graph['ingestion_state'])
        for pair in selected:
            index = pair['row_index']
            original = source_audit['fact_relations'][index]
            if (original.get('old_fact_id'), original.get('new_fact_id')) != (
                    pair['old_fact']['fact_id'], pair['new_fact']['fact_id']):
                raise ValueError('输入关系审计与源图审计不是同一顺序')
    args.output.mkdir(parents=True, exist_ok=False)
    def save(name, value):
        path = args.output / name
        temp = path.with_suffix(path.suffix + '.tmp')
        temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
        temp.replace(path)
    dependencies = [ROOT / 'coregraph/state_fact_relation.py', ROOT / 'coregraph/evidence_support/transport.py',
                    ROOT / 'base/llm.py', Path(__file__)]
    save('manifest.json', dict(source=str(args.source.resolve()), source_sha256=source_hash,
        relations_sha256=sha256(args.relations.read_bytes()).hexdigest(), selected=len(selected), skipped=skipped,
        sensitive_only=args.sensitive_only, limit=args.limit, config_sha256=sha256(args.config.read_bytes()).hexdigest(),
        code_sha256={str(p.relative_to(ROOT)): sha256(p.read_bytes()).hexdigest() for p in dependencies},
        token_limit=None, max_response_repairs=1, source_graph_modified=False))
    config = Config.parse(str(args.config)).llm
    manager = LLMManager(api_key=config.api_key, model=config.model, base_url=config.base_url,
        enable_concurrent=False, timeout=config.timeout, enable_thinking=False)
    judge = StateFactRelationJudge(AuditedLLM(manager, args.output, compact=not args.standard_json))
    outputs, links = [], []
    with (args.output / 'replayed_relations.jsonl').open('w', encoding='utf-8') as stream:
        for index, pair in enumerate(selected, 1):
            result = await judge.judge(pair['old_fact'], pair['new_fact'])
            old, new = pair['old_fact']['fact_id'], pair['new_fact']['fact_id']
            row = dict(row_index=pair['row_index'], old_fact_id=old, new_fact_id=new,
                       previous_classification=pair['original']['classification'], classification=result)
            stream.write(json.dumps(row, ensure_ascii=False) + '\n')
            stream.flush()
            outputs.append(row)
            if result.get('evaluation_status') == 'model_assessed' and result['relation'] not in ('unrelated', 'uncertain'):
                lid = sha256(json.dumps([old, new, result['relation']]).encode()).hexdigest()[:24]
                links.append(dict(link_id=lid, from_fact_id=old, to_fact_id=new, relation=result['relation'],
                                  status='candidate', review_status='pending', inference_status='not_evaluated'))
            save('links_patch.json', dict(schema='candidate_links_patch_v1', source_sha256=source_hash,
                links=links, update_applied=False, note='仅候选关联；不修改事实有效性'))
            summary = dict(completed=index, total=len(selected), finished=index == len(selected),
                evaluation_status_counts=dict(Counter(x['classification']['evaluation_status'] for x in outputs)),
                relation_counts=dict(Counter(x['classification']['relation'] for x in outputs)),
                candidate_links=len(links), source_unchanged=sha256(args.source.read_bytes()).hexdigest() == source_hash,
                update_applied=False, accuracy=None)
            save('summary.json', summary)
            print(f"REPLAY {index}/{len(selected)} relation={result['relation']} evaluation={result['evaluation_status']}", flush=True)
    if not selected:
        save('links_patch.json', dict(schema='candidate_links_patch_v1', source_sha256=source_hash,
            links=[], update_applied=False, note='没有符合条件的失败事实对'))
        save('summary.json', dict(completed=0, total=0, finished=True, skipped=skipped, source_unchanged=True))
    if args.write_graph_copy:
        # 新副本使用内容寻址审计；原图和它引用的审计文件从不覆写。
        from coregraph.lean_event_index import read_audit_snapshot, write_audit_snapshot
        copied, audit = build_candidate_graph(graph, source_audit, outputs)
        target = args.output / 'candidate_graph.pkl'
        copied['ingestion_state']['audit_file'] = write_audit_snapshot(str(target), audit)
        temporary = target.with_suffix('.pkl.tmp')
        temporary.write_bytes(pickle.dumps(copied));temporary.replace(target)
        save('candidate_graph_manifest.json', dict(source_sha256=source_hash, graph=str(target),
            graph_sha256=sha256(target.read_bytes()).hexdigest(), original_links=len(graph['event_state_layer']['links']),
            candidate_links=len(copied['event_state_layer']['links']), fact_validity_changed=False,
            original_audit_preserved=True))
    if sha256(args.source.read_bytes()).hexdigest() != source_hash:
        raise AssertionError('来源图发生变化')
    print('REPLAY_COMPLETE', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--relations', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=ROOT / 'config/Memory_EventState_sample0_rebuild.yaml')
    parser.add_argument('--sensitive-only', action='store_true')
    parser.add_argument('--write-graph-copy', action='store_true', help='将新候选边写入独立图副本，保留旧失败审计和所有原始事实')
    parser.add_argument('--limit', type=int)
    parser.add_argument('--standard-json', action='store_true')
    os.environ.setdefault('HF_HUB_OFFLINE', '1')
    os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
    asyncio.run(run(parser.parse_args()))
