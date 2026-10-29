"""证据支撑模块运行入口：读取三层图副本，保存检索、审查、可选回答的审计。

所有算法在 coregraph/evidence_support；本文件只负责参数、模型装配和输出。
--demo 使用明确标记的人工固定样例；不能据此报告 LoCoMo/LongMemEval 效果。
"""

import argparse
import asyncio
from dataclasses import asdict
from hashlib import sha256
import json
import os
from pathlib import Path
import pickle
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def save(path, value):
    """实验审计模块：仅写新输出目录，不写入来源图。"""
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def demo_graph():
    """人工协议演示：两家公司工作说法由全职/兼职说明补充，无真实模型标签。"""
    texts = {'C1': 'I work at A.', 'C2': 'I work at B.', 'C3': 'A is full time; B is part time.'}
    return dict(event_state_layer=dict(schema='event_index_v1', groups={}, links=[
        dict(from_fact_id='F1', to_fact_id='F3', relation='complement'),
        dict(from_fact_id='F2', to_fact_id='F3', relation='complement')]),
        chunk_storage={cid: dict(chunk_id=cid, text=text, session_id='D1',
                                session_time='2024-01-01') for cid, text in texts.items()},
        fact_storage=dict(facts={}, time_assertions={}, attribute_assertions={}, observations={
            fid: dict(fact_id=fid, entity_id='User', attribute='employer', value=value,
                      scope={}, modality='actual', chunk_id=cid, quote=texts[cid])
            for fid, cid, value in [('F1', 'C1', 'A'), ('F2', 'C2', 'B'), ('F3', 'C3', 'A and B')]}))


async def run(args):
    # 解析 --help 不加载模型依赖；实际运行使用项目的正式模块导入。
    from coregraph.evidence_support import EvidenceSupport, EvidenceSupportConfig
    from coregraph.evidence_support.transport import AuditedLLM

    values = json.loads(args.settings.read_text(encoding='utf-8')) if args.settings else {}
    settings = EvidenceSupportConfig(**values)
    if args.demo:
        graph, fingerprint = demo_graph(), None
        args.old_fact_id = args.old_fact_id or 'F1'
        args.new_fact_id = args.new_fact_id or 'F2'
    else:
        data = args.source.read_bytes()
        graph, fingerprint = pickle.loads(data), sha256(data).hexdigest()
    if args.mode == 'relation' and (not args.old_fact_id or not args.new_fact_id):
        raise ValueError('关系模式必须提供 old/new Fact ID')
    if args.mode == 'question' and not args.query:
        raise ValueError('问答模式必须提供 query')
    if args.answer and (args.mode != 'question' or args.retrieval_only):
        raise ValueError('--answer 仅用于启用 Judge 的 question 模式')
    args.output.mkdir(parents=True, exist_ok=False)
    dependencies = list((ROOT / 'coregraph/evidence_support').glob('*.py')) + [
        ROOT / 'coregraph/evidence_judge.py', ROOT / 'coregraph/conflict_review.py',
        ROOT / 'coregraph/graph_evidence_provider.py', ROOT / 'coregraph/state_slot.py', Path(__file__)]
    dependencies += [ROOT / 'base/llm.py', ROOT / 'base/embeddings.py']
    manifest = dict(experiment='evidence_support_v1', synthetic_fixture=args.demo,
        source=None if args.demo else str(args.source.resolve()), source_sha256=fingerprint,
        settings=asdict(settings), mode=args.mode, as_of=args.as_of,
        semantic_enabled=not args.no_semantic, judge_enabled=not args.retrieval_only,
        code_sha256={str(p.relative_to(ROOT)): sha256(p.read_bytes()).hexdigest() for p in dependencies})
    save(args.output / 'manifest.json', manifest)
    embeddings, manager = None, None
    if not args.no_semantic or not args.retrieval_only:
        from init.config import Config
        config = Config.parse(str(args.config))
        if not args.no_semantic:
            from base.embeddings import EmbeddingManager
            embeddings = EmbeddingManager(config.embedding)
            manifest['embedding_model'] = config.embedding.model
        if not args.retrieval_only:
            from base.llm import LLMManager
            lc = config.llm
            manager = LLMManager(api_key=lc.api_key, model=lc.model, base_url=lc.base_url,
                enable_concurrent=False, timeout=lc.timeout, enable_thinking=False)
            manifest['judge_model'] = lc.model
        manifest['model_config_sha256'] = sha256(args.config.read_bytes()).hexdigest()
        save(args.output / 'manifest.json', manifest)

    llm = None if manager is None else AuditedLLM(manager, args.output, compact=args.compact_json)
    support = EvidenceSupport(graph, embeddings, llm, settings)
    if args.mode == 'relation':
        report = await support.review_pair(args.old_fact_id, args.new_fact_id,
            args.query or '核查新旧事实的实际关系及其证据。', as_of=args.as_of)
    elif args.answer:
        report = await support.answer_question(args.query, llm, claim_ids=args.claim_id, as_of=args.as_of)
    else:
        report = await support.review_question(args.query, claim_ids=args.claim_id, as_of=args.as_of)
    unchanged = args.demo or sha256(args.source.read_bytes()).hexdigest() == fingerprint
    if not unchanged:
        raise AssertionError('来源图文件发生变化')
    save(args.output / 'report.json', report)
    for key, filename in [('retrieval_rounds', 'retrieval_rounds.jsonl'), ('rounds', 'judge_rounds.jsonl')]:
        (args.output / filename).write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n'
            for row in report[key]), encoding='utf-8')
    summary = dict(status=report['status'], stop_reason=report['stop_reason'],
        evidence_chunks=len(report['evidence']), judge_calls=len(report['rounds']),
        source_unchanged=unchanged, update_applied=False, synthetic_fixture=args.demo)
    save(args.output / 'summary.json', summary)
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--source', type=Path)
    source.add_argument('--demo', action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--mode', choices=['relation', 'question'], default='relation')
    parser.add_argument('--old-fact-id')
    parser.add_argument('--new-fact-id')
    parser.add_argument('--claim-id', action='append', default=[])
    parser.add_argument('--query')
    parser.add_argument('--as-of', help='ISO 观察截止时间，非事件有效时间')
    parser.add_argument('--settings', type=Path, default=ROOT / 'config/evidence_support.json')
    parser.add_argument('--config', type=Path, default=ROOT / 'config/Memory_EventState_sample0_rebuild.yaml')
    parser.add_argument('--no-semantic', action='store_true')
    parser.add_argument('--retrieval-only', action='store_true')
    parser.add_argument('--answer', action='store_true')
    transport = parser.add_mutually_exclusive_group()
    transport.add_argument('--compact-json', dest='compact_json', action='store_true',
                           help='默认：用紧凑 EBNF 约束 vLLM 输出，需 xgrammar')
    transport.add_argument('--standard-json', dest='compact_json', action='store_false',
                           help='其他 OpenAI 兼容服务可显式使用标准 JSON Schema')
    parser.set_defaults(compact_json=True)
    args = parser.parse_args()
    os.environ.setdefault('HF_HUB_OFFLINE', '1')
    os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
    asyncio.run(run(args))
