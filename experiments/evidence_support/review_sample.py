"""完整样本的关系证据审查：共享向量缓存，逐对测试全部待审查链接。

此入口只评估候选关系与证据，不把原图标签当标准答案，不进行事实覆盖。
每对单独保存报告/原始响应；根目录 summary.json 记录进度，不声称科学准确率。
"""

import argparse
import asyncio
from collections import Counter
from hashlib import sha256
import importlib.metadata
import json
import os
from pathlib import Path
import pickle
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def save(path, value):
    """实验进度模块：原子替换本次任务文件，来源图从不写回。"""
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(path)


def select_pairs(graph, limit=None):
    """选择模块：只取真实 pending 候选，按已保存写入位置稳定排序，不生成金标。"""
    from coregraph.conflict_review import ConflictPairIndex
    pairs = ConflictPairIndex(graph).pending()
    pairs.sort(key=lambda pair: (pair['new_fact'].get('position', 0),
                                pair['old_fact'].get('position', 0), pair['link_id']))
    if limit is not None and limit < 1:
        raise ValueError('limit 必须为正数')
    return pairs if limit is None else pairs[:limit]


async def run(args):
    from coregraph.evidence_support import EvidenceSupport, EvidenceSupportConfig
    from coregraph.evidence_support.transport import AuditedLLM
    data = args.source.read_bytes()
    graph = pickle.loads(data)
    source_hash = sha256(data).hexdigest()
    pairs = select_pairs(graph, args.limit)
    settings = EvidenceSupportConfig(**json.loads(args.settings.read_text(encoding='utf-8')))
    args.output.mkdir(parents=True, exist_ok=False)
    source_files = list((ROOT / 'coregraph/evidence_support').glob('*.py')) + [
        ROOT / 'base/llm.py', ROOT / 'base/embeddings.py', ROOT / 'coregraph/evidence_judge.py',
        ROOT / 'coregraph/conflict_review.py', ROOT / 'coregraph/graph_evidence_provider.py',
        ROOT / 'coregraph/evidence_confidence.py', Path(__file__)]
    manifest = dict(source=str(args.source.resolve()), source_sha256=source_hash,
        scope='all_pending_pairs_in_frozen_sample_not_qa_benchmark', pairs=len(pairs),
        settings=settings.to_dict(), as_of=args.as_of, update_applied=False,
        label_source='unlabeled_candidate_relations_not_gold',
        code_sha256={str(p.relative_to(ROOT)):sha256(p.read_bytes()).hexdigest() for p in source_files})
    save(args.output / 'manifest.json', manifest)
    embeddings, manager = None, None
    if pairs and (not args.no_semantic or not args.retrieval_only):
        from init.config import Config
        config = Config.parse(str(args.config))
        if not args.no_semantic:
            from base.embeddings import EmbeddingManager
            embeddings = EmbeddingManager(config.embedding)
        if not args.retrieval_only:
            from base.llm import LLMManager
            model = config.llm
            manager = LLMManager(api_key=model.api_key, model=model.model, base_url=model.base_url,
                enable_concurrent=False, enable_thinking=False, timeout=model.timeout)
        manifest.update(model_config_sha256=sha256(args.config.read_bytes()).hexdigest(),
                        embedding_model=config.embedding.model, judge_model=config.llm.model)
        if not args.standard_json:
            manifest['xgrammar_version'] = importlib.metadata.version('xgrammar')
        save(args.output / 'manifest.json', manifest)
    support = EvidenceSupport(graph, embeddings, config=settings)
    summaries = []
    print(f'SAMPLE_REVIEW_START pairs={len(pairs)}', flush=True)
    for index, pair in enumerate(pairs, 1):
        # 目录名不采用输入路径片段，避免旧文件中的异常 ID 越过输出目录。
        directory = args.output / ('pair_' + sha256(pair['link_id'].encode()).hexdigest()[:20])
        directory.mkdir()
        save(directory / 'pair.json', pair)
        support.llm = None if manager is None else AuditedLLM(manager, directory, compact=not args.standard_json)
        try:
            report = await support.review_pair(pair['old_fact_id'], pair['new_fact_id'], as_of=args.as_of)
            save(directory / 'report.json', report)
            decision = report.get('decision_proposal') or {}
            row = dict(link_id=pair['link_id'], old_fact_id=pair['old_fact_id'], new_fact_id=pair['new_fact_id'],
                status=report['status'], stop_reason=report['stop_reason'], relation=decision.get('relation'),
                proposed_action=decision.get('proposed_action'), evidence_chunks=len(report['evidence']),
                judge_calls=len(report['rounds']), report=str(directory.relative_to(args.output) / 'report.json'),
                update_applied=False)
        except asyncio.CancelledError:
            raise
        except Exception as error:
            row = dict(link_id=pair['link_id'], status='ERROR', error=f'{type(error).__name__}:{error}',
                       update_applied=False)
            save(directory / 'error.json', row)
        summaries.append(row)
        save(args.output / 'summary.json', dict(completed=index, total=len(pairs), finished=False,
            status_counts=dict(Counter(r['status'] for r in summaries)), pairs=summaries, update_applied=False))
        print(f"PAIR {index}/{len(pairs)} status={row['status']} action={row.get('proposed_action')}", flush=True)
    unchanged = sha256(args.source.read_bytes()).hexdigest() == source_hash
    if not unchanged:
        raise AssertionError('来源图发生变化')
    save(args.output / 'summary.json', dict(completed=len(pairs), total=len(pairs), finished=True,
        status_counts=dict(Counter(r['status'] for r in summaries)),
        relation_counts=dict(Counter(r.get('relation') for r in summaries if r.get('relation'))),
        source_unchanged=True, pairs=summaries, update_applied=False,
        accuracy=None, note='没有人工金标，不能据本文件计算关系准确率'))
    print('SAMPLE_REVIEW_COMPLETE', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=ROOT/'config/Memory_EventState_sample0_rebuild.yaml')
    parser.add_argument('--settings', type=Path, default=ROOT/'config/evidence_support.json')
    parser.add_argument('--as-of')
    parser.add_argument('--limit', type=int)
    parser.add_argument('--retrieval-only', action='store_true')
    parser.add_argument('--no-semantic', action='store_true')
    parser.add_argument('--standard-json', action='store_true')
    os.environ.setdefault('HF_HUB_OFFLINE', '1')
    os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
    asyncio.run(run(parser.parse_args()))
