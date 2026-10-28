"""新图结构实验模块：固定已写入状态事实，按历史顺序比较四路旧记忆召回。

本脚本不重新抽取、不修改原图、不执行记忆更新。测试的是存储后的状态候选
检索，不等同于完整写入质量、事件身份识别或问答效果。LLM 仅辅助标记候选
关系；没有人工标注时，不输出 Precision/Recall 等带标准答案含义的指标。
"""
from __future__ import annotations

import argparse
import asyncio
from collections import Counter
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import pickle
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from base.embeddings import EmbeddingManager
from base.llm import LLMManager
from coregraph.event_state_layer import EventStateLayer
from coregraph.hybrid_state_retriever import HybridStateRetriever, subject_key
from coregraph.state_fact_relation import StateFactRelationJudge
from coregraph.triple_fact_store import TripleFactStore, decode_upper_references
from experiments.integrated_event_state.compare_write_recall import ComparisonRetriever
from experiments.integrated_event_state.replay_upper_from_graph import RecordingLLM
from init.config import Config


def dump(path, value):
    """实验留痕模块：仅写新实验目录；小型统计文件原子替换，避免半写内容。"""
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    temp.replace(path)


def append(path, value):
    """检索审计模块：逐条持久化完整输入和结果，中断仍可查看已经完成的样本。"""
    with path.open('a', encoding='utf-8') as handle:
        handle.write(json.dumps(value, ensure_ascii=False) + '\n')


async def run(args, embeddings=None):
    """冻结回放模块：只给检索器已经出现的 Chunk，禁止自检索和未来信息泄漏。"""
    args.output.mkdir(parents=True, exist_ok=False)
    before = sha256(args.source.read_bytes()).hexdigest()
    with args.source.open('rb') as handle:
        source = pickle.load(handle)
    if source.get('event_state_layer', {}).get('schema') != 'event_index_v1':
        raise ValueError('本实验必须使用最新 event_index_v1 图，不能误用旧图')
    store = TripleFactStore(source['fact_storage'])
    decoded = decode_upper_references(source['event_state_layer'], store,
                                      source.get('ingestion_state'))
    facts = list(decoded['facts'].values())
    positions = [fact['position'] for fact in facts]
    if len(set(positions)) != len(positions):
        raise ValueError('写入位置重复，不能可靠确定历史顺序')
    chunks = {str(key): value for key, value in source['chunk_storage'].items()}
    # 状态来源必须真实可追溯；遇到错误拒绝比较，不能偷偷移除失败样本。
    for fact in facts:
        cid = str(fact['chunk_id'])
        if cid not in chunks or not fact.get('quote') or fact['quote'] not in chunks[cid]['text']:
            raise ValueError(f"事实来源或引文无效：{fact['fact_id']}")
    groups = {}
    for fact in sorted(facts, key=lambda f: f['position']):
        groups.setdefault(str(fact['chunk_id']), []).append(fact)
    # 相同 Chunk 中多个断言不能互相作为历史证据：整组先查、整组后入历史。
    last_position = 0
    for group in groups.values():
        if min(f['position'] for f in group) <= last_position:
            raise ValueError('同 Chunk 的写入位置交错，需先明确顺序再测试')
        last_position = max(f['position'] for f in group)
    config = Config.parse(args.config)
    dependency_paths = sorted(set(Path('coregraph').glob('*.py')) |
                              set(Path('base').glob('*.py')) |
                              set(Path('experiments/integrated_event_state').glob('*.py')))
    manifest = {
        'experiment': 'frozen_new_graph_state_recall_v1',
        'source': str(args.source.resolve()), 'source_sha256': before,
        'schema': 'event_index_v1', 'chunks': len(chunks),
        'raw_triples': len(store.facts), 'observations': len(facts),
        'fact_types': dict(Counter(f.get('fact_type') for f in facts)),
        'top_k': args.top_k, 'methods': ['legacy', 'structured', 'semantic', 'hybrid'],
        'history_policy': 'earlier_chunks_only_grouped_by_unique_write_position',
        'scope': 'state_observations_only; event observations excluded',
        'normalization': 'reuse_saved_values_no_reextraction_or_renormalization',
        'config_sha256': sha256(args.config.read_bytes()).hexdigest(),
        'code_sha256': {str(p): sha256(p.read_bytes()).hexdigest() for p in dependency_paths},
        'label_source': 'llm_auxiliary_not_human_gold' if args.audit else 'unlabeled',
        'started_at': datetime.now(timezone.utc).isoformat(),
    }
    dump(args.output / 'manifest.json', manifest)
    dump(args.output / 'frozen_observations.json', facts)
    print(f"START schema=event_index_v1 observations={len(facts)} "
          f"states={manifest['fact_types'].get('state', 0)} top_k={args.top_k}", flush=True)
    # 套件在不同 K 之间共享同一个只读模型实例，避免重复加载 BGE-M3 占用显存。
    if embeddings is None:
        embeddings = EmbeddingManager(config.embedding)
    retriever = ComparisonRetriever(HybridStateRetriever(embeddings, args.top_k,
        args.top_k, args.top_k, 'hybrid'), embeddings, args.output / 'write_retrieval.jsonl')
    layer = EventStateLayer({'version': 1, 'write_policy': 'append_only'})
    rows = []
    for cid, group in groups.items():
        for fact in group:
            if fact.get('fact_type') == 'event':
                continue
            await retriever.search(layer, fact)
            # ComparisonRetriever 已落盘；按源 Fact ID 固定样本身份并保存完整历史池，
            # 人工后续可在所有旧事实中找漏召回，而不仅标注各方法返回的并集。
            row = json.loads((args.output / 'write_retrieval.jsonl').read_text(
                encoding='utf-8').splitlines()[-1])
            row['new_fact_id'] = fact['fact_id']
            row['history_ids'] = sorted(layer.facts)
            row['eligible_history_ids'] = sorted(fid for fid, old in layer.facts.items()
                if retriever.driver._eligible(old, fact))
            row['new_chunk'] = chunks[cid]
            for method in row['methods'].values():
                for hit in method['hits']:
                    old = hit['fact']
                    if old['fact_id'] not in layer.facts or str(old['chunk_id']) == cid:
                        raise AssertionError('检索结果含未来事实或同 Chunk 信息')
                    hit['source_chunk'] = chunks[str(old['chunk_id'])]
            append(args.output / 'retrieval_results.jsonl', row)
            append(args.output / 'manual_labels.jsonl', {
                'new_fact_id': fact['fact_id'], 'eligible_history_ids': row['eligible_history_ids'],
                'relevant_old_fact_ids': None, 'annotation_status': 'unlabeled',
                'note': '需人工检查完整 eligible_history_ids，不能仅检查返回候选并集'})
            rows.append(row)
            print(f"RECALL {len(rows)} fact={fact['fact_id']} history={len(layer.facts)}", flush=True)
        for fact in group:
            layer.facts[fact['fact_id']] = fact
            layer._index_fact(fact)
    summaries = {}
    for name in manifest['methods']:
        populated = [r for r in rows if r['eligible_history_ids']]
        summaries[name] = {
            'queries': len(rows), 'queries_with_comparable_history': len(populated),
            'empty_with_history': sum(not r['methods'][name]['hits'] for r in populated),
            'mean_returned_with_history': sum(len(r['methods'][name]['hits'])
                for r in populated) / max(1, len(populated)),
        }
    report = ['# 新图 sample0：固定事实的历史召回测试', '',
        '不重新抽取事实；每个新事实仅访问此前 Chunk。事件类不参与本次状态消融。',
        '候选覆盖与方法差异不是检索准确率；没有人工 gold，不报告 Recall@K。', '',
        '|方法|状态输入数|有可比较历史|有历史但返回空|平均返回数（有历史）|',
        '|---|---:|---:|---:|---:|']
    for name, s in summaries.items():
        report.append(f"|{name}|{s['queries']}|{s['queries_with_comparable_history']}|"
                      f"{s['empty_with_history']}|{s['mean_returned_with_history']:.2f}|")
    cases = ['# 逐条检索及原文证据', '']
    for row in rows:
        new = row['new_fact']
        cases.extend([f"## {row['new_fact_id']}",
            f"新事实：{new['entity_id']} / {new['attribute']} / {new.get('value')}",
            f"Chunk：{new['chunk_id']}；引文：{new['quote']}", ''])
        for name, method in row['methods'].items():
            cases.append(f'### {name}')
            for hit in method['hits']:
                old = hit['fact']
                cases.extend([f"- 第 {hit['rank']}：{old['fact_id']}；Chunk {old['chunk_id']}；"
                    f"{old['attribute']} / {old.get('value')}", f"  原文：{old['quote']}"])
            if not method['hits']:
                cases.append('无候选。')
            cases.append('')
    (args.output / 'retrieval_cases.md').write_text('\n'.join(cases), encoding='utf-8')
    metrics = {'methods': summaries, 'source_unchanged': before == sha256(args.source.read_bytes()).hexdigest(),
        'hybrid_semantic_same_set': sum(
            {h['fact']['fact_id'] for h in r['methods']['hybrid']['hits']} ==
            {h['fact']['fact_id'] for h in r['methods']['semantic']['hits']} for r in rows),
        'human_gold_available': False, 'semantic_statuses': dict(Counter(
            r['methods']['hybrid']['trace']['semantic_status'] for r in rows))}
    dump(args.output / 'retrieval_metrics.json', metrics)
    (args.output / 'report.md').write_text('\n'.join(report), encoding='utf-8')
    if not metrics['source_unchanged']:
        raise AssertionError('来源图哈希发生变化')
    if any(status.startswith('unavailable:') for status in metrics['semantic_statuses']):
        raise RuntimeError('语义召回失败，不能把降级运行当作完成的对比实验')
    # 辅助判读与检索分开保存；逐对输入，不把整个人物历史塞入 LLM 上下文。
    if args.audit:
        lc = config.llm
        manager = LLMManager(api_key=lc.api_key, model=lc.model, base_url=lc.base_url,
            max_tokens=lc.max_token, enable_concurrent=True, max_concurrent=args.concurrency,
            timeout=lc.timeout, enable_thinking=lc.enable_thinking)
        judge = StateFactRelationJudge(RecordingLLM(manager, args.output / 'llm_trace.jsonl'))
        sem = asyncio.Semaphore(args.concurrency)
        async def assess(row, old):
            async with sem:
                result = await judge.judge(old, row['new_fact'])
                append(args.output / 'pooled_relation_audit.jsonl', {
                    'new_fact_id': row['new_fact_id'], 'old_fact_id': old['fact_id'],
                    'judgment': result, 'label_source': 'llm_auxiliary_not_gold'})
        for row in rows:
            pool = {h['fact']['fact_id']: h['fact'] for m in row['methods'].values() for h in m['hits']}
            await asyncio.gather(*(assess(row, pool[fid]) for fid in sorted(pool)))
            print(f"AUDIT fact={row['new_fact_id']} pairs={len(pool)}", flush=True)
    dump(args.output / 'complete.json', {
        'finished_at': datetime.now(timezone.utc).isoformat(), 'state_queries': len(rows),
        'source_unchanged': before == sha256(args.source.read_bytes()).hexdigest()})
    print('COMPLETE', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--top-k', type=int, default=5)
    parser.add_argument('--audit', action='store_true')
    parser.add_argument('--concurrency', type=int, default=4)
    args = parser.parse_args()
    if min(args.top_k, args.concurrency) < 1 or args.top_k < 2:
        parser.error('Top-k 至少为 2，并发数须为正数')
    os.environ.setdefault('HF_HUB_OFFLINE', '1')
    os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
    asyncio.run(run(args))
