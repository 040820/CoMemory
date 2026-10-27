"""sample0 写入期四路召回实验；固定新事实与历史快照，保留每次检索证据。

所属模块：独立 Event–State 实验。生产 Writer 仍由 hybrid 驱动；影子召回不
修改历史，因此方法差异不会改变下一个比较样本。LLM 池内判读只是辅助指标，
不是标准答案、因果推导或完整召回率。原始来源图和历史结果只读。
"""

from __future__ import annotations

import argparse
import asyncio
from collections import defaultdict
from hashlib import sha256
import itertools
import json
import os
from pathlib import Path
import pickle
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from base.embeddings import EmbeddingManager
from base.llm import LLMManager
from coregraph.event_state_writer import EventStateWriter
from coregraph.hybrid_state_retriever import HybridStateRetriever, subject_key
from coregraph.state_candidate_retriever import StateCandidateRetriever
from coregraph.state_fact_relation import StateFactRelationJudge
from init.config import Config
from experiments.integrated_event_state import replay_upper_from_graph as replay


METHODS = ('legacy', 'structured', 'semantic', 'hybrid')


def digest(value: Any) -> str:
    """服务于复现：稳定 JSON 指纹区分输入，而非用候选排名生成样本 ID。"""
    return sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def append_json(path: Path, value: Any) -> None:
    """服务于运行留痕：每条立即追加关闭，任务中断仍保留之前的检索结果。"""
    with path.open('a', encoding='utf-8') as handle:
        handle.write(json.dumps(value, ensure_ascii=False) + '\n')


def read_rows(path: Path) -> list[dict[str, Any]]:
    """服务于审计续跑：解析本实验已写入的完整 JSONL；损坏记录显式报错。"""
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()
            if line.strip()] if path.exists() else []


def query_id(proposal: dict[str, Any]) -> str:
    """归一化前后使用相同 ID；原始属性和值优先，避免重复统计同一新事实。"""
    return digest([proposal['entity_index'], str(proposal['chunk_id']), proposal['quote'],
                   proposal.get('source_attribute', proposal['attribute']),
                   proposal.get('source_value', proposal.get('value'))])[:24]


class ComparisonRetriever:
    """服务于影子实验：同一历史上执行四路，只将 hybrid 返回给正式写入器。"""

    def __init__(self, driver: HybridStateRetriever, embeddings: Any, path: Path):
        self.driver, self.path = driver, path
        self.legacy = StateCandidateRetriever(embeddings, driver.top_k)
        self.structured = HybridStateRetriever(embeddings, driver.top_k,
            driver.structured_k, driver.semantic_k, 'structured')
        self.semantic = HybridStateRetriever(embeddings, driver.top_k,
            driver.structured_k, driver.semantic_k, 'semantic')

    async def search(self, layer: Any, proposal: dict[str, Any], semantic_result=None) -> dict:
        """记录 bootstrap/final 两阶段；复用 B 的原始向量结果保证固定语义输入。

        legacy 保持原混合加分实现；A/B 单路使用与 hybrid 完全相同的候选和预算。
        共用缓存只用于减少计算，因此记录不用于比较独立方法的真实运行速度。
        """
        hybrid = await self.driver.search(layer, proposal, semantic_result)
        structured = await self.structured.search(layer, proposal)
        semantic = await self.semantic.search(layer, proposal, hybrid['semantic_result'])
        legacy_hits = await self.legacy.search(layer, proposal)
        results = {'legacy': {'hits': legacy_hits, 'trace': {'mode': 'legacy'}},
                   'structured': structured, 'semantic': semantic, 'hybrid': hybrid}
        views = {}
        for name, result in results.items():
            views[name] = {
                'trace': result['trace'],
                'hits': [dict(hit, rank=rank) for rank, hit in enumerate(result['hits'], 1)]}
        related = [layer.facts[fid] for fid in layer.facts_by_subject.get(subject_key(proposal), [])
                   if layer.facts[fid].get('fact_type') != 'event']
        append_json(self.path, {
            'query_id': query_id(proposal),
            'phase': 'normalized' if semantic_result is not None else 'bootstrap',
            'new_fact': proposal, 'history_count': len(layer.facts),
            'subject_history_count': len(related),
            'history_fingerprint': digest(sorted(layer.facts)), 'methods': views,
        })
        return hybrid


def final_queries(path: Path) -> list[dict[str, Any]]:
    """服务于统计：同一候选取最后一次调用，避免初始/归一后召回重复计分。"""
    latest = {}
    for row in read_rows(path):
        latest[row['query_id']] = row
    return list(latest.values())


def written_queries(output: Path) -> tuple[list, list]:
    """服务于统计分母：区分尝试召回与最终写入，拒收候选不混入写入效果指标。

    原始 JSONL 保留所有尝试。正式实验读取最终上层，按来源 ID 对齐已保存
    状态断言；无图文件的情况仅用于合成报表自检，不能当作正式实验完成。
    """
    attempts = final_queries(output / 'write_retrieval.jsonl')
    path = output / 'dynamic_memory_event_state.pkl'
    if not path.exists():
        return attempts, attempts
    with path.open('rb') as handle:
        data = pickle.load(handle)
    facts = data['event_state_layer']['facts'].values()
    written = {query_id(fact): fact['fact_id'] for fact in facts
               if fact.get('fact_type') != 'event'}
    rows = [dict(row, written_fact_id=written[row['query_id']]) for row in attempts
            if row['query_id'] in written]
    return attempts, rows


async def audit_pool(output: Path, judge: StateFactRelationJudge, concurrency: int) -> None:
    """服务于辅助评估：对四路 Top-k 的去重合并池逐对盲判，可断点继续。

    判读器只看到新旧原文/Triple，不知道方法名、排名和得分；与写入器使用同类
    LLM，结果存在模型偏差，因此只能称池内模型辅助判读，不能称人工 gold。
    """
    path = output / 'pooled_relation_audit.jsonl'
    existing = {row['pair_id'] for row in read_rows(path)}
    semaphore = asyncio.Semaphore(concurrency)

    async def one(row, old):
        pair_id = digest([row['query_id'], old, row['new_fact']])
        if pair_id in existing:
            return
        async with semaphore:
            result = await judge.judge(old, row['new_fact'])
            append_json(path, {'pair_id': pair_id, 'query_id': row['query_id'],
                'old_fact_id': old['fact_id'], 'old_fact': old,
                'new_fact': row['new_fact'], 'judgment': result,
                'label_source': 'llm_candidate_relation_not_gold'})

    for row in written_queries(output)[1]:
        # 使用 Fact ID 排序，不按任何方法的排名给判读器暗示。
        pool = {hit['fact']['fact_id']: hit['fact'] for method in row['methods'].values()
                for hit in method['hits']}
        await asyncio.gather(*(one(row, pool[fid]) for fid in sorted(pool)))
    print('POOL_AUDIT_COMPLETE', flush=True)


def summarize(output: Path) -> dict[str, Any]:
    """服务于实验报告：候选差异与辅助关联率分开记录，不构造真实 Recall@K。

    coexists 也属于可能关联线索，计入广义候选关联率；uncertain 不作负例，
    单独报告。池内覆盖率仅针对四路已召回且被模型标为关联的有限池。
    """
    attempts, rows = written_queries(output)
    audits = {(r['query_id'], r['old_fact_id']): r['judgment']
              for r in read_rows(output / 'pooled_relation_audit.jsonl')}
    eligible = [row for row in rows if row['subject_history_count'] > 0]
    statistics = {name: defaultdict(int) for name in METHODS}
    overlaps = defaultdict(list)
    cases, labels = [], []
    for row in rows:
        qid, new = row['query_id'], row['new_fact']
        sets = {name: {h['fact']['fact_id'] for h in row['methods'][name]['hits']}
                for name in METHODS}
        # 服务于续跑一致性：分母只来自本次最终候选池，不混入中断前的过期池。
        pool_ids = set.union(*sets.values())
        related_pool = {fid for fid in pool_ids if audits.get((qid, fid), {}).get('relation')
                        not in {None, 'unrelated', 'uncertain'}}
        cases.extend([f"## {qid} / Chunk {new['chunk_id']}",
                      f"新事实：{new['entity_id']} / {new['attribute']} = {new.get('value')}",
                      f"原文：{new['quote']}",
                      f"同主体历史数：{row['subject_history_count']}；阶段：{row['phase']}", ''])
        labels.append({'query_id': qid, 'new_fact': new,
                       'candidate_old_fact_ids': sorted(set.union(*sets.values())),
                       'gold_old_fact_ids': [], 'annotation_status': 'unlabeled'})
        for name in METHODS:
            hits, stats = row['methods'][name]['hits'], statistics[name]
            cases.append(f'### {name}')
            for hit in hits:
                old = hit['fact']
                label = audits.get((qid, old['fact_id']), {}).get('relation', 'not_audited')
                cases.extend([f"- #{hit['rank']} {old['fact_id']} / Chunk {old['chunk_id']} "
                              f"/ score={hit['score']:.6f} / 候选关系={label}",
                              f"  - 事实：{old['entity_id']} / {old['attribute']} = {old.get('value')}",
                              f"  - 原文：{old['quote']}",
                              f"  - Triple：{json.dumps(old.get('triple_refs'), ensure_ascii=False)}"])
            if not hits:
                cases.append('- 无候选')
            if row['subject_history_count'] == 0:
                continue
            stats['queries'] += 1
            stats['returned'] += len(hits)
            stats['empty'] += not bool(hits)
            stats['pooled_related_denominator'] += len(related_pool)
            for hit in hits:
                label = audits.get((qid, hit['fact']['fact_id']), {}).get('relation', 'not_audited')
                stats['related'] += label not in {'unrelated', 'uncertain', 'not_audited'}
                stats['uncertain'] += label == 'uncertain'
                stats['not_audited'] += label == 'not_audited'
        if row['subject_history_count'] > 0:
            for a, b in itertools.combinations(METHODS, 2):
                union = sets[a] | sets[b]
                if union:
                    overlaps[f'{a}:{b}'].append(len(sets[a] & sets[b]) / len(union))
        cases.append('')
    metrics = {'attempted_state_candidates': len(attempts), 'new_state_candidates': len(rows),
               'with_subject_history': len(eligible),
               'scope': 'fixed_fact_stream_write_recall_not_qa',
               'normalization_driver': 'hybrid', 'human_gold_available': False,
               'latency_comparison_valid': False, 'methods': {},
               'mean_jaccard': {k: sum(v) / len(v) for k, v in overlaps.items()}}
    report = ['# sample0 写入期召回对比', '',
        '固定事实流和历史快照；归一化由 hybrid 驱动。不是四套写入系统的端到端比较。',
        '模型判读仅为候选关系，不是真值。池内覆盖率不是 Recall@K；共享缓存耗时不可直接比较。', '',
        '| 方法 | 有历史的查询数 | 平均候选数 | 空结果比例 | 模型候选关联率 | 模型不确定率 | 池内关联覆盖率 |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for name, stats in statistics.items():
        values = dict(stats)
        values.update({
            'mean_returned': stats['returned'] / max(1, stats['queries']),
            'empty_rate': stats['empty'] / max(1, stats['queries']),
            'llm_candidate_relation_rate': stats['related'] / max(1, stats['returned']),
            'llm_uncertain_rate': stats['uncertain'] / max(1, stats['returned']),
            'pooled_llm_related_coverage': stats['related'] / max(1, stats['pooled_related_denominator'])})
        metrics['methods'][name] = values
        report.append(f"| {name} | {stats['queries']} | {values['mean_returned']:.2f} | "
                      f"{values['empty_rate']:.1%} | {values['llm_candidate_relation_rate']:.1%} | "
                      f"{values['llm_uncertain_rate']:.1%} | {values['pooled_llm_related_coverage']:.1%} |")
    report.extend(['', '来源 Triple 覆盖率与缺失 Chunk 见 manifest.json；不能假定旧图关系恢复完整。',
                   'write_retrieval.jsonl 保存每次召回；retrieval_cases.md 保存逐条对比。',
                   'manual_labels.jsonl 是未标注模板，不能直接作为标准答案。'])
    replay.atomic_json(output / 'comparison_metrics.json', metrics)
    # 单独导出最终参与统计的已写入状态样本，便于与原始尝试记录交叉检查。
    with (output / 'final_write_retrieval.jsonl').open('w', encoding='utf-8') as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + '\n')
    (output / 'comparison_report.md').write_text('\n'.join(report), encoding='utf-8')
    (output / 'retrieval_cases.md').write_text('\n'.join(cases), encoding='utf-8')
    label_path = output / 'manual_labels.jsonl'
    if not label_path.exists():
        for label in labels:
            append_json(label_path, label)
    return metrics


async def run(args) -> None:
    """服务于实验编排：冻结输入/脚本指纹，回放上层，再审计召回池并出报告。"""
    args.output.mkdir(parents=True, exist_ok=True)
    config = Config.parse(args.config)
    if config.state_write_policy != 'append_only' or config.state_recall_mode != 'hybrid':
        raise ValueError('对比驱动必须使用 append_only + hybrid 配置')
    identity = {'script_sha256': sha256(Path(__file__).read_bytes()).hexdigest(),
                'source': str(args.source.resolve()),
                'source_sha256': sha256(args.source.read_bytes()).hexdigest(),
                'config_sha256': sha256(args.config.read_bytes()).hexdigest(),
                'candidate_top_k': args.candidate_top_k, 'methods': list(METHODS)}
    fingerprint = args.output / 'comparison_manifest.json'
    if fingerprint.exists() and json.loads(fingerprint.read_text(encoding='utf-8')) != identity:
        raise ValueError('已有实验指纹不同，请新建目录')
    replay.atomic_json(fingerprint, identity)
    original_writer = replay.EventStateWriter

    class ComparisonWriter(EventStateWriter):
        """仅限本实验的注入适配器，不修改正式 Writer 或原版 Baseline。"""
        def __init__(self, llm, embedding_manager, **kwargs):
            super().__init__(llm, embedding_manager, **kwargs)
            self.hybrid_candidates = ComparisonRetriever(self.hybrid_candidates,
                embedding_manager, args.output / 'write_retrieval.jsonl')

    try:
        replay.EventStateWriter = ComparisonWriter
        if not (args.output / 'dynamic_memory_event_state.pkl').exists():
            await replay.run(args)
    finally:
        replay.EventStateWriter = original_writer
    llm_config = config.llm
    manager = LLMManager(api_key=llm_config.api_key, model=llm_config.model,
        base_url=llm_config.base_url, max_tokens=llm_config.max_token,
        enable_concurrent=True, max_concurrent=args.audit_concurrency,
        timeout=llm_config.timeout, enable_thinking=llm_config.enable_thinking)
    llm = replay.RecordingLLM(manager, args.output / 'pooled_audit_llm_trace.jsonl')
    await audit_pool(args.output, StateFactRelationJudge(llm), args.audit_concurrency)
    metrics = summarize(args.output)
    print(f"COMPARISON_COMPLETE queries={metrics['new_state_candidates']}", flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=Path('config/Memory_EventState_sample0_rebuild.yaml'))
    parser.add_argument('--candidate-top-k', type=int, default=5)
    parser.add_argument('--audit-concurrency', type=int, default=4)
    parser.add_argument('--allow-embedding-download', action='store_true')
    options = parser.parse_args()
    if min(options.candidate_top_k, options.audit_concurrency) < 1:
        parser.error('预算和并发必须为正数')
    if not options.allow_embedding_download:
        os.environ.setdefault('HF_HUB_OFFLINE', '1')
        os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
    asyncio.run(run(options))
