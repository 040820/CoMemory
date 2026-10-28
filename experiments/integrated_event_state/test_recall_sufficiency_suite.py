"""新图实验模块：K=5/10/20 召回消融与有限证据补查测试。

复用已实现检索器和 Judge，不改变生产写入逻辑。所有输入来自冻结 sample0；
每个查询严格隔离未来 Chunk。Judge 的“充分”仅是模型在所给证据中的判断，
不能当成人工真值、校准概率或已经执行的记忆更新。
"""
from __future__ import annotations

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
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from base.embeddings import EmbeddingManager
from base.llm import LLMManager
from coregraph.evidence_judge import EvidenceJudge, EvidenceJudgeController
from coregraph.graph_evidence_provider import GraphEvidenceProvider
from coregraph.triple_fact_store import TripleFactStore
from experiments.integrated_event_state import test_frozen_state_recall as frozen
from experiments.structural_semantic_retrieval.retriever import StructuralSemanticRetriever
from init.config import Config


def read_rows(path):
    """实验输入模块：复用已经保存的召回记录，避免再次抽取或生成另一批事实。"""
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def restrict_graph(source, target_chunk):
    """时间隔离模块：构造当前 Chunk 时刻的只读视图，彻底移除未来事实和索引。

    不仅过滤最终结果，也过滤事实表、Chunk、事件分组、链接和实体关联边。
    因而未来记录不会参与语义评分、图扩展或邻居补查。保留当前 Chunk 全文，
    因为它是新事实的原始证据，但历史种子中仍排除当前 Chunk 的其他断言。
    """
    chunks = {str(k): v for k, v in source['chunk_storage'].items()}
    order = list(chunks)
    allowed = set(order[:order.index(str(target_chunk)) + 1])
    view = dict(source)
    view['chunk_storage'] = {cid: chunks[cid] for cid in order if cid in allowed}
    table = {}
    all_ids = set()
    for name, collection in source['fact_storage'].items():
        if name in ('facts', 'observations', 'time_assertions', 'attribute_assertions'):
            table[name] = {fid: row for fid, row in collection.items()
                           if GraphEvidenceProvider.source_chunk(row) in allowed}
            all_ids.update(table[name])
        else:
            table[name] = deepcopy(collection)
    view['fact_storage'] = table
    upper = source['event_state_layer']
    view['event_state_layer'] = dict(upper,
        groups={eid: dict(group, fact_ids=[fid for fid in group['fact_ids'] if fid in all_ids])
                for eid, group in upper.get('groups', {}).items()
                if any(fid in all_ids for fid in group['fact_ids'])},
        links=[link for link in upper.get('links', [])
               if link['from_fact_id'] in all_ids and link['to_fact_id'] in all_ids])
    graph = source['graph'].copy()
    for a, b, data in list(graph.edges(data=True)):
        ids = [fid for fid in data.get('fact_ids', []) if fid in table.get('facts', {})]
        if not ids:
            graph.remove_edge(a, b)
        else:
            graph.edges[a, b]['fact_ids'] = ids
            graph.edges[a, b]['weight'] = len(ids)
    # 未来才出现的孤立实体也不进入实体匹配。当前事实的主体/客体按事实表恢复。
    nodes = {row[key] for row in table.get('facts', {}).values() for key in ('src', 'tgt')}
    graph.remove_nodes_from([node for node in graph if node not in nodes])
    view['graph'] = graph
    return view


async def run(args):
    """实验编排模块：先完成全部 K 的固定召回，再逐条运行补查并保存审计。"""
    args.output.mkdir(parents=True, exist_ok=False)
    config = Config.parse(args.config)
    if config.llm.api_key == 'demo-key':
        raise ValueError('不允许用模拟模型响应进行本实验')
    source_hash = sha256(args.source.read_bytes()).hexdigest()
    source = pickle.loads(args.source.read_bytes())
    frozen.dump(args.output / 'suite_manifest.json', {
        'source': str(args.source.resolve()), 'source_sha256': source_hash,
        'ks': [5, 10, 20], 'scope': 'state_observations_only',
        'adaptive': {'initial_top_k': 5, 'followup_top_k': 5, 'max_judge_calls': 3, 'max_chunks': 12},
        'human_gold_available': False, 'update_applied': False,
        'judge_meaning': 'model_sufficiency_within_displayed_evidence_not_real_world_verification',
        'code_sha256': sha256(Path(__file__).read_bytes()).hexdigest(),
    })
    print('SUITE_START ks=5,10,20 adaptive_judge_calls=3 max_chunks=12', flush=True)
    embeddings = EmbeddingManager(config.embedding)
    for k in (5, 10, 20):
        await frozen.run(SimpleNamespace(source=args.source, config=args.config,
            output=args.output / f'k{k}', top_k=k, audit=False, concurrency=4), embeddings)
    rows = read_rows(args.output / 'k5/retrieval_results.jsonl')
    facts = {f['fact_id']: f for f in json.loads(
        (args.output / 'k5/frozen_observations.json').read_text(encoding='utf-8'))}
    k_rows = {k: {r['new_fact_id']: r for r in read_rows(
        args.output / f'k{k}/retrieval_results.jsonl')} for k in (5, 10, 20)}
    # 导出不同 K 新增了哪些候选，便于人工检查：数量增大可能补证，也可能加噪声。
    for row in rows:
        fid = row['new_fact_id']
        changes = {}
        for method in row['methods']:
            sets = {k: [h['fact']['fact_id'] for h in k_rows[k][fid]['methods'][method]['hits']]
                    for k in (5, 10, 20)}
            changes[method] = dict(ids_by_k=sets,
                added_10_over_5=sorted(set(sets[10]) - set(sets[5])),
                added_20_over_10=sorted(set(sets[20]) - set(sets[10])),
                dropped_5_at_10=sorted(set(sets[5]) - set(sets[10])))
        frozen.append(args.output / 'k_candidate_changes.jsonl', {
            'new_fact_id': fid, 'methods': changes})
    lc = config.llm
    manager = LLMManager(api_key=lc.api_key, model=lc.model, base_url=lc.base_url,
        max_tokens=lc.max_token, max_concurrent=1, timeout=lc.timeout, enable_thinking=False)
    adaptive = args.output / 'adaptive'
    adaptive.mkdir()
    totals = Counter()
    for index, row in enumerate(rows, 1):
        target = facts[row['new_fact_id']]
        view = restrict_graph(source, target['chunk_id'])
        # 交叉检查回放顺序：固定召回的历史必须全部位于本次只读视图中的过去 Chunk。
        assert all(str(facts[fid]['chunk_id']) in view['chunk_storage'] and
                   str(facts[fid]['chunk_id']) != str(target['chunk_id']) for fid in row['history_ids'])
        case_dir = adaptive / row['new_fact_id']
        case_dir.mkdir()

        class RecordedLLM:
            """Judge 审计模块：保留实际展示文本、响应及截断元数据，供人工复核。"""
            async def generate(self, prompt, **kwargs):
                metadata = kwargs.pop('metadata_sink', {})
                trace = dict(prompt=prompt)
                try:
                    result = await manager.generate(prompt, metadata_sink=metadata, **kwargs)
                    trace['response'] = result
                    return result
                finally:
                    trace['metadata'] = metadata
                    frozen.append(case_dir / 'llm_trace.jsonl', trace)

        store = TripleFactStore(view['fact_storage'])
        memory = SimpleNamespace(graph_builder=SimpleNamespace(graph=view['graph'], fact_store=store),
            fact_store=store, chunk_storage=view['chunk_storage'], triple_storage=store.chunk_view(),
            triple_storage_origin=store.origin)
        # 只有 Judge 要求补查时才计算当前历史视图的结构化语义索引；不让未来图参与。
        retriever = StructuralSemanticRetriever(memory, embeddings)

        async def search(query, top_k):
            return await retriever.retrieve(query, top_k=top_k)

        async def seeds(fid, top_k):
            return [h['fact']['fact_id'] for h in row['methods']['hybrid']['hits'][:top_k]]

        provider = GraphEvidenceProvider(view, search, seeds)
        controller = EvidenceJudgeController(EvidenceJudge(RecordedLLM()), provider,
            max_rounds=3, top_k=5, max_chunks=12)
        query = (f"核查新事实 {target['entity_id']} / {target['attribute']} / {target.get('value')} "
                 '与历史记忆的关系。寻找支持、反驳或纠正的具体证据；证据不足则补查。'
                 '独立并存不算冲突，不要求每条事实都有变化，不执行更新。')
        result = await controller.run(query, mode='memory_relation', target_fact=target)
        allowed = set(view['chunk_storage'])
        if any(str(e['chunk_id']) not in allowed for e in result['evidence']):
            raise AssertionError('补查证据出现未来 Chunk')
        frozen.dump(case_dir / 'judge_result.json', result)
        summary = dict(new_fact_id=row['new_fact_id'], status=result['status'],
            first_status=result['rounds'][0]['status'] if result['rounds'] else None,
            stop_reason=result['stop_reason'], judge_calls=len(result['rounds']),
            chunks=len(result['evidence']),
            added_chunks=max(0, len(result['evidence']) - len(result['retrieval_actions'][0]['chunk_ids'])),
            followup_actions=len(result['retrieval_actions']) - 1,
            context_limits=[r.get('context') for r in result['rounds']], update_applied=False)
        frozen.append(args.output / 'adaptive_results.jsonl', summary)
        totals[result['status']] += 1
        print(f"ADAPTIVE {index}/{len(rows)} status={result['status']} calls={summary['judge_calls']}", flush=True)
    unchanged = source_hash == sha256(args.source.read_bytes()).hexdigest()
    if not unchanged:
        raise AssertionError('来源图被修改')
    frozen.dump(args.output / 'suite_complete.json', {
        'state_queries': len(rows), 'adaptive_statuses': dict(totals),
        'source_unchanged': unchanged, 'human_gold_available': False, 'update_applied': False})
    print('SUITE_COMPLETE', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    os.environ.setdefault('HF_HUB_OFFLINE', '1')
    os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
    asyncio.run(run(args))
