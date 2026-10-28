"""独立演化入口：读取上层待审查链接，沿图取证，保存双端关系与决策建议。

不重做初始 StateSlot/语义召回，不修改来源图。图关联不足时才启用 BGE-M3
结构化语义检索兜底。结果是未校准、未采纳的建议，不是自动记忆更新。
"""
import argparse
import asyncio
from hashlib import sha256
import json
import os
from pathlib import Path
import pickle
import sys
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from coregraph.conflict_review import ConflictPairIndex, ConflictPairJudge, PairEvidenceProvider
from coregraph.evidence_judge import EvidenceJudgeController


def save(path, value):
    """审查留痕模块：仅写独立实验目录，原图及原有实验结果均保持只读。"""
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


async def run(args):
    """图驱动演化模块：已知两端→双方来源→关联/邻居扩展→语义兜底→决策建议。"""
    content = args.source.read_bytes()
    graph = pickle.loads(content)
    pairs = ConflictPairIndex(graph).pending(args.new_fact_id)
    if args.link_id:
        pairs = [p for p in pairs if p['link_id'] == args.link_id]
    selected = pairs[:args.limit]
    if args.output.exists():
        raise ValueError('输出目录必须全新，禁止覆盖历史审查')
    args.output.mkdir(parents=True)
    from init.config import Config
    from base.llm import LLMManager
    config = Config.parse(args.config)
    if config.llm.api_key == 'demo-key':
        raise ValueError('审查不得使用模拟 LLM 响应')
    manifest = dict(source=str(args.source.resolve()), source_sha256=sha256(content).hexdigest(),
        pending_pairs=len(pairs), selected_pairs=len(selected), link_ids=[p['link_id'] for p in selected],
        scope='current_snapshot_review_not_ingestion_time_replay',
        initial_recall='upper_link_endpoints', expansion_order=['linked', 'neighbors', 'semantic'],
        max_judge_calls=args.max_rounds, top_k=args.top_k, max_chunks=args.max_chunks,
        model=config.llm.model, config_sha256=sha256(args.config.read_bytes()).hexdigest(),
        code_hashes={name: sha256((Path(__file__).resolve().parents[2] / name).read_bytes()).hexdigest()
            for name in ('coregraph/conflict_review.py', 'coregraph/evidence_judge.py',
                         'coregraph/graph_evidence_provider.py',
                         'experiments/integrated_event_state/review_conflict_pairs.py')},
        update_applied=False, confidence_fusion='not_implemented')
    save(args.output / 'manifest.json', manifest)
    if not selected:
        # 没有待审查链接不等于检索已穷尽，也不意味着没有真实冲突。
        save(args.output / 'summary.json', dict(selected_pairs=0, status='no_pending_links',
            source_unchanged=True, note='若写入阶段漏连，需要另行补建候选关联'))
        print('NO_PENDING_LINKS', flush=True)
        return
    lc = config.llm
    manager = LLMManager(api_key=lc.api_key, model=lc.model, base_url=lc.base_url,
        max_tokens=lc.max_token, max_concurrent=1, timeout=lc.timeout, enable_thinking=False)
    retriever = None

    async def search(query, top_k):
        """语义兜底模块：只有图扩展没有新增来源时，才加载模型和检索索引。"""
        nonlocal retriever
        if retriever is None:
            from base.embeddings import EmbeddingManager
            from coregraph.triple_fact_store import TripleFactStore
            from experiments.structural_semantic_retrieval.retriever import StructuralSemanticRetriever
            store = TripleFactStore(graph['fact_storage'])
            memory = SimpleNamespace(graph_builder=SimpleNamespace(graph=graph['graph'], fact_store=store),
                fact_store=store, chunk_storage=graph['chunk_storage'], triple_storage=store.chunk_view(),
                triple_storage_origin=store.origin)
            retriever = StructuralSemanticRetriever(memory, EmbeddingManager(config.embedding))
        return await retriever.retrieve(query, top_k=top_k)

    summaries = []
    for pair in selected:
        directory = args.output / pair['link_id']
        directory.mkdir()
        class RecordedLLM:
            """证据审计模块：保存真实模型输入/输出与 token 截断元数据。"""
            async def generate(self, prompt, **kwargs):
                metadata = kwargs.pop('metadata_sink', {})
                trace = dict(prompt=prompt)
                try:
                    result = await manager.generate(prompt, metadata_sink=metadata, **kwargs)
                    trace['response'] = result
                    return result
                finally:
                    trace['metadata'] = metadata
                    with (directory / 'llm_trace.jsonl').open('a', encoding='utf-8') as stream:
                        stream.write(json.dumps(trace, ensure_ascii=False) + '\n')
        provider = PairEvidenceProvider(graph, search, pair)
        judge = ConflictPairJudge(RecordedLLM(), pair)
        controller = EvidenceJudgeController(judge, provider, max_rounds=args.max_rounds,
            top_k=args.top_k, max_chunks=args.max_chunks, graph_first=True)
        result = await controller.run('核查固定新旧事实对的关系和证据，不执行覆盖。',
                                      target_fact=pair['new_fact'])
        save(directory / 'pair.json', pair)
        save(directory / 'review_result.json', result)
        decision = result['rounds'][-1].get('decision') if result['rounds'] else None
        if decision is None:
            decision = dict(link_id=pair['link_id'], old_fact_id=pair['old_fact_id'],
                new_fact_id=pair['new_fact_id'], proposed_action='DEFER', update_applied=False,
                reason=result['stop_reason'], confidence=None, validation_status='not_verified')
        save(directory / 'decision_proposal.json', decision)
        summaries.append(dict(link_id=pair['link_id'], status=result['status'],
            proposed_action=decision['proposed_action'], judge_calls=len(result['rounds']),
            chunks=len(result['evidence']), update_applied=False))
        print(json.dumps(summaries[-1], ensure_ascii=False), flush=True)
    unchanged = sha256(args.source.read_bytes()).hexdigest() == manifest['source_sha256']
    if not unchanged:
        raise AssertionError('来源图被修改')
    save(args.output / 'summary.json', dict(pairs=summaries, source_unchanged=unchanged,
                                          update_applied=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=Path('config/Memory_EventState_sample0_rebuild.yaml'))
    parser.add_argument('--new-fact-id')
    parser.add_argument('--link-id')
    parser.add_argument('--limit', type=int, default=5)
    parser.add_argument('--max-rounds', type=int, default=3)
    parser.add_argument('--top-k', type=int, default=5)
    parser.add_argument('--max-chunks', type=int, default=12)
    args = parser.parse_args()
    if args.limit < 1:
        parser.error('limit 必须为正数')
    # 先验证预算，避免无效参数启动模型。
    EvidenceJudgeController(None, None, args.max_rounds, args.top_k, args.max_chunks)
    os.environ.setdefault('HF_HUB_OFFLINE', '1')
    os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
    asyncio.run(run(args))
