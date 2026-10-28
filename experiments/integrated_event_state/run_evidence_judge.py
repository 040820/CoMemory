"""独立的 AMA 启发 Judge 实验入口：只读旧图，不改基线，不执行记忆更新。

结合当前初始事实召回、结构化语义 Chunk 检索与有限来源扩展，输出每轮判断和
精确原文引用。新事实应已按 append-only 写入图中；本入口只审查，不追加新事实。
"""

import argparse
import asyncio
from hashlib import sha256
import json
from pathlib import Path
import pickle
import sys
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from coregraph.evidence_judge import EvidenceJudge, EvidenceJudgeController
from coregraph.graph_evidence_provider import GraphEvidenceProvider


async def run(args):
    """在全新目录保存审查结果及复现材料；不向来源图或现有实验目录写入内容。"""
    source = args.source.resolve()
    content = source.read_bytes()
    graph = pickle.loads(content)
    if args.output.exists():
        raise ValueError('输出必须是全新目录')
    if args.mode == 'memory_relation' and not args.new_fact_id:
        raise ValueError('memory_relation 必须指定 --new-fact-id')
    # 先校验图与预算，避免无效输入触发模型加载和不必要的调用。
    async def unused_search(query, top_k):
        return []
    check_provider = GraphEvidenceProvider(graph, unused_search)
    if args.new_fact_id and args.new_fact_id not in check_provider.rows:
        raise ValueError('指定的新事实 ID 不存在')
    EvidenceJudgeController(None, None, args.max_rounds, args.top_k, args.max_chunks)
    target = dict(check_provider.rows[args.new_fact_id], fact_id=args.new_fact_id) if args.new_fact_id else None

    from base.llm import LLMManager
    from base.embeddings import EmbeddingManager
    from coregraph.triple_fact_store import TripleFactStore, decode_upper_references
    from coregraph.event_state_layer import EventStateLayer
    from coregraph.hybrid_state_retriever import HybridStateRetriever
    from coregraph.state_candidate_retriever import StateCandidateRetriever
    from experiments.structural_semantic_retrieval.retriever import StructuralSemanticRetriever
    from init.config import Config

    config = Config.parse(args.config)
    if config.llm.api_key == 'demo-key':
        raise ValueError('Judge 实验不接受 demo-key 模拟回答')
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = dict(source=str(source), source_sha256=sha256(content).hexdigest(), mode=args.mode,
        query=args.query, new_fact_id=args.new_fact_id, max_rounds=args.max_rounds,
        max_chunks=args.max_chunks, top_k=args.top_k, source_modified=False, update_applied=False,
        config_sha256=sha256(args.config.read_bytes()).hexdigest(),
        model=config.llm.model, code_hashes={name: sha256((Path(__file__).resolve().parents[2] / name).read_bytes()).hexdigest()
            for name in ('coregraph/evidence_judge.py', 'coregraph/graph_evidence_provider.py')})
    (args.output / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    manager = LLMManager(api_key=config.llm.api_key, model=config.llm.model, base_url=config.llm.base_url,
        max_tokens=config.llm.max_token, max_concurrent=1, timeout=config.llm.timeout, enable_thinking=False)

    class RecordedLLM:
        """服务于故障与证据审计：保留实际提示词、响应和 finish_reason，不修改生成参数。"""
        async def generate(self, prompt, **kwargs):
            metadata = kwargs.pop('metadata_sink', {})
            row = dict(prompt=prompt)
            try:
                response = await manager.generate(prompt, metadata_sink=metadata, **kwargs)
                row['response'] = response
                return response
            finally:
                row['metadata'] = metadata
                with (args.output / 'llm_trace.jsonl').open('a', encoding='utf-8') as stream:
                    stream.write(json.dumps(row, ensure_ascii=False) + '\n')

    embeddings = EmbeddingManager(config.embedding)
    store = TripleFactStore(graph['fact_storage'])
    layer = EventStateLayer(decode_upper_references(graph['event_state_layer'], store, graph.get('ingestion_state')))
    memory = SimpleNamespace(graph_builder=SimpleNamespace(graph=graph['graph'], fact_store=store),
        fact_store=store, chunk_storage=graph['chunk_storage'], triple_storage=store.chunk_view(),
        triple_storage_origin=store.origin)
    retriever = StructuralSemanticRetriever(memory, embeddings)
    hybrid = HybridStateRetriever(embeddings, top_k=max(2, args.top_k),
                                   structured_k=args.top_k, semantic_k=args.top_k)
    events = StateCandidateRetriever(embeddings, top_k=args.top_k)

    async def search(query, top_k):
        return await retriever.retrieve(query, top_k=top_k)

    async def seeds(fid, top_k):
        """服务于复用既有写入召回：状态走双路，事件走当前事件候选流程。"""
        if fid in layer.facts:
            fact = layer.facts[fid]
            if fact.get('fact_type') == 'event':
                hits = await events.search(layer, fact)
                hits = [h for h in hits if h['fact'].get('fact_type') == 'event']
            else:
                result = await hybrid.search(layer, fact)
                hits = result['hits']
                # 向量路径若不可用，不悄悄宣称语义补查成功；审计中明确留下信息。
                (args.output / 'initial_recall_trace.json').write_text(json.dumps(result['trace'],
                    ensure_ascii=False, indent=2), encoding='utf-8')
            return [h['fact']['fact_id'] for h in hits[:top_k]]
        result = await search(args.query, top_k)
        return [old for hit in result['evidence'] for old in provider.by_chunk[str(hit['chunk_id'])]
                if old != fid][:top_k]

    provider = GraphEvidenceProvider(graph, search, seeds)
    controller = EvidenceJudgeController(EvidenceJudge(RecordedLLM()), provider,
        max_rounds=args.max_rounds, top_k=args.top_k, max_chunks=args.max_chunks)
    result = await controller.run(args.query, mode=args.mode, target_fact=target)
    (args.output / 'judge_result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    for filename, rows in (('judge_rounds.jsonl', result['rounds']), ('retrieval_actions.jsonl', result['retrieval_actions'])):
        with (args.output / filename).open('w', encoding='utf-8') as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False) + '\n')
    print(json.dumps(dict(status=result['status'], stop_reason=result['stop_reason'],
        judge_calls=len(result['rounds']), chunks=len(result['evidence']), update_applied=False), ensure_ascii=False))


def main():
    """显式启用；默认审查新记忆关系，不改任何生产写入或问答入口。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--query', required=True)
    parser.add_argument('--mode', choices=['question', 'memory_relation'], default='memory_relation')
    parser.add_argument('--new-fact-id')
    parser.add_argument('--config', type=Path, default=Path('config/Memory_EventState_sample0_rebuild.yaml'))
    parser.add_argument('--max-rounds', type=int, default=3)
    parser.add_argument('--top-k', type=int, default=5)
    parser.add_argument('--max-chunks', type=int, default=8)
    asyncio.run(run(parser.parse_args()))


if __name__ == '__main__':
    main()
