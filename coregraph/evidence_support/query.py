"""新图问答适配模块：把 event_index 图接到共用证据检索、Judge 和回答流程。

只负责格式装配和门面返回值，不重建图、不执行状态更新。旧摘要 Baseline
不调用本模块；旧实验若显式使用 structural_semantic，继续走兼容检索器。
"""
from copy import deepcopy

from .service import EvidenceSupport


async def question_seed_fact(question, service, as_of=None):
    """问答候选入口：复用既有 Chunk 检索，再在首个有来源断言的 Chunk 选一个 Fact。

    算法：旧结构化语义检索取前 5 个 Chunk；在第一个具有可见、原文绑定
    Observation 的 Chunk 内按 BGE-M3 余弦选 1 条候选。它只是搜索锚点，不是
    答案真值；共用证据检索仍会补查支持与反证。没有原图/Embedding 时明确降级。
    """
    from .retriever import _date
    cutoff = _date(as_of, cutoff=True) if as_of else None
    if as_of and cutoff is None:
        raise ValueError('问题时间无法解析')
    trace = dict(method='structural_semantic_chunk_then_fact', top_chunks=5, seed_facts=1,
                 as_of=as_of, selected_ids=[])
    selector = getattr(service, 'query_seed_retriever', None)
    if service.retriever.embeddings is None or (selector is None and service.graph.get('graph') is None):
        return [], dict(trace, status='unavailable:missing_graph_or_embeddings')
    try:
        if selector is None or (hasattr(service, '_query_seed_scope') and service._query_seed_scope != as_of):
            import networkx as nx
            from types import SimpleNamespace
            from ..triple_fact_store import TripleFactStore
            from ..structural_semantic_retriever import StructuralSemanticRetriever
            # 只构造临时检索视图；先过滤来源时间，再建立可见事实关联，避免未来
            # 事实或未来图边影响种子选择。原图、实体索引和原始表均不修改。
            raw = {fid: row for fid, row in service.graph['fact_storage']['facts'].items()
                   if service.retriever._visible(fid, cutoff)}
            store = TripleFactStore(dict(version=1, origin='extracted', facts=raw))
            index = nx.Graph(memory_graph_schema='entity_fact_index_v1')
            for fid, row in raw.items():
                left, right = row['src'], row['tgt']
                if not index.has_edge(left, right):
                    index.add_edge(left, right, edge_type='entity_association', fact_ids=[], weight=0)
                index[left][right]['fact_ids'].append(fid)
                index[left][right]['weight'] += 1
            chunks = {cid: row for cid, row in service.retriever.chunks.items()
                      if any(service.retriever._visible(fid, cutoff) for fid in service.retriever.provider.by_chunk[cid])}
            memory = SimpleNamespace(graph_builder=SimpleNamespace(graph=index), chunk_storage=chunks,
                fact_store=store, triple_storage=store.chunk_view(), triple_storage_origin='extracted')
            selector = StructuralSemanticRetriever(memory, service.retriever.embeddings)
            service.query_seed_retriever, service._query_seed_scope = selector, as_of
        result = await selector.retrieve(question, top_k=5)
        trace['chunk_candidates'] = [str(row['chunk_id']) for row in result.get('evidence', [])]
        observations = service.graph['fact_storage']['observations']
        for hit in result.get('evidence', []):
            cid = str(hit['chunk_id'])
            ids = [fid for fid, row in observations.items()
                   if service.retriever.provider.source_chunk(row) == cid and
                   service.retriever._visible(fid, cutoff) and row.get('quote') and
                   row['quote'] in service.retriever.chunks[cid]['text']]
            if not ids:
                continue
            vectors = await service.retriever._encode([question] + [service.retriever._text(fid) for fid in ids])
            if any(len(v) != len(vectors[0]) for v in vectors):
                raise ValueError('问题种子向量维度不一致')
            scores = {fid: sum(a*b for a, b in zip(vectors[0], vector)) for fid, vector in zip(ids, vectors[1:])}
            selected = min(ids, key=lambda fid: (-scores[fid], fid))
            return [selected], dict(trace, status='ok', selected_ids=[selected], chunk_id=cid,
                                    cosine=scores[selected], interpretation='retrieval_seed_not_answer')
        return [], dict(trace, status='no_grounded_observation_in_top_chunks')
    except Exception as error:
        return [], dict(trace, status=f'unavailable:{type(error).__name__}:{error}')


def memory_graph_view(memory):
    """为新图生成只读服务视图；运行时 EventStateLayer 只在副本上序列化。

    pack_event_index 会更新传入的事实表，因此必须复制，不能让问答改写内存图。
    已经是持久化字典时直接读取，检索服务自身保证不修改输入。
    """
    if isinstance(memory, dict):
        return memory
    from ..lean_event_index import pack_event_index
    store = deepcopy(memory.fact_store)
    upper, _, _ = pack_event_index(memory.event_state_layer.to_dict(), store)
    return dict(graph=memory.graph_builder.graph, chunk_storage=memory.chunk_storage,
                fact_storage=store.to_dict(), event_state_layer=upper)


async def query_with_evidence_support(memory, question, question_time, embeddings, llm, *, support=None):
    """服务于 GraphRAG.query：返回答案、已引用会话和完整审查报告。

    证据不足或技术失败时不把原始模型猜测当答案。as_of 使用显式问题时间；
    没有时间时不擅自使用当前系统日期。最终来源只统计答案实际引用的 Chunk。
    """
    service = support or EvidenceSupport(memory_graph_view(memory), embeddings, llm)
    report = await service.answer_question(question, llm, as_of=question_time or None)
    by_chunk = {row['chunk_id']: row for row in report.get('evidence', [])}
    sessions = list(dict.fromkeys(by_chunk[c['chunk_id']]['session_id']
        for c in report.get('answer_citations', [])
        if c.get('chunk_id') in by_chunk and by_chunk[c['chunk_id']].get('session_id')))
    return dict(answer=report.get('answer') or '证据不足，暂时无法可靠回答。',
                top_session_ids=sessions, evidence_report=report,
                answer_status=report.get('answer_status', 'deferred'))
