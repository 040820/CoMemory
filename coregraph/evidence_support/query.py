"""新图问答适配模块：把 event_index 图接到共用证据检索、Judge 和回答流程。

只负责格式装配和门面返回值，不重建图、不执行状态更新。旧摘要 Baseline
不调用本模块；旧实验若显式使用 structural_semantic，继续走兼容检索器。
"""
from copy import deepcopy

from .service import EvidenceSupport


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
    return dict(chunk_storage=memory.chunk_storage, fact_storage=store.to_dict(), event_state_layer=upper)


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
