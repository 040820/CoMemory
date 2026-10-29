"""共用证据服务的旁路置信度适配器：关联检索审计、原文引用与逐目标角色。

本模块不调用 LLM，不修改 Judge 状态，不决定记忆更新。检索相关性单独展示；
原文已绑定的直接证据无需“再被检索命中”才能计入支持度。所有数值未经校准。
"""
from collections import defaultdict

from ..evidence_confidence import evaluate_candidate, rank_strength, route_confidence_detail


def confidence_shadow(last, retrievals, pair, mode):
    """把一次审查映射为逐目标分析；技术失败不进入不确定性分布。

    直接来源、原文定位和来源分组来自程序；supports/refutes 仍来自模型判断。
    不把时间字段存在、来源身份或同一路多次命中当作独立支持证据。
    """
    base = dict(mode='shadow', affects_decision=False, relation_confidence=None,
                calibrated=False, independence_verified=False)
    if not last or last.get('evaluation_status') == 'technical_error':
        return dict(base, status='NOT_EVALUATED', uncertainty=None, targets={},
                    reason='technical_error' if last else 'no_judge_assessment')
    candidates = defaultdict(list)
    for result in retrievals:
        for hit in result.get('candidates', []):
            candidates[hit['fact_id']].append((hit, result['config']))
    shown = {row['chunk_id']: row for row in last.get('shown_evidence', [])}
    items = []
    for role in last.get('evidence_roles', []):
        source = shown.get(role['chunk_id'])
        if source is None:
            continue
        strengths = {}
        for hit, config in candidates.get(role['fact_id'], []):
            for route, rank in hit.get('route_ranks', {}).items():
                head = route.split(':', 1)[0]
                width = config.get('structured_k', 12) if head == 'structured' else (
                    config.get('max_graph_nodes', 64) if head == 'graph' else config.get('semantic_k', 12))
                value = max(0.0, hit.get('cosine_scores', {}).get(route, 0.0)) if head == 'semantic' else rank_strength(rank, width)
                strengths[head] = max(strengths.get(head, 0.0), value)
        # 引用存在性只代表可核查，不是真值概率。检索分数不乘入支持度：
        # 否则固定锚点因为被排除在候选排序之外，反而会得到零支持。
        fields = dict(has_source_trace=1.0, quote_match='exact', field_completeness=1.0,
                      source_type=1.0, alignment_confidence=None)
        item = dict(role, independence_group=source.get('independence_group') or 'chunk:' + role['chunk_id'],
                    route_strengths={'source_bound': 1.0}, credibility_features=fields,
                    retrieval_consistency=route_confidence_detail(strengths),
                    score_basis='source_binding_and_model_role_not_retrieval_rank')
        items.append(item)
    from ..evidence_confidence import EvidenceConfidenceConfig
    # 内部 source_bound 是引用通过后的恒等入口，不是额外搜索路线或“100% 可信”。
    config = EvidenceConfidenceConfig(route_weights={'source_bound': 1.0})
    result = evaluate_candidate(items, config,
        old_fact_ids=[pair['old_fact_id']] if pair else [],
        new_fact_ids=[pair['new_fact_id']] if pair else [],
        mode='relation' if pair else 'question')
    result.update(base)
    result['retrieval_signals'] = [dict(fact_id=item['fact_id'], target_fact_id=item['target_fact_id'],
        **item['retrieval_consistency']) for item in items]
    # 关系结论与目标 Fact 支持度分开保存；没有显式关系角色就保持未评估。
    relation = result.get('targets', {}).get('relation')
    result['relation_confidence'] = relation['belief'] if relation else None
    return result
