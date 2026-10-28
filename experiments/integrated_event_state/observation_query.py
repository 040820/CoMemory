"""追加观察模式的查询适配：展示来源断言和候选关联，不伪装已确认状态。

本模块复用既有问答检索结果，不改变 Chunk 排名，也不实现完整证据链推理器。
它避免旧查询格式将“最近观察到的说法”直接标为唯一有效值或已确认事件日期。
"""

from __future__ import annotations

import json
from typing import Any, Callable

from prompt.query_prompt import QUERY_PROMPT


async def query_observations(question: str, question_time: str, layer: Any,
                             retrieved: dict[str, Any], answer_llm: Any,
                             visible_by_time: Callable[[str, str], bool],
                             confidence_records: dict[str, Any] | None = None) -> dict[str, Any]:
    """服务于实验查询入口：有限展开检索证据相关的断言与候选边。

    输入 retrieved 是既有检索器的结果，不发起第二次检索。只接受问题时间前
    已观察的来源；最多展示八条候选关联和十六条端点断言。候选边不算证据链，
    关系分数、观察先后与较新说法均不能单独证明因果或旧事实失效。
    """
    evidence = retrieved['evidence']
    chunk_ids = {str(hit['chunk_id']) for hit in evidence}
    observations = [fact for fact in layer.facts.values()
                    if str(fact['chunk_id']) in chunk_ids
                    and visible_by_time(fact.get('observed_at'), question_time)]
    observations.sort(key=lambda fact: (fact.get('position', 0), fact['fact_id']))
    observations = observations[:8]
    seed_ids = {fact['fact_id'] for fact in observations}
    # 服务于证据追溯：只展开命中事实相连的候选端点，防止整个实体历史进入提示词。
    relations = []
    for relation in layer.fact_relations:
        # 服务于查询解释：失败/无关判断属于审计，不能挤占有限的关联证据预算。
        # 这些标签仍仅是可能关系；本步骤不确认因果，也不裁决事实有效性。
        if relation.get('relation') not in {'duplicate', 'complement', 'coexists', 'conflict',
                                            'correction', 'replacement'}:
            continue
        old_id, new_id = relation.get('old_fact_id'), relation.get('new_fact_id')
        if not ({old_id, new_id} & seed_ids):
            continue
        old, new = layer.facts.get(old_id), layer.facts.get(new_id)
        if not old or not new or not all(visible_by_time(
                fact.get('observed_at'), question_time) for fact in (old, new)):
            continue
        relations.append(relation)
    relations = relations[:8]
    by_id = {fact['fact_id']: fact for fact in observations}
    for relation in relations:
        for name in ('old_fact_id', 'new_fact_id'):
            fact_id = relation[name]
            if len(by_id) < 16:
                by_id.setdefault(fact_id, layer.facts[fact_id])
    # 仅输出两端均展示的关系，避免生成悬空引用，看不到证据却误以为链条完整。
    relations = [relation for relation in relations if all(
        relation[name] in by_id for name in ('old_fact_id', 'new_fact_id'))]
    views = [{key: fact.get(key) for key in (
        'fact_id', 'entity_id', 'attribute', 'value', 'source_attribute', 'source_value',
        'scope', 'modality', 'observed_at', 'event_time', 'event_time_raw', 'time_basis',
        'chunk_id', 'event_id', 'assertion_status', 'validity_status', 'triple_refs',
        'extraction_confidence', 'extraction_evaluation_id')}
        | {'quote': str(fact.get('quote') or '')[:1000]} for fact in by_id.values()]
    links = [{key: relation.get(key) for key in (
        'relation_id', 'old_fact_id', 'new_fact_id', 'relation', 'status', 'inference_status', 'evaluation_id')}
        for relation in relations]
    # 服务于可解释回答：只展开本次已展示端点/关联的有限评价，原始响应不进入提示词。
    # 分数是未校准估计，不能按高低自动采纳关系；没有评价时仍沿用旧查询流程。
    assessment_views = {}
    if confidence_records is not None:
        selected_links = {r['relation_id'] for r in relations}
        for name in ('fact_evaluations', 'evidence_links', 'relation_evaluations'):
            selected = [r for r in confidence_records.get(name, {}).values()
                        if r.get('is_current_source', True) and
                        (r.get('fact_id') in by_id or r.get('link_id') in selected_links)]
            assessment_views[name] = [{k: r.get(k) for k in (
                'evaluation_id', 'fact_id', 'link_id', 'chunk_id', 'role', 'score', 'status',
                'calibration', 'validation_status', 'scope', 'verdict', 'checks',
                'span', 'independence_group', 'reason')} for r in selected[:16]]
    chunks = '\n'.join(f"Chunk {hit['chunk_id']} ({hit.get('session_time') or ''}): "
                       f"{hit['text'][:2800]}" for hit in evidence)
    triples = '; '.join(str(triple) for hit in evidence for triple in hit.get('triples', []))
    context = (
        '\n以下为原文断言，不是唯一当前状态。candidate 关系仅为待验证线索，'
        '没有完成证据链推导。不得因较新、相似或存在关联边就判旧事实失效或存在因果；'
        '解释原因必须指出实际原文证据，无法建立完整支持时明确说明不足。\n'
        f'来源断言：{json.dumps(views, ensure_ascii=False)}\n'
        f'候选关联：{json.dumps(links, ensure_ascii=False)}'
    )
    if assessment_views:
        context += ('\n以下分项评价均未校准，局部支持不是完整证据链验证；'
                    'null 表示未评估或无法给出有效分数，不能当作 0。'
                    '不得把分数解释为真值概率，不得因高分直接确认纠正或替换。\n'
                    + json.dumps(assessment_views, ensure_ascii=False))
    prompt = QUERY_PROMPT.format(question=question, question_time=question_time,
                                 triples=triples or 'None available',
                                 chunks=(chunks or 'None available') + context)
    answer = await answer_llm.generate(prompt)
    return {
        'question': question, 'answer': answer.strip() if answer else '',
        'top_session_ids': list(dict.fromkeys(str(hit['session_id']) for hit in evidence
                                              if hit.get('session_id'))),
        'query_analysis': retrieved.get('query_analysis', {}), 'evidence': evidence,
        'state_evidence': [], 'current_state_evidence': [], 'event_instance_evidence': [],
        'observation_evidence': views, 'candidate_relations': links,
        'confidence_evaluations': assessment_views,
        'write_policy': 'append_only', 'inference_status': 'not_evaluated',
    }
