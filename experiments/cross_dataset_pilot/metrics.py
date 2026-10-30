"""小样本诊断指标：来源覆盖是召回代理，不等于证明语义支持或事实为真。"""
import re
from statistics import mean


def source_coverage(report, gold_sessions, graph):
    """分别统计 Seed Top-5、候选、实际展示、答案引用；按 Session 去重。

    原生来源不是穷尽的支持证据集合，不把未标注 Session 当成错误证据。
    gold 为空返回 N/A；不将“全库包含答案”计为检索命中。
    """
    chunks = graph.get('chunk_storage', {})
    def sessions(rows):
        return {str(row.get('session_id') or chunks.get(str(row.get('chunk_id')), {}).get('session_id'))
                for row in rows if row.get('session_id') or str(row.get('chunk_id')) in chunks}
    seeds = report.get('query_seed_trace', {}).get('chunk_candidates')
    stages = dict(seed_top5=sessions([dict(chunk_id=cid) for cid in (seeds or [])]),
        candidate=sessions(report.get('evidence', [])),
        shown=sessions([row for turn in report.get('rounds', []) for row in turn.get('shown_evidence', [])]),
        cited=sessions(report.get('answer_citations', [])))
    gold = set(map(str, gold_sessions))
    return {stage: dict(sessions=sorted(found) if found is not None else None,
                       matched=sorted(gold & found) if found is not None else [],
                       recall=len(gold & found) / len(gold) if gold and found is not None else None,
                       all_gold_found=gold <= found if gold and found is not None else None)
            for stage, found in stages.items()}


def abstention(report):
    """仅描述暂缓行为；是否应当暂缓还需原生不可回答标签或人工标注。"""
    technical = report.get('evaluation_status') == 'technical_error' or report.get('status') == 'ERROR'
    answered = bool(report.get('answer')) and report.get('answer_status') == 'source_grounded'
    return dict(answered=answered, technical_failure=technical,
        semantic_abstention=not answered and not technical and report.get('evaluation_status') == 'semantic_uncertain')


def exact_match(answer, expected):
    """保守文本 EM，只做大小写和空白归一；日期等价交给独立评分并保留原文。"""
    normalize = lambda x: re.sub(r'\s+', ' ', str('' if x is None else x).strip().casefold())
    return bool(answer) and normalize(answer) == normalize(expected)


def locomo_turn_coverage(case, report, graph):
    """LoCoMo 原生证据轮次的严格全文覆盖：用于区分上游缺失和检索遗漏。

    只做空白归一化后的完整原句匹配。窗口可能已给出关键片段而没有整轮全文，
    因此该量不是语义支持率；LongMemEval 没有轮次金标，返回 N/A。
    """
    gold = case.get('gold_turns', {})
    if not gold:return None
    normalize = lambda text: re.sub(r'\s+', ' ', text).strip()
    chunks=graph['chunk_storage']
    def located(rows):
        return [(str(row.get('session_id') or chunks.get(str(row.get('chunk_id')),{}).get('session_id')), row['text'])
                for row in rows]
    stages = dict(stored=located([dict(row,chunk_id=cid) for cid,row in chunks.items()]),
        candidate=located(report.get('evidence', [])),
        shown=[(str(row['session_id']),window['text']) for turn in report.get('rounds', [])
               for row in turn.get('shown_evidence', []) for window in row.get('windows', [row])],
        cited=located([dict(row,text=row['quote']) for row in report.get('answer_citations', [])]))
    result={}
    for stage,texts in stages.items():
        normalized=[(sid,normalize(text)) for sid,text in texts]
        matched=[tid for tid,text in gold.items() if any(sid==tid.split(':')[0] and normalize(text) in source
                                                       for sid,source in normalized)]
        result[stage]=dict(full_turn_coverage=len(matched)/len(gold),matched_turn_ids=matched)
    return result


def summarize(rows):
    """按数据集分开汇总，失败保留分母；不把模型评分当成人工准确率。"""
    output = {}
    for dataset in sorted({row['dataset'] for row in rows}):
        group = [r for r in rows if r['dataset'] == dataset]
        ordinary = [r for r in group if not r['unanswerable']]
        values = {}
        for stage in ('seed_top5', 'candidate', 'shown', 'cited'):
            recalls = [r['source_coverage'][stage]['recall'] for r in group
                       if r.get('source_coverage', {}).get(stage, {}).get('recall') is not None]
            values[stage] = dict(mean_recall=mean(recalls) if recalls else None, measured_questions=len(recalls))
        scored = [r for r in ordinary if isinstance(r.get('grading', {}).get('correct'), bool)]
        correct = sum(r['grading']['correct'] for r in scored)
        output[dataset] = dict(questions=len(group), answerable_questions=len(ordinary),
            llm_graded_correct=correct, grading_valid=len(scored), grading_unknown=len(ordinary)-len(scored),
            llm_accuracy_on_scored=correct/len(scored) if scored else None,
            end_to_end_correct_lower_bound=correct/len(ordinary) if ordinary else None,
            technical_failures=sum(r.get('abstention', {}).get('technical_failure', False) for r in group),
            semantic_abstentions=sum(r.get('abstention', {}).get('semantic_abstention', False) for r in group),
            source_coverage=values, scoring_status='same_model_proxy_requires_human_review')
    return output
