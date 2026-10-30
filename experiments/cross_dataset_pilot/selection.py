"""实验选样模块：只按类型与固定随机种子选题，不查看模型结果或答案难易。

原生 gold 只进入评分记录；构图输入使用完整会话，不按 answer_session_ids 裁剪。
"""
from datetime import datetime
import random
import re


def timestamp(value):
    """保留原生会话时间的小时分钟；不能沿用旧转换器只截日期的行为。"""
    text = re.sub(r'\s+\([A-Za-z]{3}\)', '', str(value).strip()).replace('/', '-')
    return datetime.fromisoformat(text.replace('Z', '+00:00')).isoformat(sep=' ')


def select_locomo(native, seed=20261029, sample_index=0):
    """sample0 类别 1/2/4 各四题，类别 5 两题；排除反复调试过的第零题。

    sample0 不是独立留出会话，manifest 必须明确记录开发使用情况。
    """
    sample = native[sample_index]
    rng, selected = random.Random(seed), []
    turns = {row['dia_id']: row for key, rows in sample['conversation'].items()
             if key.startswith('session_') and isinstance(rows, list)
             for row in rows if row.get('dia_id')}
    for category, count in ((1, 4), (2, 4), (4, 4), (5, 2)):
        pool = [(i, q) for i, q in enumerate(sample['qa']) if i != 0 and int(q['category']) == category]
        if len(pool) < count:
            raise ValueError('LoCoMo 类型样本不足，不静默减少实验规模')
        for index, row in sorted(rng.sample(pool, count)):
            refs = [str(ref) for ref in row.get('evidence', [])]
            selected.append(dict(case_id=f'locomo_s{sample_index}_q{index}', dataset='locomo',
                graph_key='locomo_sample0', native_index=index, sample_id=sample.get('sample_id'),
                question=row['question'], expected_answer=row.get('answer'), question_type=str(category),
                unanswerable=category == 5, as_of=None,
                gold_sessions=sorted({ref.split(':')[0] for ref in refs}), gold_turn_ids=refs,
                gold_turns={ref: turns[ref]['text'] for ref in refs if ref in turns},
                unknown_gold_turn_ids=[ref for ref in refs if ref not in turns],
                development_conversation=True))
    return selected


def select_longmem(native, seed=20261029):
    """三个类型各随机一题；首轮拒答样本由 LoCoMo 类别 5 提供。"""
    rng, selected = random.Random(seed), []
    for kind in ('knowledge-update', 'temporal-reasoning', 'multi-session'):
        pool = [(i, row) for i, row in enumerate(native)
                if row['question_type'] == kind and not str(row['question_id']).endswith('_abs')]
        index, row = rng.choice(pool)
        qid = re.sub(r'[^A-Za-z0-9_-]', '_', str(row['question_id']))
        selected.append(dict(case_id=f'longmem_{index}_{qid}', graph_key=f'longmem_{index}_{qid}',
            dataset='longmemeval', native_index=index, question_id=row['question_id'],
            question=row['question'], expected_answer=row['answer'], question_type=kind,
            unanswerable=False, as_of=timestamp(row['question_date']),
            gold_sessions=list(row['answer_session_ids']), gold_turn_ids=[], gold_turns={},
            development_conversation=False))
    return selected


def longmem_corpus(row):
    """完整 haystack 转为独立会话记录，验证平行数组，并按完整时间稳定排序。

    不读取问题/答案字段；相同时间保持原会话顺序，不修改已有 dataset/processed。
    """
    from dataset.longmem import build_context
    dates, ids, sessions = (row.get(key, []) for key in
                           ('haystack_dates', 'haystack_session_ids', 'haystack_sessions'))
    if not len(dates) == len(ids) == len(sessions) or not ids or len(set(ids)) != len(ids):
        raise ValueError('haystack 数组缺失、不等长或会话 ID 重复')
    corpus = [dict(session_time=timestamp(date), session_id=str(sid), context=build_context(messages))
              for date, sid, messages in zip(dates, ids, sessions)]
    return sorted(corpus, key=lambda doc: doc['session_time'])
