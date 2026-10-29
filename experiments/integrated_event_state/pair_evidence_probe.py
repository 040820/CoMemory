"""固定事实对的受控证据实验：复用真实 Judge，分开评估找证据与判关系。

所属模块：独立演化实验。参考标签在运行前由实验设计给定，并非人工复核的
LoCoMo gold。预约对照组的两端完全相同，仅改变解释证据，防止把猜中标签
误当作找到了证据。本文件不修改任何生产事实或历史实验结果。
"""
import argparse
import asyncio
from copy import deepcopy
from hashlib import sha256
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def make_case(case_id, old, new, relation, *, attribute='employer', values=('Aster', 'Aster'),
              scopes=None, bridge=None, question=None):
    """样本构造模块：固定原话、主体、作用域和时间，不调用模型制造期望标签。"""
    scopes = scopes or ({}, {})
    chunks = {}
    observations = {}
    for fid, cid, text, date, value, scope in (
        ('F1', 'C1', old, '2023-04-01', values[0], scopes[0]),
        ('F2', 'C2', new, '2023-04-03', values[1], scopes[1])):
        chunks[cid] = dict(text=text, chunk_id=cid, session_id=cid, session_time=date)
        observations[fid] = dict(fact_id=fid, entity_id='Mira', entity_index=0,
            attribute=attribute, value=value, scope=scope, modality='actual',
            observed_at=date, chunk_id=cid, quote=text, fact_type='state', status='observed')
    links = [dict(link_id='Lpair', from_fact_id='F1', to_fact_id='F2',
                  relation='conflict', status='candidate')]
    if bridge:
        chunks['CB'] = dict(text=bridge, chunk_id='CB', session_id='CB', session_time='2023-04-02')
        observations['FB'] = dict(fact_id='FB', entity_id='Mira', attribute='reported_context',
            value='appointment context', chunk_id='CB', quote=bridge, scope=scopes[0], modality='actual')
        # 该候选边只提供线索，不携带正确关系或证明。Judge 必须回查 bridge 原文。
        links.append(dict(link_id='Lcontext', from_fact_id='F1', to_fact_id='FB',
                          relation='complement', status='candidate'))
    # 有关联但不能用于解释 Mira 的干扰项，用于检查人物和事项是否被混淆。
    noise = 'Mira: My friend Noah has a different appointment on May 6. That is not my appointment.'
    chunks['CN'] = dict(text=noise, chunk_id='CN', session_id='CN', session_time='2023-04-02')
    observations['FN'] = dict(fact_id='FN', entity_id='Noah', attribute='appointment_date',
        value='2023-05-06', chunk_id='CN', quote=noise, scope={'appointment': 'Noah'}, modality='actual')
    links.append(dict(link_id='Lnoise', from_fact_id='F1', to_fact_id='FN',
                      relation='coexists', status='candidate'))
    return dict(case_id=case_id, chunks=chunks, observations=observations, links=links,
        review_goal='relation',
        question=question or '判断这对新旧事实是重复、兼容补充、并存、现实变化、纠正还是未解决冲突。',
        reference=dict(relations=[relation], status='DEFER' if relation == 'uncertain' else 'SUFFICIENT',
            explanation_chunk_ids=['CB'] if bridge else [],
            label_source='controlled_probe_author_reference_not_independent_human_gold'))


def cases():
    """实验设计模块：八个受控条件，覆盖无需第三条证据与必须解释原因的情况。"""
    output = [
        make_case('duplicate', 'Mira: My employer is Aster.', 'Mira: I still work for Aster.', 'duplicate'),
        make_case('coexisting_pets', 'Mira: I have a dog.', 'Mira: I also adopted a cat; I still have my dog.',
                  'coexists', attribute='owns_pet', values=('dog', 'cat')),
        make_case('explicit_move', 'Mira: My primary home is Beijing.',
                  'Mira: I moved my primary home from Beijing to Shanghai yesterday.',
                  'replacement', attribute='primary_residence', values=('Beijing', 'Shanghai')),
        make_case('explicit_correction', 'Mira: My dental appointment is on May 7.',
                  'Mira: I was wrong about my dental appointment: it has always been May 6, not May 7.',
                  'correction', attribute='appointment_date', values=('2023-05-07', '2023-05-06'),
                  scopes=({'appointment': 'dental'}, {'appointment': 'dental'})),
        make_case('different_scope', 'Mira: I am a researcher on project Alpha.',
                  'Mira: On project Beta I am a reviewer; I also remain a researcher on Alpha.',
                  'coexists', attribute='role', values=('researcher', 'reviewer'),
                  scopes=({'project': 'Alpha'}, {'project': 'Beta'})),
    ]
    question = ('请查明这次牙科预约日期差异究竟是实际改期，还是原通知被误读后的纠正。'
                '必须寻找明确区分两种解释的原文；只有两个日期不足以判断原因，缺少依据则暂缓。')
    for cid, relation, bridge in (
        ('appointment_rescheduled', 'replacement',
         'Mira: The clinic actually rescheduled my dental appointment from May 7 to May 6. The earlier notice was correct at that time.'),
        ('appointment_misread', 'correction',
         'Mira: I misread the original dental appointment notice. It has always said May 6, not May 7; the clinic did not reschedule it.'),
        ('appointment_unexplained', 'uncertain', None)):
        case = make_case(cid, 'Mira: My dental appointment is on May 7.',
            'Mira: My dental appointment is now on May 6.', relation,
            attribute='appointment_date', values=('2023-05-07', '2023-05-06'),
            scopes=({'appointment': 'dental'}, {'appointment': 'dental'}),
            bridge=bridge, question=question)
        case['review_goal'] = 'evolution_reason'
        output.append(case)
    return output


def score_case(case, result):
    """评估模块：分开统计检索、展示、合法引用、关系判断，防止高充分率掩盖错误。

    explanation_cited 只检查准确引用来源，不把字符串存在当作语义支持已被独立验证。
    joint_pass 也只是受控参考下的探针结果，不是事实真实概率或数据集准确率。
    """
    last = result['rounds'][-1] if result.get('rounds') else {}
    cited = set()
    legal = True
    shown = {str(e['chunk_id']): e for e in last.get('shown_evidence', [])}
    for citation in last.get('citations', []):
        record = shown.get(str(citation.get('chunk_id')))
        quote = citation.get('quote')
        if (not record or not isinstance(quote, str) or not quote or quote not in record['text']
                or citation.get('fact_id') not in record.get('fact_ids', [])):
            legal = False
        else:
            cited.add(str(citation['chunk_id']))
    required = set(case['reference']['explanation_chunk_ids'])
    retrieved = {str(e['chunk_id']) for e in result.get('evidence', [])}
    available = {str(e['chunk_id']) for r in result.get('rounds', []) for e in r.get('shown_evidence', [])}
    relation = last.get('relation', 'uncertain')
    relation_ok = relation in case['reference']['relations']
    status_ok = result['status'] == case['reference']['status']
    technical_failure = (result['stop_reason'] in ('retrieval_failed', 'no_initial_evidence') or
                         any(r.get('stop_reason') in ('judge_failed', 'empty_context')
                             for r in result.get('rounds', [])))
    valid_pair = {'C1', 'C2'}.issubset(cited) if result['status'] == 'SUFFICIENT' else True
    explanation_ok = required.issubset(cited)
    return dict(case_id=case['case_id'], predicted_relation=relation,
        expected_relations=case['reference']['relations'], status=result['status'],
        expected_status=case['reference']['status'], stop_reason=result['stop_reason'],
        relation_correct=relation_ok, status_correct=status_ok,
        explanation_required=bool(required), explanation_retrieved=required.issubset(retrieved),
        explanation_shown=required.issubset(available), explanation_cited=explanation_ok,
        citation_legal=legal, technical_failure=technical_failure,
        judge_calls=len(result.get('rounds', [])),
        # 原始严格关系指标保留。原因证据缺失时，关系枚举与“是否正确暂缓”必须
        # 分开：返回并存但明确暂缓原因判断，不等于已经解释了原因，也不应强迫更新。
        task_outcome_correct=(relation_ok or case['reference']['status'] == 'DEFER')
            and status_ok and explanation_ok and legal and valid_pair and not technical_failure,
        joint_pass=relation_ok and status_ok and explanation_ok and legal and valid_pair and not technical_failure)


async def run(args):
    """真实模型实验入口：冻结输入后逐例执行，所有结果写入新目录，失败不伪装成功。"""
    import networkx as nx
    from base.llm import LLMManager
    from coregraph.conflict_review import (ConflictPairIndex, ConflictPairJudge,
                                           PairEvidenceProvider, vllm_compact_json_options)
    from coregraph.evidence_judge import EvidenceJudgeController
    from init.config import Config
    selected = [case for case in cases() if not args.case or case['case_id'] in args.case]
    if not selected:
        raise ValueError('没有匹配样本')
    args.output.mkdir(parents=True, exist_ok=False)
    config = Config.parse(args.config)
    if config.llm.api_key == 'demo-key':
        raise ValueError('探针必须使用真实模型')
    save = lambda p, v: p.write_text(json.dumps(v, ensure_ascii=False, indent=2), encoding='utf-8')
    save(args.output / 'frozen_cases.json', selected)
    root = Path(__file__).resolve().parents[2]
    save(args.output / 'manifest.json', dict(model=config.llm.model,
        experiment='fixed_pair_explanation_controlled_probe_v1', case_ids=[c['case_id'] for c in selected],
        input_sha256=sha256((args.output / 'frozen_cases.json').read_bytes()).hexdigest(),
        config_sha256=sha256(args.config.read_bytes()).hexdigest(),
        code_sha256={f: sha256((root / f).read_bytes()).hexdigest() for f in
            ('coregraph/conflict_review.py', 'coregraph/evidence_judge.py',
             'coregraph/graph_evidence_provider.py', 'experiments/integrated_event_state/pair_evidence_probe.py')},
        label_source='probe_author_reference_not_independent_human_gold', update_applied=False,
        initial_context_k=args.initial_context_k, compact_json=args.compact_json,
        max_response_repairs=args.response_repairs, max_llm_requests=3 + args.response_repairs,
        scoring_version='v2_separate_exact_relation_and_task_abstention'))
    lc = config.llm
    manager = LLMManager(api_key=lc.api_key, model=lc.model, base_url=lc.base_url,
        max_tokens=lc.max_token, max_concurrent=1, timeout=lc.timeout, enable_thinking=False)
    scores = []
    for case in selected:
        folder = args.output / case['case_id']; folder.mkdir()
        llm_requests = 0
        graph = dict(graph=nx.Graph(), chunk_storage=case['chunks'],
            fact_storage=dict(version=2, origin='extracted', facts={}, observations=case['observations'],
                              time_assertions={}, attribute_assertions={}),
            event_state_layer=dict(schema='event_index_v1', groups={}, links=case['links']))
        pair = ConflictPairIndex(graph).pending('F2')[0]
        retriever = None
        class RecordedLLM:
            """模型审计模块：保留输入、输出及 finish_reason，不修改响应来迁就标签。"""
            async def generate(self, prompt, **kwargs):
                nonlocal llm_requests
                llm_requests += 1
                metadata = kwargs.pop('metadata_sink', {})
                trace = dict(prompt=prompt)
                if args.compact_json:
                    # 服务器 OpenAPI 已确认该 vLLM 扩展存在；仅限制 JSON 语法空白，
                    # 不修改字符串原文，不修补输出，不使用参考标签。其他后端默认关闭。
                    kwargs = vllm_compact_json_options(kwargs)
                try:
                    result = await manager.generate(prompt, metadata_sink=metadata, **kwargs)
                    trace['response'] = result
                    return result
                finally:
                    trace['metadata'] = metadata
                    with (folder / 'llm_trace.jsonl').open('a', encoding='utf-8') as stream:
                        stream.write(json.dumps(trace, ensure_ascii=False) + '\n')
        async def search(query, top_k):
            """检索兜底模块：图扩展无新增证据时才加载 BGE-M3，使用原文而非参考标签。"""
            nonlocal retriever
            if retriever is None:
                from types import SimpleNamespace
                from base.embeddings import EmbeddingManager
                from coregraph.triple_fact_store import TripleFactStore
                from experiments.structural_semantic_retrieval.retriever import StructuralSemanticRetriever
                store = TripleFactStore(graph['fact_storage'])
                memory = SimpleNamespace(graph_builder=SimpleNamespace(graph=graph['graph'], fact_store=store),
                    fact_store=store, chunk_storage=graph['chunk_storage'], triple_storage={}, triple_storage_origin='extracted')
                retriever = StructuralSemanticRetriever(memory, EmbeddingManager(config.embedding))
            return await retriever.retrieve(query, top_k=top_k)
        provider = PairEvidenceProvider(graph, search, pair, initial_context_k=args.initial_context_k)
        result = await EvidenceJudgeController(ConflictPairJudge(RecordedLLM(), pair,
            review_goal=case['review_goal'], max_response_repairs=args.response_repairs), provider,
            max_rounds=3, top_k=5, max_chunks=12, graph_first=True).run(case['question'], target_fact=pair['new_fact'])
        save(folder / 'review_result.json', result)
        score = score_case(case, result)
        score['llm_requests'] = llm_requests
        score['response_repairs_used'] = max((r.get('response_repairs_used', 0) for r in result['rounds']), default=0)
        scores.append(score)
        save(folder / 'score.json', score)
        print(json.dumps(score, ensure_ascii=False), flush=True)
    save(args.output / 'summary.json', dict(cases=len(scores), joint_pass=sum(s['joint_pass'] for s in scores),
        task_outcomes_correct=sum(s['task_outcome_correct'] for s in scores),
        technical_failures=sum(s['technical_failure'] for s in scores), scores=scores,
        update_applied=False, interpretation='controlled_probe_not_dataset_generalization'))
    print('PROBE_COMPLETE', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path('config/Memory_EventState_sample0_rebuild.yaml'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--case', action='append')
    parser.add_argument('--initial-context-k', type=int, default=0)
    parser.add_argument('--compact-json', action='store_true')
    parser.add_argument('--response-repairs', type=int, choices=[0, 1], default=0)
    args = parser.parse_args()
    os.environ.setdefault('HF_HUB_OFFLINE', '1'); os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
    asyncio.run(run(args))
