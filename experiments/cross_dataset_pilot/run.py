"""双数据集 pilot 运行器：冻结输入/代码、完整构图、取证问答及代理评分。

不修改原始数据、原图或 Baseline。原生答案只在独立 grading 调用和人工核查
文件中使用；取证和 Reader 只接收问题、截止时间及由完整会话构建的图。
"""
import argparse
import asyncio
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import pickle
import shutil
import subprocess
import sys
import time
from types import SimpleNamespace

from .selection import select_locomo, select_longmem, longmem_corpus
from .metrics import source_coverage, abstention, exact_match, summarize, locomo_turn_coverage

ROOT = Path(__file__).resolve().parents[2]


def save(path, value):
    """实验落盘：原子更新当前任务的 JSON，不写回来源图或原始数据。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding='utf-8')
    temporary.replace(path)


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def prepare(args):
    """冻结选题和源码；原图复制到实验目录，LongMemEval 保存完整独立语料。"""
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    native = json.loads(args.locomo.read_text(encoding='utf-8'))
    longmem = json.loads(args.longmem.read_text(encoding='utf-8'))
    cases = select_locomo(native, args.seed) + select_longmem(longmem, args.seed)
    graph = pickle.loads(args.locomo_graph.read_bytes())
    if graph.get('event_state_layer', {}).get('schema') != 'event_index_v1' or graph['fact_storage']['origin'] != 'extracted':
        raise ValueError('LoCoMo 复用图必须是完整来源的新三层图')
    # 用原生首轮对话核对会话身份，防止同为 D1/D2 的另一份 LoCoMo 图被误复用。
    first=native[0]['conversation']['session_1'][0]['text']
    normalize=lambda text:' '.join(text.split())
    if not any(normalize(first) in normalize(row['text']) for row in graph['chunk_storage'].values()
               if str(row.get('session_id'))=='D1'):
        raise ValueError('复用图与 LoCoMo sample0 原生对话不一致')
    target = out / 'graphs/locomo_sample0.pkl'
    target.parent.mkdir(parents=True)
    shutil.copy2(args.locomo_graph, target)
    relative = graph.get('ingestion_state', {}).get('audit_file')
    if relative:
        source = (args.locomo_graph.parent / relative).resolve()
        if not source.is_relative_to(args.locomo_graph.parent.resolve()):
            raise ValueError('源图审计路径越界')
        dest = target.parent / relative;dest.parent.mkdir(parents=True, exist_ok=True);shutil.copy2(source, dest)
    for case in cases:
        if case['dataset'] == 'locomo':
            case['graph_file'] = 'graphs/locomo_sample0.pkl'
        else:
            corpus = longmem_corpus(longmem[case['native_index']])
            if not set(case['gold_sessions']) <= {row['session_id'] for row in corpus}:
                raise ValueError('LongMemEval 原生证据来源不在完整 haystack 中')
            name = f"data/{case['graph_key']}/Corpus.json"
            path = out / name;path.parent.mkdir(parents=True)
            path.write_text(''.join(json.dumps(x, ensure_ascii=False) + '\n' for x in corpus), encoding='utf-8')
            case.update(corpus_file=name, corpus_sha256=digest(path), sessions=len(corpus),
                        graph_file=f"graphs/{case['graph_key']}/graph.pkl")
    save(out / 'cases.json', cases)
    snapshot = out / 'reproducibility/source_snapshot'
    files = {}
    for folder in ('base', 'chunking', 'coregraph', 'data', 'dataset', 'evaluation', 'experiments', 'init', 'prompt', 'query', 'utils'):
        for path in (ROOT / folder).rglob('*.py'):
            if '__pycache__' in path.parts:
                continue
            relative = path.relative_to(ROOT);dest = snapshot / relative
            dest.parent.mkdir(parents=True, exist_ok=True);shutil.copy2(path, dest)
            files[str(relative)] = digest(path)
    shutil.copy2(args.config, out / 'reproducibility/base_config.yaml')
    save(out / 'manifest.json', dict(seed=args.seed, cases_sha256=digest(out/'cases.json'),
        implementation_commit=args.implementation_commit, source_code_sha256=files,
        locomo_source_sha256=digest(args.locomo), longmem_source_sha256=digest(args.longmem),
        locomo_graph_source=str(args.locomo_graph.resolve()), locomo_graph_sha256=digest(args.locomo_graph),
        base_config_sha256=digest(out/'reproducibility/base_config.yaml'),
        selection='stratified_random_before_inference', questions=len(cases),
        native_relation_gold_available=False, grading='same_qwen_model_proxy_not_human',
        history_policy='full_haystack_no_gold_session_filter', time_policy='native_full_time_stable_sort',
        development_note='LoCoMo sample0 was used in development; this is a diagnostic pilot, not held-out SOTA evaluation'))
    return out


async def build_graph(case, out, config, embeddings, manager):
    """构图装配模块：调用正式 DynamicMemory 接口，逐会话保存并支持恢复。

    只在构图期间包装 manager.generate，覆盖 batch_generate 内部的调用；方法
    结束即恢复，避免问答和 grading 重复记账。模型及 Embedding 均复用真实实例。
    """
    from coregraph.dynamic_memory import DynamicMemory
    from chunking.chunk_processor import ChunkProcessor
    from coregraph.evidence_support.transport import AuditedLLM
    path = out / case['graph_file'];path.parent.mkdir(parents=True, exist_ok=True)
    conf = deepcopy(config);conf.data_type = 'LongmemEval';conf.graph.add = True;conf.graph.force = False
    conf.working_dir=str(path.parent);conf.index_name='graph'
    conf.data_root=str((out/case['corpus_file']).parent);conf.dataset_name=case['graph_key']
    save(path.parent/'runtime_config.json',conf.to_dict())
    memory = DynamicMemory(conf, str(path.parent))
    memory.chunk_processor = ChunkProcessor(conf.chunk, data_type=conf.data_type)
    original = manager.generate
    audit = AuditedLLM(SimpleNamespace(generate=original), path.parent / 'build_audit')
    async def logged(prompt, **options):
        return await audit.generate(prompt, **options)
    manager.generate = logged
    try:
        memory.set_extractors(manager, embeddings)
        if path.exists():memory.load_graph(str(path))
        else:await memory.build_graph([])
        corpus_path = out / case['corpus_file']
        if digest(corpus_path) != case['corpus_sha256']:
            raise ValueError('冻结语料指纹不一致')
        corpus = [json.loads(line) for line in corpus_path.read_text(encoding='utf-8').splitlines() if line.strip()]
        present = {str(row['session_id']) for row in memory.chunk_storage.values()}
        for index, document in enumerate(corpus, 1):
            if str(document['session_id']) not in present:
                await memory.add_single_session([document]);memory.save_graph(str(path))
            save(out/'progress.json', dict(phase='building_longmem', case_id=case['case_id'],
                session=index, total_sessions=len(corpus), chunks=len(memory.chunk_storage)))
            print(f"BUILD {case['case_id']} session={index}/{len(corpus)} chunks={len(memory.chunk_storage)}", flush=True)
        actual = {str(row['session_id']) for row in memory.chunk_storage.values()}
        if actual != {str(row['session_id']) for row in corpus}:
            raise ValueError('完整 haystack 会话覆盖不一致')
        if set(memory.chunk_storage) != set(memory.event_state_layer.processed_chunks):
            raise ValueError('存在未处理 Chunk，不把部分图当作构建完成')
        stored = pickle.loads(path.read_bytes())
        from coregraph.lean_event_index import read_audit_snapshot
        history = read_audit_snapshot(str(path), stored['ingestion_state'])
        for name in ('decisions', 'fact_relations', 'review_queue'):
            (path.parent/(name+'.jsonl')).write_text(''.join(json.dumps(row, ensure_ascii=False, default=str)+'\n'
                for row in history[name]), encoding='utf-8')
        save(path.parent/'build_summary.json', dict(sessions=len(actual), chunks=len(memory.chunk_storage),
            triples=len(stored['fact_storage']['facts']), observations=len(stored['fact_storage']['observations']),
            candidate_links=len(stored['event_state_layer']['links']), audit_counts=stored['ingestion_state']['audit_counts']))
        return stored
    finally:
        manager.generate = original


async def grade(case, report, manager, directory, config):
    """评分模块：复用仓库 rubric，独立模型调用；gold 在本阶段才被提供。

    不调用旧 yes 子串解析器；必须输出布尔 JSON。相同 Qwen 同时作 Reader 与
    grader 会有偏差，结果始终标记为代理评分并导出人工核查表。
    """
    from coregraph.evidence_support.transport import AuditedLLM
    from evaluation.llm_evaluator import LLMEvaluator
    if case['unanswerable']:
        return dict(correct=None, method='native_unanswerable_behavior_requires_review')
    if not abstention(report)['answered']:
        return dict(correct=False, method='no_valid_answer_on_answerable_question')
    scorer = LLMEvaluator.__new__(LLMEvaluator)
    prompt = scorer.get_anscheck_prompt(case['question_type'], case['question'], str(case['expected_answer']), report['answer'])
    prompt += '\n将上述 yes/no 判断输出为 JSON：correct 是布尔值，reason 简述依据。不要将格式差异当作错误。'
    schema = dict(type='json_schema', json_schema=dict(name='pilot_answer_grade', strict=True, schema=dict(
        type='object', properties=dict(correct=dict(type='boolean'), reason=dict(type='string')),
        required=['correct', 'reason'], additionalProperties=False)))
    metadata = {}
    try:
        raw = await AuditedLLM(manager, directory).generate(prompt, max_tokens=None, temperature=0.0,
                    response_format=schema, metadata_sink=metadata)
        result = json.loads(raw)
        if type(result.get('correct')) is not bool or metadata.get('finish_reason') == 'length':
            raise ValueError('评分协议无效或截断')
        return dict(result, method='repository_rubric_same_model_proxy', grader_model=config.llm.model)
    except Exception as error:
        return dict(correct=None, method='grading_failed', error=f'{type(error).__name__}:{error}')


def annotation_pairs(graph, case, limit=8):
    """人工关系金标模板：只取原生证据来源附近的候选边，绝不把建议当标准答案。"""
    observations = graph['fact_storage']['observations'];result=[]
    gold = set(case['gold_sessions'])
    priority = {kind: i for i,kind in enumerate(('correction','replacement','conflict','coexists','complement','duplicate'))}
    counts=Counter()
    for link in sorted(graph['event_state_layer']['links'], key=lambda x: (priority.get(x['relation'],99), x['link_id'])):
        old, new = (observations.get(link[key]) for key in ('from_fact_id', 'to_fact_id'))
        if not old or not new or not ({old.get('session_id'), new.get('session_id')} & gold):continue
        if counts[link['relation']]>=2:continue
        counts[link['relation']]+=1
        source_ids={str(old['chunk_id']),str(new['chunk_id'])}
        result.append(dict(case_id=case['case_id'], dataset=case['dataset'],graph_key=case['graph_key'],
            suggested_relation=link['relation'], link_id=link['link_id'],
            old_fact=old, new_fact=new,source_chunks={cid:graph['chunk_storage'][cid] for cid in source_ids},
            gold_relation=None, gold_update_decision=None,
            gold_supporting_chunk_ids=None, annotator_notes=''))
        if len(result)>=limit:break
    return result


async def execute(out):
    """执行冻结任务：已有完成题直接读取，部分 LongMemEval 图按会话继续。"""
    from init.config import Config
    from base.embeddings import EmbeddingManager
    from base.llm import LLMManager
    from coregraph.evidence_support import EvidenceSupport
    from coregraph.evidence_support.transport import AuditedLLM
    cases = json.loads((out/'cases.json').read_text(encoding='utf-8'))
    manifest = json.loads((out/'manifest.json').read_text(encoding='utf-8'))
    if digest(out/'cases.json') != manifest['cases_sha256'] or digest(out/'reproducibility/base_config.yaml') != manifest['base_config_sha256']:
        raise ValueError('冻结选题或配置已变化')
    for name,expected in manifest['source_code_sha256'].items():
        if digest(ROOT/name)!=expected:raise ValueError('冻结源码已变化：'+name)
    if digest(out/'graphs/locomo_sample0.pkl')!=manifest['locomo_graph_sha256']:
        raise ValueError('冻结 LoCoMo 图已变化')
    config = Config.parse(str(out/'reproducibility/base_config.yaml'))
    embeddings = EmbeddingManager(config.embedding)
    manager = LLMManager(api_key=config.llm.api_key, model=config.llm.model, base_url=config.llm.base_url,
        enable_concurrent=True, max_concurrent=config.llm.max_concurrent, timeout=config.llm.timeout, enable_thinking=False)
    rows, pairs, current_key, graph, support = [], [], None, None, None
    for index, case in enumerate(cases, 1):
        directory = out/'questions'/case['case_id'];directory.mkdir(parents=True, exist_ok=True)
        save(out/'progress.json', dict(phase='question', completed=index-1, total=len(cases), case_id=case['case_id']))
        try:
            if current_key != case['graph_key']:
                graph_path = out/case['graph_file']
                complete = graph_path.parent/'build_summary.json'
                if case['dataset']=='longmemeval' and not complete.exists():
                    graph = await build_graph(case, out, config, embeddings, manager)
                else:graph = pickle.loads(graph_path.read_bytes())
                support = EvidenceSupport(graph, embeddings);current_key=case['graph_key']
            result_path = directory/'result.json'
            if result_path.exists():row=json.loads(result_path.read_text(encoding='utf-8'))
            else:
                support.llm = AuditedLLM(manager, directory/'inference')
                started = time.monotonic()
                report = await support.answer_question(case['question'], support.llm, as_of=case['as_of'])
                elapsed=time.monotonic()-started
                save(directory/'report.json', report)
                grading = await grade(case, report, manager, directory/'grading', config)
                row = dict(case_id=case['case_id'], dataset=case['dataset'], question_type=case['question_type'],
                    question=case['question'], expected_answer=case['expected_answer'], unanswerable=case['unanswerable'],
                    answer=report.get('answer'), status=report['status'], evaluation_status=report['evaluation_status'],
                    abstention=abstention(report), source_coverage=source_coverage(report,case['gold_sessions'],graph),
                    locomo_turn_coverage=locomo_turn_coverage(case,report,graph),
                    exact_match=exact_match(report.get('answer'),case['expected_answer']) if not case['unanswerable'] else None,
                    grading=grading, inference_seconds=elapsed, confidence=report.get('confidence'),
                    gold_relation=None, human_answer_correct=None)
                save(result_path,row)
            rows.append(row);pairs.extend(annotation_pairs(graph,case))
        except asyncio.CancelledError:raise
        except Exception as error:
            row=dict(case_id=case['case_id'], dataset=case['dataset'], question_type=case['question_type'],
                unanswerable=case['unanswerable'], status='ERROR', error=f'{type(error).__name__}:{error}',
                abstention=dict(answered=False,technical_failure=True,semantic_abstention=False),
                grading=dict(correct=False,method='pipeline_error'),
                source_coverage=source_coverage({},case['gold_sessions'],graph or {}))
            save(directory/'error.json',row);rows.append(row);current_key=None
        save(out/'results.json',rows)
        save(out/'metrics.json',summarize(rows))
        # 标注模板最多 30 对，按链接去重；空 gold 不参与任何关系准确率计算。
        unique={(pair['graph_key'],pair['link_id']):pair for pair in pairs}
        groups={key:[pair for pair in unique.values() if pair['graph_key']==key]
                for key in sorted({pair['graph_key'] for pair in unique.values()})}
        selected=[]
        while any(groups.values()) and len(selected)<30:
            for group in groups.values():
                if group and len(selected)<30:selected.append(group.pop(0))
        annotation_path=out/'relation_annotations.json'
        if annotation_path.exists():
            prior=json.loads(annotation_path.read_text(encoding='utf-8'))
            previous={(row['graph_key'],row['link_id']):row for row in prior}
            for row in selected:
                old=previous.get((row['graph_key'],row['link_id']),{})
                for field in ('gold_relation','gold_update_decision','gold_supporting_chunk_ids','annotator_notes'):
                    if old.get(field) is not None:row[field]=old[field]
            keys={(row['graph_key'],row['link_id']) for row in selected}
            selected += [row for row in prior if (row['graph_key'],row['link_id']) not in keys and
                         any(row.get(field) for field in ('gold_relation','gold_update_decision','gold_supporting_chunk_ids','annotator_notes'))]
        save(out/'relation_annotations.json',selected)
        save(out/'progress.json',dict(phase='running',completed=index,total=len(cases),case_id=case['case_id']))
        print(f"PILOT {index}/{len(cases)} {case['case_id']} status={rows[-1]['status']}",flush=True)
    save(out/'completion.json',dict(finished=True,questions=len(rows),errors=sum(r['status']=='ERROR' for r in rows),
        finished_at=datetime.now(timezone.utc).isoformat(),relation_accuracy=None,confidence_calibration=None))
    print('CROSS_DATASET_PILOT_COMPLETE',flush=True)
    if any(row['status']=='ERROR' for row in rows):raise RuntimeError('任务完成但包含失败样本，请检查 results.json')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--locomo',type=Path,default=ROOT/'dataset/data/locomo10.json')
    parser.add_argument('--longmem',type=Path,default=ROOT/'dataset/data/longmemeval_s_cleaned.json')
    parser.add_argument('--locomo-graph',type=Path)
    parser.add_argument('--config',type=Path,default=ROOT/'config/Memory_EventState_sample0_rebuild.yaml')
    parser.add_argument('--seed',type=int,default=20261029)
    parser.add_argument('--implementation-commit',default='uncommitted_source_fingerprint_is_authoritative')
    parser.add_argument('--execute-frozen',action='store_true')
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args();os.environ.setdefault('HF_HUB_OFFLINE','1');os.environ.setdefault('TRANSFORMERS_OFFLINE','1')
    if args.execute_frozen:asyncio.run(execute(args.output.resolve()));return
    out=args.output.resolve() if args.resume else prepare(args)
    if not (out/'manifest.json').exists():raise ValueError('恢复目录缺少冻结 manifest')
    code=out/'reproducibility/source_snapshot'
    script="import sys,runpy,os;sys.path.insert(0,"+repr(str(code))+");os.chdir("+repr(str(code))+");sys.argv=['pilot','--output',"+repr(str(out))+",'--execute-frozen'];runpy.run_module('experiments.cross_dataset_pilot.run',run_name='__main__')"
    os.execv(sys.executable,[sys.executable,'-u','-c',script])


if __name__=='__main__':main()
