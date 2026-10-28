"""服务于 sample0 独立构图实验：运行冻结代码并保留模型、写入召回与来源检查记录。

用法：python launch_sample0_rebuild.py /绝对路径/新实验目录
目录应预先放置 reproducibility/source_snapshot、config.yaml 和 data/locomo/group_1。
本入口不复用旧图，不运行问答评测，不轮询后台任务；退出码由外层 runner.sh 保存。
"""

from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import pickle
import runpy
import sys


def main():
    """服务于可复现构图：注入只记录、不改模型参数的日志钩子，然后调用正式入口。"""
    root = Path(sys.argv[1]).resolve()
    code = root / 'reproducibility/source_snapshot'
    sys.path.insert(0, str(code))
    os.chdir(code)
    from base.llm import LLMManager
    from coregraph.dynamic_memory import DynamicMemory
    from coregraph.lean_event_index import read_audit_snapshot
    from experiments.integrated_event_state.inspect_index import inspect_graph

    def write_json(name, value):
        """服务于实验记录：小型进度和汇总文件采用原子替换。"""
        path = root / name
        temporary = path.with_suffix(path.suffix + '.tmp')
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding='utf-8')
        os.replace(temporary, path)

    generate = LLMManager.generate
    save = DynamicMemory.save_graph

    async def logged_generate(self, prompt, **kwargs):
        """服务于模型故障追溯：保存输入、输出、finish_reason 和异常，不干预推理。"""
        metadata = kwargs.pop('metadata_sink', None)
        metadata = metadata if isinstance(metadata, dict) else {}
        row = {'at': datetime.now(timezone.utc).isoformat(), 'prompt': prompt,
               'stage': ('normalization' if 'STATE_SEMANTIC_NORMALIZATION' in prompt else
                         'relation' if 'STATE_FACT_RELATION' in prompt else 'extraction')}
        try:
            result = await generate(self, prompt, metadata_sink=metadata, **kwargs)
            row['response'] = result
            return result
        except BaseException as error:
            row['error'] = f'{type(error).__name__}: {error}'
            raise
        finally:
            row.update(metadata)
            with (root / 'llm_trace.jsonl').open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(row, ensure_ascii=False, default=str) + '\n')

    def logged_save(self, path):
        """服务于检查点记录：仅在正式保存成功后更新进度，不额外调用模型。"""
        save(self, path)
        write_json('progress.json', dict(saved_at=datetime.now(timezone.utc).isoformat(),
            sessions=sorted({str(c.get('session_id')) for c in self.chunk_storage.values()}),
            chunks=len(self.chunk_storage), raw_triples=len(self.fact_store.facts),
            middle_observations=len(self.fact_store.observations),
            association_edges=self.graph_builder.graph.number_of_edges(),
            event_groups=len(self.event_state_layer.event_instances),
            processed_chunks=len(self.event_state_layer.processed_chunks),
            reviews=len(self.event_state_layer.review_queue)))

    LLMManager.generate = logged_generate
    DynamicMemory.save_graph = logged_save
    sys.argv = ['main.py', '-opt', str(root / 'reproducibility/config.yaml'),
                '-dataset_name', 'group_1', '-root', str(root)]
    runpy.run_path(str(code / 'main.py'), run_name='__main__')

    # 服务于完成判定：仅审计保存结构与来源完整性；不把它当成抽取准确率验证。
    graph_path = root / 'dynamic_memory_event_state.pkl'
    stored = pickle.loads(graph_path.read_bytes())
    corpus = [json.loads(line) for line in
              (root / 'reproducibility/data/locomo/group_1/Corpus.json').read_text(encoding='utf-8').splitlines()
              if line.strip()]
    expected = {str(row['session_id']) for row in corpus}
    actual = {str(row['session_id']) for row in stored['chunk_storage'].values()}
    checkpoint, table, upper = stored['ingestion_state'], stored['fact_storage'], stored['event_state_layer']
    if expected != actual or len(expected) != 19:
        raise RuntimeError('sample0 原始会话覆盖不完整')
    if set(map(str, stored['chunk_storage'])) != set(checkpoint['processed_chunks']):
        raise RuntimeError('存在尚未处理的 Chunk，请保留检查点分析')
    if upper.get('schema') != 'event_index_v1' or table['origin'] != 'extracted':
        raise RuntimeError('本次不是原始会话重新抽取的精简三层结构')
    inspection = inspect_graph(graph_path)
    write_json('inspection.json', inspection)
    if inspection['issue_count']:
        raise RuntimeError('存在来源/事实引用断链，请查看 inspection.json')
    audit = read_audit_snapshot(str(graph_path), checkpoint)
    for filename, rows in (('write_retrieval.jsonl', audit['decisions']),
                           ('pair_judgments.jsonl', audit['fact_relations'])):
        with (root / filename).open('w', encoding='utf-8') as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False, default=str) + '\n')
    covered = {str(f['chunk_id']) for f in table['facts'].values()}
    write_json('build_summary.json', dict(sessions=len(actual), chunks=len(stored['chunk_storage']),
        raw_triples=len(table['facts']), middle_observations=len(table.get('observations', {})),
        association_edges=stored['graph'].number_of_edges(), event_groups=len(upper['groups']),
        candidate_links=len(upper['links']), time_assertions=len(table.get('time_assertions', {})),
        attribute_assertions=len(table.get('attribute_assertions', {})),
        fact_types=dict(Counter(f.get('fact_type') for f in table.get('observations', {}).values())),
        audit_counts=checkpoint['audit_counts'],
        chunks_without_middle_triples=sorted(set(map(str, stored['chunk_storage'])) - covered),
        source_origin=table['origin'], reference_issues=inspection['issue_count'], qa_evaluation_run=False))
    print('REBUILD_AND_REFERENCE_CHECK_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
