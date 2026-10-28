"""分项置信度实验入口：初始化或有限评价图副本，绝不写回源图。

服务于精简三层记忆实验。默认只注册 null 分数，不调用模型；只有明确指定
--fact-ids / --link-ids 才调用本地 vLLM。该入口不做全量扫描、记忆替换或因果确认。
"""

from __future__ import annotations

import argparse
import asyncio
from collections import Counter
from hashlib import sha256
import json
import os
from pathlib import Path
import pickle
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from coregraph.memory_confidence import register_confidence, middle_rows, MemoryConfidenceEvaluator


def _ids(text):
    """服务于实验预算控制：显式 ID 列表去重，禁止隐式扩展为所有实体历史。"""
    return list(dict.fromkeys(value.strip() for value in text.split(',') if value.strip()))


async def run(args):
    """加载用户指定图，在全新输出目录生成副本、评价记录和来源清单。"""
    source = args.source.resolve()
    content = source.read_bytes()
    graph = pickle.loads(content)
    register_confidence(graph)
    fact_ids, link_ids = _ids(args.fact_ids), _ids(args.link_ids)
    if len(fact_ids) + len(link_ids) > args.max_items:
        raise ValueError('显式评价数量超过 --max-items，未发起模型调用')
    rows = middle_rows(graph)
    links = {link['link_id'] for link in graph['event_state_layer']['links']}
    if set(fact_ids) - set(rows) or set(link_ids) - links:
        raise ValueError('指定的 Fact 或 Link ID 不存在，未发起模型调用')
    # 先检查目标和审计，再创建目录；不允许与源图同路径，不覆盖历史实验。
    output = args.output.resolve()
    if output == source.parent or output.exists():
        raise ValueError('输出必须是全新独立目录')
    audit_relative = graph.get('ingestion_state', {}).get('audit_file')
    audit_content = None
    if audit_relative:
        from coregraph.lean_event_index import read_audit_snapshot
        read_audit_snapshot(str(source), graph['ingestion_state'])
        audit_content = (source.parent / audit_relative).read_bytes()
    output.mkdir(parents=True, exist_ok=False)
    if audit_relative:
        target = output / audit_relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(audit_content)
    manifest = dict(source=str(source), source_sha256=sha256(content).hexdigest(),
        fact_ids=fact_ids, link_ids=link_ids, max_items=args.max_items,
        mode='bounded_llm_assessment' if fact_ids or link_ids else 'register_only',
        calibration='uncalibrated', source_modified=False, truth_update=False,
        code_sha256=sha256((Path(__file__).resolve().parents[2] / 'coregraph/memory_confidence.py').read_bytes()).hexdigest())
    (output / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    if fact_ids or link_ids:
        # 复用项目 LLM 配置；仅有限指定记录调用模型，不加载 BGE-M3 或重新构图。
        from base.llm import LLMManager
        from init.config import Config
        config = Config.parse(args.config)
        if config.llm.api_key == 'demo-key':
            raise ValueError('置信度实验不接受 demo-key 模拟响应')
        llm = LLMManager(api_key=config.llm.api_key, model=config.llm.model,
                         base_url=config.llm.base_url, max_tokens=config.llm.max_token,
                         max_concurrent=1, timeout=config.llm.timeout, enable_thinking=False)
        evaluator = MemoryConfidenceEvaluator(llm)
        for fid in fact_ids:
            await evaluator.assess_fact(graph, fid)
        for lid in link_ids:
            await evaluator.assess_relation(graph, lid)
    # 模型评价只更新独立记录/抽取元数据；候选 Link 状态、实体图和原始内容不改变。
    with tempfile.NamedTemporaryFile(dir=output, delete=False) as stream:
        temp = stream.name
        pickle.dump(graph, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, output / source.name)
    storage = graph['confidence_storage']
    (output / 'confidence_records.json').write_text(json.dumps(storage, ensure_ascii=False, indent=2), encoding='utf-8')
    summary = dict(graph=str(output / source.name), calibration='uncalibrated',
        complete_chain_verified=False, truth_update=False,
        records={name: dict(Counter(row['status'] for row in storage[name].values()))
                 for name in ('fact_evaluations', 'evidence_links', 'relation_evaluations')})
    (output / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main():
    """服务于独立启用：默认不调用 LLM；无 --all 选项，防止意外启动长时间评价。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=Path('config/Memory_EventState_sample0_rebuild.yaml'))
    parser.add_argument('--fact-ids', default='')
    parser.add_argument('--link-ids', default='')
    parser.add_argument('--max-items', type=int, default=10)
    asyncio.run(run(parser.parse_args()))


if __name__ == '__main__':
    main()
