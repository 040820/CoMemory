"""只读审计新上层的数量与证据指针，不调用模型或改动图。"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import pickle
import sys
from typing import Any

# 支持从项目根目录直接执行脚本，不要求先安装为 Python 包。
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from coregraph.event_state_layer import triple_ref_key
from coregraph.triple_fact_store import TripleFactStore, decode_upper_references


def inspect_graph(path: Path) -> dict[str, Any]:
    """报告事实分布，并逐条检查实体、Triple、Chunk 的反向链接。"""
    with path.open("rb") as stream:
        stored = pickle.load(stream)
    if not isinstance(stored, dict):
        raise ValueError("图文件不是含事件层的新版字典格式")
    graph = stored["graph"]
    chunks = {str(key): value for key, value in stored["chunk_storage"].items()}
    entity_index = stored["entity_name_to_index"]
    # 服务于新结构审计：先解析独立事实 ID，再检查上层到原文的证据链。
    store = TripleFactStore(stored['fact_storage']) if 'fact_storage' in stored else None
    layer = decode_upper_references(stored.get("event_state_layer"), store or TripleFactStore(),
                                    stored.get('ingestion_state'))
    if not isinstance(layer, dict) or layer.get("version") != 1:
        raise ValueError("图文件中没有新版事件状态层")
    if stored.get('event_state_layer', {}).get('schema') == 'event_index_v1':
        # 服务于审计报告：核查队列已迁出主图，计数/原因应读取对应审计而非误报为零。
        from coregraph.lean_event_index import read_audit_snapshot
        layer.update(read_audit_snapshot(str(path), stored['ingestion_state']))
    facts = layer["facts"]
    events = layer["events"]
    triple_storage = store.chunk_view() if store is not None else stored.get("triple_storage", {})
    if not isinstance(triple_storage, dict):
        triple_storage = {}
    issues: list[dict[str, str]] = []

    def issue(kind: str, record_id: str, detail: str) -> None:
        issues.append({"kind": kind, "record_id": record_id, "detail": detail})

    for fact_id, fact in facts.items():
        entity = fact["entity_id"]
        if entity not in graph or entity_index.get(entity) != fact["entity_index"]:
            issue("entity", fact_id, entity)
        chunk_id = str(fact["chunk_id"])
        if chunk_id not in chunks or fact["quote"] not in str(chunks[chunk_id].get("text") or ""):
            issue("chunk_quote", fact_id, chunk_id)
        for ref in fact["triple_refs"]:
            if ref["chunk_id"] != chunk_id or not graph.has_edge(ref["src"], ref["tgt"]):
                issue("triple", fact_id, str(ref))
            if store is not None and graph.has_edge(ref['src'], ref['tgt']):
                from coregraph.event_candidate_schema import triple_id
                fid = triple_id(ref)
                if fid not in store.facts or fid not in graph.edges[ref['src'], ref['tgt']].get('fact_ids', []):
                    issue('middle_fact_link', fact_id, fid)

    for event_id, event in events.items():
        if event["old_fact_id"] not in facts or event["new_fact_id"] not in facts:
            issue("fact_link", event_id, "旧事实或新事实不存在")
        if event_id not in layer["by_entity"].get(event["entity_id"], []):
            issue("entity_reverse", event_id, event["entity_id"])
        if event_id not in layer["by_chunk"].get(str(event["chunk_id"]), []):
            issue("chunk_reverse", event_id, str(event["chunk_id"]))
        for ref in event["triple_refs"]:
            if event_id not in layer["by_triple"].get(triple_ref_key(ref), []):
                issue("triple_reverse", event_id, str(ref))

    # 服务于精简结构来源检查：时间与属性也属于中层来源断言，必须能追溯原文。
    if stored.get('event_state_layer', {}).get('schema') == 'event_index_v1':
        for collection in ('event_time_assertions', 'event_attribute_assertions'):
            for assertion_id, assertion in layer.get(collection, {}).items():
                cid = str(assertion.get('evidence_chunk_id'))
                quote = assertion.get('evidence_quote')
                if (not quote or cid not in chunks or quote not in str(chunks[cid].get('text') or '')):
                    issue('assertion_quote', assertion_id, cid)
                if assertion.get('event_id') not in layer['event_instances']:
                    issue('assertion_event', assertion_id, str(assertion.get('event_id')))
                for ref in assertion.get('source_triples', []):
                    if str(ref.get('chunk_id')) != cid:
                        issue('assertion_triple_source', assertion_id, str(ref))

    event_examples = sorted(events.values(), key=lambda row: (row["observed_at"], row["event_id"]))[:20]
    return {
        "graph": str(path),
        "middle_memory_schema": graph.graph.get('memory_graph_schema', 'legacy'),
        "middle_facts": len(store.facts) if store is not None else None,
        "upper_schema": stored.get('event_state_layer', {}).get('schema', 'observations'),
        "middle_observations": len(store.observations) if store is not None else None,
        "event_groups": len(stored.get('event_state_layer', {}).get('groups', {})),
        "candidate_links": len(stored.get('event_state_layer', {}).get('links', [])),
        "audit_counts": stored.get('ingestion_state', {}).get('audit_counts', {}),
        "chunks": len(chunks), "graph_entities": len(graph.nodes),
        "graph_edges": len(graph.edges),
        "triple_storage_chunks": len(triple_storage),
        "triple_storage_entries": sum(
            len(entries) for entries in triple_storage.values()
            if isinstance(entries, (list, tuple))),
        "processed_chunks": len(layer["processed_chunks"]),
        "facts": len(facts), "current_states": len(layer["current"]),
        "events": len(events), "review_queue": len(layer["review_queue"]),
        "fact_kinds": dict(Counter(row["kind"] for row in facts.values())),
        "event_types": dict(Counter(row["update_type"] for row in events.values())),
        "event_statuses": dict(Counter(row["status"] for row in events.values())),
        "attributes": dict(Counter(row["attribute"] for row in facts.values()).most_common(30)),
        "review_reasons": dict(Counter(row["reason"] for row in layer["review_queue"]).most_common(30)),
        "issue_count": len(issues), "issues": issues[:100],
        "event_examples": event_examples,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="检查事件状态上层及其证据引用")
    parser.add_argument("--graph", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = inspect_graph(args.graph)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items()
                      if key not in {"issues", "event_examples"}}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
