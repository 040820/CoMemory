"""从已有 CogniGraph 只读收集证据候选，不改变原图。"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from typing import Any


@dataclass(frozen=True)
class EvidenceCandidate:
    """待 LLM 核对的三元组与原文组合；图边本身不保证在每个 Chunk 中成立。"""

    source_key: str
    chunk_id: str
    session_id: str
    observed_at: str
    text: str
    subject: str | None
    predicate: str | None
    object: str | None
    entity_names: tuple[str, ...]


def _chunk_ids(edge: dict[str, Any]) -> set[str]:
    """兼容首次构图的单值与增量构图时合并的列表。"""
    found: set[str] = set()
    for field in ("chunk_id", "chunk_ids"):
        value = edge.get(field)
        if isinstance(value, (list, tuple, set)):
            found.update(str(item) for item in value if item is not None and str(item).strip())
        elif value is not None and str(value).strip():
            found.add(str(value))
    return found


def collect_candidates(memory: Any, namespace: str) -> list[EvidenceCandidate]:
    """只读构造候选，保留无图边的 Chunk 以识别文本纠正。

    同一图边若指向多个 Chunk，会产生多个候选；后续判定必须逐个核对
    三元组是否真的受该 Chunk 原文支持，不能直接继承合并边的断言。
    """
    graph = memory.graph_builder.graph
    storage = {str(key): raw for key, raw in memory.chunk_storage.items()
               if isinstance(raw, dict) and str(raw.get("text") or "").strip()}
    names = tuple(sorted(str(node) for node in graph.nodes))
    links: dict[str, list[tuple[str, str, str]]] = {key: [] for key in storage}
    for src, tgt, edge in graph.edges(data=True):
        relation = str(edge.get("relation_name") or "").strip()
        if not relation:
            continue
        for chunk_id in _chunk_ids(edge):
            if chunk_id in links:
                links[chunk_id].append((str(src), relation, str(tgt)))

    rows: list[EvidenceCandidate] = []
    for chunk_id, raw in storage.items():
        triples = sorted(set(links[chunk_id])) or [(None, None, None)]
        for subject, predicate, object_ in triples:
            identity = "\x1f".join((namespace, chunk_id, subject or "", predicate or "", object_ or ""))
            rows.append(EvidenceCandidate(
                source_key=sha256(identity.encode("utf-8")).hexdigest(),
                chunk_id=chunk_id,
                session_id=str(raw.get("session_id") or ""),
                observed_at=str(raw.get("session_time") or ""),
                text=str(raw["text"]),
                subject=subject,
                predicate=predicate,
                object=object_,
                entity_names=names,
            ))
    return sorted(rows, key=lambda row: (row.observed_at, row.session_id, row.chunk_id,
                                         row.subject or "", row.predicate or ""))
