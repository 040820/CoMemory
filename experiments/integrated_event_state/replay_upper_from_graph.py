"""复用已有 LoCoMo 图与 Chunk，独立重建事件实例上层。

原始构建的完整 Triple 抽取列表没有单独保存。本脚本只从同一份图的
最终关系边和旧上层事实的来源快照恢复带 Chunk ID 的 Triple；不从其他
运行的图中拼接关系。原始 pickle 只读，新上层和运行日志写入独立目录。
"""

from __future__ import annotations

import argparse
import asyncio
from hashlib import sha256
from collections import defaultdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import pickle
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from base.embeddings import EmbeddingManager
from base.llm import LLMManager
from coregraph.event_state_layer import EventStateLayer
from coregraph.event_state_writer import EventStateWriter
from coregraph.event_candidate_schema import triple_id
from coregraph.graph_builder import GraphBuilder
from coregraph.triple_fact_store import (TripleFactStore, encode_upper_references,
                                        decode_upper_references)
from init.config import Config


def atomic_json(path: Path, value: Any) -> None:
    """避免中断时留下半写入的进度文件。"""
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temp, path)


def atomic_pickle(path: Path, value: Any) -> None:
    """只替换本实验目录中的文件，不接触来源图。"""
    temp = path.with_name(path.name + ".tmp")
    with temp.open("wb") as handle:
        pickle.dump(value, handle, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(temp, path)


def restore_relationships(source: dict[str, Any]) -> list[dict[str, str]]:
    """恢复本图 Chunk 的关系出现记录，按四元键去重。

    优先级：
        1. ``triple_storage``：写入时保存的原始出现记录，最可靠；
        2. 最终图边：旧 pkl 没有 triple_storage 时的兜底；
        3. 旧上层 fact 的 Triple 快照：补回图边覆盖掉的一部分关系。

    仅接受能指向本图 Chunk 的关系，防止无证据的上层断言。
    """
    known_chunks = {str(key) for key in source["chunk_storage"]}
    refs: dict[tuple[str, str, str, str], dict[str, Any]] = {}

    def add(src: Any, relation: Any, tgt: Any, chunk_id: Any,
            extra: dict[str, Any] | None = None) -> None:
        if chunk_id is None:
            return
        key = (str(src or "").strip(), str(relation or "").strip(),
               str(tgt or "").strip(), str(chunk_id))
        if not (all(key[:3]) and key[3] in known_chunks):
            return
        entry = dict(zip(("src", "relation", "tgt", "chunk_id"), key))
        for name in ("session_id", "session_time", "event_time", "create_time", "strength", "weight", "description"):
            value = (extra or {}).get(name)
            if value is not None and str(value).strip():
                entry[name] = value
        # setdefault 保留先加入的来源：triple_storage > 图边 > 旧 fact 快照。
        refs.setdefault(key, entry)

    # 1) 原始出现记录（新格式 pkl）
    # 新中层的主表是唯一事实来源，association 边不含谓词，不可据此恢复事实。
    store = TripleFactStore(source['fact_storage']) if 'fact_storage' in source else None
    if (source['graph'].graph.get('memory_graph_schema') == 'entity_fact_index_v1'
            and store is None):
        raise ValueError('关联索引图缺少事实表，不能用普通边进行回放')
    stored = store.chunk_view() if store is not None else source.get("triple_storage")
    if isinstance(stored, dict):
        for chunk_id, entries in stored.items():
            if not isinstance(entries, (list, tuple)):
                continue
            for ref in entries:
                if isinstance(ref, dict):
                    add(ref.get("src"), ref.get("relation") or ref.get("predicate"),
                        ref.get("tgt"), ref.get("chunk_id", chunk_id), ref)

    if source.get("triple_storage_origin") != "extracted":
        # 2) 旧格式最终图边，可能已被覆盖，只有原始抽取记录缺失时才兜底。
        for src, tgt, data in source["graph"].edges(data=True):
            ids = data.get("chunk_ids") or []
            if not isinstance(ids, (list, tuple, set)):
                ids = [ids]
            if data.get("chunk_id") is not None:
                ids = list(ids) + [data["chunk_id"]]
            for chunk_id in ids:
                add(src, data.get("relation_name") or data.get("relation"), tgt, chunk_id, data)

        # 3) 旧上层事实快照只能补回一部分丢失的关系。
        upper = decode_upper_references(source.get('event_state_layer'), store or TripleFactStore()) or {}
        for fact in upper.get("facts", {}).values():
            for ref in fact.get("triple_refs") or []:
                if isinstance(ref, dict):
                    add(ref.get("src"), ref.get("relation"), ref.get("tgt"),
                        ref.get("chunk_id", fact.get("chunk_id")),
                        {"session_id": fact.get("session_id"),
                         "session_time": fact.get("observed_at")})

    return [refs[key] for key in sorted(refs,
            key=lambda key: (int(key[3]) if key[3].isdecimal() else 10**12,
                             key[3], key[0], key[1], key[2]))]


def group_relationships(relationships: list[dict[str, str]],
                        chunks: dict[str, dict[str, Any]]) -> dict[str, list[dict[str, str]]]:
    """把扁平关系列表重新按 Chunk 分组，写回结果 pkl 的 triple_storage。

    这样即使来源 pkl 没有原始记录，重放后的新 pkl 也会自带一份可继续
    使用的 Triple 出现记录。
    """
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for ref in relationships:
        chunk_id = str(ref["chunk_id"])
        chunk = chunks.get(chunk_id, {})
        item = {
            "src": str(ref["src"]),
            "relation": str(ref["relation"]),
            "tgt": str(ref["tgt"]),
            "chunk_id": chunk_id,
        }
        item["triple_id"] = triple_id(item)
        session_id = chunk.get("session_id") or ref.get("session_id")
        if session_id:
            item["session_id"] = str(session_id)
        session_time = chunk.get("session_time") or ref.get("session_time")
        if session_time:
            item["session_time"] = str(session_time)
        for name in ("event_time", "create_time", "strength", "weight", "description"):
            if ref.get(name) is not None:
                item[name] = ref[name]
        grouped[chunk_id].append(item)
    for entries in grouped.values():
        entries.sort(key=lambda row: (row["src"], row["relation"], row["tgt"]))
    return dict(grouped)


class RecordingLLM:
    """逐次记录模型输入输出；每次调用完成即落盘，便于事后查错。"""

    def __init__(self, manager: LLMManager, path: Path):
        self.manager = manager
        self.path = path
        self.lock = asyncio.Lock()

    async def generate(self, prompt: str, **kwargs: Any) -> str:
        metadata: dict[str, Any] = {}
        response = await self.manager.generate(prompt, temperature=0.0,
                                               metadata_sink=metadata, **kwargs)
        # 服务于实验审计：关系分类与替换门控分开计数，避免混入抽取调用。
        stage = ("relation" if "STATE_FACT_RELATION" in prompt else
                 "normalize" if "STATE_SEMANTIC_NORMALIZATION" in prompt else
                 "repair" if "EXTRACTION_REPAIR" in prompt else
                 "identity" if "EVENT_IDENTITY_GATE" in prompt else
                 "state_gate" if "STATE_SLOT_GATE" in prompt else "extract")
        row = {"at": datetime.now(timezone.utc).isoformat(), "stage": stage,
               "prompt": prompt, "response": response, **metadata}
        async with self.lock:
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        return response


async def run(args: argparse.Namespace) -> None:
    source_path = args.source.resolve()
    output = args.output.resolve()
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    if (output / "dynamic_memory_event_state.pkl").exists():
        raise FileExistsError(f"实验结果已存在，拒绝覆盖: {output}")
    output.mkdir(parents=True, exist_ok=True)

    # 载入原图只读；恢复的关系快照与当前图来自同一个 pickle。
    with source_path.open("rb") as handle:
        source = pickle.load(handle)
    relationships = restore_relationships(source)
    chunks = [dict(value, chunk_id=str(key))
              for key, value in source["chunk_storage"].items()]
    chunks.sort(key=lambda row: (str(row.get("session_time") or ""),
                 str(row.get("session_id") or ""),
                 int(row["chunk_id"]) if row["chunk_id"].isdecimal() else 10**12,
                 row["chunk_id"]))
    chunk_map = {row["chunk_id"]: row for row in chunks}
    triple_storage = group_relationships(relationships, chunk_map)
    # 旧实体字典可能留有图中已合并/删除的名称；只采用当前图节点。
    entity_index = {name: index for name, index in source["entity_name_to_index"].items()
                    if name in source["graph"]}
    covered = {ref["chunk_id"] for ref in relationships}
    stat = source_path.stat()
    identity = {"source": str(source_path), "source_size": stat.st_size,
                "source_mtime_ns": stat.st_mtime_ns,
                "config_sha256": sha256(args.config.read_bytes()).hexdigest(),
                "candidate_top_k": args.candidate_top_k}
    code_root = Path(__file__).resolve().parents[2]
    signature_files = ["coregraph/event_state_writer.py", "coregraph/event_state_layer.py",
                       "coregraph/event_instance_time.py", "coregraph/event_candidate_schema.py",
                       "coregraph/state_slot.py", "coregraph/state_candidate_retriever.py"]
    signature_files.append('coregraph/state_semantic_normalizer.py')
    # 中层结构及上层引用编码属于构图流程的一部分，变化后禁止沿用旧检查点。
    signature_files.extend(['coregraph/graph_builder.py', 'coregraph/triple_fact_store.py',
                            'coregraph/lean_event_index.py'])
    # 双路召回和关系判断的源码也进入检查点指纹，禁止不同写入流程混用同一实验。
    signature_files.extend(['coregraph/hybrid_state_retriever.py',
                            'coregraph/state_fact_relation.py'])
    # 追加模式查询只改变断言展示，不参与构图；写入策略由配置指纹和 v4 协议隔离。
    identity["pipeline_sha256"] = sha256(b"".join(
        (code_root / name).read_bytes() for name in signature_files)).hexdigest()
    manifest = {**identity, "source_graph_nodes": source["graph"].number_of_nodes(),
                "source_graph_edges": source["graph"].number_of_edges(),
                "source_chunks": len(chunks), "restored_triples": len(relationships),
                "triple_storage_chunks": len(triple_storage),
                "triple_coverage_source": source.get("triple_storage_origin", "graph_and_facts"),
                "triple_covered_chunks": len(covered),
                "chunks_without_restored_triple": sorted(
                    {row["chunk_id"] for row in chunks} - covered,
                    key=lambda cid: int(cid) if cid.isdecimal() else 10**12),
                "config": str(args.config.resolve()), "candidate_top_k": args.candidate_top_k}
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if any(previous.get(key) != value for key, value in identity.items()):
            raise ValueError("已有检查点的来源图、配置或写入代码不一致，请使用新实验目录")
    else:
        atomic_json(manifest_path, manifest)

    checkpoint = output / "upper_layer_checkpoint.pkl"
    if checkpoint.exists():
        with checkpoint.open("rb") as handle:
            layer = EventStateLayer(pickle.load(handle))
    else:
        layer = EventStateLayer()

    config = Config.parse(args.config)
    # 此实验复用服务器上已缓存的 BGE-M3；离线加载能避免每次访问模型站点。
    if not args.allow_embedding_download:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    llm = RecordingLLM(LLMManager(
        api_key=config.llm.api_key, model=config.llm.model,
        base_url=config.llm.base_url, max_tokens=config.llm.max_token,
        enable_concurrent=config.llm.enable_concurrent,
        max_concurrent=config.llm.max_concurrent, timeout=config.llm.timeout,
        enable_thinking=config.llm.enable_thinking), output / "llm_trace.jsonl")
    writer = EventStateWriter(llm, EmbeddingManager(config.embedding),
                              candidate_top_k=args.candidate_top_k,
                              enable_state_normalization=config.state_semantic_normalization,
                              # 回放入口与正式写入使用同一双路实现；CLI 总预算优先。
                              state_recall_mode=config.state_recall_mode,
                              structured_k=config.state_structured_top_k,
                              semantic_k=config.state_semantic_top_k,
                              # 服务于回放策略隔离：v4 只追加，不能续写非空旧协议上层。
                              state_write_policy=config.state_write_policy)
    sessions: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for chunk in chunks:
        sessions[(str(chunk.get("session_time") or ""),
                  str(chunk.get("session_id") or ""))].append(chunk)

    print(f"START chunks={len(chunks)} triples={len(relationships)} "
          f"covered={len(covered)} sessions={len(sessions)}", flush=True)
    for session_number, (key, group) in enumerate(sessions.items(), 1):
        # 只有完整处理过的 Chunk 才会跳过；失败或空响应的 Chunk 留给续跑。
        pending = [row for row in group if row["chunk_id"] not in layer.processed_chunks]
        if pending:
            layer = await writer.write_chunks(layer, pending, relationships, entity_index)
            atomic_pickle(checkpoint, layer.to_dict())
        progress = {"session": key[1], "session_number": session_number,
                    "total_sessions": len(sessions), "processed_chunks": len(layer.processed_chunks),
                    "total_chunks": len(chunks), "facts": len(layer.facts),
                    "event_instances": len(layer.event_instances),
                    "time_assertions": len(layer.event_time_assertions),
                    "attribute_assertions": len(layer.event_attribute_assertions),
                    "reviews": len(layer.review_queue),
                    "updated_at": datetime.now(timezone.utc).isoformat()}
        atomic_json(output / "progress.json", progress)
        print(f"SESSION {session_number}/{len(sessions)} {key[1]} "
              f"processed={len(layer.processed_chunks)}/{len(chunks)} "
              f"instances={len(layer.event_instances)} reviews={len(layer.review_queue)}",
              flush=True)

    # 未处理 Chunk 不能被当作成功完成；留下检查点供续跑。
    missing = sorted({row["chunk_id"] for row in chunks} - layer.processed_chunks)
    if missing:
        atomic_json(output / "incomplete.json", {"unprocessed_chunk_ids": missing})
        raise RuntimeError(f"仍有 {len(missing)} 个 Chunk 未处理，检查点可用于续跑")

    final = dict(source)
    final["event_state_layer"] = layer.to_dict()
    # 结果 pkl 自带一份按 Chunk 分组的 Triple 出现记录，供后续重放/增量使用。
    final["triple_storage"] = triple_storage
    final["triple_storage_origin"] = source.get("triple_storage_origin", "recovered_partial")
    if config.middle_memory_mode == 'entity_fact_index':
        # 新回放结果落盘时同时生成真实的新中层，不能只替换上层后仍保存旧 DiGraph。
        builder = GraphBuilder(config)
        builder.restore_graph(source['graph'], triple_storage, final['triple_storage_origin'])
        final['graph'] = builder.graph
        final['fact_storage'] = builder.fact_store.to_dict()
        final['event_state_layer'] = encode_upper_references(layer.to_dict(), builder.fact_store)
        if getattr(config, 'upper_memory_layout', 'observations') == 'event_index':
            # 服务于新格式回放：与正式 DynamicMemory 保存入口使用同一打包协议。
            from coregraph.lean_event_index import pack_event_index, write_audit_snapshot
            index, checkpoint, audit = pack_event_index(layer.to_dict(), builder.fact_store)
            checkpoint['audit_file'] = write_audit_snapshot(str(output / 'dynamic_memory_event_state.pkl'), audit)
            final['event_state_layer'] = index
            final['fact_storage'] = builder.fact_store.to_dict()
            final['ingestion_state'] = checkpoint
        final.pop('triple_storage', None)
    atomic_pickle(output / "dynamic_memory_event_state.pkl", final)
    atomic_json(output / "summary.json", {
        **manifest, "processed_chunks": len(layer.processed_chunks),
        "facts": len(layer.facts), "event_instances": len(layer.event_instances),
        "time_assertions": len(layer.event_time_assertions),
        "attribute_assertions": len(layer.event_attribute_assertions),
        "link_decisions": len(layer.event_link_decisions),
        "reviews": len(layer.review_queue),
        "completed_at": datetime.now(timezone.utc).isoformat()})
    print("COMPLETE", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("config/Memory_EventState.yaml"))
    parser.add_argument("--candidate-top-k", type=int, default=5)
    parser.add_argument("--allow-embedding-download", action="store_true",
                        help="本机尚无 BGE-M3 缓存时允许在线下载")
    asyncio.run(run(parser.parse_args()))
