"""复用已有 LoCoMo 图与 Chunk，独立重建事件实例上层。

原始构建的完整 Triple 抽取列表没有单独保存。本脚本只从同一份图的
最终关系边和旧上层事实的来源快照恢复带 Chunk ID 的 Triple；不从其他
运行的图中拼接关系。原始 pickle 只读，新上层和运行日志写入独立目录。
"""

from __future__ import annotations

import argparse
import asyncio
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
    """合并同源的图边与旧事实来源，按四元键去重。

    图边可能被后续同实体关系覆盖；旧事实保留的 Triple 快照可补回其中
    一部分。仅接受能指向本图 Chunk 的关系，防止无证据的上层断言。
    """
    known_chunks = {str(key) for key in source["chunk_storage"]}
    refs: dict[tuple[str, str, str, str], dict[str, str]] = {}

    def add(src: Any, relation: Any, tgt: Any, chunk_id: Any) -> None:
        if chunk_id is None:
            return
        key = (str(src or "").strip(), str(relation or "").strip(),
               str(tgt or "").strip(), str(chunk_id))
        if all(key[:3]) and key[3] in known_chunks:
            refs[key] = dict(zip(("src", "relation", "tgt", "chunk_id"), key))

    for src, tgt, data in source["graph"].edges(data=True):
        ids = data.get("chunk_ids") or []
        if not isinstance(ids, (list, tuple, set)):
            ids = [ids]
        if data.get("chunk_id") is not None:
            ids = list(ids) + [data["chunk_id"]]
        for chunk_id in ids:
            add(src, data.get("relation_name") or data.get("relation"), tgt, chunk_id)

    for fact in source.get("event_state_layer", {}).get("facts", {}).values():
        for ref in fact.get("triple_refs") or []:
            if isinstance(ref, dict):
                add(ref.get("src"), ref.get("relation"), ref.get("tgt"),
                    ref.get("chunk_id", fact.get("chunk_id")))

    return [refs[key] for key in sorted(refs,
            key=lambda key: (int(key[3]) if key[3].isdecimal() else 10**12,
                             key[3], key[0], key[1], key[2]))]


class RecordingLLM:
    """逐次记录模型输入输出；每次调用完成即落盘，便于事后查错。"""

    def __init__(self, manager: LLMManager, path: Path):
        self.manager = manager
        self.path = path
        self.lock = asyncio.Lock()

    async def generate(self, prompt: str, **kwargs: Any) -> str:
        response = await self.manager.generate(prompt, temperature=0.0, **kwargs)
        stage = ("identity" if "EVENT_IDENTITY_GATE" in prompt else
                 "state_gate" if "STATE_SLOT_GATE" in prompt else "extract")
        row = {"at": datetime.now(timezone.utc).isoformat(), "stage": stage,
               "prompt": prompt, "response": response}
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
    # 旧实体字典可能留有图中已合并/删除的名称；只采用当前图节点。
    entity_index = {name: index for name, index in source["entity_name_to_index"].items()
                    if name in source["graph"]}
    covered = {ref["chunk_id"] for ref in relationships}
    stat = source_path.stat()
    identity = {"source": str(source_path), "source_size": stat.st_size,
                "source_mtime_ns": stat.st_mtime_ns}
    manifest = {**identity, "source_graph_nodes": source["graph"].number_of_nodes(),
                "source_graph_edges": source["graph"].number_of_edges(),
                "source_chunks": len(chunks), "restored_triples": len(relationships),
                "triple_covered_chunks": len(covered),
                "chunks_without_restored_triple": sorted(
                    {row["chunk_id"] for row in chunks} - covered,
                    key=lambda cid: int(cid) if cid.isdecimal() else 10**12),
                "config": str(args.config.resolve()), "candidate_top_k": args.candidate_top_k}
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if any(previous.get(key) != value for key, value in identity.items()):
            raise ValueError("已有检查点的来源图与当前输入不一致")
    else:
        atomic_json(manifest_path, manifest)

    checkpoint = output / "upper_layer_checkpoint.pkl"
    if checkpoint.exists():
        with checkpoint.open("rb") as handle:
            layer = EventStateLayer(pickle.load(handle))
    else:
        layer = EventStateLayer()

    config = Config.parse(args.config)
    llm = RecordingLLM(LLMManager(
        api_key=config.llm.api_key, model=config.llm.model,
        base_url=config.llm.base_url, max_tokens=config.llm.max_token,
        enable_concurrent=config.llm.enable_concurrent,
        max_concurrent=config.llm.max_concurrent, timeout=config.llm.timeout,
        enable_thinking=config.llm.enable_thinking), output / "llm_trace.jsonl")
    writer = EventStateWriter(llm, EmbeddingManager(config.embedding),
                              candidate_top_k=args.candidate_top_k)
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
    asyncio.run(run(parser.parse_args()))
