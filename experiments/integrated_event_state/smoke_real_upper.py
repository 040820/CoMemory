"""用指定真实 Chunk 检查上层写入；只生成诊断文件，不生成完整记忆图。"""

import argparse
import asyncio
import json
import os
from pathlib import Path
import pickle
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from base.llm import LLMManager
from base.embeddings import EmbeddingManager
from coregraph.event_state_layer import EventStateLayer
from coregraph.event_state_writer import EventStateWriter
from init.config import Config
from experiments.integrated_event_state.replay_upper_from_graph import (
    RecordingLLM, atomic_json, restore_relationships)


async def main(args):
    args.output.mkdir(parents=True, exist_ok=False)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    with args.source.open("rb") as handle:
        source = pickle.load(handle)
    requested = args.chunk_ids.split(",")
    storage = {str(key): value for key, value in source["chunk_storage"].items()}
    if any(cid not in storage for cid in requested):
        raise ValueError("指定 Chunk 不存在")
    chunks = [dict(storage[cid], chunk_id=cid) for cid in requested]
    refs = [ref for ref in restore_relationships(source) if ref["chunk_id"] in requested]
    atomic_json(args.output / "inputs.json", {"source": str(args.source.resolve()),
                "chunks": chunks, "relationships": refs, "scope": "selected_chunks_only"})
    config = Config.parse(args.config)
    manager = LLMManager(api_key=config.llm.api_key, model=config.llm.model,
                         base_url=config.llm.base_url, enable_thinking=False,
                         max_concurrent=config.llm.max_concurrent)
    llm = RecordingLLM(manager, args.output / "llm_trace.jsonl")
    writer = EventStateWriter(llm, EmbeddingManager(config.embedding))
    layer = await writer.write_chunks(EventStateLayer(), chunks, refs,
        {name: index for name, index in source["entity_name_to_index"].items()
         if name in source["graph"]})
    atomic_json(args.output / "layer.json", layer.to_dict())
    summary = {"scope": "selected_chunks_only", "chunk_ids": requested,
               "processed_chunks": len(layer.processed_chunks), "facts": len(layer.facts),
               "events": len(layer.event_instances), "changes": len(layer.events),
               "current_slots": len(layer.current), "reviews": layer.review_queue,
               "skipped_candidates": layer.skipped_candidates}
    atomic_json(args.output / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    if len(layer.processed_chunks) != len(chunks):
        raise RuntimeError("小样本有未处理 Chunk，请检查 trace 与日志")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("config/Memory_EventState.yaml"))
    parser.add_argument("--chunk-ids", default="0,1,2,3,4,5,6,7")
    asyncio.run(main(parser.parse_args()))
