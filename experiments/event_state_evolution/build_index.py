"""在独立数据库中构建状态索引；现有 CogniGraph 文件始终只读。"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
import sys
from typing import Any

# 可从项目根目录直接运行脚本，不要求安装为 Python 包。
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.event_state_evolution.decision import ATTRIBUTES, classify, resolve_entity_id
from experiments.event_state_evolution.evidence import collect_candidates
from experiments.event_state_evolution.store import StateStore


async def index_memory(memory: Any, store: StateStore, llm: Any, namespace: str,
                       limit: int | None = None,
                       identity_map: dict[str, str] | None = None,
                       chunk_ids: set[str] | None = None) -> dict[str, int]:
    """顺序处理已有会话；每个候选分属性判断，支持同一事件修改多个属性。"""
    candidates = collect_candidates(memory, namespace)
    if chunk_ids is not None:
        candidates = [candidate for candidate in candidates if candidate.chunk_id in chunk_ids]
    # limit 对未处理候选生效，因此多次小批量运行可持续向后推进。
    candidates = [candidate for candidate in candidates
                  if any(not store.has_source(candidate.source_key, attr) for attr in ATTRIBUTES)]
    if limit is not None:
        if limit < 1:
            raise ValueError("limit must be positive")
        candidates = candidates[:limit]
    counts = {"candidates": len(candidates), "written": 0, "review": 0,
              "ignored": 0, "skipped": 0}
    for candidate in candidates:
        # 全图节点列表仅用于判定输出是否合法；送给模型的上下文只含原文中
        # 出现的实体，从而控制提示词长度与数据库查询数量。
        names = tuple(name for name in candidate.entity_names
                      if name.casefold() in candidate.text.casefold())
        stable_ids = {name: resolve_entity_id(name, candidate, identity_map) for name in names}
        with store.batch():
            stored_context = store.context(tuple(item for item in stable_ids.values() if item))
            context = {name: stored_context.get(stable_id, {}) for name, stable_id in stable_ids.items()}
            for attribute in sorted(ATTRIBUTES):
                if store.has_source(candidate.source_key, attribute):
                    counts["skipped"] += 1
                    continue
                decision = await classify(candidate, context, llm, attribute_hint=attribute,
                                          identity_map=identity_map)
                if decision.kind == "ignore":
                    counts["ignored"] += 1
                    store.mark_processed(candidate.source_key, attribute, "ignore")
                    continue
                if decision.kind == "review":
                    counts["review"] += 1
                    # 同一 Chunk 对不同属性的核查理由必须独立保存。
                    from dataclasses import replace
                    store.apply(replace(candidate, source_key=f"{candidate.source_key}:{attribute}"), decision)
                    store.mark_processed(candidate.source_key, attribute, "review")
                    continue
                if (decision.kind != "correction" and decision.entity_id and decision.attribute and
                        store.has_matching_fact(candidate.chunk_id, decision.entity_id,
                                                decision.attribute, decision.value)):
                    counts["skipped"] += 1
                    store.mark_processed(candidate.source_key, attribute, "duplicate")
                    continue
                store.apply(candidate, decision)
                counts["written"] += 1
                store.mark_processed(candidate.source_key, attribute, decision.kind)
                # 下一属性判定看到刚写入的状态，避免同一 Chunk 的相互矛盾判断。
                stored_context = store.context(tuple(item for item in stable_ids.values() if item))
                context = {name: stored_context.get(stable_id, {}) for name, stable_id in stable_ids.items()}
    return counts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="从已有 LiCoMemory 图构建独立状态索引")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--output-db", type=Path, required=True)
    parser.add_argument("--entity-map", type=Path, required=True,
                        help="人工核定的图节点名到稳定实体 ID 的 JSON 映射；未列出的主体进入待核查")
    parser.add_argument("--limit", type=int, default=None, help="只处理前 N 个候选，用于小样本核查")
    parser.add_argument("--chunk-id", action="append", default=None,
                        help="只处理指定 Chunk，可重复传入以验证真实模型")
    return parser.parse_args()


async def main(args: argparse.Namespace) -> None:
    from base.llm import LLMManager
    from coregraph.dynamic_memory import DynamicMemory
    from init.config import Config

    if not args.graph.is_file():
        raise FileNotFoundError(args.graph)
    if args.output_db.resolve() == args.graph.resolve():
        raise ValueError("output-db cannot overwrite the graph")
    config = Config.parse(args.config)
    identity_map = json.loads(args.entity_map.read_text(encoding="utf-8"))
    if (not isinstance(identity_map, dict) or
            any(not isinstance(key, str) or not isinstance(value, str) or not value.strip()
                for key, value in identity_map.items())):
        raise ValueError("entity-map must be a JSON object mapping names to nonempty stable IDs")
    memory = DynamicMemory(config, base_dir=str(args.graph.parent))
    memory.load_graph(str(args.graph))
    llm = LLMManager(api_key=config.llm.api_key, model=config.llm.model,
                     base_url=config.llm.base_url, max_tokens=config.llm.max_token,
                     enable_concurrent=False, max_concurrent=1, timeout=config.llm.timeout,
                     enable_thinking=config.llm.enable_thinking)
    store = StateStore(args.output_db)
    try:
        result = await index_memory(memory, store, llm, str(args.graph.resolve()), args.limit,
                                    identity_map=identity_map,
                                    chunk_ids=set(args.chunk_id) if args.chunk_id else None)
        result["review_total"] = store.review_count()
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    finally:
        store.close()


if __name__ == "__main__":
    asyncio.run(main(parse_args()))
