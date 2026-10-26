"""用六段可核对的对话调用本地 vLLM，记录事件实例与纠正链。

此脚本只构造小型输入并输出 JSON，不读取或覆盖已有 LoCoMo 图。
固定关系快照用于隔离事件身份/断言逻辑与下层 Triple 抽取误差。
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from base.llm import LLMManager
from coregraph.event_state_layer import EventStateLayer
from coregraph.event_state_writer import EventStateWriter


def _chunk(number: int, day: str, text: str) -> dict:
    return {"chunk_id": str(number), "session_id": f"D{number}",
            "session_time": day, "text": "Caroline: " + text}


CHUNKS = [
    _chunk(1, "2023-05-08", "I attended the LGBTQ support group at the Community Center yesterday."),
    _chunk(2, "2023-05-09", "That LGBTQ support group I attended was helpful."),
    _chunk(3, "2023-05-15", "I attended a school event last Friday."),
    _chunk(4, "2023-05-20", "I actually went to that LGBTQ support group on May 6, not May 7."),
    _chunk(5, "2023-05-21", "It was at the Library, not the Community Center, for that LGBTQ support group."),
    _chunk(6, "2023-05-22", "I attended the LGBTQ support group again on May 14."),
]
RELATIONSHIPS = [
    {"src": "Caroline", "relation": "attended", "tgt": "LGBTQ support group", "chunk_id": "1"},
    {"src": "Caroline", "relation": "attended on", "tgt": "2023-05-07", "chunk_id": "1"},
    {"src": "Caroline", "relation": "attended", "tgt": "LGBTQ support group", "chunk_id": "2"},
    {"src": "Caroline", "relation": "attended", "tgt": "school event", "chunk_id": "3"},
    {"src": "Caroline", "relation": "attended", "tgt": "LGBTQ support group", "chunk_id": "4"},
    {"src": "Caroline", "relation": "attended on", "tgt": "2023-05-06", "chunk_id": "4"},
    {"src": "Caroline", "relation": "attended", "tgt": "LGBTQ support group", "chunk_id": "5"},
    {"src": "Caroline", "relation": "attended", "tgt": "LGBTQ support group", "chunk_id": "6"},
    {"src": "Caroline", "relation": "attended on", "tgt": "2023-05-14", "chunk_id": "6"},
]


class RecordingLLM:
    def __init__(self, manager: LLMManager):
        self.manager = manager
        self.calls: list[dict] = []

    async def generate(self, prompt: str, **kwargs) -> str:
        response = await self.manager.generate(prompt, temperature=0.0, **kwargs)
        self.calls.append({"stage": "identity" if "EVENT_IDENTITY_GATE" in prompt else
                           "state_gate" if "STATE_SLOT_GATE" in prompt else "extract",
                           "prompt": prompt, "response": response})
        return response


async def run(output: Path, base_url: str, model: str) -> None:
    output.mkdir(parents=True, exist_ok=False)
    manager = LLMManager(api_key="EMPTY", model=model, base_url=base_url,
                         enable_concurrent=False, enable_thinking=False)
    llm = RecordingLLM(manager)
    layer = await EventStateWriter(llm).write_chunks(
        EventStateLayer(), CHUNKS, RELATIONSHIPS,
        {"Caroline": 0, "LGBTQ support group": 1, "school event": 2})
    summary = {
        "processed_chunks": len(layer.processed_chunks),
        "event_instances": [
            {"event_id": event_id, "actor": event["actor"],
             "action": event["action"], "target": event["target"],
             "event_time": event["event_time"],
             "evidence_chunk_ids": event["evidence_chunk_ids"],
             "current_attributes": {
                 name: layer.event_attribute_assertions[assertion_id]["value"]
                 for name, assertion_id in event["current_attributes"].items()
             }}
            for event_id, event in layer.event_instances.items()
        ],
        "time_assertions": list(layer.event_time_assertions.values()),
        "attribute_assertions": list(layer.event_attribute_assertions.values()),
        "link_decisions": layer.event_link_decisions,
        "review_queue": layer.review_queue,
    }
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "llm_trace.json").write_text(json.dumps(llm.calls, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "layer.json").write_text(json.dumps(layer.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8011/v1")
    parser.add_argument("--model", default="qwen3-8b")
    args = parser.parse_args()
    asyncio.run(run(args.output, args.base_url, args.model))
