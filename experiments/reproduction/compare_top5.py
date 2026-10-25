"""对照原方法与结构化语义方法在 Top-5、Top-15 下的回答和来源覆盖。"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT))

from evaluation.evaluator import Evaluator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="比较 LoCoMo group_1 的四组运行")
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def gold_sessions(origin: Any) -> list[str]:
    """与项目 SessionMatchingEvaluator 相同地解析来源会话标签。"""
    if isinstance(origin, list):
        return [str(item).strip() for item in origin if item]
    if isinstance(origin, str):
        return [item.strip() for item in origin.replace(";", ",").split(",") if item.strip()]
    return []


def load_run(root: Path, name: str, nested: bool = False) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """读取一轮运行的逐题答案与同口径 LLM 评测指标。"""
    folder = root / name / "results" if nested else root / name
    return (json.loads((folder / "results.json").read_text(encoding="utf-8")),
            json.loads((folder / "metrics.json").read_text(encoding="utf-8")))


def make_chunk_map(graph_path: Path) -> dict[str, list[tuple[str, str]]]:
    """按原检索器的提示词格式，将返回的文本反查到原始 Chunk ID 与会话。"""
    # 仅加载由本项目在当前服务器生成、已知来源的 pickle 文件。
    with graph_path.open("rb") as source:
        storage = pickle.load(source)["chunk_storage"]
    mapping: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for chunk_id, raw in storage.items():
        if not isinstance(raw, dict) or not raw.get("text"):
            continue
        text = str(raw["text"])
        session_id = str(raw.get("session_id") or "")
        session_time = str(raw.get("session_time") or "")
        mapping[f"{session_time} {text}"].append((str(chunk_id), session_id))
    return mapping


def main() -> None:
    args = parse_args()
    root = args.results_root
    runs = {
        "original_15": load_run(root, "locomo_group_1", nested=True),
        "original_5": load_run(root, "original_top5_locomo_group_1"),
        "semantic_15": load_run(root, "structural_semantic_locomo_group_1"),
        "semantic_5": load_run(root, "structural_semantic_top5_locomo_group_1"),
    }
    old_rows = runs["original_5"][0]
    if not all(len(rows) == len(old_rows) for rows, _ in runs.values()):
        raise ValueError("四组结果的题数不一致")
    for index in range(len(old_rows)):
        if len({rows[index]["question"] for rows, _ in runs.values()}) != 1:
            raise ValueError(f"第 {index + 1} 题的顺序不一致")

    chunk_map = make_chunk_map(root / "locomo_group_1/dynamic_memory_graph.pkl")
    scorer = Evaluator("", "group_1")
    exact = Counter()
    chunk_counts = Counter()
    mapping_failures: list[int] = []
    source_denominator = 0
    old_source_hits = 0
    new_source_hits = 0
    old_only = 0
    new_only = 0
    per_question: list[dict[str, Any]] = []

    for index, original5 in enumerate(old_rows):
        gold = gold_sessions(original5.get("origin"))
        mapped: list[tuple[str, str]] = []
        ambiguous = False
        for text in original5.get("retrieved_chunks", []):
            candidates = chunk_map.get(text, [])
            if len(candidates) != 1:
                ambiguous = True
            else:
                mapped.append(candidates[0])
        if ambiguous:
            mapping_failures.append(index + 1)
        chunk_counts[len(original5.get("retrieved_chunks", []))] += 1

        new5 = runs["semantic_5"][0][index]
        old_hit = bool(gold and any(session in gold for _, session in mapped))
        new_hit = bool(gold and any(str(hit.get("session_id")) in gold
                                    for hit in new5.get("evidence", [])))
        if gold and not ambiguous:
            source_denominator += 1
            old_source_hits += int(old_hit)
            new_source_hits += int(new_hit)
            old_only += int(old_hit and not new_hit)
            new_only += int(new_hit and not old_hit)

        flags = {}
        for name, (rows, _) in runs.items():
            row = rows[index]
            flags[name] = scorer._check_answer_match(str(row.get("answer", "")),
                                                     str(row.get("output", "")))
            exact[name] += int(flags[name])
        per_question.append({
            "number": index + 1,
            "question": original5["question"],
            "gold_sessions": gold,
            "original_5_chunk_ids": [item[0] for item in mapped],
            "original_5_session_ids": [item[1] for item in mapped],
            "semantic_5_chunk_ids": new5.get("retrieved_chunk_ids", []),
            "original_5_gold_session_hit": old_hit if not ambiguous and gold else None,
            "semantic_5_gold_session_hit": new_hit if not ambiguous and gold else None,
            "exact_answer_match": flags,
        })

    summary = {
        "question_count": len(old_rows),
        "llm_evaluation": {name: {"correct": metrics["correct_answers"],
                                  "total": metrics["total_questions"],
                                  "accuracy": metrics["accuracy"]}
                           for name, (_, metrics) in runs.items()},
        "deterministic_exact_match": dict(exact),
        "original_5_returned_chunk_counts": dict(sorted(chunk_counts.items())),
        "original_5_unmapped_or_ambiguous_questions": mapping_failures,
        "gold_session_comparison": {
            "fully_mapped_labeled_questions": source_denominator,
            "original_5_hit": old_source_hits,
            "semantic_5_hit": new_source_hits,
            "original_only": old_only,
            "semantic_only": new_only,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.with_suffix(".json").write_text(
        json.dumps({"summary": summary, "questions": per_question}, ensure_ascii=False, indent=2),
        encoding="utf-8")

    lines = ["# 原方法与新方法的 Top-5 对照", "",
             "四组运行使用同一份 LoCoMo `group_1` 图和 199 道问题。",
             "`原方法 Top-5` 直接调用项目的 `GraphRAG.query()`，只覆盖 `retriever.top_chunks=5`。", "",
             "| 方法 | LLM 评测答对 | 确定性答案匹配 |", "| --- | ---: | ---: |"]
    for name in ["original_15", "original_5", "semantic_15", "semantic_5"]:
        result = summary["llm_evaluation"][name]
        lines.append(f"| {name} | {result['correct']}/{result['total']} "
                     f"({result['accuracy']:.2%}) | {exact[name]}/{len(old_rows)} |")
    lines.extend(["", f"原方法 Top-5 实际返回 Chunk 数量分布：{dict(sorted(chunk_counts.items()))}。",
                  f"原方法文本无法唯一映射回 Chunk 的题号：{mapping_failures or '无'}。",
                  f"在 {source_denominator} 道来源标签可用且文本可映射的题中，"
                  f"原方法 Top-5 命中来源会话 {old_source_hits} 题，"
                  f"新方法 Top-5 命中 {new_source_hits} 题。",
                  f"只被原方法命中 {old_only} 题；只被新方法命中 {new_only} 题。", "",
                  f"原方法从 Top-15 到 Top-5 的答对数变化为 "
                  f"{summary['llm_evaluation']['original_5']['correct'] - summary['llm_evaluation']['original_15']['correct']:+d} 题；"
                  f"新方法的变化为 "
                  f"{summary['llm_evaluation']['semantic_5']['correct'] - summary['llm_evaluation']['semantic_15']['correct']:+d} 题。"
                  f"同为 Top-5 时，新方法比原方法 "
                  f"{summary['llm_evaluation']['semantic_5']['correct'] - summary['llm_evaluation']['original_5']['correct']:+d} 题。", "",
                  "这说明单纯减少原方法返回的 Chunk 并没有带来同样的提升；新方法的候选证据覆盖也更高。"
                  "但这是单次运行，回答提示词和 Session 摘要使用方式亦有差异，不能将全部收益归因于检索排序。", "",
                  "逐题 Chunk ID、来源会话和确定性匹配标记见同名 JSON 文件。", ""])
    args.output.write_text("\n".join(lines), encoding="utf-8")
    # 原方法的 Chunk 原文保存在 results.json；另导出便于逐题阅读的版本。
    detail_path = args.output.with_name("original_top5_review.md")
    detail = ["# 原始 LiCoMemory Top-5 检索证据逐题记录", "",
              "按原方法返回顺序列出 Chunk 原文。Chunk ID 与会话由同一图中的原文精确映射得到。", ""]
    for index, row in enumerate(old_rows):
        comparison = per_question[index]
        detail.extend([f"## 第 {index + 1} 题：{row['question']}", "",
                       f"- 标准答案：{row.get('answer', '')}",
                       f"- 原方法 Top-5 答案：{row.get('output', '')}",
                       f"- 新方法 Top-5 答案：{runs['semantic_5'][0][index].get('output', '')}",
                       f"- 来源会话：{', '.join(comparison['gold_sessions']) or '未提供'}",
                       f"- 原方法 Chunk ID：{', '.join(comparison['original_5_chunk_ids'])}",
                       f"- 新方法 Chunk ID：{', '.join(map(str, comparison['semantic_5_chunk_ids']))}", ""])
        for rank, text in enumerate(row.get("retrieved_chunks", []), start=1):
            candidates = chunk_map.get(text, [])
            label = (f"Chunk {candidates[0][0]}；会话 {candidates[0][1]}"
                     if len(candidates) == 1 else "Chunk ID 未能唯一映射")
            detail.extend([f"### 证据 {rank}：{label}", "",
                           "\n".join("> " + line for line in str(text).splitlines()), ""])
    detail_path.write_text("\n".join(detail), encoding="utf-8")
    print(f"Wrote {args.output}; original_5={runs['original_5'][1]['correct_answers']}/"
          f"{len(old_rows)}; semantic_5={runs['semantic_5'][1]['correct_answers']}/{len(old_rows)}")


if __name__ == "__main__":
    main()
