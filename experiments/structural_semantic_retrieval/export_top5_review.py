"""把 Top-5 实验的逐题证据展开为可直接阅读的 Markdown 报告。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="导出 Top-5 检索内容及与 Top-15 的对照")
    parser.add_argument("--top5", type=Path, required=True, help="Top-5 结果 JSON")
    parser.add_argument("--top15", type=Path, required=True, help="同一批问题的 Top-15 结果 JSON")
    parser.add_argument("--output", type=Path, required=True, help="输出 Markdown 文件")
    return parser.parse_args()


def gold_sessions(origin: Any) -> list[str]:
    """沿用原评测器的来源会话字段语义，缺失标签时返回空列表。"""
    if isinstance(origin, list):
        return [str(item).strip() for item in origin if str(item).strip()]
    if isinstance(origin, str):
        return [item.strip() for item in origin.replace(";", ",").split(",") if item.strip()]
    return []


def quote(text: str) -> str:
    """把原文写为 Markdown 引用，保留换行且避免数据中的 Markdown 标记破坏标题。"""
    return "\n".join("> " + line for line in text.splitlines())


def main() -> None:
    args = parse_args()
    top5 = json.loads(args.top5.read_text(encoding="utf-8"))
    top15 = json.loads(args.top15.read_text(encoding="utf-8"))
    if len(top5) != len(top15):
        raise ValueError("Top-5 与 Top-15 的题目数量不一致")

    lines = ["# Top-5 检索证据逐题记录", "",
             f"问题数：{len(top5)}。每条证据列出 Chunk ID、会话、得分、三元组和完整原文。",
             "`来源会话名次` 用于检查检索覆盖，答案正确性仍以原项目评测文件为准。", ""]
    first_five_same = 0
    gold_available = 0
    top5_gold_hit = 0
    top15_gold_hit = 0
    missing_gold_examples: list[int] = []
    missed: list[tuple[int, dict[str, Any], int | None]] = []

    for index, (row5, row15) in enumerate(zip(top5, top15), start=1):
        if row5.get("question") != row15.get("question"):
            raise ValueError(f"第 {index} 题的问题文本不一致")
        hits5 = row5.get("evidence", [])
        hits15 = row15.get("evidence", [])
        if len(hits5) > 5 or len(hits15) > 15:
            raise ValueError(f"第 {index} 题的返回数超过实验设定")
        ids5 = [str(hit["chunk_id"]) for hit in hits5]
        ids15 = [str(hit["chunk_id"]) for hit in hits15]
        if ids5 == ids15[:len(ids5)]:
            first_five_same += 1

        gold = gold_sessions(row5.get("origin"))
        gold_rank = next((rank for rank, hit in enumerate(hits15, start=1)
                          if str(hit.get("session_id")) in gold), None)
        if gold:
            gold_available += 1
            top5_gold_hit += int(gold_rank is not None and gold_rank <= 5)
            top15_gold_hit += int(gold_rank is not None)
            if gold_rank is None or gold_rank > 5:
                missed.append((index, row5, gold_rank))
        else:
            missing_gold_examples.append(index)

        lines.extend([f"## 第 {index} 题：{row5['question']}", "",
                      f"- 标准答案：{row5.get('answer', '')}",
                      f"- Top-5 生成答案：{row5.get('output', '')}",
                      f"- 来源会话：{', '.join(gold) if gold else '未提供'}",
                      f"- 来源会话在 Top-15 的首次名次：{gold_rank if gold_rank is not None else '未命中或无标签'}",
                      f"- Top-5 Chunk ID：{', '.join(ids5)}",
                      f"- Top-15 第 6–15 位 Chunk ID：{', '.join(ids15[5:])}", ""])

        for rank, hit in enumerate(hits5, start=1):
            lines.extend([f"### 证据 {rank}：Chunk {hit['chunk_id']}", "",
                          f"得分：{hit['score']:.4f}；会话：{hit.get('session_id') or '无'}；"
                          f"时间：{hit.get('session_time') or '无'}", ""])
            triples = hit.get("triples") or []
            if triples:
                lines.append("三元组：")
                lines.append("")
                for fact in triples:
                    lines.append(f"- {fact['subject']} — {fact['predicate']} — {fact['object']}")
            else:
                lines.append("三元组：无（仅有原文）")
            lines.extend(["", "原文：", "", quote(str(hit["text"])), ""])

    # 将总览置于逐题正文之前，便于先判断 Top-5 是否丢失了第 6–15 位的来源证据。
    summary = ["## 总览", "",
               f"- Top-5 ID 顺序与 Top-15 前五位完全一致：{first_five_same}/{len(top5)} 题",
               f"- 有来源会话标签：{gold_available}/{len(top5)} 题",
               f"- Top-5 命中至少一个来源会话：{top5_gold_hit}/{gold_available} 题" if gold_available else "- Top-5 来源命中：无可用标签",
               f"- Top-15 命中至少一个来源会话：{top15_gold_hit}/{gold_available} 题" if gold_available else "- Top-15 来源命中：无可用标签",
               f"- 无来源标签的题号：{missing_gold_examples if missing_gold_examples else '无'}", ""]
    lines[4:4] = summary
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines), encoding="utf-8")
    # 单列 Top-5 未命中的题，优先展示第 6–15 位可以补回来源证据的情况。
    missed_path = args.output.with_name("top5_missed_gold.md")
    missed_lines = ["# Top-5 未命中来源会话的题", "",
                    f"有来源标签的 {gold_available} 题中，Top-5 未命中 {len(missed)} 题；"
                    f"其中 {sum(rank is not None for _, _, rank in missed)} 题能由第 6–15 位补回。", ""]
    for index, row, rank in sorted(missed, key=lambda item: (item[2] is None, item[2] or 99, item[0])):
        missed_lines.extend([f"## 第 {index} 题：{row['question']}", "",
                             f"- 来源会话：{', '.join(gold_sessions(row.get('origin')))}",
                             f"- 来源证据首次名次：{rank if rank is not None else 'Top-15 均未命中'}",
                             f"- 标准答案：{row.get('answer', '')}",
                             f"- Top-5 答案：{row.get('output', '')}", ""])
        for position, hit in enumerate(row.get("evidence", []), start=1):
            preview = " ".join(str(hit["text"]).split())[:250]
            missed_lines.append(f"{position}. Chunk {hit['chunk_id']}；会话 {hit.get('session_id')}；"
                                f"分数 {hit['score']:.4f}；{preview}")
        missed_lines.append("")
    missed_path.write_text("\n".join(missed_lines), encoding="utf-8")
    print(f"Wrote {args.output}; prefix_match={first_five_same}/{len(top5)}; "
          f"gold_session_hit={top5_gold_hit}/{gold_available} at 5, "
          f"{top15_gold_hit}/{gold_available} at 15; missed={missed_path}")


if __name__ == "__main__":
    main()
