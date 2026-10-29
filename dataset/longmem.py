#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
把 LongMemEval 式的「多 session 问答数据集」拆成按题目类型分组的逐题文件夹。

【整体数据结构】
输入是一个 JSON 数组，每个元素是一道题（一个 item），结构如下：

    {
      "question_id":        "题目唯一 ID",
      "question_type":      "single-session-user",   # ★ 决定放到哪个类型文件夹
      "question":           "What degree did I graduate with?",
      "answer":             "Business Administration",
      "question_date":      "2023/05/30 (Tue) 23:40",  # 提问时间
      "haystack_dates":     ["2023/05/20 (Sat) 02:21", "2023/05/20 (Sat) 02:57"],
      "haystack_session_ids": ["sharegpt_yywfIrx_0", "85a1be56_1"],
      "haystack_sessions":  [                       # 与上面两个数组一一对应
          [ {"role":"user","content":"..."},
            {"role":"assistant","content":"..."} ],  # 第 1 个 session
          [ {"role":"user","content":"..."},
            {"role":"assistant","content":"..."} ]   # 第 2 个 session
      ],
      "answer_session_ids": ["answer_280352e9"],    # 证据所在 session
    }

注意：一道题自带一整包 haystack_sessions（可能几十个 session），
其中只有极少数 session 是回答该问题真正需要的证据。

【输出布局】
outdir/
  longmem_ssu/            # question_type = single-session-user
    lm_1/                 # 该类型下的第 1 道题
      Corpus.json         # 该题 haystack 里的所有 session，一行一条（NDJSON）
      Question.json       # 该题本身，一行（NDJSON）
    lm_2/ ...
  longmem_ssa/ ...        # single-session-assistant
  longmem_ms/  ...        # multi-session
  longmem_tr/  ...        # temporal-reasoning
  longmem_ku/  ...        # knowledge-update
  longmem_ssp/ ...        # single-session-preference

【"分割"体现在三层】
  1) 第一层：按 question_type 分到不同的一级目录（数据集类型划分）。
  2) 第二层：同一类型内，每道题各占一个 lm_N 子目录（数据项划分）。
  3) 第三层：每道题的 haystack_sessions 再按 session 拆成
     Corpus.json 中的多行（session 级划分）；
     而题目本身单独写进 Question.json。

用法:
  python longmem.py --input data.json --outdir ./out
"""

import argparse
import json
import os
import re
from typing import Any, Dict, Iterable, List, Optional

# 从 "2023/05/20 (Sat) 02:21" 这类字符串中，只匹配开头的 "YYYY/MM/DD"。
# 注意：小时/分钟会被丢弃，只保留日期。
DATE_RE = re.compile(r"^(\d{4}/\d{2}/\d{2})")

# question_type（完整题目类型名） -> 文件夹后缀（缩写）
# 作用：把长类型名映射成目录名后缀，例如 single-session-user -> ssu
QUESTION_TYPE_MAPPING = {
    "single-session-user": "ssu",
    "single-session-assistant": "ssa",
    "multi-session": "ms",
    "temporal-reasoning": "tr",
    "knowledge-update": "ku",
    "single-session-preference": "ssp",
}


def get_folder_suffix(question_type: str) -> str:
    """把 question_type 映射成文件夹后缀；未知类型返回 "unknown"。"""
    return QUESTION_TYPE_MAPPING.get(question_type, "unknown")


def get_next_folder_index(parent_dir: str) -> int:
    """
    计算某个类型目录下下一个可用的 lm_N 编号。

    规则：
      - 目录不存在，或里面还没有 lm_* 子目录 -> 返回 1；
      - 否则取所有 lm_<数字> 里最大的数字 + 1；
      - 名字不是 lm_<整数> 的目录（如 lm_abc）会被忽略。

    注意：编号是"每个类型目录各自独立计数"的，
    并且每次调用都重新扫一遍磁盘，所以重复运行不会覆盖旧结果，
    而是接着往后追加（不具备幂等性）。
    """
    if not os.path.exists(parent_dir):
        return 1

    existing_folders = []
    for item in os.listdir(parent_dir):
        if os.path.isdir(os.path.join(parent_dir, item)) and item.startswith("lm_"):
            try:
                idx = int(item[3:])  # 取 "lm_" 之后的数字部分
                existing_folders.append(idx)
            except ValueError:
                continue  # 非数字后缀，跳过

    return max(existing_folders) + 1 if existing_folders else 1


def extract_date_prefix(dt: Optional[str]) -> Optional[str]:
    """
    从 '2023/05/20 (Sat) 02:21' 中提取出 'YYYY/MM/DD'。
    输入为空 / 非字符串 / 匹配失败时返回 None。
    """
    if not dt or not isinstance(dt, str):
        return None
    m = DATE_RE.match(dt.strip())
    return m.group(1) if m else None


def build_context(session_messages: Iterable[Dict[str, Any]]) -> str:
    """
    把一个 session 的消息列表拼成一个 context 字符串。

    拼接格式形如（不带任何分隔符，直接首尾相接）：
        "user": "..." "assistant": "..."
    即每条消息变成：  "<role>": "<content>"

    注意：
      - 这里刻意不用 json.dumps(content)，否则内容里的 \n 会被二次转义；
      - 因为消息之间没有分隔符，content 本身又可能含引号，
        所以这个字符串严格说不是合法 JSON，仅作为纯文本语料使用。
    """
    parts: List[str] = []
    for msg in session_messages:
        role = str(msg.get("role", ""))
        content = str(msg.get("content", ""))
        parts.append(f"\"{role}\": \"{content}\"")
    return "".join(parts)


def process_item(item: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    """
    处理单个题目 item，产出两份记录：

      - corpus_records: 该题 haystack 中每个 session 一条记录
                        （session 级切分就在这里完成）
      - question_record: 该题本身的一条记录

    返回 {"corpus": [...], "question": [...]}
    """
    # 三个平行数组：日期、session_id、session 消息列表
    haystack_dates = item.get("haystack_dates") or []
    haystack_session_ids = item.get("haystack_session_ids") or []
    haystack_sessions = item.get("haystack_sessions") or []

    # 三个数组理论上等长，取最短长度对齐，避免越界（多余的数据被丢弃）
    n = min(len(haystack_dates), len(haystack_session_ids), len(haystack_sessions))

    corpus_records: List[Dict[str, Any]] = []
    for i in range(n):
        session_time = extract_date_prefix(haystack_dates[i])  # 只留日期
        session_id = haystack_session_ids[i]
        session_msgs = haystack_sessions[i] or []
        context = build_context(session_msgs)                  # 整个 session 压成一段文本
        corpus_records.append({
            "session_time": session_time,
            "context": context,
            "session_id": session_id,
        })

    # ---------- 题目记录 ----------
    question_text = item.get("question", "")
    answer_text = item.get("answer", "")
    question_type = item.get("question_type", "")
    question_time = item.get("question_date", "")  # 原样保留提问时间字符串

    # origin 表示"证据所在 session"：
    #   - 只有一个证据 session -> 存成裸字符串（与示例格式一致）
    #   - 多个证据 session     -> 存成列表
    # 下游读取时需要同时兼容这两种类型。
    answer_session_ids = item.get("answer_session_ids", [])
    if isinstance(answer_session_ids, list):
        if len(answer_session_ids) == 1:
            origin: Any = answer_session_ids[0]
        else:
            origin = answer_session_ids
    else:
        origin = answer_session_ids  # 形状异常时原样透传

    question_record = {
        "question": question_text,
        "answer": answer_text,
        "question_type": question_type,
        "question_time": question_time,
        "origin": origin,
        "label": answer_text,  # label 直接复用 answer（便于评测时对齐）
    }

    return {"corpus": corpus_records, "question": [question_record]}


def write_ndjson(path: str, rows: Iterable[Dict[str, Any]]) -> None:
    """
    写 NDJSON：每行一个 JSON 对象。
    ensure_ascii=False 保证中文直接写入，不使用 Unicode 转义序列。
    （注意：文件名是 .json，但内容其实是 NDJSON 格式）
    """
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False))
            f.write("\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", required=True, help="Path to input dataset JSON file")
    parser.add_argument("--outdir", "-o", required=True, help="Output directory to place per-item folders")
    parser.add_argument("--basename", default="longmem", help="Folder base name prefix (default: longmem)")
    args = parser.parse_args()

    in_path = args.input
    out_root = args.outdir
    base = args.basename

    # 输出根目录不存在则创建
    os.makedirs(out_root, exist_ok=True)

    # 一次性把整个数据集读进内存（数据集较大时注意内存占用）
    with open(in_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("Input JSON must be a list of items.")

    # 仅用于最后打印统计信息
    total_items = 0
    total_sessions = 0
    type_counts = {}

    # ===== 逐题处理：这里体现第 1、2 层切分 =====
    for item in data:
        if not isinstance(item, dict):
            continue  # 跳过非法元素

        # ① 按 question_type 决定一级目录后缀
        question_type = item.get("question_type", "")
        folder_suffix = get_folder_suffix(question_type)

        if folder_suffix == "unknown":
            print(f"Warning: Unknown question type '{question_type}', skipping item")
            continue  # 未知类型直接跳过，不写入任何文件

        # ② 一级目录：<outdir>/<base>_<suffix>，例如 out/longmem_ssu
        type_folder = os.path.join(out_root, f"{base}_{folder_suffix}")
        os.makedirs(type_folder, exist_ok=True)

        # ③ 二级目录：该类型下下一个可用编号 lm_N
        next_idx = get_next_folder_index(type_folder)
        item_folder = os.path.join(type_folder, f"lm_{next_idx}")
        os.makedirs(item_folder, exist_ok=True)

        # ④ 第 3 层切分：把该题的所有 session 拆成多条 corpus 记录
        result = process_item(item)
        corpus_records = result["corpus"]
        question_records = result["question"]

        # ⑤ 写入：Corpus.json 一行一个 session；Question.json 一行（只有 1 题）
        corpus_out = os.path.join(item_folder, "Corpus.json")
        question_out = os.path.join(item_folder, "Question.json")

        write_ndjson(corpus_out, corpus_records)
        write_ndjson(question_out, question_records)

        total_items += 1
        total_sessions += len(corpus_records)
        type_counts[question_type] = type_counts.get(question_type, 0) + 1

        print(f"[OK] {item_folder} -> {len(corpus_records)} sessions, 1 question (type: {question_type})")

    # ===== 汇总输出 =====
    print(f"\nDone. Processed {total_items} items with {total_sessions} sessions in total.")
    print(f"Base folder: {out_root}, prefix: '{base}_'")
    print("Items by type:")
    for qtype, count in type_counts.items():
        suffix = get_folder_suffix(qtype)
        print(f"  {qtype} ({base}_{suffix}): {count} items")


if __name__ == "__main__":
    main()
