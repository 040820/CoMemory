"""在已有 CogniGraph 上运行结构化语义检索、回答生成和项目原有评测。"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any

# 允许从项目根目录直接执行此脚本，不要求先安装为 Python 包。
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT))

from base.embeddings import EmbeddingManager
from base.llm import LLMManager
from coregraph.dynamic_memory import DynamicMemory
from data.query_dataset import RAGQueryDataset
from evaluation.evaluator import Evaluator
from experiments.structural_semantic_retrieval import StructuralSemanticRetriever
from init.config import Config
from init.logger import logger
from prompt.query_prompt import QUERY_PROMPT


def parse_args() -> argparse.Namespace:
    """显式指定已有图、数据和独立输出目录，避免重新建图或覆盖 Baseline。"""
    parser = argparse.ArgumentParser(description="评测结构化语义检索实验")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=15)
    parser.add_argument("--limit", type=int, default=None, help="仅供独立小样本检查；正式运行不设置")
    parser.add_argument("--max-chunk-chars", type=int, default=2800)
    return parser.parse_args()


def make_prompt(question: str, question_time: str, evidence: list[dict[str, Any]],
                max_chunk_chars: int, max_triples: int) -> tuple[str, int]:
    """复用原项目的无摘要问答提示词，并记录为模型窗口截短的 Chunk 数。"""
    if max_chunk_chars < 1:
        raise ValueError("max_chunk_chars must be positive")

    chunks = []
    truncated = 0
    unique_triples: dict[tuple[str, str, str], None] = {}
    for index, hit in enumerate(evidence, start=1):
        original_text = str(hit["text"])
        if len(original_text) > max_chunk_chars:
            truncated += 1
        session_time = str(hit.get("session_time") or "").strip()
        shown_text = original_text[:max_chunk_chars]
        chunks.append(f"Chunk {index}: {session_time} {shown_text}".strip())

        # 同一个事实可能挂在多个 Chunk 上；提示词只保留一次，证据文件仍保留原始映射。
        for fact in hit.get("triples", []):
            key = (str(fact["subject"]), str(fact["predicate"]), str(fact["object"]))
            unique_triples[key] = None

    triples = "; ".join(f"({subject}, {relation}, {object_})"
                        for subject, relation, object_ in list(unique_triples)[:max_triples])
    prompt = QUERY_PROMPT.format(
        question_time=question_time,
        question=question,
        triples=triples or "None available",
        chunks="\n".join(chunks) or "None available",
    )
    return prompt, truncated


def create_query_llm(config: Config) -> LLMManager:
    """沿用原查询阶段的 vLLM 参数，保持回答模型一致。"""
    query = config.query_llm
    return LLMManager(
        api_key=query.api_key or config.llm.api_key,
        model=query.model or config.llm.model,
        max_tokens=query.max_token if query.max_token > 0 else config.llm.max_token,
        base_url=query.base_url or config.llm.base_url,
        enable_concurrent=False,
        max_concurrent=1,
        timeout=query.timeout if query.timeout > 0 else config.llm.timeout,
        enable_thinking=(query.enable_thinking if query.enable_thinking is not None
                         else config.llm.enable_thinking),
    )


def load_progress(path: Path, dataset: RAGQueryDataset, total: int) -> list[dict[str, Any]]:
    """中断后从完整的 JSONL 行继续，核对题目次序以防误用其他数据。"""
    if not path.exists():
        return []
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows) > total:
        raise ValueError("Progress file has more rows than the requested dataset")
    for index, row in enumerate(rows):
        if row.get("question") != dataset[index]["question"]:
            raise ValueError(f"Progress file question mismatch at row {index + 1}")
    return rows


async def run(args: argparse.Namespace) -> None:
    """逐题保存检索轨迹；全部完成后再用原项目评测器生成指标。"""
    if args.top_k < 1:
        raise ValueError("top_k must be positive")
    if not args.graph.is_file():
        raise FileNotFoundError(args.graph)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    config = Config.parse(args.config)
    dataset = RAGQueryDataset(str(args.dataset_dir))
    total = min(len(dataset), args.limit) if args.limit is not None else len(dataset)
    if total < 1:
        raise ValueError("No questions found")

    memory = DynamicMemory(config, base_dir=str(args.graph.parent))
    memory.load_graph(str(args.graph))
    retriever = StructuralSemanticRetriever(memory, EmbeddingManager(config.embedding))
    answer_llm = create_query_llm(config)
    progress_path = args.output_dir / "progress.jsonl"
    results_path = args.output_dir / "results.json"
    metrics_path = args.output_dir / "metrics.json"
    rows = load_progress(progress_path, dataset, total)
    logger.setLevel("WARNING")
    print(f"Loaded {len(memory.chunk_storage)} chunks; {len(rows)}/{total} questions already saved", flush=True)

    with progress_path.open("a", encoding="utf-8") as progress:
        for index in range(len(rows), total):
            # 项目数据集中部分字段由 pandas 读成 NumPy 标量；转为标准 JSON 类型。
            question_row = json.loads(json.dumps(dataset[index], ensure_ascii=False, default=str))
            question = str(question_row["question"])
            question_time = str(question_row.get("question_time") or "")
            retrieved = await retriever.retrieve(question, top_k=args.top_k)
            evidence = retrieved["evidence"]
            prompt, truncated = make_prompt(question, question_time, evidence,
                                            args.max_chunk_chars, config.retriever.top_k_triples)
            output = await answer_llm.generate(prompt)
            if not output.strip():
                raise RuntimeError(f"Question {index + 1}: answer model returned empty text")

            # 逐题保留完整证据和得分，便于核查候选 ID、顺序与正确答案的关系。
            session_ids = list(dict.fromkeys(str(hit["session_id"]) for hit in evidence
                                                 if hit.get("session_id")))
            question_row.update({
                "output": output.strip(),
                "top_session_ids": session_ids,
                "retrieved_chunk_ids": [hit["chunk_id"] for hit in evidence],
                "prompt_truncated_chunks": truncated,
                "query_analysis": retrieved["query_analysis"],
                "evidence": evidence,
            })
            progress.write(json.dumps(question_row, ensure_ascii=False, default=str) + "\n")
            progress.flush()
            rows.append(question_row)
            print(f"Answered {index + 1}/{total}; evidence={len(evidence)}; truncated={truncated}", flush=True)

    # 评测器读取与 Baseline 相同的 JSON 数组格式；只写入本实验目录。
    results_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    evaluator = Evaluator(str(results_path), args.dataset_dir.name, config)
    metrics = await evaluator.evaluate()
    if "error" in metrics:
        raise RuntimeError(metrics["error"])
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Finished: {metrics.get('correct_answers')}/{metrics.get('total_questions')} correct; "
          f"results={results_path}; metrics={metrics_path}", flush=True)


if __name__ == "__main__":
    asyncio.run(run(parse_args()))
