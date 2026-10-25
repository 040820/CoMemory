"""复用已有图和摘要，运行 LiCoMemory 原始检索方法的 Top-5 对照实验。"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

# 从项目根目录之外启动时，也要能导入原项目模块。
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT))

from data.query_dataset import RAGQueryDataset
from evaluation.evaluator import Evaluator
from init.config import Config
from init.graph_rag import GraphRAG
from init.logger import update_logger_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="原始 LiCoMemory 检索 Top-5 对照")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=5,
                        help="原方法返回 Chunk 的上限；其余检索配置保持原值")
    parser.add_argument("--limit", type=int, default=None,
                        help="独立小样本检查用；正式运行不设置")
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    """记录图和摘要的哈希，证明不同 k 值实验读取的是同一份记忆。"""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_progress(path: Path, dataset: RAGQueryDataset, total: int) -> list[dict[str, Any]]:
    """中断后按问题文本核对已保存的记录，避免混用其他数据。"""
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
    if args.top_k < 1:
        raise ValueError("top_k must be positive")
    if not args.graph.is_file():
        raise FileNotFoundError(args.graph)
    summary_path = args.graph.parent / "session_summaries.json"
    if not summary_path.is_file():
        raise FileNotFoundError(f"原检索启用 Session 摘要，需要已有文件：{summary_path}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    update_logger_path(str(args.output_dir))

    config = Config.parse(args.config)
    # 只覆盖原方法最终的 Chunk 数量；实体候选、三元组、摘要和 CogniRank 参数沿用配置。
    config.retriever.top_chunks = args.top_k
    dataset = RAGQueryDataset(str(args.dataset_dir))
    total = min(len(dataset), args.limit) if args.limit is not None else len(dataset)
    if total < 1:
        raise ValueError("No questions found")

    metadata = {
        "method": "original_LiCoMemory",
        "retriever_top_chunks": config.retriever.top_chunks,
        "retriever_top_k_entities": config.retriever.top_k,
        "retriever_top_k_triples": config.retriever.top_k_triples,
        "enable_summary": config.retriever.enable_summary,
        "enable_CogniRank": config.retriever.enable_CogniRank,
        "rerank_k": config.retriever.rerank_k,
        "embedding_model": config.embedding.model,
        "query_model": config.query_llm.model or config.llm.model,
        "graph_path": str(args.graph.resolve()),
        "graph_sha256": file_sha256(args.graph),
        "session_summaries_path": str(summary_path.resolve()),
        "session_summaries_sha256": file_sha256(summary_path),
        "dataset_dir": str(args.dataset_dir.resolve()),
        "question_count": total,
    }
    metadata_path = args.output_dir / "run_config.json"
    if metadata_path.exists() and json.loads(metadata_path.read_text(encoding="utf-8")) != metadata:
        raise ValueError("Output directory contains a run with different configuration")
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    # GraphRAG.query() 是项目原始检索和答案生成入口。此处直接加载图，不调用 insert()。
    graph_rag = GraphRAG(config, base_dir=str(args.graph.parent))
    graph_rag.core.graph.load_graph(str(args.graph))
    progress_path = args.output_dir / "progress.jsonl"
    rows = read_progress(progress_path, dataset, total)
    print(f"Loaded original graph; {len(rows)}/{total} questions already saved", flush=True)

    with progress_path.open("a", encoding="utf-8") as progress:
        for index in range(len(rows), total):
            # pandas 的 NumPy 标量转换为标准 JSON 类型，保证评测器可以再次读取。
            row = json.loads(json.dumps(dataset[index], ensure_ascii=False, default=str))
            result = await graph_rag.query(
                str(row["question"]), question_time=str(row.get("question_time") or ""))
            if not isinstance(result, dict):
                raise RuntimeError(f"Question {index + 1}: original query returned unexpected type")
            answer = str(result.get("answer") or "").strip()
            if not answer or answer.startswith("Error processing query:"):
                raise RuntimeError(f"Question {index + 1}: original query failed: {answer}")

            # main.py 原本只保存答案；这里额外记录 QueryProcessor 实际返回的证据以便对照。
            chunks = result.get("chunks") or []
            row.update({
                "output": answer,
                "top_session_ids": result.get("top_session_ids", []),
                "retrieved_chunk_count": len(chunks),
                "retrieved_chunks": chunks,
                "retrieved_triples": result.get("triples") or [],
                "retrieved_summaries": result.get("summaries") or [],
            })
            progress.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
            progress.flush()
            rows.append(row)
            print(f"Answered {index + 1}/{total}; chunks={len(chunks)}", flush=True)

    results_path = args.output_dir / "results.json"
    metrics_path = args.output_dir / "metrics.json"
    results_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    metrics = await Evaluator(str(results_path), args.dataset_dir.name, config).evaluate()
    if "error" in metrics:
        raise RuntimeError(metrics["error"])
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Finished: {metrics.get('correct_answers')}/{metrics.get('total_questions')} correct; "
          f"results={results_path}; metrics={metrics_path}", flush=True)


if __name__ == "__main__":
    asyncio.run(run(parse_args()))
