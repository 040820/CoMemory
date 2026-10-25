# Query-driven Structural-Semantic Retrieval

Independent retrieval experiment over a **previously built** LiCoMemory
`DynamicMemory`. It reads `graph_builder.graph` and `chunk_storage`; it does not
change graph storage, memory construction, the existing `QueryProcessor`, or
baseline evaluation scripts.

## Method

1. Match entity and relation labels from the query against the existing graph.
   The remaining query text is exposed as `target`. This deterministic analysis
   needs no LLM; paraphrased or implicit labels may not be extracted.
2. Gather every relationship edge by its `chunk_id` and `chunk_ids`. Each Chunk
   becomes a semantic unit containing all available triples plus its original
   context. Chunks with no stored triples retain a text-only fallback.
3. Embed the **complete query**, each Chunk unit, and each individual triple.
   Query-to-unit and Query-to-Triple cosine scores are combined with explicit
   entity/relation matches. There is no Session prefilter.
4. Expand from the top ranked Chunk seeds through shared entities and one graph
   hop. Rerank and return distinct original Chunks with score components.

The process-local embedding cache is keyed by text; no index files are written.
Call `clear_cache()` if the embedding model changes. Since LiCoMemory currently
uses a `DiGraph`, an edge merged or overwritten during original construction
can expose only the relation stored on that edge. This retriever cannot recover
relations absent from the saved graph.

## Use

From the repository root, with an existing graph pickle and a working embedding
backend:

```python
import asyncio
from pathlib import Path

from base.embeddings import EmbeddingManager
from coregraph.dynamic_memory import DynamicMemory
from init.config import Config
from experiments.structural_semantic_retrieval import StructuralSemanticRetriever

config = Config.parse(Path("config/Memory.yaml"))
memory = DynamicMemory(config)
memory.load_graph("/path/to/existing/dynamic_memory_graph.pkl")
embedding_manager = EmbeddingManager(config.embedding)
retriever = StructuralSemanticRetriever(memory, embedding_manager)

result = asyncio.run(retriever.retrieve("Your complete question here", top_k=15))
for item in result["evidence"]:
    print(item["chunk_id"], item["score"], item["text"])
```

The module returns evidence only. Answer generation can consume
`result["evidence"]` separately. An available embedding backend is required;
vLLM and a query extraction LLM are not.

## 在已有图上运行完整问答评测

`run_evaluation.py` 只读取已经保存的 CogniGraph，逐题调用本目录的检索器，使用原项目的 `QUERY_PROMPT` 和当前配置中的查询 LLM 生成答案，再调用原项目的 `Evaluator`。它不会重新建图或修改原始检索器。每题完整证据、Chunk ID、分数及答案保存在独立输出目录的 `progress.jsonl` 和 `results.json`，指标保存在 `metrics.json`。

在服务器项目根目录执行：

```bash
conda activate licomemory
CUDA_VISIBLE_DEVICES=1 HF_HUB_OFFLINE=1 python -u \
  experiments/structural_semantic_retrieval/run_evaluation.py \
  --config config/Memory.yaml \
  --graph results/locomo_group_1/dynamic_memory_graph.pkl \
  --dataset-dir dataset/processed/locomo/group_1 \
  --output-dir results/structural_semantic_locomo_group_1
```

LongMemEval 样例使用 `experiments/reproduction/Memory_LongMemEval.yaml`、`results/longmem_ku_lm_1/dynamic_memory_graph.pkl`、`dataset/processed/longmem/longmem_ku/lm_1`，输出目录建议使用 `results/structural_semantic_longmem_ku_lm_1`。

默认检索 15 个 Chunk。为适配当前 vLLM 的 16K 上下文窗口，回答提示词中每个 Chunk 最多使用 2800 个字符；`results.json` 中的 `evidence` 仍保存完整原文，`prompt_truncated_chunks` 记录实际被截短的数量。LoCoMo `group_1` 本次 199 题均未截短；LongMemEval 样例有 10 个 Chunk 在回答提示词中截短。

本次服务器运行：LoCoMo `group_1` 的 LLM 评测为 81/199（40.70%），原流程在同一图和问题上的结果为 78/199（39.20%）；LongMemEval 单题样例两种方法均为 1/1。LoCoMo 的确定性答案匹配分别为 32/199 和 31/199。原流程使用 Session 摘要提示词，新方法使用不含摘要的 `QUERY_PROMPT`；这两个准确率是各自完整问答流程的结果，不能单独归因于检索排序。原流程没有输出 `top_session_ids`，因此两种方法的 session matching 分数也不具可比性。

## Top-5 检查实验

保持图、问题、Embedding、回答模型与提示词不变，只把上述命令的输出目录改为 `results/structural_semantic_top5_locomo_group_1`，并加上 `--top-k 5`。LongMemEval 样例同样使用 `--top-k 5`，输出到 `results/structural_semantic_top5_longmem_ku_lm_1`。`results.json` 保留每题全部五条证据的原文及分数。

逐题阅读报告可重新生成：

```bash
python experiments/structural_semantic_retrieval/export_top5_review.py \
  --top5 results/structural_semantic_top5_locomo_group_1/results.json \
  --top15 results/structural_semantic_locomo_group_1/results.json \
  --output results/structural_semantic_top5_locomo_group_1/top5_review.md
```

该命令还生成 `top5_missed_gold.md`，集中列出 Top-5 没有命中标准答案来源会话的题。LoCoMo `group_1` 的本次结果：Top-5 LLM 评测 87/199（43.72%），Top-15 为 81/199（40.70%）；两者确定性答案匹配均为 32/199。199 题的 Top-5 顺序都与 Top-15 前五位一致。197 题有来源会话标签，其中 Top-5 命中 166 题，Top-15 命中 187 题；有 21 题的来源证据排在第 6–15 位。Top-5 回答准确率较高并不表示其证据覆盖更好。LongMemEval 单题样例在两种 k 值下均为 1/1；其回答提示词为适配模型窗口截短了 5 条较长 Chunk，保存的证据原文未截短。
