# 服务器复现记录

本目录只保存复现配置和操作说明。运行结果保存在服务器的 `results/` 下，不覆盖已有实验结果。

## 环境与模型

- 服务器项目目录：`/home/cwf/Code/LiCoMemory`
- Conda 环境：`licomemory`
- 本地 vLLM 接口：`http://127.0.0.1:8011/v1`，服务模型名为 `qwen3-8b`
- Embedding：服务器 Hugging Face 默认缓存中的 `BAAI/bge-m3`，1024 维；`HF_HUB_OFFLINE=1` 禁止运行时重新下载
- `config/Memory.yaml` 用于 LoCoMo；本目录的 `Memory_LongMemEval.yaml` 用于 LongMemEval 的 `longmem_ku` 子集

## 运行 LoCoMo

在服务器项目根目录执行：

```bash
conda activate licomemory
HF_HUB_OFFLINE=1 python -u main.py -opt config/Memory.yaml \
  -dataset_name group_1 -root locomo_group_1 -query 1
```

该命令读取 `dataset/processed/locomo/group_1`，在 `results/locomo_group_1/` 下保存图、逐题答案和评测指标。更换组别时，同时更换 `-dataset_name` 和 `-root`，避免覆盖原有结果。

## 运行 LongMemEval 单个样例

LongMemEval 转换后每个问题都有独立目录。例如 `longmem_ku/lm_1`：

```bash
HF_HUB_OFFLINE=1 python -u main.py \
  -opt experiments/reproduction/Memory_LongMemEval.yaml \
  -dataset_name lm_1 -root longmem_ku_lm_1 -query 1
```

此配置的 `data_root` 固定指向 `longmem_ku`；其他问题类型应另设对应路径。完整 LongMemEval 包含 500 个独立问题目录，单个样例运行结果不能代表完整数据集指标。

## 使用新增检索模块

新增的结构化语义检索模块读取已经生成的 `.pkl` 图，不参与上述 `main.py` 的原始检索评测。调用方式见 `experiments/structural_semantic_retrieval/README.md`。其默认返回 15 个原始 Chunk；若要比较两种检索算法，需要另行采用相同问题和评价口径进行评测。

本次使用 Qwen3-8B 本地模型。论文使用的建图与回答模型不同，因此此处验证程序流程和输出产物，不能把数值直接当作论文结果复现。

## 本次服务器运行结果

| 数据 | 结果目录 | 图文件 | 答案数 | LLM 评测准确率 |
| --- | --- | ---: | ---: | ---: |
| LoCoMo `group_1` | `results/locomo_group_1/` | 8,039,681 字节 | 199/199 | 78/199，39.20% |
| LongMemEval `longmem_ku/lm_1` | `results/longmem_ku_lm_1/` | 48,156,790 字节 | 1/1 | 1/1 |

两个运行的 `results/results.json` 都没有空答案或查询处理错误，`results/metrics.json` 已生成。新增检索模块在 LoCoMo 保存的图上独立返回 15 条原文证据；示例输出见 `results/locomo_group_1/structural_semantic_demo.log`。

当前原始流程写出的每条 `top_session_ids` 均为空，因此 `average_matching_score=0.0` 不能解释为真实的会话检索命中率。LoCoMo 的数字型 `question_type` 在 LLM 评测器中走默认提示词模板；上述准确率是本次 Qwen3-8B 配置的输出，不适合与论文指标直接比较。

## 新增检索方式的独立评测

执行脚本和详细口径见 `experiments/structural_semantic_retrieval/README.md`。它复用上述图文件，在新目录保存结果：

| 数据 | 结果目录 | 答案数 | LLM 评测准确率 |
| --- | --- | ---: | ---: |
| LoCoMo `group_1` | `results/structural_semantic_locomo_group_1/` | 199/199 | 81/199，40.70% |
| LongMemEval `longmem_ku/lm_1` | `results/structural_semantic_longmem_ku_lm_1/` | 1/1 | 1/1 |

新方法的 `results.json` 包含每题完整证据、Chunk 顺序、分数和生成答案。LoCoMo 两种结果的问题顺序一致；本次新方法的 199 题每题均返回 15 个 Chunk，回答提示词没有截短原文。准确率差异是一次运行的观察结果，尚不能证明新检索方式优于原流程。

### Top-5 复查

相同图和数据，设置新检索脚本的 `--top-k 5` 后，结果分别在 `results/structural_semantic_top5_locomo_group_1/` 与 `results/structural_semantic_top5_longmem_ku_lm_1/`。LoCoMo 的 LLM 评测为 87/199（43.72%），LongMemEval 单题样例为 1/1。`top5_review.md` 展开了每题五条证据的原文、三元组和分数；`top5_missed_gold.md` 集中列出来源会话未进入 Top-5 的题。

LoCoMo 有 197 题带来源会话标签：Top-5 命中 166 题，Top-15 命中 187 题；21 题的来源证据只出现在第 6–15 位。这是来源会话覆盖的检查，不等同于答案正确率。两种 k 值的确定性答案匹配同为 32/199，而 LLM 评测分别为 87/199 和 81/199；目前不宜据此断言 Top-5 的检索质量更高。

### 原始检索方法的 Top-5 对照

`run_original_top5.py` 直接加载已保存的图和 Session 摘要，并逐题调用原项目 `GraphRAG.query()`。它只把内存中的 `retriever.top_chunks` 覆盖为 5，不调用 `insert()`，也不修改已有图、摘要、原始检索代码或先前的结果。

```bash
CUDA_VISIBLE_DEVICES=1 HF_HUB_OFFLINE=1 python -u \
  experiments/reproduction/run_original_top5.py \
  --config config/Memory.yaml \
  --graph results/locomo_group_1/dynamic_memory_graph.pkl \
  --dataset-dir dataset/processed/locomo/group_1 \
  --output-dir results/original_top5_locomo_group_1 \
  --top-k 5
```

输出目录包含 `run_config.json`（图和摘要的 SHA256）、`progress.jsonl`、`results.json`（逐题答案、原方法实际返回的 Chunk、三元组和摘要）、`metrics.json` 与 `run.log`。运行完成后可用 `compare_top5.py --results-root results --output results/original_top5_locomo_group_1/comparison.md` 生成四组问答和来源覆盖对照。

本次同图、同 199 题的结果如下：

| 检索方法 | Top-15 LLM 评测 | Top-5 LLM 评测 | Top-15 确定性匹配 | Top-5 确定性匹配 |
| --- | ---: | ---: | ---: | ---: |
| 原始 LiCoMemory | 78/199 | 75/199 | 31/199 | 27/199 |
| 结构化语义检索 | 81/199 | 87/199 | 32/199 | 32/199 |

原方法 Top-5 在全部 199 题都返回了五个 Chunk。197 道有来源会话标签的题中，原方法 Top-5 命中 143 题，新方法 Top-5 命中 166 题；前者独有 10 题，后者独有 33 题。原方法减少 Chunk 后没有提升答案指标，因此本次新方法 Top-5 的优势不能仅由上下文数量减少解释。回答提示词、Session 摘要使用方式和 LLM 评测波动仍是混杂因素，不能把 12 题的差额全部归因于检索排序。完整的逐题 Chunk ID 和证据原文保存在 `comparison.json` 与 `original_top5_review.md`。
