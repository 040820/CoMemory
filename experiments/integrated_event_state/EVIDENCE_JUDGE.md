# AMA 启发的证据 Judge 调度 v1

复用 AMA 的证据充分性、事实冲突判断和有限补查思想，以及问题精度、同一事件/状态、
相对时间精度保持等判定原则。独立实现，不依赖 AMA 目录，不直接复制其提示词正文、
数据库操作或 Refresher。原版检索/写入默认行为不变，通过独立 CLI 显式启用。

## 流程

```text
已追加的新事实 / 用户问题
  → 初始历史召回＋规范 Chunk 原文
  → Judge：SUFFICIENT / NEED_MORE / CONFLICT / DEFER
  → NEED_MORE 时根据具体缺口补查
     ├─ 后续问题：原任务＋缺口 → 现有结构化语义检索
     ├─ 来源扩展：同事件分组、候选关联两端 → 中层 → 原始 Chunk
     └─ 上下文扩展：同 Session 的前后 Chunk
  → 有新增证据才再次 Judge
  → 保存证据、判断、引用及每轮检索轨迹
```

状态类初始候选复用 StateSlot＋BGE-M3，事件类复用当前事件候选流程。
Question 模式和非 Observation 类型的来源回查使用现有结构化语义检索器。
候选关联和实体相似仅用于定位，不能直接证明关系。检索返回的文本必须按本图
Chunk ID 回查，不能将其他图或模型编造的片段作为证据。

## 状态与边界

| 状态 | 含义 |
| --- | --- |
| SUFFICIENT | 在所给范围内，证据足以完成当前检查；不是现实真值或更新已确认 |
| NEED_MORE | 存在明确证据缺口，需要有限补查 |
| CONFLICT | 所给证据存在可比较的事实矛盾；不代表新说法正确，也不执行替换 |
| DEFER | 解析/检索失败、引用无效、缺少关键依据、无新增证据或预算耗尽 |

任何适用的身份、作用域、模态、精度、时间检查为 fail/unknown，都不能通过。
精度检查始终适用，记忆关系模式中的身份核对也始终适用，不允许用 not_applicable 跳过。
引用必须绑定展示的 Fact ID、Chunk 和连续原文；无结构事实的纯原文引用可以使用空 Fact ID。
判定冲突必须绑定双方不同 Fact 的证据，记忆关系模式还必须包含被审查的新 Fact。
同 Chunk 可以含多条断言，不强制将两个 Chunk 当成两份独立证据。

默认最多 3 次 Judge 调用（包括首次，不是额外 3 次）、每次 Top-5、最多 8 个 Chunk。
输入按 12000 UTF-8 字节保守限额，单个原文窗口最多 1200 字节，输出最多 1200 token。
模型仅可引用实际展示窗口，截断和未展示 Chunk 会留档；原始 Chunk 不被修改。
不会对哈希 Chunk ID 做加减找邻居；同 Session 中按局部序号/插入顺序定位前后文。

本阶段仅 Judge 调度，不实现最终 ADD/CORRECT/REPLACE 决策，不改历史有效性、
当前状态或事件身份。不进行数值置信度融合。所有结果 `update_applied=false`、
`validation_status=not_verified`；后续证据融合与决策模块可消费本模块产物。

## 命令

审查已经追加到图中的新事实：

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python \
  experiments/integrated_event_state/run_evidence_judge.py \
  --source /绝对路径/dynamic_memory_event_state.pkl \
  --output /绝对路径/全新Judge实验目录 \
  --new-fact-id 实际中层FactID \
  --query '核查这条新记忆与历史记忆的关系，并检查证据是否充分' \
  --max-rounds 3 --top-k 5 --max-chunks 8
```

检查问答证据时使用 `--mode question --query '具体问题'`，不需要 new-fact-id。
本入口只返回证据审查结果，不生成最终答案，也不会自动挂接现有问答入口。
请使用 event_index_v1 图；源图不改动，输出目录必须全新。

输出：
- `manifest.json`：来源哈希、代码指纹、模型和预算。
- `llm_trace.jsonl`：实际模型输入、输出和 finish_reason。
- `judge_result.json`：状态、停止原因、收集的来源和每轮判断。
- `judge_rounds.jsonl`：每轮展示窗口、引用、检查结果。
- `retrieval_actions.jsonl`：补查问题、路线和命中 Chunk。
- `initial_recall_trace.json`：状态类初始双路召回轨迹（适用时）。

这是新的检索调度实验，与既有冻结 sample0 构图无关，不重启或修改该任务。

## AMA 对照来源

- `python/AdaptiveMemory/Model/model.py:144`：inferenceJudge 的输入组织。
- `python/AdaptiveMemory/Settings/prompt.py:177`：充分性、精度和冲突规则。
- `python/AdaptiveMemory/Core/AMA.py:577`、`:997`：有界 Judge/补查循环。
- 上述路径来自用户提供的 `D:/Code/AMA-main`；这是思想/规则参考，不是原实现的直接移植。
