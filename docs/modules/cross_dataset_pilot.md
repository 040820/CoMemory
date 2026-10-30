# LoCoMo / LongMemEval 小样本衡量模块说明书

## 本轮目的

用真实数据检查当前系统能否找到来源、给出有引用的答案，以及失败发生在哪一步。原生问答标签不能直接作为 Fact 演化关系或证据真实性标签，因此关系准确率与置信度校准保持未评估，另导出人工标注模板。

## 样本与防泄漏

固定随机种子 `20261029`，在模型运行前冻结：

- LoCoMo sample0：类别 1/2/4 各 4 题，类别 5 另取 2 题，共 14 题。复用既有完整来源的新图；排除反复调试过的第 0 题。sample0 已用于开发，不能声称独立留出集。
- LongMemEval：knowledge-update、temporal-reasoning、multi-session 各 1 题，共 3 题。固定选中原始下标 434、289、98，对应 48、42、47 个会话。
- 每个 LongMemEval 问题使用其完整 haystack，包含干扰会话；绝不根据 answer_session_ids 只保留正确来源。
- 问答题的 gold 答案/来源只在评分、覆盖统计及人工核查中使用，不传给构图、取证或 Reader/Judge。
- 原始 LongMemEval 日期保留小时分钟，按时间稳定排序。新语料独立保存在实验目录，已有 dataset/processed 不修改。

LongMemEval 虽只有三题，仍需构建 137 个会话；不能把“题目少”理解为“只构建几个证据片段”。

## 文件与职责

| 代码 | 所属实验模块及功能 |
|---|---|
| `experiments/cross_dataset_pilot/selection.py` | 固定种子分层选题；验证会话数组完整性；保留原生时间精度 |
| `experiments/cross_dataset_pilot/metrics.py` | 四阶段来源覆盖、严格原生证据轮次全文覆盖、文本 EM、技术/语义暂缓区分及分数据集汇总 |
| `experiments/cross_dataset_pilot/run.py` | 冻结代码/输入、复用 LoCoMo 图、完整构建 LongMemEval、逐题问答、独立评分及人工候选对导出 |
| `tests/test_cross_dataset_pilot.py` | 固定输入验证选样不依赖答案、完整历史、时间精度和指标分母；不评价语义效果 |

## 调用流程与检索方法

选题冻结 → 保存独立完整语料 → 冻结源码并重新启动进程 → LoCoMo 复用图／LongMemEval 逐会话构图 → 当前 EvidenceSupport 问答 → 原生来源覆盖统计 → 独立 LLM 评分 → 导出人工检查内容。

构图直接调用现有 `DynamicMemory.build_graph/add_single_session/save_graph`，复用真实 BGE-M3 和 vLLM 实例；不另写一套构图算法。每个会话结束保存图和进度，崩溃后从该题检查点恢复。

问答算法复用当前正式实现：原结构化语义 Chunk 检索选入口 → BGE 选择候选 Fact → 结构索引/向量/有限 BFS/RRF 补查 → Judge 按片段编号审查 → 回答按编号引用。具体参数及边界见 [证据模块说明](evidence_protocol_repair.md)。不把检索分数当来源真实性，也不把 candidate 关联当成已确认更新。

## 指标如何理解

1. **原生证据 Session 覆盖**：分别计算 Seed Top-5、候选池、实际展示窗口、最终答案引用命中了多少标注 Session。按 Session 去重；gold source 为空时 N/A。某阶段失败且有 gold 来源时计零，不只统计成功题。
2. **LoCoMo 证据轮次全文覆盖**：对原生 evidence 中 dia_id 对应的完整文本做同 Session、空白归一后的匹配，分别检查存储/候选/窗口/引用。用于发现上游丢失或窗口遗漏。关键片段可能足以回答却没有整轮全文，因此它不是语义支持率。
3. **答案评分**：保留保守文本 EM，另复用 `LLMEvaluator.get_anscheck_prompt` 的类型 rubric。使用当前 Qwen3-8B 独立调用、温度 0、结构化布尔输出。模型与 Reader 相同，必须标为代理评分，不能称人工准确率或官方复现分数。
4. **失败与暂缓**：technical_error、语义暂缓、合法回答分别统计。不可回答题的行为单列；技术错误不能算正确拒答。
5. **端到端分母**：失败保留在总题数内；评分协议失败为未知，报告有效评分数及保守正确率下界，不能只报告有效评分中的高比例。

这些指标没有直接给出“新旧 Fact 召回率”“支持关系正确率”“更新误改率”或 ECE/Brier。后四类需要人工标注 Fact 对、证据角色和正确动作。

## 参数与评价边界

| 参数 | 值 |
|---|---|
| 随机种子 | 20261029 |
| 本轮问答数 | LoCoMo 14；LongMemEval 3 |
| LongMemEval 独立图 | 3 张；48/42/47 会话；完整 haystack |
| Reader/Judge/Grader | 当前配置的 qwen3-8b；评分温度 0 |
| 额外人为输出 Token 上限 | 本实验问答/评分不新增，显式 None；构图沿用现有调用参数，审计保留实际 finish_reason/Token |
| 人工候选对 | 默认最多 30，按 graph_key/link_id 去重，各图轮流取；优先覆盖纠正、替换、冲突、并存、补充等建议 |
| 每题候选对 | 默认最多 8，同建议类型最多 2；只导出原生证据 Session 附近候选 |
| 自动记忆失效/覆盖 | 不启用，沿用 append_only |

人工模板中的 suggested_relation 不是金标；gold_relation、gold_update_decision 和 supporting chunks 初始为空。导出会保留已有人工填写，不能把空值当成错误标签计算准确率。

## 运行与恢复

```powershell
python -m experiments.cross_dataset_pilot.run --output <新实验目录> --locomo-graph <完整来源的新图.pkl> --config <服务器模型配置.yaml> --implementation-commit <本次代码版本>
python -m experiments.cross_dataset_pilot.run --output <同一目录> --resume
```

准备阶段冻结 cases、配置、原图副本和源码 SHA256；执行阶段从 source_snapshot 重新启动，后续编辑项目代码不会改变运行中的实验。恢复时校验冻结选题/代码/语料，不隐式换题。原图和旧实验不覆盖。

## 输出

- `manifest.json`、`cases.json`：选题及输入/配置/源码指纹。
- `progress.json`、`run.log`、`exit_code.txt`：任务进度、日志、退出标记。
- `graphs/<case>/`：逐会话图检查点、构图审计、写入召回和关系轨迹、build_summary。
- `questions/<case>/report.json`：完整检索/Judge/回答/引用；`inference/` 与 `grading/` 分别保留模型调用。
- `results.json`、`metrics.json`：逐题对照与分数据集指标。
- `relation_annotations.json`：最多约 30 对人工标注材料；人工已填写行保留。
- `completion.json`：全部题已尝试及错误数。存在执行错误时进程非零退出；不能只看文件存在就称全部通过。

任务启动后不持续轮询。完成后再结合原文和 gold 审查错误，不能由小样本代理分数宣布 SOTA。
