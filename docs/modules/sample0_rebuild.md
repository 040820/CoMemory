# sample0 原始对话重建与证据测试流程说明

本流程服务于“修复 Chunk 原始顺序与紧凑生成后，从原始数据重新测试”的实验。旧图不作为新写入的来源。

## 涉及代码与职责

| 代码 | 作用 |
|---|---|
| `experiments/integrated_event_state/launch_sample0_rebuild.py` | 从冻结源代码启动原始会话构图，保存模型审计、会话检查点及来源引用检查 |
| `coregraph/event_state_writer.py` | 按会话时间及 source_chunk_id 顺序提交，哈希 Chunk ID 只用于唯一引用 |
| `coregraph/evidence_support/transport.py` | 真正限制任意空白的 EBNF 约束，以及完整原始响应日志 |
| `experiments/evidence_support/review_sample.py` | 构图成功后，逐对审查新图的全部待审查关联，不覆盖记忆 |

## 数据与执行阶段

sample0 为 LoCoMo conv-26，即 group_1，Caroline/Melanie 共 19 个 Session。复制原始 Corpus.json/Question.json 到新实验目录并记录哈希；代码与配置也冻结在 reproducibility 中。

1. 验证编译器、正式模型传输、原始数据会话数及代码可导入。
2. 在全新输出目录依次写入 Session；每会话保存检查点，完全不加载旧图。
3. 检查 19 会话覆盖、处理过的 Chunk 集合、原始 Triple 来源及跨层引用。
4. 仅在构图检查成功后自动执行新图的全部 pending Fact 对审查。
5. 写入 build_exit_code.txt、review_exit_code.txt、exit_code.txt 和完成时间；失败停止后续阶段，日志保留。

## 参数和功能边界

构图沿用 Memory_EventState_sample0_rebuild.yaml 的三层结构、append_only 与写入期双路候选预算；不自动改变抽取或状态归一化规则。LICOMEMORY_COMPACT_JSON=1 只把结构化请求转换为同一 Schema 的紧凑语法，不改抽取与证据校验条件。

证据测试采用 config/evidence_support.json，默认结构/语义各 12、BFS 2 跳、最多 64 个图状态、RRF 常数 60、最多 12 个来源、Judge 最多 3 轮。Judge/默认回答不发送人为输出 Token 上限，仍受模型服务物理上下文限制。细节见 evidence_support.md。

此任务验证真实构图和候选关系的取证流程；没有人工关系标签，不报告关系准确率，不等同于完整问答评测，也不自动采纳更新建议。

## 输出位置

本次任务目录：`results/locomo_sample0_fresh_20261029/`。

- run.log：整个构图及后续审查的标准输出和错误日志，直接放在任务目录内。
- dynamic_memory_event_state.pkl：本次从原始对话重新生成的图。
- progress.json / build_summary.json / inspection.json：逐会话进度、构图统计与引用完整性。
- llm_trace.jsonl：建图模型请求与原始响应、截断和空白审计。
- evidence_review/：新图逐对审查的报告、模型审计、语法和总进度。
- reproducibility/：冻结代码、原始输入、实际 YAML、代码/数据哈希和实验清单。
- cleanup_manifest.json：旧 sample0 图文件的绝对路径、大小和删除结果。

旧图清理只作用于 results 中已确认属于旧 sample0 的 .pkl 图/图检查点（包括复制的图），不删除原始数据、代码、模型、JSON/Markdown 报告与历史日志。新目录不进入清理范围。启动后不轮询，用户报告结束后读取退出码和结果。
