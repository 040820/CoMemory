# 新图的历史召回与证据补查实验

输入必须是 `event_index_v1` 的 sample0 图。固定图中已抽取的来源断言，不重新
运行抽取或归一化，不修改旧图、原始检索模块或现有实验结果。

1. 对每条状态事实，分别运行 K=5、10、20 的 legacy、StateSlot、BGE-M3、双路召回。
2. 按原写入位置回放，只允许此前 Chunk 的历史；同一 Chunk 的断言不互相充当历史。
3. 将双路 K=5 的结果作为证据初始集合，用现有 Judge 检查缺口并有限补查。
   补查复用结构化语义 Chunk 检索、来源关联和同 Session 相邻 Chunk。
4. 每条状态最多 3 次 Judge 调用、每次补查 Top-5、最多 12 个证据 Chunk。
   Judge 仍采用现有输入窗口预算，保存未展示和截断记录。

本实验只验证状态类历史候选和模型的证据补查行为；事件类身份识别、完整写入
抽取质量和问答准确率不在本轮范围。K 增大可能同时引入噪声，不预设增大更好。
三组 K 的两路候选预算与最终预算均为 K，不代表已经单独控制候选池大小的消融。

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python \
  experiments/integrated_event_state/test_recall_sufficiency_suite.py \
  --source /绝对路径/dynamic_memory_event_state.pkl \
  --config config/Memory_EventState_sample0_rebuild.yaml \
  --output /绝对路径/全新实验目录
```

## 结果怎么看

- `suite_manifest.json`：来源哈希、预算和测试边界。
- `k5/`、`k10/`、`k20/`：各路候选、原文、历史池、逐条对照及代码哈希。
- `k_candidate_changes.jsonl`：扩大 K 后新增或丢失的候选 ID。
- `adaptive_results.jsonl`：首轮与最终 Judge 状态、调用次数、证据数、停止原因。
- `adaptive/<新FactID>/judge_result.json`：补查问题、命中来源、每轮判断和原文引用。
- `adaptive/<新FactID>/llm_trace.jsonl`：实际提示词、输出和响应元数据。
- `suite_complete.json`：完成标记与来源图未修改检查。

`manual_labels.jsonl` 只是待人工标注模板；提供完整可比较历史 ID，以检查未被
任何方法召回的事实。没有人工 gold，不把候选数量、Judge 的 SUFFICIENT 比例
或 LLM 自述当作 Precision、Recall@K、正确概率或已经确认的记忆更新。

实验构造受限历史图时，过滤未来 Chunk、事实、分组、链接和实体关联边；全图
不会先参与评分再过滤返回结果。来源分组和候选关联只是补查线索，不作为证明。

## 历史顺序修复与续跑

现有 Writer 在同 Session 内按 Chunk ID 排序，哈希 ID 的顺序与 Chunk 字典
插入顺序、原始 source_chunk_id 顺序可能不同。本实验复现的是已有写入历史，
补查范围必须采用 Writer 的排序；不能拿原始对话顺序混用已保存的写入位置。
因此本轮的“未来”指相对已有 Writer 的处理顺序，并非保证原对话轮次顺序。

首次补查因两种顺序不一致触发 AssertionError；旧日志和部分 Judge 输出保留，
不作为修复后结果。固定 K 召回若已完整完成，可传入 `--reuse-recall-from 旧full目录`，
校验来源、配置、检索依赖哈希后复制到全新输出目录。全部补查重新运行，不复用
可能受到旧历史范围影响的 Judge 判断；在调用模型前核验全部状态的历史范围。
