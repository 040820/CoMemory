# 当前上层写入：断言保留，解释延后

## 当前流程

新会话 → Chunk/Triple 来源校验 → 少量历史召回 → 可选状态语义归一化 → 独立保存新断言 → 逐对提出候选关联 → 保存原文、Triple、Chunk 与关系轨迹。

原文引用有效，只能证明“来源里确实这样说”，不能证明现实真值，也不能证明新旧说法之间的因果、纠正、替换或事件身份关系。因此当前模式不调用状态转换门控，不使旧事实失效，不维护唯一当前值。时间解析仍保留相对时间的基准和范围，时间先后不作为写入裁决依据。

## 三层保存什么

| 层级 | 内容及作用 |
|---|---|
| Chunk | 原始对话、会话、时间；继续保留来源证据 |
| Triple | 原始逐 Chunk 关系及多关系图；继续关联实体索引 |
| 上层 | 每条来源断言、候选新旧关联、活动观察锚点及其时间/属性断言 |

同一事实在不同 Chunk 重复出现，仍保存独立来源断言；同一 Chunk 的同来源重跑使用确定性 ID 去重。StateSlot 单值/多值标签仅辅助召回与后续分析，不导致拒绝保存不同新值。

## 关键字段

- Fact：`operation=APPEND`、`status=observed`、`assertion_status=source_grounded`、`validity_status=unresolved`。来源与归一化视图并存，归一化不能覆盖原始表达。
- 关系：`status=candidate`、`outcome=recorded_only`、`inference_status=not_evaluated`、`applied_to_old=false`，包含新旧 Fact ID、候选关系类型、原文、Triple 和召回轨迹。`replacement` 等类型只表示模型建议的“可能变化”，不是执行指令。
- `current`、`members`、已确认变化 `events`：新模式不自动填充。全版本断言通过 `facts_by_slot`、`facts_by_subject` 找到。
- 纠正意图：保存 `asserted_intent`、`proposed_target_fact_id`，不写成已执行的 CORRECT。
- 活动：每次来源建立独立观察锚点；可能是同一事件的联系也只作为候选。不合并事件身份，不撤销旧日期/属性。`observed_time_assertion_ids` 和 `observed_attribute_assertion_ids` 保留该来源的断言，`current_*` 指针为空。

## 代码职责

- `event_state_writer.py`：按策略分流；新路径保存新断言和候选关系后返回，不进入历史替换/纠正路径。
- `event_state_layer.py`：`append_observation()` 只追加；`record_fact_relation()` 强制候选状态；旧裁决接口在 append_only 模式下显式拒绝调用。
- `hybrid_state_retriever.py`：A/B 继续独立找历史线索；`observed` 断言可参与召回，召回命中不代表证据已证明结论。
- `state_fact_relation.py`：逐对提出可能关系，不判断最终有效性。
- `observation_query.py`：查询适配层保留原 Chunk 排名，展示有限来源断言和候选端点；不把最近断言显示为当前有效状态。**本轮没有实现完整证据链推理器或证据置信度模型。**

## 配置与兼容性

```yaml
state_write_policy: append_only
state_recall_mode: hybrid
state_semantic_normalization: true
state_structured_top_k: 5
state_semantic_top_k: 5
state_candidate_top_k: 5
```

当前两份 Event–State 配置已启用此策略，正式构建、上层回放和小样本入口都传递该字段。缺省 `gated` 仅为保留旧配置与 Baseline 的历史逻辑，不代表当前实验配置。

新写入协议为 v4。请用新目录从中层来源重建上层；非空 v2/v3 上层不能直接混写。既有图、实验结果和报告不修改，也不会自动把过去已失效的事实恢复为新模式结果。

按用户要求，本轮只修改代码，不运行测试、LLM 或 sample0。
