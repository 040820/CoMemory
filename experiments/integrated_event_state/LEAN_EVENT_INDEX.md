# 精简三层持久化（event_index_v1）

仅在 `middle_memory_mode: entity_fact_index`、`state_write_policy: append_only`、
`upper_memory_layout: event_index` 同时启用时使用。默认与旧实验格式不变。

| 层次 | 实际文件字段 | 职责 |
| --- | --- | --- |
| Chunk | `chunk_storage` | 原始对话、会话、时间、来源 ID |
| 中层 | `fact_storage.facts` | 不同方向、关系、Chunk 的原始 Triple 独立保存 |
| 中层 | `fact_storage.observations` | 有来源支持的事件/状态断言、必要时间、模态、作用域、规范化语义和 Triple ID |
| 中层 | `fact_storage.time_assertions` / `attribute_assertions` | 绑定具体事件的来源时间/属性断言，保留证据 ID |
| 上层 | `event_state_layer.groups` | `event_id + fact_ids`，把同一来源锚点的断言组织起来 |
| 上层 | `event_state_layer.links` | 两个中层断言的可能关联，统一标记 candidate |

中层 Observation 是原始 Triple 之上的来源解释，保留区间时间、模态等原 Triple
没有表达的内容；不是另一份已裁决现实状态。普通状态断言不额外建立事件分组。
上层不再重复保存属性值、引文、来源快照、归一化诊断或核查队列。

示例：`E1 → [F1, A1]`；F1 在中层记录 Caroline 参加 support group，
A1 在中层记录本次活动的日期，两者各自引用原始 Triple/Chunk。
如果另一条来源纠正日期，只追加新断言与可能纠正链接，保留旧记录。
事件分组是来源锚点，不能据此宣称不同 Chunk 的事件身份已经确认。

`ingestion_state` 保存处理过的 Chunk、协议版本、审计计数和审计路径，是运行检查点。
审计快照位于 `<图文件名>_audit/<内容哈希>.json`，保留全部写入检索结果、
关系模型输出、失败与核查原因。每次会话保存独立快照，避免中断破坏旧检查点。
移动或归档图时必须一起保留该审计目录。主图外保留审计是为了可复现，不代表图节点。

运行时通过中层生成旧接口兼容视图，因此抽取器、双路历史召回和已修改的 Chunk
检索主流程沿用现有逻辑。保存/加载检查不能证明语义准确率提升。
无关和 uncertain 判断只留在审计；查询时也不占用候选关联预算。
写入阶段不推导因果、不确认完整证据链、不使旧断言失效。
