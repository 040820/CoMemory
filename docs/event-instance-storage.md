# 事件实例上层存储结构

`dynamic_memory_event_state.pkl` 的主要字段为 `graph`、`chunk_storage`、
`entity_name_to_index`、`triple_storage`、`triple_storage_origin`、`event_state_layer`。旧图和原文
仍原样保存。以下 ID 和内容用于解释字段，并非一次新实验的输出。

```text
chunk_storage["1"]                      原始对话及观察时间
graph                                 event_state 的逐来源 MultiDiGraph（检索索引）
entity_name_to_index                  中层实体名称索引
triple_storage[chunk_id]              原始抽取关系出现记录（append-only）
event_state_layer
├── facts[F1]                         活动事实；含 event_id、chunk_id、Triple 快照
├── event_instances[E1]               一次活动的稳定身份和当前有效日期
├── event_time_assertions[T1]         日期断言、原表达、解析依据及证据
├── event_attribute_assertions[A1]    地点等可独立纠正的事件属性断言
├── event_instance_by_actor           actor → [event_id]
├── event_instance_by_chunk           chunk_id → [event_id]
├── event_instance_by_triple          Triple 来源快照键 → [event_id]
├── superseded_event_time_triples     已纠正旧日期关系的来源快照键
├── event_link_decisions              同一事件/不同事件的逐对判断记录
├── current、members、decisions         原持续状态槽及操作日志
└── events、by_entity/by_chunk/by_triple  原状态变化边及其索引
```

`event_state` 的 `graph` 使用 NetworkX `MultiDiGraph`：同一对实体可以有多种
关系，同一关系在不同 Chunk 也各保留一条边；边 key 为稳定 Triple ID。
原版模式保留 `DiGraph`。`triple_storage` 按 Chunk 保存每次抽取的原始关系
出现记录（`src`、`relation`、`tgt`、`chunk_id`、会话元数据），append-only，
不会被后续关系覆盖。上层事件层优先从这份记录获取关系与时间证据；旧 pkl
没有该字段时，加载阶段会从现存图边和旧 fact 快照做一次不完整的兜底恢复。
结构化语义检索优先读取标记为 extracted 的原始来源，避免旧图压缩造成关系串源。

`triple_storage_origin=extracted` 表示在图边合并前保存的原始出现记录；
`recovered_partial` 表示旧图的部分恢复，不能据此声称关系完整。Triple 的
稳定 ID 由 `(chunk_id, src, relation, tgt)` 生成，模型仅引用当前 Chunk 的 ID。
会话观察时间保存在 `session_time`，关系抽取器标注的发生时间另存 `event_time`。
事件实例的 `time_mentions` 保留时间范围或无法精确解析的原始表达；只有精确
日期进入 `event_time`。`skipped_candidates` 记录预期过滤的对话/附件行为，
`review_queue` 记录证据不足或处理失败。事件属性的 `evidence_basis` 区分
专门 Triple 与同一事件原文引文，空 Triple 来源不能当作有 Triple 支持。

开启 `state_semantic_normalization` 后，state Fact 另存开放的 `semantic_state`
比较表示及 source_* 原始字段，支持有证据的同义/正反表达对齐；当前值与变化
记录使用规范语义值，原始 value、quote 和 Triple 仍保留。新上层写入协议为
v3，已有 v2 上层需独立重建；关闭该功能仍保留旧协议。见 `docs/open-state-normalization.md`。

## 独立活动和同一事件的纠正

第一次活动形成 `E1`，活动事实 `F1` 指向 `E1`。事件实例保存 `actor`、
`action`、`target`、`fact_ids`、`evidence_chunk_ids`、`source_triples` 和
`current_time_assertion_id`。时间断言 `T1` 保存 ISO 日期、原文时间表达（例如
`yesterday`）、解析依据、观察时间、原文引文、Chunk ID 和时间 Triple 快照。
`E1.event_time` 是对外读取的当前值；未能确定时为 `null`。
`E1.current_attributes` 按属性名保存当前有效断言 ID，例如
`{"location": "A2"}`；`event_attribute_assertions` 保存每次的值、原文引文、
来源 Chunk、状态与被纠正断言 ID。属性名不固定为地点，其他可由原文支持的
事件属性也使用同一结构。

同一人物另一次参加活动会得到新的 `E2`。共享主体或动作只用来召回候选；
逐对判断必须确认同一次活动，不能自动建立替换关系。若后续原文明确说同一次活动
是 May 6 而不是 May 7，旧 `T1` 标记 `corrected`，新增指向 `T1` 的 `T2`，
`E1.current_time_assertion_id` 改指 `T2`。原始 Chunk、旧断言和旧 Triple 快照
不删除；新查询入口依提问时间判断旧日期关系是否仍可见。
地点等普通属性只更新对应的 `current_attributes` 指针，不改变日期或其他属性。
查询证据组装会按提问时间选择有效断言；若旧属性有可识别的 Triple 来源，
纠正后也会从当前查询证据中隐藏该旧关系。原始 Chunk 和断言历史仍保留。

Triple 引用以主体、关系、客体和来源 Chunk 共同计算稳定键。这样即使中层
`DiGraph` 的同一对实体关系后来被覆盖，上层仍保留当时的关系快照与来源。

## 时间依据及边界

- `explicit_date`：原文给出完整日期。
- `explicit_month_day_observed_year`：原文给出月日，年份取观察时间，
  但不会擅自跨年。
- `relative_observed`：`yesterday`、`today` 或 `last Friday` 相对会话时间解析。
- `middle_triple`：同一 Chunk 中存在唯一且能关联当前活动的日期 Triple。

无法唯一归属的日期不写为确定日期。中层图和原始 Chunk 不会被物理改写；
现有结构化语义检索仍按原规则排序 Chunk，纠正覆盖在答案证据组装时生效。
旧版已存的普通事实不会自动生成事件实例，完整结构需在独立目录重建。
