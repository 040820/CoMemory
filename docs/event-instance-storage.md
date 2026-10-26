# 事件实例上层存储结构

`dynamic_memory_event_state.pkl` 仍有四个同级字段：`graph`、`chunk_storage`、
`entity_name_to_index`、`event_state_layer`。本次仅扩展最后一个字段；旧图和原文
仍原样保存。以下 ID 和内容用于解释字段，并非一次新实验的输出。

```text
chunk_storage["1"]                      原始对话及观察时间
graph                                 原有实体关系 DiGraph
entity_name_to_index                  中层实体名称索引
event_state_layer
├── facts[F1]                         活动事实；含 event_id、chunk_id、Triple 快照
├── event_instances[E1]               一次活动的稳定身份和当前有效日期
├── event_time_assertions[T1]         日期断言、原表达、解析依据及证据
├── event_instance_by_actor           actor → [event_id]
├── event_instance_by_chunk           chunk_id → [event_id]
├── event_instance_by_triple          Triple 来源快照键 → [event_id]
├── superseded_event_time_triples     已纠正旧日期关系的来源快照键
├── event_link_decisions              同一事件/不同事件的逐对判断记录
├── current、members、decisions         原持续状态槽及操作日志
└── events、by_entity/by_chunk/by_triple  原状态变化边及其索引
```

## 独立活动和同一事件的纠正

第一次活动形成 `E1`，活动事实 `F1` 指向 `E1`。事件实例保存 `actor`、
`action`、`target`、`fact_ids`、`evidence_chunk_ids`、`source_triples` 和
`current_time_assertion_id`。时间断言 `T1` 保存 ISO 日期、原文时间表达（例如
`yesterday`）、解析依据、观察时间、原文引文、Chunk ID 和时间 Triple 快照。
`E1.event_time` 是对外读取的当前值；未能确定时为 `null`。

同一人物另一次参加活动会得到新的 `E2`。共享主体或动作只用来召回候选；
逐对判断必须确认同一次活动，不能自动建立替换关系。若后续原文明确说同一次活动
是 May 6 而不是 May 7，旧 `T1` 标记 `corrected`，新增指向 `T1` 的 `T2`，
`E1.current_time_assertion_id` 改指 `T2`。原始 Chunk、旧断言和旧 Triple 快照
不删除；新查询入口依提问时间判断旧日期关系是否仍可见。

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
