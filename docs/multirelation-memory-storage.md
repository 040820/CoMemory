# 一条记忆从输入到三层存储

本文按 event_state 模式的实际结构说明。示例中的 C1、T1、F1 为便于阅读的短 ID；实际 Triple/Fact ID 由程序生成。示例没有调用 LLM，不代表一次实际抽取结果。

## 输入与写入顺序

会话 D1：`User: 我对牛奶过敏。`

1. 对话分块后保存原话及会话时间。
2. LLM 抽取实体和关系，关系端点归一化到实体索引。
3. 关系合并前保存逐 Chunk 的原始 Triple；同一实体对可以有多条关系出现边。
4. LLM 提出上层事件或状态候选，程序核对 quote、主体及 Triple 来源，再召回少量旧事实决定操作。
5. 按会话将三层一起保存到 `dynamic_memory_event_state.pkl`。

## 下层：原始 Chunk

```json
{
  "chunk_id": "C1",
  "session_id": "D1",
  "session_time": "2025-01-01",
  "text": "User: 我对牛奶过敏。"
}
```

位于 `chunk_storage["C1"]`。这是原始证据，不会因为后续状态变化被改写。

## 中层：实体和逐来源关系

实体节点：`User`、`牛奶`；`entity_name_to_index` 分配实体编号。Triple 示例：

```json
{
  "triple_id": "T1",
  "src": "User",
  "relation": "allergic_to",
  "tgt": "牛奶",
  "chunk_id": "C1",
  "session_id": "D1",
  "session_time": "2025-01-01"
}
```

这份记录保存在 `triple_storage["C1"]`。同时，中层 `graph` 保存一条 MultiDiGraph 边：

```text
User --[key=T1, relation_name=allergic_to, chunk_id=C1]--> 牛奶
```

实际边属性采用 `src_id/tgt_id/relation_name`，Triple 原始记录采用 `src/tgt/relation`。
关系发生时间若能抽取则另存 `event_time`，不会覆盖会话观察时间 `session_time`。

## 上层：有证据的事实、当前值和演化

上层候选通过原文核对后，可以形成如下 Fact（省略部分管理字段）：

```json
{
  "fact_id": "F1",
  "entity_id": "User",
  "entity_index": 0,
  "attribute": "allergic_to",
  "value": "牛奶",
  "fact_type": "state",
  "kind": "add",
  "status": "active",
  "scope": {},
  "modality": "actual",
  "chunk_id": "C1",
  "session_id": "D1",
  "observed_at": "2025-01-01",
  "event_time": null,
  "quote": "我对牛奶过敏。",
  "triple_refs": [{"src": "User", "relation": "allergic_to", "tgt": "牛奶", "chunk_id": "C1", "triple_id": "T1"}]
}
```

位于 `event_state_layer.facts["F1"]`。上层还包含：

- `current`：已识别单值状态槽到当前 Fact ID 的映射。
- `members`：多值槽到有效 Fact ID 集合的映射。
- `events`：经过门控的 REPLACE/CORRECT 变化记录，保存 old/new Fact 和证据。
- `event_instances`：一次现实活动的 actor/action/target 和证据锚点。
- `event_time_assertions/event_attribute_assertions`：活动的日期、地点等可独立补充/纠正的断言。
- `by_entity/by_chunk/by_triple` 等反向索引：用于追溯。
- `review_queue/skipped_candidates`：证据不足的候选及预期过滤项。

“过敏”是持续事实，不会因此额外创建一个“参加活动”式的 Event 实例。

## 后来又说“现在我对牛奶不过敏了”

下层新增 C2；中层新增 T2。例如抽取结果为 `User --not_allergic_to--> 牛奶`，则图中同时保留：

```text
User --[T1: allergic_to,     C1, D1]--> 牛奶
User --[T2: not_allergic_to, C2, D2]--> 牛奶
```

仍然只有两个实体节点，但两条不同关系有独立来源。若同一关系在 C3 再次出现，也会新增独立的 T3；重复提交同一 Chunk 的相同 Triple 不会无限加边。

本次中层修改保证两条关系不覆盖、不串源。上层是否形成 `F1 → F2`，仍需把二者识别为同一状态槽并通过变化证据门控。**当前过敏属性尚未加入专门的单值槽规范，开放属性默认 set；不能承诺这个示例已自动更新 current 或产生 REPLACE。** 新旧谓词或 actual/negated 模态不同，也可能导致分槽。原文始终保留，后续需要完善的是上层语义归一化，而不是再合并掉图中的事实。

## 持久化及兼容

同一个 pickle 的主要字段为：

```text
dynamic_memory_event_state.pkl
├── chunk_storage               C1、C2 原文
├── entity_name_to_index        User、牛奶的实体编号
├── triple_storage              T1、T2 的独立出现记录
├── triple_storage_origin       extracted / recovered_partial
├── graph                       MultiDiGraph，T1/T2 为不同边
└── event_state_layer           Fact、Event、断言、当前状态与反向索引
```

event_state 加载旧 DiGraph 时，只在内存中适配：有 extracted 原始记录可重建完整平行边；来源不完整时仅保留旧文件实际存在的边，标记 legacy_partial。加载过程不改写旧文件。原版 session_summary 模式仍使用 DiGraph，不能静默把新多边图压缩给 Baseline。

本轮按用户要求不再追加验证。提出该要求前已启动的 8 项结构回归结束并通过；没有据此声称已完成全量 sample0 或 LLM 语义验证。此前启动的 sample0 任务不受本轮代码同步影响，其启动版本不代表本次新图结构。
