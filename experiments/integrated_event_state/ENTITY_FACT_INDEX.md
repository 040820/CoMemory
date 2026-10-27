# 当前三层结构：简单实体索引＋独立事实表

## 借鉴 HippoRAG 的部分

已阅读本地 `D:/Code/HippoRAG-main/src/hipporag/HippoRAG.py`：`index` 将完整事实单独向量化，`add_fact_edges` 累积实体对连接，`add_passage_edges` 连接实体与段落。这里借鉴事实与图索引分离，不复制其 PageRank，不更换我们的 Chunk 排序算法，也不要求新增图数据库。

## 三层职责

| 层级 | 持久化内容 | 职责 |
|---|---|---|
| 底层 | `chunk_storage` | 原话、会话、观察时间、稳定来源 ID |
| 中层 | `fact_storage`＋`graph` | 完整事实独立保存；实体关联图只帮助找事实 |
| 上层 | `event_state_layer` | 来源断言、事件观察锚点、候选演化联系；通过中层事实 ID 追溯证据 |

同一对实体在图中只保留一条无向 `entity_association` 边：

```text
User ── fact_ids=[T1,T2,T3] ── Milk

T1: User --allergic_to--> Milk       / Chunk C1 / 时间 t1
T2: User --not_allergic_to--> Milk   / Chunk C2 / 时间 t2
T3: Milk --affects--> User          / Chunk C3 / 时间 t3
```

边不保存某个 `relation_name`，也不保存某一条事实的向量。无向关联不意味着事实对称；实际主客方向、谓词、时间和来源必须从 `fact_storage` 读取。weight 是不同来源事实数，不是证据可信度。

## 独立事实记录

事实字段为 `fact_id/triple_id`、`src/relation/tgt`、`chunk_id`、会话/时间元数据，以及可选 description、strength、embedding。ID 沿用原 `triple_id(chunk_id,src,relation,tgt)`：同来源同 Triple 幂等，不同 Chunk 的相同事实也分别保留。它是来源出现记录，不代表现实真值已经确定。

`TripleFactStore.by_chunk/by_entity` 为可重建的内存索引。新文件不再重复保存整个 `triple_storage`；加载时提供共享主表记录的兼容视图。

增量写入发现相同局部 Chunk ID 对应不同会话或不同文本时，生成稳定的新全局 ID，并保留 `source_chunk_id`；后续抽取、事实和事件都使用新 ID，避免跨会话覆盖原文。

## 上层如何引用

上层自身的断言 ID、断言内容、引文和候选关系保留。中层 Triple 证据在落盘时编码为 ID：

| 旧运行时字段 | 新文件字段 |
|---|---|
| `triple_refs` | `middle_fact_ids` |
| `source_triples` | `source_middle_fact_ids` |
| `event_triple_refs` | `event_middle_fact_ids` |
| `time_triple_refs` | `time_middle_fact_ids` |

这些 ID 都引用唯一 `fact_storage.facts`。加载后恢复轻量来源引用，供已有 Writer/Query 使用；中层向量不复制进每条上层引用。上层断言可能由多个 Triple 共同支持，所以保留上层断言 ID 和中层 Triple ID 两种身份。

写入仍是 `append_only`：不调用状态替换、纠正失效或事件身份合并，不自动维护唯一 current。结构迁移不修复已有的分类/归一化错误，也不意味着证据链推理已经实现。

## 配置与操作

```yaml
upper_memory_mode: event_state
middle_memory_mode: entity_fact_index
state_write_policy: append_only
```

两份 `Memory_EventState*.yaml` 已启用。旧配置缺省 `middle_memory_mode=legacy`，原版 DiGraph 和旧 event_state MultiDiGraph 逻辑保留。新索引文件不能混用旧查询入口。

显式迁移已有图，输出必须是新文件：

```bash
python experiments/integrated_event_state/migrate_entity_fact_index.py \
  --source results/旧目录/dynamic_memory_event_state.pkl \
  --output results/新目录/dynamic_memory_event_state.pkl
```

这不会调用 LLM。旧图没有完整来源时输出仍标注 `recovered_partial`，不能找回历史上被覆盖的关系。要验证完整新构图，仍需以后从原始对话建立全新的图。

只读检查：

```bash
python experiments/integrated_event_state/inspect_index.py \
  --graph results/新目录/dynamic_memory_event_state.pkl \
  --output results/新目录/inspection.json
```

## 模块职责

- `coregraph/triple_fact_store.py`：事实主表、反向索引、上层 ID 引用编码。
- `coregraph/graph_builder.py`：构造简单实体关联索引，保留旧图模式。
- `coregraph/dynamic_memory.py`：Chunk 防覆盖、事实存储、逐事实向量和三层保存/加载。
- `experiments/structural_semantic_retrieval/retriever.py`：从事实表读取完整语义，适配无向邻接扩展。
- `replay_upper_from_graph.py`：回放来源表，最终结果真实采用新中层，不只替换上层。
- `migrate_entity_fact_index.py`：来源只读的格式迁移，不能代替重建或效果评测。
