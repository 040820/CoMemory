# 写入式事件与状态演化层

这是与旧版 Session Summary **二选一**的上层写入模式。使用 `upper_memory_mode: event_state` 时，新会话仍先构建原始 Chunk 和实体关系图；随后立即从本次 Chunk、关系和已有状态判断持续事实的基线、变化或纠正。新上层随图一起保存于 `dynamic_memory_event_state.pkl`，而不是在建图后运行独立 SQLite 扫描。

## 三层对应

1. 下层 `chunk_storage`：完整原话、会话 ID、观察时间。
2. 中层 NetworkX 图与 `entity_name_to_index`：实体节点及关系边。
3. 上层 `event_state_layer`：持续状态事实与变化边，以及独立的历史活动实例 `event_instances`、日期断言 `event_time_assertions` 和来源索引。

`events` 继续表示持续状态的变化/纠正边；普通历史活动由 `event_instances` 表示，不再与状态替换混淆。事件实例绑定活动事实、原始 Chunk 与本次抽取的 Triple 快照。日期、地点等事件属性另存为可独立纠正的断言；纠正会改变对外有效值，但保留旧断言与原话。具体字段见 [事件实例存储结构](../../docs/event-instance-storage.md)。属性名由证据和已有状态决定，不限于雇主、职位、住处。模型返回的属性需要统一成简短英文标识；同义属性的自动归并尚未实现，正式数据应检查名称一致性。

## 使用

服务器需先启动 `config/Memory_EventState.yaml` 指向的 vLLM 和 Embedding 服务。新配置采用独立的 `index_name`；建议再用独立 `-root` 隔离整个实验目录：

```bash
python main.py -opt config/Memory_EventState.yaml -root event_state_group_1
```

若同时需要运行检索与原项目评测，在命令末尾加 `-query 1`。新模式的 `GraphRAG.query()` 会调用已有的 `StructuralSemanticRetriever`，保留 Query→Triple/Chunk 排名与 Top-5 原文证据；事件实例只按检索出的实体、Chunk 附加，不作为候选筛选门槛。日期被明确纠正后，查询组装阶段按提问时间过滤失效的日期 Triple，并补入该事件的有效日期；原始 Chunk 始终保留。旧模式仍由原 `QueryProcessor` 运行。结构化语义检索器不读取原 `CogniRank` 的 `rerank_k` 时间衰减参数。**不要传 `-query 0`**：当前 `main.py` 将非空字符串视为开启查询。

输出图为 `results/event_state_group_1/dynamic_memory_event_state.pkl`。其中 `event_state_layer` 与 `graph`、`chunk_storage`、`entity_name_to_index`、`triple_storage` 同级。`triple_storage` 按 Chunk 保存原始关系出现记录；新模式的 `graph` 使用 `MultiDiGraph`，每条出现记录以稳定 Triple ID 作边 key，同一实体对的不同关系和同一关系的不同来源均保留。结构化语义检索优先使用 extracted 原始来源。旧 pkl 只有完整的原始记录才能补回被 DiGraph 压缩的关系，否则仍是部分恢复。`session_summaries.json` 在此模式下不会新生成；请使用独立 `-root`。原 `experiments/event_state_evolution/` 是早期离线旁路原型。三层字段示例见 `docs/multirelation-memory-storage.md`。

建图后可只读检查事件与证据指针：

```bash
python experiments/integrated_event_state/inspect_index.py \
  --graph results/event_state_group_1/dynamic_memory_event_state.pkl \
  --output results/event_state_group_1/index_audit.json
```

## 增量更新与边界

`GraphRAGCore.insert()` 的逐会话增量路径使用相同写入器，在每次会话后连同图一起保存上层。构建完初始图后，增量运行须将配置中的 `graph.force` 改为 `False`、`graph.add` 改为 `True`，并沿用同一个 `-root`。旧格式 `.pkl` 仍能加载，但向旧图切换到新模式只会处理**之后**写入的会话，不会暗中重建历史事件；完整上层应在独立目录重新构建。纠正不会覆盖原始 Chunk；缺证据、身份歧义或模型结构错误进入上层 `review_queue`。目前实体身份继承中层图的名称合并结果；若中层已经把同名不同人合并，需先解决实体消歧。

事件实例锚点已完成[六段对话的小样本验证](reports/event_instance_probe_2026-10-26.md)，但尚未运行完整 LoCoMo 实验或回答效果评测。不能把该小样本结果等同于全数据集的事件归并率、纠正准确率或回答提升。
