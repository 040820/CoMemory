# 写入式事件与状态演化层

这是与旧版 Session Summary **二选一**的上层写入模式。使用 `upper_memory_mode: event_state` 时，新会话仍先构建原始 Chunk 和实体关系图；随后立即从本次 Chunk、关系和已有状态判断持续事实的基线、变化或纠正。新上层随图一起保存于 `dynamic_memory_event_state.pkl`，而不是在建图后运行独立 SQLite 扫描。

## 三层对应

1. 下层 `chunk_storage`：完整原话、会话 ID、观察时间。
2. 中层 NetworkX 图与 `entity_name_to_index`：实体节点及关系边。
3. 上层 `event_state_layer`：事实基线、当前状态、变化/纠正事件及 `by_entity`、`by_chunk`、`by_triple` 反向索引。

事件记录含 `event_type`、`update_type`、`attribute`、`old_state`、`new_state`、`event_time`、`event_group_id`、旧新事实 ID、实体索引、Triple 快照和 Chunk ID。没有明确当前状态变化的普通活动不产生上层事件；首次状态只作基线。属性名由证据和已有状态决定，不限于雇主、职位、住处。模型返回的属性需要统一成简短英文标识；同义属性的自动归并尚未实现，正式数据应检查名称一致性。

## 使用

服务器需先启动 `config/Memory_EventState.yaml` 指向的 vLLM 和 Embedding 服务。新配置采用独立的 `index_name`；建议再用独立 `-root` 隔离整个实验目录：

```bash
python main.py -opt config/Memory_EventState.yaml -root event_state_group_1
```

若同时需要运行检索与原项目评测，在命令末尾加 `-query 1`。新模式的 `GraphRAG.query()` 会调用已有的 `StructuralSemanticRetriever`，保留 Query→Triple/Chunk 排名与 Top-5 原文证据；事件记录只按检索出的实体、Chunk 附加，不作为候选筛选门槛。旧模式仍由原 `QueryProcessor` 运行。结构化语义检索器不读取原 `CogniRank` 的 `rerank_k` 时间衰减参数。**不要传 `-query 0`**：当前 `main.py` 将非空字符串视为开启查询。

输出图为 `results/event_state_group_1/dynamic_memory_event_state.pkl`。其中 `event_state_layer` 与 `graph`、`chunk_storage`、`entity_name_to_index` 同级。`session_summaries.json` 在此模式下不会新生成；若在已有目录中发现旧文件，应以独立 `-root` 运行，避免混淆。原 `experiments/event_state_evolution/` 是早期离线旁路原型，新模式不依赖它。

建图后可只读检查事件与证据指针：

```bash
python experiments/integrated_event_state/inspect_index.py \
  --graph results/event_state_group_1/dynamic_memory_event_state.pkl \
  --output results/event_state_group_1/index_audit.json
```

## 增量更新与边界

`GraphRAGCore.insert()` 的逐会话增量路径使用相同写入器，在每次会话后连同图一起保存上层。构建完初始图后，增量运行须将配置中的 `graph.force` 改为 `False`、`graph.add` 改为 `True`，并沿用同一个 `-root`。旧格式 `.pkl` 仍能加载，但向旧图切换到新模式只会处理**之后**写入的会话，不会暗中重建历史事件；完整上层应在独立目录重新构建。纠正不会覆盖原始 Chunk；缺证据、身份歧义或模型结构错误进入上层 `review_queue`。目前实体身份继承中层图的名称合并结果；若中层已经把同名不同人合并，需先解决实体消歧。

本轮按用户要求先完成代码，未运行数据集实验或效果测试。不能将模块存在等同于 LoCoMo 上的变化抽取率或回答提升。
