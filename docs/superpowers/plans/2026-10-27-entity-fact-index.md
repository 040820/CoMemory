# Entity–Fact Index 图结构修改计划

用户已同意保留三个逻辑层，借鉴本地 HippoRAG 的“完整事实独立存储、图连接用于检索”实现。当前工作区有用户尚未提交的修改；在现有目录修改相关干净文件，提交时排除这些已有改动，不另建或切换工作树。

## 参考与设计

本地 `D:/Code/HippoRAG-main/src/hipporag/HippoRAG.py`：`index` 独立向量化 facts；`add_fact_edges` 只累积实体对连接；`add_passage_edges` 维护实体与段落联系。我们借鉴数据职责，不复制其 PPR 或更换已有问答排序。

- Chunk 层：完整原文；新模式增量写入检测局部 Chunk ID 冲突，产生稳定新 ID，禁止覆盖历史。
- 中层：独立逐来源事实表，ID 兼容原有 triple_id；无向实体关联索引一对实体最多一条边，边仅保存 fact_ids。谓词、方向、时间、来源、向量全部保存在事实表。
- 上层：继续追加来源断言和候选关系，落盘时把 Triple 快照编码成中层事实 ID，加载时解析；不在写入期裁决真假。
- 新配置显式 `middle_memory_mode: entity_fact_index`。旧 session_summary 和旧 event_state 配置缺省逻辑保留。
- 查询优先从事实表读取完整关系，邻接扩展适配无向图，不将 association 当作事实谓词。
- 迁移在新输出文件完成，标注 recovered_partial，不恢复或臆测已丢失事实。

## 实施与检查

1. 新增 `coregraph/triple_fact_store.py`：稳定 ID、原子 upsert、按实体/Chunk 反查、持久化与向量归属。
2. 接入 GraphBuilder、DynamicMemory 的构建、增量、保存/加载及 Chunk ID 防覆盖。
3. 接入上层 ID 引用编码、结构化语义查询和 replay/inspect 工具。
4. 新增无 LLM 的关键检查：不同谓词/方向/来源一边多事实，增量幂等，Chunk 冲突，向量不串写，保存/加载与证据回溯，旧模式不变。
5. 提供独立迁移入口和中文结构说明；提交 GitHub，同步服务器。不自动重跑 sample0 或启动完整 LLM 构图。
