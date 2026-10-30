# 项目结构与运行入口

本仓库同时保留原始 LiCoMemory 路径和事件状态记忆的研究实现。实际使用哪条路径由配置选择；实验脚本不等于正式入口。

## 当前主流程

| 目录或文件 | 职责 |
| --- | --- |
| `main.py`、`init/` | 加载配置、数据和模型，分发建图与查询。 |
| `chunking/`、`data/`、`dataset/` | 读取和转换数据，按对话或文档切分来源 Chunk。 |
| `coregraph/dynamic_memory.py`、`coregraph/graph_builder.py` | 增量构图、保存来源 Chunk、实体和关系索引。 |
| `coregraph/triple_fact_store.py`、`coregraph/lean_event_index.py` | 独立事实表与精简事件索引的持久化和恢复。 |
| `coregraph/event_state_writer.py` 及状态、时间相关模块 | 从原文抽取来源断言，召回历史并记录候选关系。 |
| `coregraph/evidence_support/`、`coregraph/evidence_judge.py` | 检索来源、审查证据、生成带引用的回答。 |
| `query/`、`evaluation/` | 原始检索、重排序、回答与评估；新旧配置的查询入口不同。 |
| `base/`、`prompt/`、`utils/` | 模型和向量接口、提示词以及统计辅助代码。 |
| `tests/` | 项目回归测试；测试中部分导入实验工具或兼容路径。 |

事件状态实验的配置示例是 `config/Memory_EventState_sample0_rebuild.yaml`：使用 `entity_fact_index`、`event_index` 和 `append_only`。该写入策略保存新断言和候选关联，不自动提交旧状态替换；证据服务的置信度目前为旁路分析，未经校准。

## 实验与结果

| 位置 | 用途 |
| --- | --- |
| `experiments/cross_dataset_pilot/` | LoCoMo / LongMemEval 小样本运行、来源覆盖统计与人工标注材料导出。 |
| `experiments/integrated_event_state/` | 事件状态相关的建图检查、诊断和旧查询兼容入口；部分文件被项目或测试直接调用。 |
| `experiments/evidence_support/` | 证据服务的独立运行、样本审查与关系重放。 |
| `experiments/event_state_evolution/` | 早期 SQLite 旁路原型，保留用于历史对照；不属于当前三层写入主路径。 |
| `experiments/reproduction/` | 原版结果复现与对照。 |
| `experiments/structural_semantic_retrieval/` | 旧导入路径的兼容层；算法实现位于 `coregraph/structural_semantic_retriever.py`。 |
| `results/` | 本地生成结果，不纳入 Git；目录内的模型原始输出、图、指标和复现材料应按实验核对。 |
| `docs/modules/`、`docs/reports/` | 分模块说明与实验报告。`plans/` 和 `superpowers/` 记录历史计划，不能替代当前源码。 |

详细的双数据集实验口径见 [cross_dataset_pilot.md](modules/cross_dataset_pilot.md)，证据服务见 [evidence_support.md](modules/evidence_support.md)。
