# 分项置信度 v1：记录、局部评价与解释

本模块实现已讨论的三类评价，不改变三层图的职责，不自动裁决现实真值。

| 对象 | 存储位置 | 字段 |
| --- | --- | --- |
| 中层原始 Triple / Observation / Assertion | 原中层记录 | `extraction_confidence`、`extraction_evaluation_id` |
| 断言与 Chunk 的证据关联 | `confidence_storage.evidence_links` | `fact_id`、`chunk_id`、`role`、`score`、`span`、`independence_group` |
| 事实抽取评价详情 | `confidence_storage.fact_evaluations` | `score`、`reason`、`method`、`model`、`calibration`、`status` |
| 新旧事实关系评价 | `confidence_storage.relation_evaluations` | 两端 Fact ID、分项检查、局部证据、分数、理由和评价状态 |
| 上层候选 Link | 原上层链接 | 只增加 `evaluation_id`，不堆放评价正文 |

`score: null` 表示未评估、失效、失败或无法给出有效分数，不能当作 0。
模型输出均为未校准估计 `calibration: uncalibrated`；不能解释为事实正确概率。
支持方向 `supports/refutes/insufficient` 和分数分开；Evidence 的分数评价方向判断，
不是来源现实可信度。原话被准确抽取，不代表原话必然真实。

关系评价检查同主体、同一次事件或同一状态槽、时间可比较性、关系依据和所给反证。
纠正/替换等关键条件未知时，即使模型输出高分，也将 verdict 降为 uncertain。
状态始终 `validation_status: not_verified`，范围为 `pair_local`；“支持候选”不是确认关系。
不会自动覆盖旧事实、合并事件、生成因果链或把最新说法当作当前有效值。

同一 Chunk 派生的多个证据共享 `independence_group`，为以后去重留依据；当前不做
多证据独立性推断或分数融合。实体/槽位归一化尚无独立评分器，关系检查不能替代它。
模型原始响应及元数据留在 trace；源内容变化时旧分数失效，旧评价保留在 history。

## 使用

既有构图默认不启用；新实验配置可设置 `memory_confidence_enabled: true`，
需要同时启用 `upper_memory_layout: event_index`。保存时只初始化记录，不调用额外模型。
新建构图实验可使用 `config/Memory_EventState_confidence.yaml`（独立 index_name，必须配合全新 -root）。
正在运行的冻结 sample0 不修改配置、不重新启动。

完成构图后，也可以在副本中只初始化记录（不调用 LLM）：

```bash
python experiments/integrated_event_state/assess_confidence.py \
  --source /绝对路径/dynamic_memory_event_state.pkl \
  --output /绝对路径/全新的置信度实验目录
```

有限调用本地 vLLM 评价指定中层和上层 ID：

```bash
python experiments/integrated_event_state/assess_confidence.py \
  --source /绝对路径/dynamic_memory_event_state.pkl \
  --output /绝对路径/全新的小样本评价目录 \
  --fact-ids F1,F2 --link-ids R1 --max-items 3
```

F1/F2/R1 是示意 ID，实际应读取图中的中层 ID 和 Link ID。默认最多显式指定 10 条，
不提供隐式全量评价。输出新图、`confidence_records.json`、`summary.json` 和来源清单，
同时复制本检查点引用的审计文件；源图不改动。

已存评价可在现有 append_only 问答入口中有限展示，不改变 Chunk 排序，不自动重新评价。
下一阶段仍需人工标注来评价分项准确性、误采纳率和分数校准，当前不宣称效果提升。
