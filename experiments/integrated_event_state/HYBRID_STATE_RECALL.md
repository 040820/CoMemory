# 写入期 StateSlot + BGE-M3 双路历史召回（v4 追加观察模式）

当前 `Memory_EventState*.yaml` 使用 `state_write_policy: append_only`。写入只保存原文断言和**候选关联**，不在没有完整证据链时决定事实有效性。下面旧版门控说明仅适用于显式选择 `gated` 的历史复现；当前流程以 [APPEND_ONLY_MEMORY.md](APPEND_ONLY_MEMORY.md) 为准。

这是**新记忆写入时找相关旧事实**的实现。问答阶段仍使用已有 Structural–Semantic 检索器，原版 LiCoMemory 和事件实例身份识别流程不变。

## 数据流程与代码职责

1. 当前 Chunk 抽取新 Fact，先经过既有原文、主体和 Triple 来源校验。
2. `coregraph/hybrid_state_retriever.py` 并行取候选：A 精确查 StateSlot 历史索引；B 使用原始谓词、对象、scope、Triple 和引文计算 BGE-M3 余弦相似度。
3. `state_semantic_normalizer.py` 使用少量候选对齐开放状态维度。对齐成功后重新查 A；B 复用原始表达的召回结果，避免两路共同依赖归一化标签。
4. 按 Fact ID 去重，通过 RRF 排序。保护两路首位，预算允许时保护当前状态，再补入其余排名靠前候选。默认 A/B 各取 5，最终最多 5。RRF 不是事实置信度。
5. `coregraph/state_fact_relation.py` 每次只比较一对事实，区分重复、补充、并存、冲突、纠正、替换、无关或不确定。模型输出必须带当前候选引文中的连续证据。
6. `event_state_writer.py` 执行程序槽位、基数、更新目标约束；替换/纠正还要通过原有独立 LLM 三道证据门控和时间门控。多个目标均通过时保留待核查，不按检索排名强行选择。
7. `event_state_layer.py` 保存 Fact、当前状态、历史演化和关系审计。冲突不自动覆盖；同一单值槽的补充若尚不能安全合并，保存为 pending 关系和待核查候选。

## 存储新增字段

| 字段 | 作用 |
|---|---|
| `facts_by_slot` | StateSlot → 全部历史 Fact ID；失效版本也保留，召回时过滤不可比较版本 |
| `facts_by_subject` | 主体名称与中层实体 ID → Fact ID，供语义路径限制主体 |
| `fact_relations` | 旧/新 Fact ID、模型关系、实际操作、是否真正作用于该旧事实、原文/Triple/Chunk 证据 |
| `decisions[].retrieval_hits` | 最终候选的 A/B 来源、各路名次、余弦和融合分数 |
| `decisions[].retrieval_trace` | 初始/归一后 A/B 池、融合池、选中/丢弃 ID、预算、B 路可用状态 |

`fact_relations[].new_fact_id=null` 表示候选没有获准写入 Fact 表；其原始内容在 `candidate`、`quote`、`triple_refs` 中保留。`status=pending` 的关系不能当作已确认的状态变化。重复事实可能复用旧 ID；该记录表示新增证据的审计关联，不是图中的自环事件。

对象的作用取决于语义：雇主 B→C 中对象是变化值，不能要求对象相等；牛奶过敏中“牛奶”是作用域，应落在 scope。时间不做相等过滤，替换必须有证据。实际活动事件继续沿用原事件实例匹配及时间断言流程。

## 配置

```yaml
state_semantic_normalization: true
state_recall_mode: hybrid        # legacy / hybrid / structured / semantic
state_structured_top_k: 5
state_semantic_top_k: 5
state_candidate_top_k: 5
```

正式构建、上层回放和小样本入口都传递上述配置；上层回放的 `--candidate-top-k` 优先指定最终预算。旧配置默认 legacy。Embedding 使用项目现有配置，设为 BGE-M3 即复用该模型，不另建第二套模型服务。

## 运行与评估边界

本次仅实现代码，按用户要求未运行测试、LLM 或 sample0 重建。旧实验结果不代表新实现效果。复用来源中层图时请使用新的上层输出目录，回放检查点源码指纹包含新模块，不能继续旧代码的未完成检查点。

以后评估两路 Recall@K，需要人工标注每个新 Fact 应关联的旧 Fact ID，然后分别与 `structured_ids`、`semantic_ids`、`union_ids`、`selected_ids` 比较；现阶段只有召回轨迹，没有捏造召回率。B 路不可用时日志和轨迹会明确显示降级。当前语义实现按同主体事实计算余弦，使用进程内向量缓存；大规模数据可以以后增加独立 ANN 索引，不修改原始记忆。
