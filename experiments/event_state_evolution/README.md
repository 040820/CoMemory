# 事件与状态演化层（第一版）

本目录是独立实验模块。它读取已有的 CogniGraph `.pkl`，把持续状态的基线、变化、纠正和证据写入**新的 SQLite 文件**。它不修改原有图、Chunk、写入代码、检索器或 Baseline。

## 处理范围

- 当前雇主 `employer`、当前职位 `job_title`、当前常住地 `residence`。
- 首次确认的值只建立基线；旧值与新值均有证据时追加变化。若仅从前后当前状态推断出变化，标记 `inferred`。
- 明确纠正会使旧事实失效，并重算受影响的变化；后续独立证据支持的当前状态保留。
- 普通活动、并存兴趣及没有持续状态含义的事件不写入演化层。
- 同一 Chunk 可产生多个属性事实，并可用共同的 `event_group_id` 关联同一事件。

图边可能合并过多个 Chunk，因此模型必须逐条核对原文。无图边的 Chunk 也会检查纠正，但只有能从原文给出可核对三元组、且主体能映射到现有实体节点时才入库。无法核对的结果进入 `review_queue`。这意味着原图已丢失的 Triple 不一定能恢复，待核查数量需要人工检查。

## 构建索引

服务器 vLLM 按 `config/Memory.yaml` 中 `llm.base_url` 提供 OpenAI 兼容接口。先准备人工核定的实体映射，例如 `results/event_state_locomo_group_1/entity_map.json`：

```json
{
  "Caroline": "conv-26:participant_a",
  "Melanie": "conv-26:participant_b"
}
```

只有映射中的主体可以进入有效状态；未列出的主体进入待核查。若原图把同名人物合并成一个节点，可用 `"ChunkID|姓名"` 或 `"SessionID|姓名"` 覆盖普通姓名映射，优先级为 Chunk、Session、普通姓名。映射必须由源数据或人工核定，不能只凭图节点名称猜身份。

用少量候选检查输出，再去掉 `--limit` 运行完整索引：

```bash
python experiments/event_state_evolution/build_index.py \
  --config config/Memory.yaml \
  --graph results/locomo_group_1/dynamic_memory_graph.pkl \
  --entity-map results/event_state_locomo_group_1/entity_map.json \
  --output-db results/event_state_locomo_group_1/state.sqlite \
  --limit 5
```

命令输出 `candidates`、`written`、`review`、`ignored`、`skipped` 和 `review_total`。相同来源重跑不会重复写入事实；`--limit` 计算的是**尚未处理**的候选数，不是会话数。可用 `--chunk-id 42` 定点检查已有 Chunk。正式实验建议使用**新的输出库路径**，把小样本试跑与正式运行分开审计。索引文件放在 `results/` 下，不提交 Git。

## 可选检索

现有 `StructuralSemanticRetriever` 继续工作。新入口仅在调用时附加状态证据：

```python
from experiments.event_state_evolution.retrieve import retrieve_with_state
from experiments.event_state_evolution.store import StateStore

store = StateStore("results/event_state_locomo_group_1/state.sqlite")
result = await retrieve_with_state(
    "Where does Alice work now?", retriever, store,
    top_k=5, mode="current", entity_id="conv-26:participant_a", attribute="employer",
)
print(result["state_evidence"])
store.close()
```

`mode="off"` 完全返回原检索器结果；`current` 只返回当前有效状态；`history` 返回状态事实和变化（包括已纠正、已撤销的审计记录及其状态标记）。回答当前状态时应使用 `state_evidence`，不能把原检索 `evidence` 中出现的旧 Chunk 当作当前值。使用实体映射时，`entity_id` 应传入映射后的稳定 ID（上例是 `conv-26:participant_a`）；未传实体时仅在原检索分析恰好识别一个实体的情况下推定。

## 时间和局限

`observed_at` 来自会话时间；只有原文出现的日期才进入 `event_time`。第一版按会话时间顺序处理现有图，不自动支持乱序补录。纠正重算仅针对同一实体、同一属性。图节点本身若已错误合并同名人物，旁路索引无法自动拆开；需要在实体映射中按 Chunk 或 Session 消歧。模型判定可能遗漏隐含状态或纠正，因此正式实验应保存待核查与逐题证据，不能只看最终准确率。

无需 vLLM 的逻辑测试：

```bash
python -m unittest discover -s experiments/event_state_evolution/tests -v
```
