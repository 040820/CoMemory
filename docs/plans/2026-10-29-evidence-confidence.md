# 证据置信度实验说明书（evidence_confidence_v1）

日期：2026-10-29
状态：v2 已按目标断言隔离，并以 shadow 接入；未校准、不改变决策。本文保留 v1 设计记录，当前行为见 `docs/modules/evidence_protocol_repair.md`。
关联：`coregraph/evidence_support/`、`coregraph/memory_confidence.py`、`coregraph/state_fact_relation.py`

## 1. 背景与目标

现有检索模块能给出候选事实、原文和路线（structured / semantic / graph），但：

- `coregraph/evidence_support/retriever.py` 的 `rrf_score` 和 `cosine_scores` 明确只是检索分数，不是可信度；
- `coregraph/memory_confidence.py` 有 Fact / EvidenceLink / Relation 三类评价，但没有回流到检索或决策；
- `EvidenceJudge` 只输出 SUFFICIENT / NEED_MORE / CONFLICT / DEFER 等离散状态，没有可比较的数值；
- 多路命中的“一致性”没有单独建模，同一 Chunk 派生的多条证据仍可能被当成多份独立证据。

本实验的目标是给每个候选证据和每个关系/问答判断增加一组可审计的数值：

```text
(检索一致性, 来源可信度, 支持信念, 反驳信念, 冲突质量, 不确定性, 双侧覆盖)
```

研究假设：把“检索相关性”和“来源可信度”分开、把“支持”和“反驳”分开，再用独立性折扣和双侧覆盖约束，比单一 RRF 排序更能区分“相关材料”和“足以支持关系判断的证据”。

## 2. 范围与明确不做的事

- P1 只实现纯函数和默认参数，不读取图、不调用模型、不修改任何记忆或证据。
- P0 只做离线分析，不改变线上 SUFFICIENT / DEFER。
- P2 才以 shadow 模式接入 `EvidenceSupport`，数值先只记录，不参与最终决策。
- 不把任何分数解释为事实为真的概率；默认参数是工程初值，未做真实样本校准。
- 不因数值阈值自动令旧事实失效、不自动执行 REPLACE / CORRECT。
- 不把候选链接、BFS 路径或摘要命中当成已确认因果。

## 3. 与现有模块的关系

| 现有模块 | 提供什么 | 新模块如何使用 |
|---|---|---|
| `coregraph/evidence_support/retriever.py` | 候选 Fact、路线、排名、余弦、原文 | 由调用方转成 `route_strengths` |
| `coregraph/state_fact_relation.py` | 关系判断的 `quote_match` | 映射为来源可信度的 exact / inexact / none |
| `coregraph/memory_confidence.py` | 抽取分数、来源评价 ID | 未来作为 `alignment_confidence` 等特征 |
| `coregraph/evidence_judge.py` | 检查项、引用、状态 | 作为标签/基线，不直接当概率 |
| `coregraph/evidence_support/service.py` | `evidence_roles`（supports/refutes/...） | 作为带符号信念的角色输入 |

新模块 `coregraph/evidence_confidence.py` 不反向依赖上述模块，保持纯函数。

## 4. 输入

### 4.1 单条证据

```python
{
  "fact_id": "F1",
  "chunk_id": "C1",
  "target_fact_id": "F1",
  "side": "old",                     # 可选；不填时按 target_fact_id 推断
  "role": "supports",                # supports / refutes / explains_change / background / uncertain
  "independence_group": "chunk:C1",  # 同源证据共享一个组
  "route_strengths": {               # 取值 [0,1]
      "structured": 1.0,
      "bm25": 0.8,
      "semantic": 0.62,
      "graph": 0.4
  },
  "credibility_features": {
      "has_source_trace": 1.0,
      "quote_match": "exact",        # exact / inexact / none
      "field_completeness": 1.0,     # [0,1]
      "source_type": 1.0,            # 1.0 显式原文，0.65 派生断言
      "alignment_confidence": 0.9,   # 可选
      "has_time": true,
      "has_quantity": false,
      "llm_self_confidence": null    # 可选，默认权重为 0
  }
}
```

### 4.2 配置

`EvidenceConfidenceConfig` 包含四组参数：`route_weights`、`credibility_weights`、`role_weights`、`thresholds`。所有值必须位于 [0,1]；角色权重必须覆盖五种角色。

## 5. 计算定义

### 5.1 路线强度

结构化、BM25、图扩展使用排名强度：

```text
strength(rank, K) = (K - rank + 1) / K      # rank 从 1 开始；超出 K 为 0
```

语义路线使用余弦：

```text
strength = max(0, min(1, cosine))
```

### 5.2 路线一致性（Noisy-OR）

```text
c_route(u) = 1 - Π_h (1 - w_h · s_h(u))
```

- `w_h` 是路线权重；
- `s_h` 是该路线的命中强度；
- 未配置的路线权重视为 0，并记录为 ignored；
- 与 SodaMem 的区别：不采用“命中即 1”，排名会影响强度。

### 5.3 来源可信度

```text
cred = clip(Σ β_k · f_k, 0.05, 0.98)
```

默认特征权重：

| 特征 | 权重 |
|---|---:|
| 有来源链 `has_source_trace` | 0.30 |
| 引文精确连续匹配 `quote_match=exact` | 0.20 |
| 引文可定位但非精确 `inexact` | 0.08 |
| 字段完整度 | 0.20 |
| 来源类型（显式 / 派生） | 0.15 |
| 抽取/对齐置信度 | 0.10 |
| 有时间 | 0.03 |
| 有数量/单位 | 0.02 |
| LLM 自评 | 0.00（默认关闭） |

`quote_match` 映射：

```text
state_fact_relation: exact                         -> exact
state_fact_relation: stripped_* / *+whitespace     -> inexact
state_fact_relation: None / 失败                   -> none
```

### 5.4 独立性折扣

```text
同一 independence_group：c_group = max(c_i)
不同 group：             c_all = 1 - Π_g (1 - c_group)
```

同一 Chunk、同一引文或同一来源派生出的多条事实不重复计票。

### 5.5 支持 / 反驳 / 冲突

角色权重：

| 角色 | 权重 |
|---|---:|
| supports | 1.0 |
| explains_change | 1.0 |
| refutes | 1.0 |
| background | 0.2 |
| uncertain | 0.0 |

每条证据的有效置信度：

```text
c_i = c_route(i) · cred(i) · role_weight(role)
```

按角色分组合并：

```text
b = 1 - Π_{i∈supports} (1 - c_i)      # belief
d = 1 - Π_{i∈refutes}  (1 - c_i)      # disbelief
K = b · d                              # conflict mass
u = (1 - b)(1 - d)                     # uncertainty
```

### 5.6 双侧覆盖

关系任务要求旧、新两侧都有证据：

```text
coverage_side = 1 - Π (1 - c_i)，i 作用于该侧
bilateral = min(coverage_old, coverage_new)
```

问答模式不要求双侧覆盖。

### 5.7 状态映射（建议值，不自动执行）

默认阈值：

| 阈值 | 值 |
|---|---:|
| `sufficient_belief` | 0.60 |
| `refute_ceiling` | 0.25 |
| `uncertainty_ceiling` | 0.30 |
| `conflict_mass` | 0.35 |
| `bilateral_floor` | 0.50 |
| `uncertainty_defer` | 0.50 |

映射顺序：

```text
无有效证据                                      -> NOT_EVALUATED
K >= conflict_mass                              -> CONFLICT
b >= sufficient 且 d <= refute_ceiling
  且 u <= uncertainty_ceiling
  且（问答 或 bilateral >= bilateral_floor）    -> SUFFICIENT
u >= uncertainty_defer                          -> DEFER
关系任务且 bilateral < bilateral_floor           -> NEED_MORE
其余                                            -> DEFER
```

输出只是建议，`update_applied=false`，不改变现有 Judge 状态。

## 6. 默认参数

路线权重（起步值，按用户确认）：

```text
structured 0.35 / bm25 0.30 / semantic 0.20 / graph 0.15 / summary 0.05
```

这些值不是论文最优值，也不是概率；P0 之后需要根据真实 pair 的相关性和校准结果调整。

## 7. P1 实现

文件：`coregraph/evidence_confidence.py`
测试：`tests/test_evidence_confidence.py`

公开接口：

```python
EvidenceConfidenceConfig
rank_strength(rank, budget)
route_confidence(strengths, weights=None)
route_confidence_detail(strengths, weights=None)
source_credibility(features, weights=None)
source_credibility_detail(features, weights=None)
combine_independent(grouped)
signed_belief(support_groups, refute_groups)
bilateral_coverage(side_groups)
quote_credibility_match(match)
evaluate_candidate(items, config=None, *, old_fact_ids=(), new_fact_ids=(), mode='relation')
```

P1 不读取文件、不调用模型、不导入 graph 模块；只用标准库。

## 8. P0 评测协议

### 8.1 数据来源

- quote-fix 重跑：`results/locomo_sample0_fresh_quote_fix_20261029/`
- `evidence_review/` 中实际审查的关系对（预计约 15 对，按 pending conflict/correction/replacement 筛选）
- `pair_judgments.jsonl` 中 1058 条写入期关系调用（含被引文校验拒绝的原始输出）
- `results/fixed_pair_explanation_probe_20261029/` 的 8 个受控样本
- 现有 `tests/` 固定输入

### 8.2 指标

- 与 Judge 关系/状态的一致性（弱标签，不是准确率）
- 关系标签上的 AUROC / 分组均值（若有足够标签）
- risk-coverage：不同置信度阈值下可自动处理的 pair 比例与其一致率
- 路线消融：仅 structured、+BM25、+semantic、+graph、+independence discount
- 后续有 gold 后再做 Brier / ECE 和阈值选择

### 8.3 边界

- 15 对审查样本仍然很小；P0 只能看趋势，不能报告数据集准确率。
- `pair_judgments` 的标签来自模型，不是人工 gold。
- 8 个受控样本的设计标签不是独立留出集。

## 9. 测试计划

- 路线强度与 Noisy-OR 数值
- 来源可信度的 exact / inexact / none、缺失特征、裁剪
- 同源证据只计一次
- 支持/反驳/冲突/不确定性的组合
- 双侧覆盖
- SUFFICIENT / NEED_MORE / CONFLICT / DEFER / NOT_EVALUATED 的边界
- 未知角色与非法权重报错
- `quote_match` 映射

## 10. 风险与边界

1. **独立性假设**：Noisy-OR 假设路线独立，但 BM25 和向量可能命中同一 Chunk；独立性折扣是必要但不充分的修正。
2. **角色语义**：supports/refutes 来自模型判断，未经独立人工核验；置信度不是真值概率。
3. **权重未校准**：0.35/0.30/0.20/0.15/0.05 是工程初值；不同语料和模型需要重新估计。
4. **同源重复**：同一 Chunk 的多条派生事实共享 `independence_group` 时只取最大值，可能低估真正独立的多次陈述。
5. **任务偏移**：关系审查可能只验证新事实；双侧覆盖只约束“有没有两侧证据”，不证明证据语义正确。
6. **输出用途**：P1/P0 不改变线上决策；P2 必须先 shadow 运行并保留回退。

## 11. 复现

```bash
python -m unittest discover -s tests -p "test_evidence_confidence.py" -v
```

P0 脚本将在 quote-fix 重跑完成后另行添加，输出目录不得覆盖历史运行。

## 12. 变更记录

- 2026-10-29：初版说明书；确定路线权重 0.35 / 0.30 / 0.20 / 0.15 / 0.05；P1 纯函数实现。
