# 写入标记冲突对，沿图取证，延后决策

当前流程：新 Fact 追加 → 召回历史 → 逐对提出候选关系 → 上层登记待审查对。
后续运行独立审查入口：直接读两端 → 双方原文 → 图关联/邻居扩展 → 语义兜底
→ 保存关系判断和决策建议。原始 Fact 不覆盖，旧版本和证据仍可追溯。

## 上层只保存轻量标记

```json
{
  "link_id": "L1",
  "from_fact_id": "旧FactID",
  "to_fact_id": "新FactID",
  "relation": "conflict",
  "status": "candidate",
  "review_kind": "potential_conflict",
  "review_status": "pending"
}
```

`conflict/correction/replacement` 都进入待审查索引；并存/补充不自动变成冲突。
加载未保存 review 字段的旧 event_index_v1 图时，可从候选标签识别待审查对。
判定失败或写入期漏连，不能靠本索引补出不存在的端点，仍需初始召回补建关联。

## 审查约束

- 每次只核查固定 old_fact_id、new_fact_id，模型不能更换比较对象。
- 初始证据直接回查双方 Chunk，不重新泛搜同主体。相同 Chunk 合并但保留两端 ID。
- SUFFICIENT 必须引用双方 Fact 及各自展示窗口中的连续原文，且比较条件通过。
- 已有候选关系标签不传给 Judge，以减少照抄上层标签；图连通不证明因果。
- 缺证据先扩展候选关联/事件分组，再查同 Session 前后文，最后语义检索兜底。
- 仍使用已有 Judge 的窗口、轮数和引用约束，输出截断一律暂缓，不取消证据检查。

## 显式运行

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python \
  experiments/integrated_event_state/review_conflict_pairs.py \
  --source /绝对路径/dynamic_memory_event_state.pkl \
  --output /绝对路径/全新审查目录 \
  --new-fact-id 实际新FactID --limit 5
```

也可使用 `--link-id` 指定单条候选关联。若不指定新 Fact，按图中待审查链接
顺序检查至多 limit 条。默认最多 3 轮、每路 Top-5、12 个证据 Chunk。
此入口面向当前快照的延后审查，不作为写入时历史召回的无泄漏评测。

每对结果：`pair.json`、`review_result.json`、`decision_proposal.json`、`llm_trace.jsonl`。
根目录有来源/代码哈希清单和汇总。上层不重复存储长解释和证据文本。

## 决策含义与边界

关系为重复/并存/无关时提出 RETAIN；兼容补充提出 REFINE；现实变化提出
REPLACE；明确纠错提出 CORRECT；冲突未解决、关键条件未知或技术失败则 DEFER。
这些都是未采纳建议：`update_applied=false`、`validation_status=not_verified`。
置信度字段为 null，不捏造分数，也不自动使旧事实失效；后续需确定可信度评估
与决策采纳政策，再实现当前状态投影及版本更新。

旧默认问答、Judge 调度顺序和 gated Baseline 不因这个独立入口改变。写入时
新增的 pending 标记只服务于 append_only 的后续审查。
