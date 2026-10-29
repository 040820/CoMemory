# 固定新旧 Fact 的证据解释实验：第一轮开发结果

## 结论与边界

已实现并真实运行“固定两端 → 预取图关联上下文 → 核对原文 → 判断关系/原因 → 保存建议”。核心预约对照能根据不同中间原文区分现实改期与原记录误读；没有原因依据时，结果暂缓，不执行事实覆盖。

这不是完整数据集评测。8 个受控样本的参考标签由本实验设计给定，未经独立人工复核；同一批输入被用于调试协议，不是留出测试集。严格关系标签与证据要求完整通过 6/8；若把“要求原因但缺少依据时正确暂缓”单列为任务结果，则为 7/8。不能将其解释为泛化准确率或事实真实概率。

两组 LoCoMo 固定事实对另作定性检查：旧猫与新猫 Bailey 判为并存并保留；鞋与 Bailey 的比较因引用错误暂缓。真实复杂对话上的稳定性尚未解决。

## 实验输入与评估设计

- 模型：服务器真实 vLLM `qwen3-8b`，关闭 thinking；语义兜底使用本地 BGE-M3。
- 三层存储不变，不重新抽取已有图，不改变原始数据或历史报告。
- 受控样本：重复、不同宠物并存、明确迁居、明确纠正、不同项目角色，以及同一预约 Fact 对的三种解释条件。
- 预约对照三例的旧/新 Fact 完全相同，只改变中间解释：真实改期、原通知误读、没有解释。解释原文在独立 Chunk，模型不能从两端直接看到。
- 图关联只是来源定位线索；候选关系标签不提供给 Judge。每例保留原文、参考关系、检索动作、实际展示窗口、连续引文及模型原始输出。
- 比较基线与最终版本的 `chunks/observations/links/question/reference`，8 例内容全部相同；后续增加的显式任务目标和协议选项在清单中记录。
- 原始严格指标保留。单独的任务结果指标用于区分“关系枚举必须是 uncertain”和“原因未确定，因此正确暂缓”，不把兼容关系误当成原因已经解释。

## 固定两端，证据如何发挥作用

两端均为：

```text
旧 F1：“My dental appointment is on May 7.”
新 F2：“My dental appointment is now on May 6.”
```

| 关联原文 | 结果 | 解释 |
|---|---|---|
| “The clinic actually rescheduled ... from May 7 to May 6. The earlier notice was correct at that time.” | replacement / SUFFICIENT | 旧说法在当时正确，后来实际改变 |
| “I misread the original ... It has always said May 6, not May 7; the clinic did not reschedule it.” | correction / SUFFICIENT | 旧记录源自误读，不是后来改期 |
| 不提供上述解释，仅有别人的预约干扰项 | coexists / DEFER | 两段记录可以在时间上兼容，但仍不能说明差异原因 |

前两例的解释 Chunk 均实际被检索、展示并在最终结果中准确引用。第三例没有被强制推为变化或纠正；其关系枚举与参考 uncertain 不一致，故严格指标仍记为未完全通过。

## 最终受控结果

目录：`bounded_repair_v6/`。8 例，9 次实际模型请求，其中 1 次为有界引用修复；未恢复的技术失败为 0。

| 样本 | 参考关系 | 输出关系 | 最终状态 | 严格通过 |
|---|---|---|---|---|
| 同雇主重复陈述 | duplicate | complement | SUFFICIENT | 否 |
| 狗与新增猫 | coexists | coexists | SUFFICIENT | 是 |
| 明确从北京迁居上海 | replacement | replacement | SUFFICIENT | 是 |
| 明确承认原日期说错 | correction | correction | SUFFICIENT | 是 |
| Alpha 研究员与 Beta 评审员 | coexists | coexists | SUFFICIENT | 是 |
| 预约真实改期解释 | replacement | replacement | SUFFICIENT | 是 |
| 预约原通知误读解释 | correction | correction | SUFFICIENT | 是 |
| 预约无解释 | uncertain | coexists | DEFER | 否；任务层面正确暂缓 |

“I still work for Aster”被判断为对旧雇主记录的补充，重复/补充标签边界仍不稳定，也需独立人工确认标签口径。本轮不事后改参考答案来提高通过率。

## 调试过程全部保留

| 版本目录 | 严格完整通过 | 未恢复技术失败 | 主要发现 |
|---|---:|---:|---|
| baseline | 2/8 | 4 | 输出被截断，原因证据未进入上下文 |
| context_first_v2 | 2/8 | 2 | 能找到并引用解释，但把真实变化也叫作纠正 |
| context_first_v3 | 1/8 | 0 | 关系多已判对，却照抄模板的 unknown 检查项 |
| context_first_v4 | 6/8 | 0 | 改为字段允许值规范；缺原因时仍可能过早充分 |
| goal_guard_v5 | 5/8 | 0 | 显式限定原因任务；一次正确纠正因漏引旧端被拒绝 |
| bounded_repair_v6 | 6/8 | 0 | 一次有界协议修复恢复正确引文；另报任务结果 7/8 |

多个协议、提示和调度设置在开发中改变，不能把差异单独归因于某一个设置，更不能声称 PPR 带来提升。本轮使用已有图扩展，没有接入 PPR。

## 两组真实 LoCoMo 检查

使用冻结 sample0 新图：`new_graph_recall_sufficiency_20261028/reproducibility/dynamic_memory_event_state.pkl`。

1. **旧猫—Bailey**：旧原话含 “We've got a pup and a kitty ... here's our cat”；新原话含 “we got another cat named Bailey too”。最终 coexists / RETAIN，双方原文被引用；1 个审查轮次、2 次实际模型请求（含一次协议修复）。
2. **旧鞋—Bailey**：初次输出截断；一次修复后又把其他猫的原文绑定到了鞋的来源，严格引用检查拒绝，最终 DEFER。不是证据不存在的证明，也不是已经得到可靠关系判断。

这两个定性样本均未覆盖旧 Fact，来源图哈希检查保持不变。它们不是随机抽样，也没有独立人工 gold，不能作为数据集准确率。

## 代码改动

- `explicit_fact_pair()`：直接指定两个 Observation，不要求已有冲突链接，防止只测曾被模型标冲突的样本。
- `initial_context_k`：第一次判断前可从上层关联预取有限原文，减少只看两端就结束的情况。
- 完整字段规范及长度/数组上限：帮助模型完成输出；没有取消证据检查或提高上下文上限。
- `review_goal`：关系分类与解释演化原因分开。原因目标下，单纯重复/并存/无关不算完成原因问题。
- `max_response_repairs`：一对事实整个过程至多一次格式/引用修复，保留初始失败响应；证据不足不触发修复。
- vLLM 紧凑 JSON 适配仅显式启用，Schema 内容不变；非兼容后端默认关闭。
- 决策仍为建议，`update_applied=false`、`confidence=null`，没有自动失效、覆盖或采纳。

## 产物与复现

服务器根目录：

```text
/home/cwf/Code/LiCoMemory/results/fixed_pair_explanation_probe_20261029/
```

- `bounded_repair_v6/summary.json`：最终受控汇总。
- `bounded_repair_v6/<case>/review_result.json`：每轮证据、引用、判断与协议修复。
- `bounded_repair_v6/<case>/llm_trace.jsonl`：真实提示词、输出和 token 元数据。
- `real_locomo_v6_cat_bailey/`、`real_locomo_v6_sneakers_bailey/`：真实固定事实对。
- `real_pair_commands_v6.json`：实际 Fact ID 和运行参数。
- `input_equivalence_report.json`：受控原文与参考标签未改动检查。
- `reproducibility/source_snapshot*`：各轮代码快照，历史产物不覆盖。
- `test_checks.json` 及各测试日志：相关单元/集成检查。

下一步应在少量独立标注的真实事实对上核查解释证据与关系，优先处理长对话中的错绑引用、重复/补充边界及漏证据。之后再将 PPR 作为候选证据检索分支对比；目前不增加存储复杂度或自动更新策略。
