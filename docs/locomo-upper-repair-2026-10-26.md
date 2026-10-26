# LoCoMo 上层写入排查与修复记录

## 已核对的原实验

来源：服务器 `results/locomo_sample0_upper_event_v1/`，LoCoMo sample0。

- 已处理 214 个 Chunk；保存 162 条事实、56 个事件实例、6 条精确日期断言、30 条属性断言。
- 162 条事实全部按 set/ADD 保存，当前单值状态索引为空。
- 222 条 review 涉及 173 个 Chunk。队列按“Chunk＋原因”去重，因此不是 222 条不同事实，也不是 222 个失败 Chunk。
- 194 次可解析抽取响应共有 786 条候选，其中 492 条缺 quote；另有 20 次抽取返回不完整 JSON。旧 trace 没保存 finish_reason，结合 768 token 上限与响应在字符串/对象中间结束，可怀疑截断，但不能逐条断言其终止原因。
- 图边与同源旧事实快照共恢复 574 条 Triple，覆盖 198/214 个 Chunk。旧文件不能恢复被覆盖的全部关系。

原 review 原因汇总：

| 原因 | 条数 | 占 222 条的比例 |
|---|---:|---:|
| 引文缺失或不是连续原文 | 100 | 45.0% |
| 新值不在引文中 | 29 | 13.1% |
| kind 不属于允许类型 | 22 | 9.9% |
| JSON 结构无效 | 20 | 9.0% |
| Triple 对象/主体/索引及活动对象不匹配 | 26 | 11.7% |
| 说话人归属不足 | 8 | 3.6% |
| 将事件身份或日期重复写成普通属性 | 8 | 3.6% |
| 地点变化没有明确纠正证据 | 5 | 2.3% |
| 属性值没有原文依据 | 2 | 0.9% |
| 纠正目标不成立 | 1 | 0.5% |
| 无依据的 scope 被忽略 | 1 | 0.5% |

需要修正的统计解释：

1. `162/786` 不能直接作为事实准确率或完整接受率。同一事件的补充证据、重复事实、无长期价值对话与错误候选会经过不同处理。
2. 88 个 Chunk 是已保存 fact 的来源 Chunk 数；事件补充证据可能只进入事件实例，不产生新 fact，所以不能据此判断其余 Chunk 都抽取失败。
3. 已确认 `Melanie talked_to Caroline` 汇集了 11 个 Chunk；“21 个错误对话事件”仍需依据逐条原文标签判定，不能按动词一律判错。对话中报告的历史交谈可以是真实事件。
4. 24 条属性断言没有专门 Triple 支持，这一数量成立；若同一事件引文明示了属性，也可以有原文证据。新版本显式区分这种证据与 Triple 证据。
5. “35 个带时间表达却无日期的候选”需要指定统计口径。按旧版已保存 event fact 的 `event_time_raw` 非空、`event_time` 为空统计，本次读取得到 12 条；这与扫描原文时间词或统计全部 LLM 候选不是同一指标。

## 代码修复

### 抽取与可追溯引用

- 缩短提示；每次最多抽取 4 条候选，输出上限从 768 提高到 2048 token。
- 使用服务器实际支持的 JSON Schema 格式约束，要求 quote、字段类型和当前来源 ID；缺 quote 或 JSON 不完整时只重试一次。
- Triple ID 由 `(chunk_id, src, relation, tgt)` 稳定生成。兼容旧整数及对象引用时，程序核对精确来源；不同 Chunk 或改写的 Triple 不会被接受。
- 修正旧“引用越界”错误将格式错误混为一谈的问题。记录每次模型调用的 finish_reason、token 数及 API 错误，修复调用单独标记为 repair。
- `214/8` 属于把全局累计值除以本批大小的日志显示错误；现在分别显示本批与全局进度。

### 原始 Triple 保存

用户新增 `triple_storage` 的方向正确；已接通其上下游：在关系合并前按 Chunk 保存，上层直接消费这份原始记录。

- `session_time` 取会话时间；抽取器的 `create_time` 另存为关系发生时间，避免污染观察时间。
- `triple_storage_origin=extracted` 表示合并前保存的记录；`recovered_partial` 表示旧图部分恢复。
- 已有原始记录时不再从压缩图边混入额外关系。旧记录的部分恢复不会因写回新 pickle 就变成“完整原始来源”。
- 新旧写入协议不能混写；旧非空上层需要在独立目录重建。replay 检查点校验来源、配置和写入代码摘要。

### 状态与事件语义

- 将明确同义的 `works_at/employed_by/current_employer` 等归一到雇主槽。未知开放属性仍默认 set；不为了产生更新而改变基数。
- 雇主/居所槽需要对应关系的原文依据，团体成员身份不能直接成为雇主。
- 明确纠正可以先召回同槽旧事实，再通过 CORRECT 门控；初次抽取不需要猜旧 fact_id。
- 普通提问、评论、附件动作与一般赞叹单独过滤并记录到 `skipped_candidates`。原文报告的历史交谈不因谓词相似就全部删除。
- 附件说明中的人物动作不能自动归给说话人；主体自身感受也需原文支持。
- 支持、成员身份、习惯性活动等不直接提升为一次历史事件。不同日期且无纠正证据的活动不合并。
- 事件候选必须具有对象名称或中层对象锚点的交集，再由 LLM 判断身份，防止仅因主体或向量相似合并不相干活动。

### 时间与属性

- 相对时间以 Session 观察时间为基准，支持 last night、last Tues/Fri 和带序数的月日等表达。
- last week、last year、next month、过去周末保存区间；季节与 a few weeks ago 保留模糊表达，不假定精确边界。
- 区间或模糊表达不会因中层恰好有单个日期就被擅自精确化。事件实例的 `time_mentions` 保存这些证据。
- 属性必须出现在对应事件引文中；地点另需地点表达与类型依据。拒绝不合格可选属性时保留已经通过校验的核心事件，并记录具体原因。
- 属性的 `evidence_basis` 区分 `event_quote` 与 `triple_and_event_quote`。

## 验证结果与边界

服务器相关回归测试共 **65 项通过**：状态相关 23、事件与写入器 19、Triple 持久化 5、本次真实输入回归 18。

最终真实小样本：Chunk `1,6,18,36,77,92,99,100`，使用 qwen3-8b、缓存 BGE-M3 和原图同源恢复 Triple。

- 8/8 个指定 Chunk 处理完成，保存 11 条事实、8 个事件实例。
- 8 次抽取产生 30 条候选，缺 quote 为 0；8 次抽取和 4 次身份门控均以 `finish_reason=stop` 结束，无 JSON 截断或引用协议错误。
- LGBTQ support group 日期为 2023-05-07；counseling workshop 为 2023-06-23；新加入 activist group 为 2023-07-18。
- 绘画的 last year 保存为 2022 年区间；school event 的 last week 保存为 2023-05-29 至 2023-06-04；a few weeks ago 保留原始模糊表达。
- 错误雇主、当前提问、照片中的撑伞动作和错误主观感受被拒绝；不同对象的活动没有再通过“支持 Mental Health”错误合并。
- 当前槽与变化边均为 0。此处没有足够证据证明发生单值状态替换，不能把非零更新数量作为通过条件。
- 仍有 12 条去重 review：5 条值缺少逐字依据、2 条主观感受归属不足、3 条地点属性无对应引文、1 条将时间当事件对象、1 条错误雇主。7 条预期过滤另记 skipped。

这些结果验证了已定位错误的拦截及协议修复，不构成完整 LoCoMo 写入准确率评测。事件子动作的粒度、无词面/中层对象锚点的语义别名匹配，以及每段最多 4 条候选对召回的影响，仍需完整标注实验分析。更严格的门控也可能拒绝本来正确但被模型改写的值。

原 sample0 结果保持原样。本次没有重跑完整 sample0，也没有补回旧中层已经丢失的 Triple；要验证完整三层写入，需要从原始对话在新目录重建中层并保存原始出现记录。

## 产物与复现

服务器目录均以 `/home/cwf/Code/LiCoMemory/` 为根：

- 最终小样本：`results/locomo_sample0_upper_smoke_fix7/`，包含 inputs、summary、layer 与逐调用 llm_trace。
- 运行日志：`results/locomo_sample0_upper_smoke_fix7.log`。
- 测试日志：`results/upper_repair_complete_tests.log`。
- 排查中间结果 fix2 至 fix6 保留，仅用于追踪修复过程，不视作最终结果。

```bash
python -u experiments/integrated_event_state/smoke_real_upper.py \
  --source results/event_state_candidate_sample0_scopefix/dynamic_memory_event_state.pkl \
  --output results/locomo_sample0_upper_smoke_new \
  --chunk-ids 1,6,18,36,77,92,99,100
```

输出目录必须是新目录。BGE-M3 默认离线读取缓存；独立 replay 如需首次下载，可显式传 `--allow-embedding-download`。
