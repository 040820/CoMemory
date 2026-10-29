# 证据协议修复与逐目标置信度模块说明书

## 功能与边界

本次修复服务于三层新图的写入、关系审查和问答：减少“原文已经保存，但模型引用格式错误导致判断丢失”的情况。保留 Chunk、原始 Triple、观察断言和候选链接的既有结构；不把协议通过理解为事实真实，不自动执行覆盖、纠正或失效。

## 代码职责

| 文件 | 所属模块与职责 |
|---|---|
| `coregraph/state_fact_relation.py` | 写入期逐对关系分类；LLM 选择 old/new 证据编号，程序回填原句；保留旧输出严格兼容 |
| `coregraph/evidence_judge.py` | 按事实原句开窗口、生成片段编号与原始字符位置、核对引用、区分技术失败与语义不确定 |
| `coregraph/evidence_support/service.py` | 共用取证/Judge/回答控制器；片段编号 Schema、白名单解析、证据角色和 shadow 置信度装配 |
| `coregraph/evidence_support/retriever.py` | 保留结构索引、向量、BFS、RRF 取证；额外携带各 Fact 原句和必须展示的双端 ID |
| `coregraph/evidence_support/confidence.py` | 将真实检索轨迹、已校验引用和模型角色适配为旁路评分，不参与排序或判断 |
| `coregraph/evidence_confidence.py` | 纯函数评分；按 target_fact_id 隔离支持/反驳，关系目标和两端断言分别计算 |
| `coregraph/evidence_support/query.py` | 正常新图问答入口的装配；返回答案、引用会话和完整审查报告 |
| `coregraph/structural_semantic_retriever.py` | 从已删除实验目录迁移的历史算法；供旧布局/显式历史实验复现，算法参数不变 |
| `experiments/structural_semantic_retrieval/retriever.py` | 旧路径导入兼容，不复制算法、不恢复已删除的评测程序 |
| `experiments/evidence_support/replay_relations.py` | 从现有图和失败审计重放关系；只输出审计及候选边补丁，不写回来源图 |

## 写入期关系协议

输入是新旧两条已通过原文校验的 Fact。LLM 判断 duplicate、complement、coexists、conflict、correction、replacement、unrelated、uncertain；确定关系必须选择 `evidence_refs=["old","new"]`。程序从输入事实回填原文，旧字段 `evidence_quote` 仍使用新 Fact 原句，兼容既有关系存储与上层引用校验。

编号只是“使用哪条证据”，不表示这条证据蕴含模型结论。`same_slot` 和关系语义仍由 LLM 局部判断，保存为候选。输出 Token 上限为 None；仍检查服务端 `finish_reason=length`。协议错误最多补发一次，不对真实语义 uncertain 反复请求。

## 检索与引用算法

候选寻找继续由算法承担：精确槽位/事件索引、BGE-M3 余弦、有界 BFS 和 RRF。不是 PPR，也尚未实现 BM25 路线。LLM 负责核对身份、作用域、模态、粒度、时间和证据角色。

每个来源保存原始全文，但模型只看有预算的窗口。程序围绕每个 Fact 原句前 80 个字符开始开窗，默认 1200 UTF-8 字节；如果原句更长，会完整保留该原句，仍受总预算限制。同一 Chunk 内相距很远的双端分别开窗，校验禁止跨窗口拼接连续引文。必要端点没有进入窗口时，记录技术失败，而不是声称语义证据不足。

每个可选片段包含 `quote_id / fact_id / chunk_id / quote / span`。quote_id 由 Chunk、Fact 和字符位置稳定哈希产生；仅实际展示的编号进入 Schema 枚举。Judge 的 citations 只返回编号；程序绑定原句及 `[start,end)` 字符位置。角色同样绑定编号。未知编号、错绑来源、改写及省略号不被自动修复为“有效证据”。

双端任务额外必填 `endpoint_citations.old_quote_id/new_quote_id`，分别限制为旧/新 Fact 的片段编号；不能在自由数组中两次选择新端点冒充双端。模型给角色标注的合法片段编号也是显式引用，由程序统一合并，避免同一引用在两个数组重复填写。这个步骤只修复引用协议，不自动判定角色语义正确。

提示词中同一段原文只在 windows 展示一次；片段选项仅给编号、Fact ID 和原文位置，完整原句保留在审计记录。真正重叠的窗口合并为同一连续原文，相隔很远的窗口不拼接。这样减少重复文本占用，不增加原来的 12000 字节预算，也不放宽来源校验。

问答分支额外说明字段含义，并仅对技术协议失败补发一次；模型语义不确定不重试。展示窗口优先保留显式断言来源，再按原问题 `semantic:task` 名次分配预算，图扩展材料随后展示。这仅改变提示词装入顺序，候选集合、RRF 排名和证据支持度计算不变；展示名次不是置信度。

最终回答同样只选择已审查窗口的 quote_id，至少引用一个片段；程序还原 Fact、Chunk、原文和位置后再校验。旧答案格式仍严格兼容，不能因为模型写出了答案就跳过引用验证。

## 参数

| 参数 | 当前值/含义 |
|---|---|
| 写入关系最大输出 Token | None，服务端物理上下文仍有效 |
| 新证据 Judge/回答最大输出 Token | None |
| 协议修复次数 | 写入分类、双端审查、问答审查各最多 1 次；语义不足不重试 |
| 温度 | 0.0 |
| 单 Fact 写入视图原句 | 最多 1800 字符，原始存储不截断 |
| 窗口前缀 | 80 字符 |
| 默认窗口 | 1200 UTF-8 字节，完整锚点可扩大 |
| Judge 总输入预算 | 默认 12000 UTF-8 字节，配置允许 6000–12000 |
| 单次结构/向量候选 | 各 12 |
| BFS | 2 跳、64 个访问状态、每节点最多 12 个邻居 |
| RRF 常数 | 60；只用于候选排名 |
| 最终候选/来源 | 最多 48 Fact、12 Chunk，每 Chunk 最多 4 Fact |
| Judge 调度轮数 | 最多 3 轮；整次请求共享最多 1 次协议修复，因此最多 4 次模型调用尝试；每轮最多 2 条缺口查询，语义查询总计最多 7 条 |

## 失败分类

- `evaluation_status=technical_error`：传输、服务端截断、JSON/编号/来源绑定错误、必要端点缺失。
- `evaluation_status=semantic_uncertain`：协议合法，但模型不能确认关系或仍有具体材料缺口。
- `evaluation_status=model_assessed`：协议合法且给出判断；没有人工金标时仍未独立验证语义。
- `not_evaluated`：只有检索，没有 Judge。

兼容状态仍可为 DEFER，但统计必须分开技术失败与语义暂缓。技术失败不生成 uncertainty 数值，也不作为置信度校准的负例。

报告中的 `rounds` 是取证调度轮次，`judge_calls` 是模型调用尝试数，包含协议修复；尚未调用模型的窗口/输入失败不计为模型调用。精确服务端响应及 Token 用量仍以 llm_trace 为准。

## 逐目标旁路置信度

支持 F1 的证据只进入 F1，支持 F2 的证据只进入 F2。若要评价两者的某种关系，角色必须显式指向 `target_fact_id="relation"`，并说明具体关系命题；没有这类证据，relation_confidence 保持 None。双端覆盖不是关系成立的概率。

每个目标内部保留原有工程公式：同来源组取最大值，不同组 Noisy-OR，支持与反驳分别聚合。引用来源完整度和精确定位形成可核查性特征，**不是现实真值可信度**。模型角色仍需人工评估。来源分组默认按 Chunk/source_id，不保证跨 Chunk 转述独立，报告 `independence_verified=false`。

检索一致性（结构化 0.35、预留 BM25 0.30、语义 0.20、图 0.15、预留摘要 0.05）单独保存。已绑定的直接来源不因没有检索排名而归零；检索分数不乘入当前支持度。来源特征权重仍是工程初值：来源链 0.30、精确引文 0.20、完整度 0.20、来源类型 0.15；未评估的抽取对齐、时间/数量等特征不虚构输入。

所有结果设置 `mode=shadow / affects_decision=false / calibrated=false`。不改变原 Judge 状态、排序或记忆有效性。原有数值阈值只保留为离线诊断值；下一步需要人工标注集和独立校准，不能宣称 0.9 等于 90% 正确。

## 新图问答与重放

`event_state_query_mode=auto` 时，event_index 布局使用共用 EvidenceSupport；旧摘要 Baseline 不改变。显式 `structural_semantic` 可复现旧检索。缓存只在当前图上复用，写入后失效。

问答入口先复用原结构化语义 Chunk 检索（语义 0.65、Triple 0.20、词面 0.10、图扩展 0.05，图种子 20），取前 5 个 Chunk。在首个有可见、原文绑定 Observation 的 Chunk 中，以 BGE-M3 余弦选择 1 个 Fact 作候选锚点，再交给共用模块查支持和反证。锚点不是已确认答案；Judge 可以否决。`as_of` 在种子阶段先过滤事实与原文，用临时可见关联图检索，不修改原图，也不让未来事实进入种子。

入口轨迹记录在 `query_seed_trace`；缺少原图或 Embedding 时明确记录降级，显式提供 claim_ids 时不再自动选种子。问答候选入口与更新侧双端取证是不同任务，不把更新期全库 Fact 排名直接替代原问答检索。

截止时间、会话时间和观察时间统一支持 ISO，以及数据集实际使用的 `YYYY/MM/DD`、`YYYY/MM/DD (Tue) HH:MM`；星期不参与推算，保留小时分钟。只有日期的截止值包含当天，不以清空 as_of 绕过过滤。主程序使用本地 vLLM 时可设置 `LICOMEMORY_COMPACT_JSON=1` 复用现有紧凑解码；独立实验入口默认使用紧凑语法并保存模型审计。

失败重放命令：

```powershell
python -m experiments.evidence_support.replay_relations --source <现有图.pkl> --relations <pair_judgments.jsonl> --config <原模型配置.yaml> --output <新目录>
```

先使用 `--sensitive-only --limit 5` 预检，再重放全部技术失败。输出 `manifest.json / llm_trace.jsonl / replayed_relations.jsonl / links_patch.json / summary.json`。补丁绑定原图 SHA256，只有双端存在且原句可定位的记录才进入模型；缺失项明确列为 skipped。原图不会改写；候选边补丁也不是更新执行日志。

增加 `--write-graph-copy` 可生成独立 `candidate_graph.pkl`，原始事实与有效性完全保留，只补充候选链接。旧失败审计不覆盖，新判断追加到独立内容寻址审计，确保后续 load/save 不丢掉新增链接。重放前核对输入审计行号，结束时再次核对源图 SHA256。

评测应分别报告协议通过率、候选关系变化和人工语义准确率。没有人工金标时 accuracy=null，不能用“程序通过”替代“推理正确”。
