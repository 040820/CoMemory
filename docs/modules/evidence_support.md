# 共用证据支撑模块功能说明书

版本：`evidence_support_v1`。本文对应本仓库新增的独立运行入口，不代表旧主程序已经自动切换。

## 1. 功能与研究目的

同一套取证能力服务两个任务：

- 更新侧：给定新旧两个 Fact，查找解释两者关系的材料，输出候选关系、更新建议或暂缓。
- 问答侧：给定问题及可选候选断言，查找证据、审查是否充分，再交给回答模型。

核心研究问题是“证据是否能帮助正确区分并存、变化、纠正和冲突”，不是把新记忆默认判为正确。
图继续使用下层原始 Chunk、中层事实与断言、上层分组与候选链接。模块不增加存储层，不修改来源图，不执行覆盖、删除、失效或当前状态更新。

## 2. 涉及代码及职责

| 文件 | 核心接口 | 职责 |
|---|---|---|
| `coregraph/evidence_support/config.py` | `EvidenceSupportConfig` | 统一默认超参数、检查有效范围、导出实际配置 |
| `coregraph/evidence_support/retriever.py` | `EvidenceRetriever.retrieve` | 只读取证、结构化索引、余弦检索、BFS、RRF、来源回查、观察截止过滤 |
| `coregraph/evidence_support/service.py` | `EvidenceSupport` | 关系/问答公共服务、有界缺口补查、回答适配 |
| 同上 | `RelationEvidenceJudge` | 扩展既有双端关系 Judge，增加具体证据角色 |
| 同上 | `QuestionEvidenceJudge` | 审查问题所需证据是否充分，增加具体证据角色 |
| `experiments/evidence_support/run.py` | `run` | 参数读取、模型注入、独立输出、模型调用审计、来源文件哈希核验 |
| `config/evidence_support.json` | 参数文件 | 可修改的实验预算；模型地址与名称另由现有 YAML 提供 |
| `tests/test_evidence_support.py` | 固定输入回归 | 检索边界、时间过滤、补查、引用、下游回答协议 |
| `base/llm.py` | `_generate_internal` | 新模块显式传 None 时省略输出上限参数；旧调用的默认/显式预算兼容 |
| `coregraph/evidence_judge.py` | `_generation_options` | 隔离旧实验生成策略与新模块取消上限的策略 |
| `tests/test_llm_output_budget.py` | 传输参数回归 | 验证 None 真正省略 HTTP 参数、旧默认和显式限制仍兼容 |

复用代码：`GraphEvidenceProvider` 提供规范 Fact/Chunk 视图；`EvidenceJudge` 提供身份、作用域、模态、精度、时间及连续引文校验；`ConflictPairJudge` 提供关系分类和双端引用要求。新代码不依赖旧 `experiments/structural_semantic_retrieval` 包。

## 3. 执行流程与 LLM 分工

```text
更新：新旧 Observation ID                问答：问题＋可选 Fact ID
                ↓                                  ↓
直接回查锚点原文 → 结构/语义/图候选检索 → RRF 融合与来源去重
                ↓
Judge：核对比较条件、支持/反驳/解释角色、缺少什么
                ↓
NEED_MORE → 将具体缺口交回算法检索 → 有限轮数再次审查
                ↓
关系建议/未解决冲突/暂缓                  可选：依据已展示证据生成回答
```

| 步骤 | 是否调用 LLM | 输出性质 |
|---|---|---|
| Fact ID → Chunk 原文 | 否 | 已存数据回查 |
| StateSlot/事件索引查询 | 否 | 候选检索 |
| BGE-M3 编码＋余弦排名 | 不调用生成式 LLM；调用 Embedding 模型 | 语义相关候选 |
| 有限 BFS、RRF、去重、来源时间过滤 | 否 | 检索路径与排名 |
| 判断证据作用、比较条件与缺口 | 是，复用项目 vLLM/OpenAI 兼容接口 | 未独立验证的模型判断 |
| 根据缺口寻找材料 | LLM 提供检索请求，算法实际执行 | 补充候选材料 |
| 原文引用存在性、Fact/Chunk 绑定、预算检查 | 否 | 可机械验证的格式/来源检查 |
| 最终可选回答 | 是，独立一次调用 | 带引用回答，语义正确性另测 |

## 4. 各检索算法的具体含义

### 4.1 直接证据回查

通过中层 Fact ID 查其 `chunk_id` 或 `evidence_chunk_id`，再查 `chunk_storage`。新旧两端来源固定排在候选首部，不由余弦或 RRF 淘汰。同一 Chunk 的两个端点合并为一个来源，但保留两个 Fact ID。

这是精确索引读取，无需模型。引用的真实原文来自当前图，不能采用检索器或 LLM 提供的外部片段替代。

### 4.2 结构化索引查询

每个锚点独立查：

- 同 `event_id` 的断言；
- 同主体名称、规范属性、作用域、模态的 StateSlot 候选。

槽位不包含属性值，否则 A/B 不同值无法互相召回。模态和作用域参与检索键，不能据此直接判冲突；主体别名未统一仍可能漏检，语义路径可以跨主体名称补候选，最终身份由 Judge 核对。

索引使用 Python 字典，不是对所有同实体历史进行 LLM 比较。缺少主体或属性的时间/属性断言不建立空主体 StateSlot，仅通过明确 event_id 等路线定位。候选去重后按 ID 稳定截断；该路没有学习排序器，也没有按“较新”判真伪。

### 4.3 BGE-M3 稠密向量余弦召回

实际模型由现有 YAML 的 `embedding.model` 选择；当前实验配置是 `BAAI/bge-m3`。本实现只使用稠密向量，没有接入 BGE-M3 的稀疏或多向量检索。

Fact 编码内容包含主体、关系、值、作用域、模态、事件时间、原文引文，最多 `max_embedding_chars` 个字符。完整 Chunk 不截断存储。

初始查询分别使用原请求、新锚点＋原请求、旧锚点＋原请求。问答无锚点时只使用问题；补查使用“原问题＋需要核查的具体缺口”。候选池可跨主体，避免漏掉其他人物报告的反证或说明。

先对向量做 L2 归一化，计算：

\[
\mathrm{cosine}(q,f)=\frac{q\cdot f}{\|q\|\|f\|}
\]

每个查询独立取 Top-k。首版采用精确扫描，复杂度约为 O(QNd)，没有使用 FAISS/ANN。文本向量仅在当前实例缓存；更换模型应创建新实例或调用 `clear_cache()`。空向量、零向量、NaN/Inf、维度不一致和编码数量不一致明确记为不可用，不填假分数。

### 4.4 有限广度优先搜索 BFS

更新时以两个 Fact 分别作为锚点；问答没有显式锚点时，以语义候选中最多 `max_question_seeds` 个 Fact 为搜索种子。

允许经过的关联：

1. 上层事件分组中共同出现的 Fact；
2. 上层候选关联链接的两端；
3. 断言到中层 Triple 的明确引用，以及反向定位。

不直接遍历实体关联图中的所有同实体事实，也不把同 Chunk 的所有 Fact 构造成完全图。默认最多 2 跳、访问 64 个 `(锚点, Fact)` 状态，每个节点最多处理 12 个不同邻居；同目标的多种边合并记录，不重复占名额。不可见的未来节点不能充当中转。每条命中保留起点、Fact 路径和边类型。

**路径只解释“怎么找到”，不证明“为什么成立”。** 分组不代表已经确认同一事件，候选链接不代表确认因果。第一版不使用 PPR，后续可以在相同候选/证据预算下替换 BFS 做消融。

### 4.5 RRF 融合、路线保留和去重

每个锚点、每种查询保留独立排名。对候选 f：

\[
\mathrm{RRF}(f)=\sum_{r:f\in r}\frac{1}{k_{rrf}+\mathrm{rank}_r(f)}
\]

默认 `k_rrf=60`，路线等权；不直接把余弦与规则分数相加。先保留强制锚点，再尽量保留每路前 `reserve_per_route` 个独有候选，再按 RRF 补足；因此最终顺序是“保留约束＋RRF”，不是单一纯 RRF 排序。

Fact 按 ID 去重；随后按 Chunk ID 合并原文。不同 Fact 可保留相同 Chunk 引用。补查在各路 Top-k 截断前排除已见来源，但已见节点仍可作为图中转，避免已见高排名候选挤掉新材料。最大 Fact 数、Chunk 数和每 Chunk 展示 Fact 数分别限制。预算不足时记录被截断的来源及结构视图，不能声称检索穷尽。

## 5. 全部超参数

下列数值均为**工程初值，未通过真实样本调优，不是论文最优配置**。

| 参数 | 默认值 | 含义与作用 |
|---|---:|---|
| `structured_k` | 12 | 每锚点精确槽位/同事件路线的最大候选 Fact 数 |
| `semantic_k` | 12 | 每条语义查询的最大候选 Fact 数 |
| `max_hops` | 2 | BFS 最大跳数，0 关闭，允许 0–3 |
| `max_graph_nodes` | 64 | 每轮图检索最多访问的锚点/Fact 状态数，双端共享预算 |
| `max_neighbors` | 12 | 每个 BFS 节点最多考虑的邻居数 |
| `max_question_seeds` | 2 | 无显式锚点的问答自动图种子数，最多 4 |
| `rrf_k` | 60 | RRF 平滑常数，必须大于 0 |
| `reserve_per_route` | 1 | 截断前优先保留每条非空路线的候选数，0 关闭 |
| `max_candidate_facts` | 48 | 每轮融合 Fact 数上限，包含两个直接锚点 |
| `max_chunks` | 12 | 整次审查累计不同原文来源上限，允许 2–32 |
| `max_facts_per_chunk` | 4 | 每个原文展示的结构化 Fact 数，允许 2–4，匹配既有 Judge 的视图上限 |
| `embedding_batch_size` | 32 | 每批 Embedding 文本数 |
| `max_embedding_chars` | 1000 | 每条编码输入的字符上限，包含查询与 Fact 表示 |
| `min_cosine` | 0.0 | 语义候选余弦下限，允许 [-1,1]；不是可信度阈值 |
| `max_semantic_queries` | 7 | 整次审查语义查询路线预算，失败尝试也保守扣减；不是编码批次数 |
| `max_rounds` | 3 | Judge 最大轮数，允许 1–5，首版每轮一次调用，不额外协议修复 |
| `max_followups_per_round` | 2 | 每轮最多接纳的非重复缺口查询，允许 1–2 |
| `max_prompt_bytes` | 12000 | Judge 输入 UTF-8 字节预算，允许 6000–12000；不是精确 Token 计数 |
| `max_answer_tokens` | null | 默认取消可选回答的人为输出 Token 上限；显式正整数才启用限制 |

新模块 Judge 和默认回答**不发送 max_tokens/max_completion_tokens 参数**，取消原先 1200 Token 人为输出上限。内部以显式 None 绕过旧传输默认值，实际 HTTP 请求省略该参数；服务端依据剩余上下文和自身生成配置决定输出长度，并非物理无限。仍保留 finish_reason=length 的失败检查。旧 EvidenceJudge 入口保留原实验生成策略，不修改历史 Baseline。

其他约束保持明确：temperature=0；每个来源只展示以锚点附近开始的最多 1200 UTF-8 字节，窗口前留 80 个字符；最多 6 条证据角色与 6 条引用，原因最多 300 字符。这些是输入/结构协议约束，不是输出 Token 上限。遗漏或截断来源写入 `context`；同 Chunk 的两个端点相距过远可能无法同时展示，应暂缓，不放宽引文校验。

Embedding 不做在线训练；维度由真实模型返回，BGE-M3 通常为 1024。模型配置、名称、代码哈希和全部算法参数在 manifest 中记录。

## 6. 输入输出与调用示例

输入是已加载的精简图字典：`event_state_layer.schema=event_index_v1`、`fact_storage` 四张表、`chunk_storage`。关系入口首版要求两个已存在的 Observation ID；不要求两者已有 conflict 标签，也不预设关系。

```python
from coregraph.evidence_support import EvidenceSupport, EvidenceSupportConfig

settings = EvidenceSupportConfig(semantic_k=12, max_hops=2, max_chunks=12)
service = EvidenceSupport(graph, embedding_manager, llm_manager, settings)

relation_report = await service.review_pair(old_id, new_id, as_of='2024-01-15')
question_report = await service.review_question('目前在哪里工作？', as_of='2024-01-15')
answer_report = await service.answer_question('目前在哪里工作？', answer_llm,
                                               as_of='2024-01-15')
```

`as_of` 指**来源观察截止时间**，不是事件发生时间或事实有效时间。日期包含当天；带时区时间统一比较，不带时区视作同一 UTC 参考。截止时间启用时，来源 Chunk 时间必须可解析且不晚于截止，Fact 自带观察时间也不能更晚。未知时间来源排除；不能用 Fact 自报旧时间绕过未来 Chunk。`as_of=None` 表示当前整个快照；回放历史写入时必须传实际截止时间，不能用完整未来快照替代在线历史。

输出核心字段：

```json
{
  "schema": "evidence_support_v1",
  "status": "SUFFICIENT / NEED_MORE / CONFLICT / DEFER / NOT_EVALUATED",
  "retrieval_rounds": [],
  "rounds": [],
  "evidence_roles": [
    {
      "fact_id": "F3",
      "chunk_id": "C3",
      "quote": "A is full time; B is part time.",
      "target_fact_id": "F2",
      "claim": "B 的工作是兼职，可以与 A 的全职工作并存",
      "role": "supports"
    }
  ],
  "missing_evidence": [],
  "checks": {},
  "decision_proposal": null,
  "confidence": null,
  "update_applied": false,
  "validation_status": "not_verified"
}
```

上例只展示字段形式，不是实际模型结果。角色取 `supports/refutes/explains_change/background/uncertain`。每个角色须对应 citations 中完全相同的 `(Fact ID, Chunk ID, quote)`，并明确目标命题。角色通过机械绑定后仍标记 `citation_validated_semantics_model_assessed`，不是独立语义证明。

关系建议复用现有 `duplicate/complement/coexists/conflict/correction/replacement/unrelated/uncertain`。对应动作可为 RETAIN/REFINE/CORRECT/REPLACE/DEFER，但永远 `update_applied=false`。未解决冲突不自动选新 Fact。问答仅在本次审查 SUFFICIENT 时调用回答模型；回答引用必须来自 Judge 本次实际展示的来源，否则拒绝输出该答案。

## 7. 停止条件、失败与置信度边界

- Judge 返回 SUFFICIENT、CONFLICT 或 DEFER：停止，保留真实状态，不将冲突改成通过。
- NEED_MORE：仅补查其明确缺口；查询去重，来源按 Chunk 去重。
- 没有新查询、没有新增来源、来源预算满、Judge 轮数耗尽：DEFER。
- 没有 LLM：仅返回取证报告，状态 NOT_EVALUATED，不能称证据充分。
- Embedding 不可用：记录 `unavailable:*`，保留结构和图路线，不能宣称语义召回正常。
- 假引文、错 Fact/Chunk 绑定、错误角色、输出截断：保守 DEFER；非法角色单独留审计，不交给回答器。
- 回答调用失败或引用不合法：保留审查报告，answer 为 null，状态 failed/rejected。

`rrf_score`、`cosine_scores` 仅反映检索相关性，不是可信度。尚未实现来源真实性评分、多来源支持/反驳概率融合或分数校准。来源有 `source_id` 时仅据其分组，否则按 Chunk 分组；`independence_verified=false`，不同 Chunk 可能只是同一信息的转述，不计独立证据票数。

当前机械检查可确认引用存在及身份绑定；**不能独立证明 quote 蕴含 claim、回答每句话正确、现实事实真实或因果关系成立**。这些是后续人工标注、独立核验和真实数据实验的对象。

## 8. 独立运行与输出文件

服务器项目根目录 `/home/cwf/Code/LiCoMemory`，先激活 `licomemory`。仅算法演示，无 vLLM/BGE 调用：

```bash
python -m experiments.evidence_support.run --demo --retrieval-only --no-semantic --output results/evidence_support_demo_01
```

演示应返回 NOT_EVALUATED。它是人工协议样例，不是 LoCoMo。输出目录必须不存在，避免覆盖历史。

真实图关系审查（把路径、ID 替换为实际值）：

```bash
python -m experiments.evidence_support.run --source /path/to/graph.pkl --old-fact-id OLD_ID --new-fact-id NEW_ID --as-of 2023-10-22 --output results/evidence_support_pair_01
```

问答与证据审查：

```bash
python -m experiments.evidence_support.run --source /path/to/graph.pkl --mode question --query "用户目前在哪里工作？" --answer --output results/evidence_support_question_01
```

`--settings` 指定算法 JSON，`--config` 指定现有模型 YAML。`--retrieval-only` 关闭 Judge；`--no-semantic` 关闭 Embedding；`--compact-json` 仅用于已经确认支持扩展的 vLLM。离线缓存环境通过入口默认开启，不会自动下载模型。

| 文件 | 内容 |
|---|---|
| `manifest.json` | 来源哈希、实际超参数、代码哈希、使用的模型及是否人工演示 |
| `report.json` | 共用完整报告、候选路径、每轮审查、可选回答和机械核验状态 |
| `retrieval_rounds.jsonl` | 每轮各路 Fact、排名、余弦、BFS 路径、过滤/降级/截断 |
| `judge_rounds.jsonl` | Judge 检查项、引用、角色、缺口、实际展示窗口及原始输出 |
| `llm_trace.jsonl` | 有模型调用时保存完整提示词、响应、finish_reason 和 Token 用量 |
| `summary.json` | 运行状态、来源数、Judge 调用数及来源文件是否保持不变 |

## 9. 验证、版本与下一步实验

固定输入测试使用人工向量和固定 JSON，检验真实算法及边界，不检验 BGE/vLLM 的推理质量。测试包括双端保留、不同取值槽位、跨主体候选、问答自动图种子、BFS 深度/节点预算、未来/未知时间过滤、同 Chunk 双端、错误向量、不可变性、按缺口实际新增反证、错误角色、无 Judge、回答失败与假引用。

本次目标是实现协议与可复现运行，不重建图、不宣称 SOTA。后续使用冻结的独立样本标注比较：仅双端原文、普通语义检索、结构＋语义＋BFS＋RRF、加入缺口补查；保持模型与证据/调用预算一致，分别统计关键证据召回、关系正确率、错误更新建议率、暂缓表现、问答准确性和成本。

版本追溯使用运行时 `code_sha256`，以及 `git log -- coregraph/evidence_support docs/modules/evidence_support.md`。修改算法、参数或下游采纳策略时必须同步更新本说明书。

## 10. 本次实施核查（2026-10-29）

- 本地：新模块 27 项、既有证据 Judge/冲突审查/固定事实对协议 38 项，共 65 项通过。
- 服务器：新模块 27 项通过，正式 Python 包与独立 CLI 能实际导入运行。
- 服务器人工无模型演示：3 个原文来源、0 次 Judge 调用、NOT_EVALUATED、update_applied=false。
- 本地完整项目环境缺少依赖，回归采用测试中已有的轻量包加载方式；服务器演示使用正常导入，不使用此适配。
- 独立审查发现并修复：空主体跨事件错槽、多类型边重复占邻居预算、已见来源挤占补查 Top-k、错误回答类型逃出校验。各项均先复现失败，再运行通过。
- 服务器演示目录：`/home/cwf/Code/LiCoMemory/results/evidence_support_implementation_20261029/server_demo/`。
- 本轮没有调用真实 BGE-M3/vLLM、没有运行 LoCoMo 全量评测或重建图。支持/反证识别质量、关系准确率、错误更新率与问答收益仍待独立实验。

### 后续输出策略修订

真实图首例运行发现原 1200 Token 输出预算导致 finish_reason=length。现按用户要求取消新模块 Judge 与默认回答的人为输出上限，增加传输参数回归；不会悄悄退回底层的 1000 默认值。`llm_trace.jsonl` 记录 output_token_policy 和 requested_max_tokens，便于核对运行是否使用新策略。原首轮结果保留，不覆盖或改写。
