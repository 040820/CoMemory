# 事件与状态演化层实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在不修改原有写入、图存储和检索逻辑的条件下，新增可追溯的单值状态与变化索引，并让独立检索器可选择使用它。

**Architecture:** 新实验目录提供只读 CogniGraph 适配器、LLM 判定器、SQLite 状态库及独立检索入口。适配器从已有图边和 `chunk_storage` 产生可核对的三元组候选；判定器只处理受控属性；状态库在事务内保存基线、变化、纠正和证据。原始 `.pkl` 与 Baseline 代码保持原样。

**Tech Stack:** Python、标准库 `sqlite3` / `unittest`、现有 `DynamicMemory` / `LLMManager` / `StructuralSemanticRetriever`。

**Spec:** `docs/superpowers/specs/2026-10-25-event-state-evolution-design.md`

## Global Constraints

- 只新增 `experiments/event_state_evolution/` 及其测试、说明；不改 `coregraph/`、`query/`、原有实验脚本或数据。
- 新增代码写详细中文注释；状态数据库放独立输出目录，不提交 Git。
- 第一版属性仅为 `employer`、`job_title`、`residence`；一个变化对应一个实体和一个属性。
- 必须关联已有实体索引、三元组语义和 Chunk；不能定位到原文的候选不写入有效状态。
- 首次出现是基线；隐含的新旧状态差异可记推断变化；纠正不算变化，保留原始 Chunk。
- 第一版按现有会话顺序处理，不自动支持乱序旧会话补录；`event_time` 未知时保持空值。
- 每完成一项独立改动即提交 Git；最终推送到 `origin/main`（用户已授权后续修改保留 GitHub 版本）。

## Review Focus

- 同名不同人：不能仅凭名字相同合并，无法唯一定位时进入待核查。
- 合并后的图边：关系可能并非属于其列出的每个 Chunk，必须让 LLM 结合原文核对。
- 回顾性陈述：较晚观察到的过去状态不得覆盖当前状态。
- 历史纠正：一月事实被纠正后，二月、三月独立支持的当前值必须保留。
- 重跑任务：同一 Chunk / 三元组来源不得重复产生基线或变化。

---

### Task 1: 只读证据适配器与稳定来源键

**Files:**
- Create: `experiments/event_state_evolution/evidence.py`
- Create: `experiments/event_state_evolution/tests/test_evidence.py`

**Interfaces:**
- `collect_candidates(memory) -> list[EvidenceCandidate]`；候选包含 `entity_name`、三元组主谓宾、`chunk_id`、`session_id`、`observed_at`、原文。
- `EvidenceCandidate.source_key` 由图文件的命名空间、Chunk ID 和三元组内容生成稳定摘要，供状态库幂等去重；无图边的 Chunk 保留为待判定候选，以免漏掉纯文本纠正。

- [ ] 先写测试：一条边的多个 `chunk_ids` 只产出有原文的候选；无图边的 Chunk 仍可作为待判定证据；同一输入来源键稳定；缺失 Chunk 不虚构证据。
- [ ] 运行 `python -m unittest discover -s experiments/event_state_evolution/tests -p 'test_evidence.py' -v`，确认因接口缺失失败。
- [ ] 实现只读适配器；保留图边原始主谓宾和 Chunk 原文，不把图边与全部 Chunk 的对应关系当作已证实事实。
- [ ] 重跑该测试，确认通过；提交 `feat: add event state evidence adapter`。

### Task 2: LLM 结构化判定与保守校验

**Files:**
- Create: `experiments/event_state_evolution/decision.py`
- Create: `experiments/event_state_evolution/tests/test_decision.py`

**Interfaces:**
- `async classify(candidate, current, llm) -> Decision`；`Decision` 含受控 `attribute`、规范化值、`kind`（baseline/change/correction/same/retrospective/review/ignore）、`detection`、目标旧事实、`event_time`、`event_group_id`、证据片段。
- 判定结果必须通过字段白名单、原文支持、主体一致性、三元组与 Chunk 一致性校验；无已有图边时，只有能从原文明确抽取三元组并映射到已有实体索引，才可建立新的旁路事实快照，否则进入 `review`。

- [ ] 先写测试：空/坏 JSON、越界属性、幻造时间、无证据关系、同名歧义转待核查；当前 B 与新 C 可产生推断变化；明确“从来不是 B”为纠正；回顾性语句不覆盖当前值。
- [ ] 运行对应 `unittest`，确认因接口缺失失败。
- [ ] 实现中文提示词、LLM 调用和纯函数解析校验；`LLMManager.generate()` 返回空串时不写有效事实。
- [ ] 重跑测试，确认通过；提交 `feat: classify supported state changes`。

### Task 3: SQLite 状态索引与时间线

**Files:**
- Create: `experiments/event_state_evolution/store.py`
- Create: `experiments/event_state_evolution/tests/test_store.py`

**Interfaces:**
- `StateStore(path)` 提供 `apply(candidate, decision)`、`get_current(entity_id, attribute)`、`get_history(entity_id, attribute)`、`get_by_fact(fact_id)`、`get_by_chunk(chunk_id)`。
- 表包括实体映射、事实快照、当前状态、变化、证据链接、待核查；约束 `source_key` 唯一，索引覆盖正向与反向查询。

- [ ] 先写临时库测试：首次事实只有基线；B→C 留新旧证据；相同事实重复写入幂等；纠正历史 B 后当前 D 保留且受影响推断变化重算；多属性共用事件组；写入失败整体回滚。
- [ ] 运行对应 `unittest`，确认因接口缺失失败。
- [ ] 实现 SQLite schema、事务和受影响实体属性的时间线重建；无证据的决定只进待核查表，不覆盖当前状态。
- [ ] 重跑测试，确认通过；提交 `feat: persist event state timeline`。

### Task 4: 独立写入命令与可选检索入口

**Files:**
- Create: `experiments/event_state_evolution/build_index.py`
- Create: `experiments/event_state_evolution/retrieve.py`
- Create: `experiments/event_state_evolution/README.md`
- Create: `experiments/event_state_evolution/tests/test_workflow.py`

**Interfaces:**
- 命令接收 `--config`、`--graph`、`--output-db`，只读加载图、按会话顺序处理候选、断点续跑并输出处理/待核查计数。
- `retrieve_with_state(query, retriever, store, top_k, mode)` 的 `mode` 为 `off/current/history`，由调用者显式选择；返回原有 Chunk 证据及独立的 `state_evidence`。`current` 的状态证据只含有效事实，`history` 保留旧版本，`off` 与原检索结果一致；原检索器及评测脚本不改。

- [ ] 先写工作流测试：合成图+假 LLM 的端到端索引；关闭状态入口时与原检索结果一致；当前问题不返回已纠正事实作为当前证据；重跑无重复记录。
- [ ] 运行对应 `unittest`，确认因接口缺失失败。
- [ ] 实现独立 CLI、检索包装器和中文操作说明，明确现有合并图缺失的 Triple 无法恢复。
- [ ] 重跑工作流测试和 `python -m unittest discover -s experiments/event_state_evolution/tests -v`；运行项目已有轻量测试；确认 `git diff --check` 通过后提交 `feat: expose event state index workflow`。

## 交付检查

- 在无 vLLM 的本地环境，用合成 Chunk、Triple 和假 LLM 响应验证存储与检索逻辑；真实模型效果必须待服务器可用后另做逐题实验。
- 核对新索引数据库、日志和待核查输出均位于独立目录；原图 `.pkl`、原检索模块、实验结果无差异。
- 推送所有提交到 `origin/main`，记录提交号与测试命令；说明尚未验证的真实数据质量与 LLM 误判风险。
