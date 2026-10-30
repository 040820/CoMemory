# Fact-grounded Memory Evolution for QA Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将已校验 Fact 与可追溯的关联、更新决策及状态版本连接，并通过固定输入的 QA 对照检验效果。

**Architecture:** 以现有 `event_index_v1` 和事实表为输入，增加独立 v2 演化状态与决策；新增路线通过显式配置启用，旧模式保持一致。写入先保存有来源的 Observation，在隔离视图审查旧新证据，再由可测试的策略提交版本；查询在统一时间过滤下按需补充关联及状态证据。

**Tech Stack:** Python、NetworkX、pickle、unittest、现有 BGE-M3/vLLM 接口。

**Spec:** `docs/superpowers/specs/2026-10-30-fact-grounded-evolution-qa-design.md`

## Global Constraints

- 不覆盖已有用户改动、原图、实验结果和人工标注；新实验独立目录，冻结代码和输入。
- 引文、主体、Triple、时间和模态校验失败不能降级为写入；候选关系不等于事实或更新。
- 旧配置及 v1 查询兼容；默认关闭新路径或只产生 shadow 建议。
- Gold 不进入构建、检索、Judge 或 Reader；LongMemEval 使用完整 haystack。
- 重要新增/修改逻辑使用中文注释；实际效果仅由完成的验证和对照实验报告。

## Review Focus

- 跨主体纠正和否定表达不能被普通同主体过滤静默漏掉或误更新。
- 后到证据不能通过状态、关联、原始 Chunk、Judge 或 Reader 泄漏至过去问题。
- 失败和待确认不得假装更新成功或让旧状态保持确定；未知有效时间不得变成无限有效期。
- 并发、崩溃和重复执行不能产生半次提交或重复决策。
- QA 对照共享基础图、数据和预算；相同样本应逐步记录候选与实际展示，不能用全库包含答案冒充命中。

## Task 1: 证据与更新策略

**Files:** create `coregraph/update_policy.py`, test `tests/test_update_policy.py`.

**Interface:** `decide(candidate: dict, evidence_report: dict, context: dict) -> dict` returns action, checks, reason, evidence and status without modifying input. Covers `ADD`, `SUPPORT`, `KEEP_BOTH`, `UPDATE`, `CORRECT`, `PENDING`.

- [ ] Add failing tests for explicit change, explicit correction, independent coexistence, absent or invalid quote, unknown/conflicting evidence, technical error, ambiguous targets and cross-subject correction.
- [ ] Run tests and verify expected failures; implement strict source/identity/scope/modality/time/transition gates with Chinese comments.
- [ ] Run policy tests; verify uncalibrated confidence alone cannot approve an update.

## Task 2: 版本化存储与提交

**Files:** create `coregraph/state_version_store.py`, modify `coregraph/lean_event_index.py`, `coregraph/dynamic_memory.py` and `coregraph/graph_evidence_provider.py`; test `tests/test_state_version_store.py`, existing lean/provider tests.

**Interfaces:** `StateVersionStore.commit(decision, expected_revision)`, `get_state(slot, valid_time, knowledge_cutoff)`, `history(slot, knowledge_cutoff)`; versioned event-index view with preserved v1 behavior.

- [ ] Add failing tests for raw evidence span integrity, idempotent transitions, unknown validity, pending conflict, as-of history, v1 isolation and v2 round trip.
- [ ] Add failing tests for revision conflict, lock, interrupted commits and missing audit snapshot; implement atomic single-writer commit with content-addressed audit and independent idempotency key.
- [ ] Run new and existing persistence/provider tests; confirm no implicit promotion of v1 candidates.

## Task 3: Fact 构建连接及配置

**Files:** `coregraph/event_state_writer.py`, `coregraph/dynamic_memory.py`, `init/config.py`, isolated config, tests for integration.

**Interfaces:** opt-in `off/shadow/review/gated`; staged, source-validated new Fact is available to pair review; `shadow` records proposals without applying updates.

- [ ] Add failing tests that successful observation survives failed review, invalid extraction remains review-only, shadow never changes state, explicit cross-subject target is found, negation and planned/actual comparisons remain distinct.
- [ ] Implement staged read-only graph and bounded review; wire policy/store, preserve existing path in off mode; add Chinese comments.
- [ ] Run integration and current writer/config regressions.

## Task 4: 关联检索和时间可见性

**Files:** `coregraph/evidence_support/query.py`, `retriever.py`, `service.py`, provider; focused evidence-support tests.

**Interfaces:** existing query with opt-in related-fact/state/decision overlay; trace seed, added and dropped candidates, shown windows and citations; distinct valid time and knowledge cutoff.

- [ ] Add failing tests for multi-session addition, shared evidence budget, no duplicate source, future-dependent graph/decision/chunk omission, temporal unknown and baseline off-mode identity.
- [ ] Implement bounded expansion and unified time visibility through every stage, never treat candidate edges as proof; preserve the final reader’s actual source windows.
- [ ] Run evidence and temporal regression tests.

## Task 5: 固定样本评估与报告

**Files:** create isolated experiment runner and metrics under `experiments/`, focused tests, Chinese module documentation and results directory.

**Interfaces:** frozen A/B/C configuration, per-case retrieval and response trace, source/decision metrics, matched denominators and same context/model budgets.

- [ ] Add failing tests for gold isolation, candidate headroom, failed/unknown denominator, unanswerable handling and paired trajectory differences.
- [ ] Implement reproducible runner: A baseline, B association overlay, C association plus decisions; preserve original inputs, logs, audits, graph hashes and prompts.
- [ ] Run appropriate unit/integration checks and a feasible real-data pilot with identical frozen examples and existing server models; report complete versus partial results, raw metrics, per-type QA, evidence and costs. Do not claim calibration without independent labels.

## Final verification

- [ ] Run targeted tests and full relevant regression suite in an environment with project dependencies; record failures and environment.
- [ ] Check every user-facing code change for Chinese comments, source/time safety and preserved old modes.
- [ ] Review the implementation against the approved spec, especially graph provenance and experiment trace.
- [ ] Summarize actual effect and limits; never infer QA improvement solely from synthetic tests.
