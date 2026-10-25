# Integrated Event State Layer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在新写入模式中以事件与状态演化层替代 Session Summary，保持 Query-driven Structural-Semantic Retrieval 为主要检索方式。

**Architecture:** 原始 Chunk 和实体关系图照旧构建。图构建后立即从本次 Chunk、关系、既有状态判定状态事实与变化；把事件、实体反向索引和证据引用随图一起持久化。旧摘要模式仍可选择，以便复现实验。

**Tech Stack:** Python、现有 `DynamicMemory`、`GraphRAGCore`、`LLMManager`、NetworkX、pickle。

**Spec:** 本次对话中用户确认的三层架构：上层事件状态、中层实体关系、下层原始 Chunk；原结构化语义检索保持主入口。

## Global Constraints

- 新模式不生成 Session Summary，不依赖其检索候选筛选。
- 状态属性开放，但只记录持续事实的新增、变化或纠正；普通活动跳过。
- 每条事件须关联现有实体索引，记录相关关系和 Chunk 原文证据。
- 不覆盖原文和历史事实，图的旧格式仍可加载。
- 用户明确允许此轮先写代码、不做测试；交付时不得声称已验证。

## Tasks

- [x] 新建事件状态数据层：事实、变化、纠正、当前状态与实体/关系/Chunk 反向索引。
- [x] 新建 LLM 写入判定器：根据本次 Chunk 和既有状态抽取任意属性，核验主体、证据、关系与时间。
- [x] 把新模式接入批量和增量记忆写入；图保存/加载包含事件层，摘要模式独立保留。
- [x] 提供使用现有结构化语义检索器的事件查询包装器与新模式配置说明。
- [x] 提交 GitHub 版本，代码同步服务器；本轮不运行测试。
