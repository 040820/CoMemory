# Evidence Support Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 为新旧 Fact 关系审查和问题回答建立共用的只读证据支撑模块，并提供中文模块说明。

**Architecture:** 三层图不变。独立模块以 Fact/Chunk ID 回查原文，进行结构化索引查询、BGE-M3 稠密向量余弦召回、有限 BFS 图扩展及 RRF 融合；复用已有 Judge，由有界控制器按缺口补查，输出分项证据角色、引用和决策建议。回答仅消费本次通过审查的证据。

**Tech Stack:** Python 标准库；现有 EmbeddingManager、EvidenceJudge、ConflictPairJudge；运行入口延迟加载现有模型依赖。

**Spec:** 用户已确认的共用证据模块及算法方案；完整接口、默认超参数和边界记入 `docs/modules/evidence_support.md`。

## Global Constraints

- 保持 Chunk/Fact/上层索引及历史实验不变；不自动执行更新，不伪造置信度。
- 新增代码有中文注释，注明所属模块与用途；不导入用户正在删除的旧实验检索包。
- 在当前授权工作区增量增加独立文件，保留用户未提交修改；只提交本次文件。
- 初始双端原文固定保留；语义补查可以跨实体，候选关联和 BFS 路径不作为证明。
- as_of 过滤同时约束 Fact 和 Chunk；缺时间无法证明可见时排除，非法截止时间直接报错。
- 保存实际参数、检索路由/排名/图路径、降级、预算和模型原始响应，所有输出另存新目录。

## Review Focus

- 同 Chunk 双端、重复路径、哈希 ID 不得丢失 Fact 身份。
- 历史过滤不得通过图扩展、原文拼接或语义索引泄漏未来信息。
- 一侧独有证据和反证不得被“同时接近两端”的要求过滤。
- 向量失败、引文伪造、Judge 格式错误、预算不足必须可见并保守暂缓。
- 模型的支持标签不等于独立验证；重复来源不视为独立佐证。

## Tasks

### 1. 检索器与参数
- [x] 新增 `tests/test_evidence_support.py`，先检验结构召回、跨主体语义、BFS 深度/预算、RRF 去重、双端保留、时间过滤和不可变性。
- [x] 运行失败用例，确认新 API 尚未实现。
- [x] 新增 `coregraph/evidence_support/config.py`、`retriever.py`、`__init__.py`；执行固定输入回归。

### 2. 共用证据审查与下游接口
- [x] 测试有界缺口补查、重复查询停止、引用/角色绑定、预算停止、关系建议不落盘、回答来源校验。
- [x] 新增 `coregraph/evidence_support/service.py`，复用原 Judge，明确区分模型判断和机械校验。
- [x] 新增独立 CLI `experiments/evidence_support/run.py` 和参数 JSON；保留完整审计，不改主流程。

### 3. 中文说明、验证与交付
- [x] 完成 `docs/modules/evidence_support.md`：职责/文件/接口/算法公式/参数/示例/失败路径/能力边界/运行方法/测试与版本。
- [x] 执行新模块回归、已有 Judge 回归及独立 CLI 固定输入演示；不调用真实 vLLM、不重建图。
- [x] 审查差异；提交本次独立文件并推送 GitHub；服务器同步前核对现有文件，验证只读演示。

## Execution ledger

- 设计按用户批准的接口与算法继续执行，不再引入新存储层或 PPR。
- 目前本地 `experiments/structural_semantic_retrieval` 被用户删除；新模块不依赖该目录，不恢复其文件。
- 本地完整项目导入缺依赖；纯模块测试采用既有轻量包加载方式；服务器另验实际包导入。

- 完成核查：27 项新模块＋38 项既有协议回归通过；服务器 27 项通过，正常包导入的人工 CLI 演示返回 NOT_EVALUATED/3 来源/0 次模型调用。
- 独立审查发现四项 P2，均以失败回归复现并修复；新增 RRF 数值和已见节点中转检查。
- 置信度未决仍保留 null，关系建议不自动采纳；未运行真实模型与数据集效果验证。
