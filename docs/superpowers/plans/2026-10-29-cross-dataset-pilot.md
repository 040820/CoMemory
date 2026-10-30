# Cross-dataset Memory Pilot Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 用 LoCoMo 和 LongMemEval 的少量真实样本衡量当前系统，保留可人工核查的证据与演化候选。

**Architecture:** 冻结分层随机选题；LoCoMo 复用完整来源的新图，LongMemEval 每题使用完整 haystack 独立建图。运行当前 EvidenceSupport 问答，按原生来源标签计算覆盖率，使用独立评分调用复用现有问答 rubric；关系正确性与置信度校准无金标时保留空值。

**Tech Stack:** Python、unittest、现有 BGE-M3/vLLM、JSON/NDJSON。

**Spec:** 用户要求先用两个数据集的一小批样本衡量上一轮讨论的研究挑战。

## Global Constraints

- 固定随机种子 20261029：LoCoMo sample0 类别 1/2/4 各 4 题，类别 5 额外 2 题；LongMemEval ku/tr/ms 各 1 题。
- 不删会话、不按标准答案筛上下文、不截断 LongMemEval 的整个 haystack。
- Gold 答案和来源仅供评分/人工核查，不进入构图、检索或 Reader/Judge 提示词。
- 保留原始 LongMemEval 时间精度并按时间稳定排序，使用独立实验语料，不改既有转换数据。
- 原图/旧结果保留；每会话保存检查点；开始后只确认启动，不持续轮询。
- 同模型自动评分明确标为代理指标，不冒充人工准确率或 SOTA 结果。

## Review Focus

- 缺失/非法标签不可默认为命中；不可回答题没有 gold source 时来源覆盖率为 N/A。
- Seed、候选、实际窗口、最终引用四个阶段分别统计，不能用全库包含答案冒充已展示证据。
- 技术 DEFER 与语义 DEFER 分开；失败仍在端到端分母内。
- 恢复任务须使用冻结选题、代码和配置，不能隐式换样本。
- LongMemEval 独立图不混入其他问题的来源；原生标注不能推出 Fact 纠正/替换金标。

## Tasks

- [ ] 1. 纯函数选样、完整会话转换和来源覆盖指标；先写失败回归。
- [ ] 2. 独立运行器、模型审计、逐会话检查点、问答与评分报告、人工关系标注导出。
- [ ] 3. 模块说明、冻结快照、服务器预检并启动；提交代码，交付结果目录。
