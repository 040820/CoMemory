# State Slot 与证据门控实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 只有同一状态槽位、互斥的新旧值及有原文依据的转换同时成立，才写入上层变化事件。

**Architecture:** Chunk 和 Triple 不变。候选事实增加作用域、模态与属性基数；纯逻辑模块先判槽位与默认更新语义。对可能替换的单值状态，独立 LLM 判定新旧事实和原文，并由程序复核三道门控及证据引文。事件层保存操作与证据；并存事实保留为集合成员，不进入单值 current 槽。

**Tech Stack:** Python 3.11、现有 vLLM/OpenAI 兼容接口、pickle 图文件、unittest。

**Spec:** 用户当前提供的 State Slot 与三道证据门控描述；sample0 误判审计 `experiments/integrated_event_state/reports/sample0_event_misclassification_audit.md`。

## 全局约束

- 不改变原始 Chunk、Triple 和旧检索流程；新构建使用独立结果目录。
- 旧版图可读取；旧版事件不得被自动当成新门控的结果继续演化。
- 所有新增注释使用中文；代码、测试和验证记录保存到 GitHub，并同步服务器。
- 缺少作用域、模态或转换依据时采取保守处理，不能仅因时间更晚或字符串不同写入 REPLACE。

## Review Focus

- 同人养狗又养猫：两个 ADD，无 REPLACE。
- 同人不同项目角色：作用域不同，不能互相替换。
- 计划与已经发生：模态不同，不能互相替换。
- 雇主 B 到 C：只有门控三项和引文校验通过才 REPLACE。
- 历史纠正：撤销相关旧事件，不凭剩余事实的时间顺序重新造变化。

---

### Task 1: 槽位与属性语义

**Files:** Create `coregraph/state_slot.py`; Test `tests/test_state_slot.py`。

- [ ] 写失败测试：集合属性、项目作用域、模态、单值属性、未知属性与值细化。
- [ ] 运行测试，确认按预期失败。
- [ ] 实现标准化槽位键、属性基数与保守预判。
- [ ] 运行测试并提交。

### Task 2: 事件层操作语义

**Files:** Modify `coregraph/event_state_layer.py`; Test `tests/test_integrated_event_writer.py`。

- [ ] 写失败测试：ADD 集合、REPLACE 门控、REFINE/NOOP/UNCERTAIN、纠正与撤销。
- [ ] 运行测试，确认按预期失败。
- [ ] 扩充可持久化字段；只有三道门控全通过才创建 REPLACE；保持旧版读取能力。
- [ ] 运行测试并提交。

### Task 3: 写入器独立证据门控

**Files:** Modify `coregraph/event_state_writer.py`; Test `tests/test_integrated_event_writer.py`。

- [ ] 写失败测试：抽取候选后按旧事实和原文判断，拒绝假纠正和无依据替换。
- [ ] 运行测试，确认按预期失败。
- [ ] 提示词生成带 scope/modality 的候选；对潜在替换执行单独判断并校验引文。
- [ ] 运行测试并提交。

### Task 4: sample0 独立验证

**Files:** New config or result directory; new audit report under `experiments/integrated_event_state/reports/`。

- [ ] 在服务器新目录构建 sample0，不覆盖历史图和日志。
- [ ] 审计全部新变化事件的证据与误判；抽查查询入口。
- [ ] 保存原始日志、审计 JSON 与报告，同步 GitHub。
