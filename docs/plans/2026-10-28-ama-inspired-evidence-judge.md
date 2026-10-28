# AMA 启发的证据 Judge 调度计划

目标：复用 AMA 的证据充足性、同一事件/状态的冲突检查与有限补检索思路，
接入当前三层记忆，不移植 AMA Refresher，不执行覆盖/删除，不修改冻结实验。

设计：Judge 接收任务、新事实及有限的事实/Chunk 证据，输出 SUFFICIENT、
NEED_MORE、CONFLICT 或 DEFER，附连续原文引用、缺口和后续检索问题。
Controller 在上限内补查关联来源、同 Session 前后文或语义检索，再次 Judge。
CONFLICT 只是发现待处理冲突，不等于接受新说法；预算耗尽、解析失败或无新增证据
明确 DEFER。无数值置信度融合或自动状态更新。

文件职责：
- coregraph/evidence_judge.py：标准库 Judge 协议、引用校验与有界控制器。
- coregraph/graph_evidence_provider.py：从既有图只读组织事实、来源和局部扩展。
- experiments/integrated_event_state/run_evidence_judge.py：独立启用并复用现有检索器，输出审计。
- tests/test_evidence_judge.py：固定输入、预算、引用和失败行为检查。

实施顺序：先失败测试，再协议/控制器，再图适配与独立入口，最后检查、审阅、提交与同步。
默认最多 3 次 Judge 调用、每次检索 Top-5、最多 8 个 Chunk；模型输入按 UTF-8 字节
保守限额，保留丢弃/截断标记，不把轮数用完当作证据充分。

参考来源：D:/Code/AMA-main/python/AdaptiveMemory/{Model/model.py,Settings/prompt.py,Core/AMA.py}。
复用调度思想与判定原则，代码独立实现，不直接复制其数据库更新或提示词正文。
当前工作区的用户修改保留；新增模块通过 CLI 显式启用，不自动挂接既有写入/问答路径。

完成记录：新增 19 项 Judge/图适配固定输入检查通过，连同既有 43 项共 62 项
在服务器隔离快照中通过。独立审阅发现并修复“失败检查被忽略”和“冲突缺少双方证据”
两类问题；接口检查发现模型滥用 not_applicable，身份/精度必要项已增加明确约束。
人工养猫新增案例在真实 BGE-M3＋vLLM 下独立运行，最终严格版返回 SUFFICIENT，
两段来源保留且 update_applied=false；该结果仅验证接口，不是 LoCoMo 效果评测。
来源/模型输入输出/代码指纹保留在服务器 results/evidence_judge_probe_20261028。
