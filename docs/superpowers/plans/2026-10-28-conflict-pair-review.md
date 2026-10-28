# 冲突事实对与图优先取证 Implementation Plan

**Goal:** 写入阶段标记潜在冲突，后续从上层链接锁定两端，沿图取证并保存可解释决策建议。

**Architecture:** 复用 append-only 写入与已有候选 links；新增待审查索引、双端 Judge 和独立入口。只在新入口启用图优先补查，既有问答和默认 Judge 行为不变。没有校准置信度与完整采纳政策，不自动使历史事实失效。

**Tech Stack:** Python、既有三层 pickle/事实表、BGE-M3、vLLM、unittest。

- [x] 先写测试：冲突类链接才进入队列；SUFFICIENT 只引新事实不能通过；双端证据合法且语义条件通过才生成决策建议；图扩展先于语义兜底。
- [x] 新增 conflict_review.py：链接索引、双端 Judge、绑定两端的证据适配器。
- [x] 为通用 Judge 提供可覆写 schema，为 Controller 增加默认关闭的图优先选项。
- [x] 上层冲突/纠正/替换链接保存 pending 标记，旧格式链接按标签兼容识别。
- [x] 新增独立 review_conflict_pairs.py：记录证据、原始响应、逐轮结果与决策建议，不修改来源图。
- [x] 运行针对性测试，保存代码版本并同步服务器；不重跑既有 sample0 构图。

验证：新增冲突审查 9 项、原 Judge 19 项、精简存储 7 项均通过。只读读取已有
sample0 新图得到 41 条待审查候选，不代表已确认冲突或已完成真实 LLM 效果评测。
