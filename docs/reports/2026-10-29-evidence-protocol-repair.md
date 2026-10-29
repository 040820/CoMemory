# 五项证据协议修复与验证记录

## 修改结果

1. 写入关系使用 old/new 证据编号并由程序绑定原句，取消 512 Token 上限，保留服务端截断检查和一次协议修复。
2. Judge 使用实际展示片段的编号枚举；双端分别必填、原始字符位置可回溯；重叠窗口合并，避免重复原文挤占预算。技术失败单独标注，不进入语义 uncertainty。
3. 复用现有 sample0 图重放失败关系，追加候选链接和审计到独立图副本，保留旧失败记录及全部原始事实。
4. 修复旧检索导入、LongMemEval docstring 和日期格式；问答保留原结构化语义 Chunk 候选入口，再接共用证据审查及编号引用回答。
5. 按目标 Fact 分开计算支持/反驳，关系命题单独评估；作为 shadow 输出，不改变 Judge、排序或事实有效性。

算法、代码职责、配置及边界见 [模块说明书](../modules/evidence_protocol_repair.md)。

## 验证结果与限制

| 验证 | 结果 | 能说明什么 |
|---|---|---|
| 服务器相关回归 | 153 项通过，0 failure、0 error | 协议、存储兼容、日期、旁路评分和边界行为通过固定输入测试 |
| 五条敏感失败预检 | 4 条有关系判断、1 条语义不确定，0 技术失败 | 新编号协议可运行，不证明关系标签正确 |
| 原十对复查 v2 | 9 SUFFICIENT、1 端点窗口预算技术失败 | 诊断结果；不作为最终准确率 |
| 上述预算个案 v3 | SUFFICIENT，端点已实际展示 | 该输入预算问题修复；没有拼成同版本的十题指标 |
| 全量失败重放 | 299/299；271 model_assessed、28 semantic_uncertain、0 technical_error | 技术协议问题解决；未经人工金标验证关系语义 |
| 候选图副本 | 链接 351→468，新增 117 条 | 候选关联增加，不代表执行了 117 次更新 |
| 单题完整流程 v6 | 2023-05-07，D1/Chunk 1，source_grounded | 该调试题恢复正确；不能推出数据集总体效果 |

全量重放的模型关系分布：complement 45、coexists 8、replacement 47、correction 17、unrelated 154、uncertain 28。replacement/correction **均为待审查候选**，没有确认旧事实失效。

原始三层来源仍是 214 Chunk、912 原始 Triple、292 来源断言；全量候选副本保留原始事实与有效性。五条预检副本已做实际 load/save 验证，新增链接能恢复；既有加载器会把某些缺省空引用列表规范化为 []，这不是事实内容或有效性变化。

开发诊断 v1–v5 都保留：其中 v4 曾审查充分但答案引用失败，v5 曾合法引用却混淆 support group 与 activist group。修复后 v6 复用了原查询入口。该题被多次用于调试，不是未见测试集，不能报告为泛化准确率提升。

## 产物位置

服务器根目录：

```text
/home/cwf/Code/LiCoMemory/results/evidence_protocol_repairs_20261029/
```

- `unit_tests_final_153.json` / `.log`：最终回归。
- `verification_v2/replay_all_failures/`：299 对逐条判断、原始模型响应、候选链接补丁及独立图。
- `verification_v2/replay_all_failures/candidate_graph.pkl`：本次补充候选的图副本；来源图不覆写。
- `window_repair_v3/pair_window/`：窗口预算个案复查。
- `question_entry_v6/`：保留原查询入口后的单题检索、Judge、回答和引用审计。
- `reproducibility/`：同步前备份、源码快照和文件指纹。

部分摘要及报告同步到本地 `results/evidence_protocol_repairs_20261029/`。置信度尚未做人工标注、校准或独立效果评测；默认仅旁路记录。
