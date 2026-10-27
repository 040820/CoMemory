# sample0 写入检索对比实验

## 比较对象与控制变量

比较 `legacy`（之前的语义与槽位加分）、`structured`（仅 StateSlot）、`semantic`（仅原始 BGE-M3）、`hybrid`（双路融合）。四路最终 Top-k 均为 5。事件观察走统一活动候选流程，不计入状态召回消融分母。

四种方法对每条新状态候选使用相同新事实和相同历史。统一抽取、归一化和追加写入由 hybrid 驱动，影子召回不修改存储；因此这是固定事实流下的条件召回比较，不能当作四套写入系统各自独立运行的端到端效果。共享 B 路向量/缓存，不做速度优劣结论。

## 保留的结果

| 文件 | 内容 |
|---|---|
| `write_retrieval.jsonl` | 所有尝试的归一前/后四路召回，新旧事实全文、Triple、排名、分数、历史指纹 |
| `final_write_retrieval.jsonl` | 最终已写入状态断言所对应的最后一次召回，作为主统计样本 |
| `retrieval_cases.md` | 每条新事实下四种方法分别找到了什么，附原文与 Triple |
| `pooled_relation_audit.jsonl` | 四路 Top-5 合并池的模型逐对盲判，不向判读器提供方法名和排名 |
| `comparison_metrics.json` / `comparison_report.md` | 空结果率、平均候选数、方法重叠度、模型辅助关联率与池内覆盖率 |
| `manual_labels.jsonl` | 未填写的人工 gold 模板；不能直接当标准答案 |
| `llm_trace.jsonl` / `pooled_audit_llm_trace.jsonl` | 构建和辅助判读各自的完整提示词、输出、finish_reason 与 token 数 |
| `manifest.json` / `comparison_manifest.json` | 来源覆盖、缺失 Chunk、输入/代码/配置指纹 |
| `upper_layer_checkpoint.pkl` / `dynamic_memory_event_state.pkl` | 会话检查点和新上层，来源图保持只读 |
| `exit_code.txt` | 启动器退出码；0 才代表构建及对比阶段全部完成 |

`uncertain` 单独统计；duplicate/complement/coexists/conflict/correction/replacement 计入“模型认为可能有关联”的宽口径。池内覆盖率分母只包含四路已召回并被模型判相关的候选，**不是真实 Recall@K**。模型与写入器同源，仍需人工复核，不支持据此宣称方法优于原版问答检索。

## 本次来源限制

服务器上 `locomo_sample0_rebuild_event_v2_20261026` 产物只剩 20 个 Chunk、6 个 Session，本实验不将其作为完整 sample0。改用 `locomo_group_1/dynamic_memory_graph.pkl` 中的 214 Chunk、19 Session，在新目录回放。该旧图没有原始逐 Chunk Triple 存档，恢复的关系可能不完整；覆盖率、缺失和拒收候选必须随结果报告，不能跨图拼接关系。

## 启动

```bash
bash experiments/integrated_event_state/run_write_recall_comparison.sh \
  results/独立实验目录 \
  results/locomo_group_1/dynamic_memory_graph.pkl
```

先做前八个 Chunk 的真实预检，确认召回轨迹与存储协议，再后台启动完整样本。启动后不持续轮询，用户通知完成后读取报告和退出码。
