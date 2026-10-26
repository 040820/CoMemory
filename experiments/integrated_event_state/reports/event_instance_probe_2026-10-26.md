# 事件实例三项验证（2026-10-26）

## 输入与边界

- 在服务器用本地 `qwen3-8b` vLLM 处理六段人工构造的 Caroline 对话。
- 固定提供同 Chunk 的 Triple 快照，隔离下层关系抽取误差；上层事实抽取、
  事件身份判断和纠正判断均由真实 vLLM 完成。
- 同时运行 42 项相关程序测试，覆盖事件分开、跨 Chunk 关联、断言历史、
  提问时间可见性和通用属性局部纠正。
- 这不是 LoCoMo 全集实验，也不能估计整体精确率或召回率。

## 结果

| 验证问题 | 结果 | 证据 |
| --- | --- | --- |
| 同一活动跨 Chunk 是否共用 E1 | 通过 | support group 第 1、2、4、5 个 Chunk 归入同一事件 |
| 同主体同类活动是否错误合并 | 通过 | school event 独立；再次参加同一 support group 的 May 14 活动也独立 |
| 时间纠正是否只更新对应断言 | 通过 | May 7 断言 `corrected`，May 6 断言 `active`，同一 E1 |
| 地点纠正是否只更新对应断言 | 通过 | Community Center `corrected`，Library `active`，E1 的日期仍为 May 6 |
| 其他属性能否使用同一机制 | 程序级通过 | `organizer: Pat → Alex`，地点保持原值；未单独做真实模型输入 |
| 纠正前后的提问能否看到对应版本 | 程序级通过 | 纠正前返回 May 7 / Community Center，纠正后返回 May 6 / Library，并屏蔽旧日期 Triple |

最终小样本处理 6/6 个 Chunk，生成 3 个事件实例、4 条日期断言、
2 条地点断言；`review_queue` 为 0。原 May 7 的时间 Triple 来源键被标记失效。

前两次真实模型试验暴露了事件客体被误抽成 `yesterday`、模型漏填纠正引文等
问题。已加入事件客体与中层关系对象的一致性检查、事件字段示例，且仅在当前
说话人原文唯一时从原文补齐漏填引文。最终结果来自 `v4` 六段输入。

## 原始文件与复现

服务器目录：`/home/cwf/Code/LiCoMemory/results/event_instance_probe_v4/`

- `summary.json`：事件、断言、逐对判断与待核查结果。
- `layer.json`：完整上层序列化结构和反向索引。
- `llm_trace.json`：模型输入提示词与原始输出。
- 相邻文件 `event_instance_probe_v4.log`：运行日志。
- 归档 `event_instance_probe_v4.tar.gz`：上述目录与日志。

复现命令（需先启动配置对应的本地 vLLM）：

```bash
/home/cwf/miniconda3/envs/licomemory/bin/python \
  experiments/integrated_event_state/probe_event_instances.py \
  --output results/event_instance_probe_reproduce
```

程序测试命令：

```bash
/home/cwf/miniconda3/envs/licomemory/bin/python -m unittest \
  tests.test_event_instances tests.test_integrated_event_writer \
  tests.test_state_candidate_retriever tests.test_state_slot \
  tests.test_state_slot_layer -q
```

## 尚未验证

真实 LoCoMo 数据包含更长、更模糊的指代、同名活动和多说话人 Chunk。
本次没有重建整份数据，也没有测量回答质量。小样本的 LLM 判断成功只说明
这六段输入可以形成预期的事件链接与断言历史。
