# 第一版验证记录（2026-10-25）

## 环境与版本

- 服务器：`/home/cwf/Code/LiCoMemory`，Conda 环境 `licomemory`。
- 模型：本机 vLLM 的 `qwen3-8b`，按 `config/Memory.yaml` 调用。
- 原图：`results/locomo_group_1/dynamic_memory_graph.pkl`，含 214 个 Chunk。
- 本模块只读加载原图，输出在 `results/event_state_locomo_group_1/`。

## 检查

本地和服务器均执行：

```bash
python -m unittest discover -s experiments/event_state_evolution/tests -v
```

LoCoMo 小样本实测命令：

```bash
python experiments/event_state_evolution/build_index.py \
  --config config/Memory.yaml \
  --graph results/locomo_group_1/dynamic_memory_graph.pkl \
  --entity-map results/event_state_locomo_group_1/entity_map.json \
  --output-db results/event_state_locomo_group_1/state_f63950b.sqlite \
  --chunk-id 113
```

该 Chunk 涉及艺术创作，但未明确给出当前雇主、职位或常住地。它产生 5 个候选、15 次属性判定：写入 0，待核查 13，忽略 2。待核查原因中 12 条为模型输出的三元组无法由所选原文片段支持，1 条为主体或片段无法核对。原始运行日志是 `results/event_state_locomo_group_1/smoke_f63950b.log`；旁路库是上述 SQLite 文件。该结果是接口及保守校验的检查，**不是** LoCoMo 全集抽取率或回答准确率的评估。

另用明确的合成话语 `Caroline: I worked as an artist at Blue Studio.` 和 `Caroline: I now work as a teacher at Green School.` 经真实 vLLM 验证：旧值 `artist`、新值 `teacher` 构成 `job_title` 变化，当前值变为 `teacher`，变化表保留旧新事实 ID。模型把观察日期填为事件日期时，校验器将 `event_time` 置空。原始模型响应及库查询保存在 `results/event_state_locomo_group_1/synthetic_change.log`；运行脚本保存在同目录 `synthetic_change.py`。合成样例只验证模型接入和写库链路，不能替代真实语料验证。

## 后续实验边界

进行大规模索引前，应先人工核定每组对话的实体 ID 映射，并抽样检查 `review_queue`。当前图的合并边可能不受所关联每个 Chunk 支持；程序故意不把这些边直接写成当前事实。若要报告召回或问答提升，应另建带人工标注的状态变化集合，逐条核对变化、纠正和证据。
