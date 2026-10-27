# 中层多关系记忆图实施计划

**目标：** 同一实体对的不同关系、同一关系在不同 Chunk 的出现记录均独立保留，支持检索和上层追溯。

**设计：** event_state 使用 MultiDiGraph，边 key 为稳定 Triple ID（Chunk、主体、谓词、客体）。同 Chunk 相同 Triple 幂等，不同 Chunk 独立成边。节点仍按规范实体名合并。原版 session_summary 保留 DiGraph。

**边界：** 本轮修复中层事实保全；过敏等开放属性的状态槽归一化仍由上层规则和证据门控负责，不能凭平行边自动判断谁是当前状态。不检查或中断先前启动的 sample0 任务；验证在独立目录中进行。

## 修改范围

- GraphBuilder：新图类型、按来源写入多边、观察时间与发生时间分离。
- DynamicMemory：三条建图路径使用原始出现记录；关系向量准确写入对应边；旧图在内存中兼容转换并保持来源完整性标记。
- StructuralSemanticRetriever：优先读取 extracted 的逐 Chunk Triple；兼容旧 DiGraph 与新 MultiDiGraph。
- Inspect/文档：统计边出现记录、检索来源和格式限制。

## 验证与交付

- [x] 先运行失败回归：相同实体不同关系、相同关系不同来源、重复写入、关系向量隔离。
- [x] 实现事件状态模式多边存储，保持旧模式行为。
- [x] 接通旧图加载、pickle 往返及新检索的逐 Chunk 来源。
- [x] 用户要求停止追加验证；此前已启动的 8 项回归完成，后续不再运行测试或 LLM 验证。
- [ ] 同步经验证代码与说明，提交并推送 GitHub。

**限制：** 旧文件未保存的关系不可恢复；只有 extracted 原始记录才能用于完整重建旧中层。旧模式不直接消费新多边图，避免隐藏的检索语义变化。

存储说明见 `docs/multirelation-memory-storage.md`。过敏等属性的状态槽规范化不在本轮中层改造范围，不能用多边本身自动推定 REPLACE。
