"""GraphRAG 门面层（Facade）：对上层屏蔽建图与查询的内部实现。

类位置与职责：
    main.py
      └── GraphRAG（本文件）          ← 只负责“装配 + 分发”，不实现具体算法
            ├── GraphRAGCore          （coregraph/graph_rag_core.py）
            │     ├── LLMManager      建图阶段使用的 LLM
            │     ├── EmbeddingManager
            │     ├── ChunkProcessor
            │     └── DynamicMemory   动态记忆图（graph_builder / chunk_storage / 事件层）
            ├── QueryProcessor        （query/query_processor.py）session_summary 模式的查询
            └── query_with_event_state（experiments/integrated_event_state/query.py）
                                     event_state 模式的查询

为什么需要这一层：
    1. 建图与查询共用同一个 GraphRAGCore，但两阶段使用的 LLM 配置可以不同；
       本层负责在查询时挑选正确的 LLM，而不是改动核心对象的状态。
    2. 项目支持两种“上层记忆模式”（由 config.upper_memory_mode 控制），
       查询入口需要根据模式分发到不同实现，同时保持 main.py 的调用方式不变。
    3. 把可选的实验性依赖（事件层检索器）做成惰性加载，
       未启用 event_state 的普通运行不会付出额外的导入/初始化成本。
"""

import asyncio
from typing import List, Dict, Any
from .config import Config
from .logger import logger


class GraphRAG:
    """GraphRAG 系统的统一入口：初始化核心组件、写入语料、回答查询。

    实例通常由 main.py 创建一次并复用：
        graph_rag = GraphRAG(config, base_dir)
        await graph_rag.insert(corpus)                 # 建图/增量写入
        answer = await graph_rag.query(question, ...)  # 查询

    本类不持有查询结果或会话状态，除 event_state 模式下的检索器缓存
    （self._event_retriever）外，调用 query() 不会修改自身状态。
    """

    def __init__(self, config: Config, base_dir: str = "./results"):
        """初始化 GraphRAG 门面。

        Args:
            config:   全局配置对象（init/config.py 中的 Config）。
            base_dir: 本次运行的输出目录，用于存放图快照 pkl、摘要 JSON 等。
                      由 main.py 的 check_dirs() 根据 -root 参数生成。

        注意：
            - 这里会一次性构造 LLMManager / EmbeddingManager / 分块器 / 动态记忆图，
              因此初次初始化有一定开销（尤其是加载本地嵌入模型时）。
            - 查询阶段的 LLM 不在这里创建，而是在 query() 中按需创建，
              以免未配置 query_llm 时产生多余对象。
        """
        self.config = config
        self.base_dir = base_dir

        # Import here to avoid circular imports
        # 延迟导入 GraphRAGCore：coregraph 内部又会引用 init.config / init.logger，
        # 在模块顶层导入容易形成循环依赖，因此放到实例化时再导入。
        try:
            # 正常包内调用（如 `from init.graph_rag import GraphRAG`）走这里。
            from ..coregraph.graph_rag_core import GraphRAGCore
        except ImportError:
            # 以脚本方式直接运行或从项目根目录之外启动时，包层级不成立，
            # 回退为把项目根目录加入 sys.path 后再按顶层包名导入。
            import sys
            import os
            sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            from coregraph.graph_rag_core import GraphRAGCore

        # 核心对象一次性装配完成：LLM、Embedding、分块器、动态记忆图都在其构造函数中创建。
        # 建图与查询共享同一个 core，保证查询时看到的就是已建好的图。
        self.core = GraphRAGCore(config, base_dir)

        # event_state 模式下的结构化语义检索器采用惰性创建：
        # 第一次查询时才实例化并缓存，避免 session_summary 模式下的无效初始化。
        # 缓存的原因是检索器内部维护进程内向量缓存，复用可减少重复嵌入计算。
        self._event_retriever = None

        logger.info("GraphRAG initialized successfully")

    async def insert(self, corpus: List[Dict[str, Any]]) -> None:
        """把语料写入系统（内部委托给 GraphRAGCore.insert）。

        Args:
            corpus: 文档/会话片段列表。每条通常包含 session_id、对话文本、
                    session_time 等字段，具体格式由 ChunkProcessor 与数据集决定。

        具体行为（重建 / 增量 / 无操作）由配置决定，本层不做判断：
            - graph.force = True：忽略已有图，重新构建；
            - graph.add   = True：加载已有图后按会话逐条增量写入；
            - 两者都未开启：通常直接加载已有图或空图。
        详见 coregraph/graph_rag_core.py 的 insert 实现。
        """
        logger.info(f"Inserting {len(corpus)} documents")
        await self.core.insert(corpus)
        # 写入后清空只读服务缓存，防止后续问答漏掉刚追加的事实和候选链接。
        self._evidence_support = None

    async def query(self, question: str, question_time: str = "") -> str:
        """处理一次查询并返回结果。

        Args:
            question:      用户问题文本。
            question_time: 提问时间（字符串，可为空）。用于需要“截至某时刻”的检索，
                           例如 event_state 模式会过滤掉在提问时间之后才观察到的事件。

        Returns:
            正常路径下为 dict（QueryProcessor / query_with_event_state 的统一返回），
            典型键包括：
                answer           模型生成的答案
                top_session_ids  检索命中的会话 ID（用于会话命中率评测）
            异常路径下返回 {"answer": 错误信息, "top_session_ids": []}。
            注意：类型标注写的是 str，但实际返回 dict，调用方（main.py）按 dict 处理。

        查询流程：
            1. 选择查询阶段使用的 LLM：
               - 若配置了 query_llm（任一字段非默认值），单独创建一个 LLMManager；
               - 否则复用建图阶段的 core.llm_manager。
            2. 按 upper_memory_mode 分发：
               - event_state    → experiments.integrated_event_state.query；
               - session_summary→ QueryProcessor（原版摘要 + 图检索流程）。
            3. 捕获所有异常，返回带错误信息的 dict，避免单题失败中断整轮评测。
        """
        logger.info(f"Processing query: {question}")
        try:
            # 查询相关模块按需导入：main.py 在“只建图不查询”时不会加载这些依赖。
            import sys
            import os
            sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            from query.query_processor import QueryProcessor
            from base.llm import LLMManager

            query_llm_config = self.config.query_llm

            # 判断用户是否真正配置了独立的查询 LLM。
            # QueryLLMConfig 的默认值分别是：model=""、api_key=""、base_url=""、
            # max_token=0、temperature=-1.0，均表示“沿用主 LLM 配置”。
            # 只要有一个字段被改成有效值，就认为需要单独创建查询 LLM。
            # 注意 temperature >= 0.0 这一条：设置 temperature=0.0 也算显式配置。
            use_query_llm = (
                query_llm_config.model or
                query_llm_config.api_key or
                query_llm_config.base_url or
                query_llm_config.max_token > 0 or
                query_llm_config.temperature >= 0.0
            )

            if use_query_llm:
                # 逐字段回退：查询 LLM 未设置的字段沿用主 LLM 的同名配置，
                # 这样用户只需覆盖想改的部分（例如只换模型或只换 base_url）。
                query_llm_manager = LLMManager(
                    api_key=query_llm_config.api_key or self.config.llm.api_key,
                    model=query_llm_config.model or self.config.llm.model,
                    max_tokens=query_llm_config.max_token if query_llm_config.max_token > 0 else self.config.llm.max_token,
                    base_url=query_llm_config.base_url or self.config.llm.base_url,
                    # 查询阶段逐个问题串行回答，不需要并发池，因此关闭并发，
                    # 既简化限流，也避免查询请求互相挤占建图阶段的并发配额。
                    enable_concurrent=False,  # Query phase doesn't need concurrent
                    max_concurrent=1,
                    timeout=query_llm_config.timeout if query_llm_config.timeout > 0 else self.config.llm.timeout,
                    # enable_thinking 是三态（True/False/None）：
                    # None 表示“未配置”，此时沿用主 LLM 的设置。
                    enable_thinking=(query_llm_config.enable_thinking
                                     if query_llm_config.enable_thinking is not None
                                     else self.config.llm.enable_thinking)
                )
            else:
                # 未单独配置查询 LLM：复用建图时的实例，省去重复初始化和额外的客户端连接。
                query_llm_manager = self.core.llm_manager

            # 动态记忆图（DynamicMemory）既承担图检索，也保存上层记忆
            # （session 摘要或事件状态层），两种查询模式都要用到它。
            # 用 getattr 兜底：某些实验分支可能构造了不完整的 core。
            agent_memory = getattr(self.core, 'graph', None)
            if getattr(self.config, 'upper_memory_mode', 'session_summary') == 'event_state':
                mode = getattr(self.config, 'event_state_query_mode', 'auto')
                if mode not in ('auto', 'evidence_support', 'structural_semantic'):
                    raise ValueError('未知 event_state_query_mode')
                if mode == 'evidence_support' or (mode == 'auto' and
                        getattr(self.config, 'upper_memory_layout', '') == 'event_index'):
                    # 新图入口：共用取证→Judge→带引用回答；只缓存当前已加载图的视图。
                    from coregraph.evidence_support import EvidenceSupport
                    from coregraph.evidence_support.query import memory_graph_view, query_with_evidence_support
                    if getattr(self, '_evidence_support', None) is None:
                        self._evidence_support = EvidenceSupport(memory_graph_view(agent_memory),
                            self.core.embedding_manager, query_llm_manager)
                    return await query_with_evidence_support(agent_memory, question, question_time,
                        self.core.embedding_manager, query_llm_manager, support=self._evidence_support)
                # 新写入模式继续使用已有的 Query→Triple/Chunk 检索实现。
                # 上层只补充演化证据，不能成为 Chunk 的候选筛选门槛。
                #
                # 设计要点：
                #   - 底层检索（结构化语义检索）决定哪些 Chunk 入选；
                #   - 事件/状态层只作为附加证据拼进上下文，不参与候选过滤或重排，
                #     避免事件抽取的误差直接影响召回。
                from experiments.integrated_event_state.query import query_with_event_state
                from coregraph.structural_semantic_retriever import StructuralSemanticRetriever

                # 首次进入该分支时创建检索器并缓存；后续查询复用同一实例，
                # 从而复用其进程内向量缓存（见 StructuralSemanticRetriever.clear_cache）。
                if self._event_retriever is None:
                    self._event_retriever = StructuralSemanticRetriever(
                        agent_memory, self.core.embedding_manager)

                # top_k 取配置中的 top_chunks：与 session_summary 模式保持一致的
                # “最终喂给 LLM 的 Chunk 数量”语义。
                return await query_with_event_state(
                    question, question_time, agent_memory, self._event_retriever,
                    query_llm_manager, top_k=self.config.retriever.top_chunks)

            # session_summary 模式（默认）：走原版查询流程——
            # 提取问题实体 → 图/三元组检索 → 摘要检索与重排 → 组织上下文 → 生成答案。
            query_processor = QueryProcessor(self.config, query_llm_manager, agent_memory)

            # QueryProcessor 构造时会按 config.embedding 新建一个 EmbeddingManager。
            # 这里用 core 中已经加载好的实例覆盖，避免重复加载嵌入模型，
            # 同时保证查询与建图使用完全相同的向量空间。
            query_processor.embedding_manager = self.core.embedding_manager

            result = await query_processor.process_query(question, question_time=question_time)
            logger.info(f"Query processed successfully")
            return result

        except Exception as e:
            # 查询失败不向上抛出：main.py 的逐题评测依赖“每题都有结果”，
            # 由调用方记录错误答案并继续下一题。错误信息会原样放进 answer，
            # 便于在 results.json 中定位失败原因。
            logger.error(f"Error processing query: {e}")
            return {"answer": f"Error processing query: {str(e)}", "top_session_ids": []}
