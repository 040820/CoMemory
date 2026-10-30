"""建图（写入）阶段的调度层：把“语料”变成“图 + 原文 + 上层记忆”。

在整体流程中的位置：
    GraphRAG.insert（init/graph_rag.py）
      └── GraphRAGCore.insert（本文件）
            ├── DynamicMemory.generate_session_summaries   仅 session_summary 且启用摘要时
            ├── ChunkProcessor.process_corpus              切块
            └── DynamicMemory.build_graph                 建图 / 加载已有图
                  ├── _build_graph         从头建图
                  ├── add_single_session   按会话增量建图（graph.add=True）
                  └── _dynamic_add_chunks  按 Chunk 增量建图（已废弃）

两种写入模式：
    - add=True：逐会话增量。先加载已有 pkl，再对每个 session 依次调用
      add_single_session，每处理完一个会话就落盘一次。
    - 其余情况：一次性写入。force=True 或图不存在时从头构建，
      否则直接加载已有 pkl，不重新建图。
"""

from typing import List, Dict, Any
import asyncio
import sys
import os

# 允许以脚本/模块两种方式导入项目内其它包。
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from init.config import Config
from init.logger import logger
from base.llm import LLMManager
from base.embeddings import EmbeddingManager
from chunking.chunk_processor import ChunkProcessor
from .dynamic_memory import DynamicMemory


class GraphRAGCore:
    """建图核心：装配各组件，并按配置选择建图模式。

    与 GraphRAG 的分工：
        - GraphRAG 面向调用方，负责查询模式分发；
        - 本类面向写入流程，只关心“怎么把语料写进图”。
    """

    def __init__(self, config: Config, base_dir: str = "./results"):
        """创建 LLM/Embedding/切块器/动态记忆图，并完成依赖注入。

        Args:
            config:   全局配置。
            base_dir: 输出目录，图快照 pkl、摘要 JSON 都写在这里。
        """
        # 上层记忆模式只有两种取值，提前校验，避免运行到一半才报 KeyError。
        if getattr(config, 'upper_memory_mode', 'session_summary') not in {'session_summary', 'event_state'}:
            raise ValueError('upper_memory_mode must be session_summary or event_state')
        self.config = config
        self.base_dir = base_dir

        # 建图阶段统一使用的 LLM（查询阶段可能另建，见 init/graph_rag.py）。
        self.llm_manager = LLMManager(
            api_key=config.llm.api_key,
            model=config.llm.model,
            max_tokens=config.llm.max_token,
            base_url=config.llm.base_url,
            enable_concurrent=config.llm.enable_concurrent,
            max_concurrent=config.llm.max_concurrent,
            timeout=config.llm.timeout,
            enable_thinking=config.llm.enable_thinking
        )
        
        self.embedding_manager = EmbeddingManager(config.embedding)

        # 切块方式由 config.chunk.dialogue_input 决定：
        #   True  → 按“对话轮次”切（DialogChunkProcessor）；
        #   False → 按 token 数滑动窗口切。
        data_type = getattr(config, 'data_type', 'LongmemEval')
        self.chunk_processor = ChunkProcessor(config.chunk, data_type=data_type)

        # 动态记忆图：内部持有 NetworkX 图、chunk_storage、上层记忆（摘要或事件层）。
        self.graph = self._create_graph(config)

        # 把 LLM / Embedding 注入图对象，同时实例化各抽取器。
        if hasattr(self.graph, 'set_extractors'):
            self.graph.set_extractors(self.llm_manager, self.embedding_manager)

        logger.info("GraphRAG Core initialized")

    def _create_graph(self, config: Config):
        """创建 DynamicMemory，并复用已装配好的切块器（避免重复初始化）。"""
        graph = DynamicMemory(config, self.base_dir)
        graph.chunk_processor = self.chunk_processor
        return graph

    async def insert(self, corpus: List[Dict[str, Any]]) -> None:
        """把语料写入系统入口。

        Args:
            corpus: 文档/会话列表，每条至少包含 session_id、context，
                    对话数据集还会带 session_time。

        分支说明：
            add=True   → 增量模式：加载已有图后逐会话处理并保存；
            其它情况   → 一次性模式：可能先建摘要，再切块、建图、保存。
        """
        self.graph.time_manager.start_total_graph_building()

        add = getattr(self.config.graph, 'add', False)
        force = getattr(self.config.graph, 'force', False)

        if add and corpus:
            logger.info("🔄 ADD MODE: Processing sessions sequentially")

            graph_path = os.path.join(self.base_dir, f"{self.config.index_name}.pkl")
            if os.path.exists(graph_path):
                # 增量模式必须基于已有图，否则新旧实体无法合并。
                logger.info(f"Loading existing graph from {graph_path}")
                self.graph.load_graph(graph_path)
                stats = self.graph.get_graph_stats()
                logger.info(f"Loaded graph with {stats['num_nodes']} nodes and {stats['num_edges']} edges")
            else:
                # 没有旧图时先建一个空图，后续 add_single_session 会往里加内容。
                logger.info("No existing graph found, will create new graph")
                await self.graph.build_graph([])

            # 按 session_id 聚合文档：一个会话可能由多条 doc 组成。
            sessions_map = {}
            for doc in corpus:
                session_id = doc.get('session_id', 'unknown')
                if session_id not in sessions_map:
                    sessions_map[session_id] = []
                sessions_map[session_id].append(doc)

            logger.info(f"Found {len(sessions_map)} unique sessions to process")

            for idx, (session_id, session_docs) in enumerate(sessions_map.items(), 1):
                logger.info(f"\n{'='*80}")
                logger.info(f"Processing session {idx}/{len(sessions_map)}: {session_id}")
                logger.info(f"{'='*80}")

                # 单个会话内部完成：摘要更新 → 切块 → 抽取 → 去重 → 增量并入图 → 事件层。
                await self.graph.add_single_session(session_docs)

                # 每处理完一个会话立即落盘：中途失败时已完成的会话不会丢失。
                self.graph.save_graph(graph_path)
                stats = self.graph.get_graph_stats()
                logger.info(f"Graph saved with {stats['num_nodes']} nodes and {stats['num_edges']} edges")

            logger.info("\n" + "="*80)
            logger.info("✅ All sessions processed successfully")
            logger.info("="*80)

        else:
            # 一次性建图前的可选步骤：为每个会话生成摘要。
            # event_state 模式用事件层替代摘要，因此只在 session_summary + enable_summary 时执行。
            if (self.config.upper_memory_mode == 'session_summary'
                    and self.config.retriever.enable_summary):
                logger.info("Generating session summaries...")
                summaries = await self.graph.generate_session_summaries(corpus)
                logger.info(f"Generated {len(summaries)} session summaries")
            else:
                logger.info("Summary generation disabled in config")
                summaries = []

            # 切块：对话数据按轮次切，普通文档按 token 切；同时分配全局唯一 chunk_id。
            chunks = self.chunk_processor.process_corpus(corpus)
            logger.info(f"Processed {len(chunks)} chunks from {len(corpus)} documents")

            # 建图内部会判断 force / 已有图：从头构建或直接加载。
            await self.graph.build_graph(chunks)

            graph_path = os.path.join(self.base_dir, f"{self.config.index_name}.pkl")
            self.graph.save_graph(graph_path)

            stats = self.graph.get_graph_stats()
            logger.info(f"Graph built with {stats['num_nodes']} nodes and {stats['num_edges']} edges")
            logger.info(f"Graph saved to {graph_path}")

        self.graph.time_manager.end_total_graph_building()
        logger.info("Document insertion completed")
