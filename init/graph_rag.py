import asyncio
from typing import List, Dict, Any
from .config import Config
from .logger import logger

class GraphRAG:
    """Main GraphRAG class that orchestrates the entire system."""

    def __init__(self, config: Config, base_dir: str = "./results"):
        """Initialize GraphRAG with configuration."""
        self.config = config
        self.base_dir = base_dir
        # Import here to avoid circular imports
        try:
            from ..coregraph.graph_rag_core import GraphRAGCore
        except ImportError:
            import sys
            import os
            sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            from coregraph.graph_rag_core import GraphRAGCore
        self.core = GraphRAGCore(config, base_dir)
        self._event_retriever = None
        logger.info("GraphRAG initialized successfully")

    async def insert(self, corpus: List[Dict[str, Any]]) -> None:
        """Insert documents into the system."""
        logger.info(f"Inserting {len(corpus)} documents")
        await self.core.insert(corpus)

    async def query(self, question: str, question_time: str = "") -> str:
        """Process a query and return answer."""
        logger.info(f"Processing query: {question}")
        try:
            import sys
            import os
            sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            from query.query_processor import QueryProcessor
            from base.llm import LLMManager
            
            query_llm_config = self.config.query_llm
            use_query_llm = (
                query_llm_config.model or 
                query_llm_config.api_key or 
                query_llm_config.base_url or
                query_llm_config.max_token > 0 or
                query_llm_config.temperature >= 0.0
            )
            
            if use_query_llm:
                query_llm_manager = LLMManager(
                    api_key=query_llm_config.api_key or self.config.llm.api_key,
                    model=query_llm_config.model or self.config.llm.model,
                    max_tokens=query_llm_config.max_token if query_llm_config.max_token > 0 else self.config.llm.max_token,
                    base_url=query_llm_config.base_url or self.config.llm.base_url,
                    enable_concurrent=False,  # Query phase doesn't need concurrent
                    max_concurrent=1,
                    timeout=query_llm_config.timeout if query_llm_config.timeout > 0 else self.config.llm.timeout,
                    enable_thinking=(query_llm_config.enable_thinking
                                     if query_llm_config.enable_thinking is not None
                                     else self.config.llm.enable_thinking)
                )
            else:
                query_llm_manager = self.core.llm_manager

            agent_memory = getattr(self.core, 'graph', None)
            if getattr(self.config, 'upper_memory_mode', 'session_summary') == 'event_state':
                # 新写入模式继续使用已有的 Query→Triple/Chunk 检索实现。
                # 上层只补充演化证据，不能成为 Chunk 的候选筛选门槛。
                from experiments.integrated_event_state.query import query_with_event_state
                from experiments.structural_semantic_retrieval.retriever import StructuralSemanticRetriever
                if self._event_retriever is None:
                    self._event_retriever = StructuralSemanticRetriever(
                        agent_memory, self.core.embedding_manager)
                return await query_with_event_state(
                    question, question_time, agent_memory, self._event_retriever,
                    query_llm_manager, top_k=self.config.retriever.top_chunks)
            query_processor = QueryProcessor(self.config, query_llm_manager, agent_memory)
            query_processor.embedding_manager = self.core.embedding_manager
            result = await query_processor.process_query(question, question_time=question_time)
            logger.info(f"Query processed successfully")
            return result

        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return {"answer": f"Error processing query: {str(e)}", "top_session_ids": []}

