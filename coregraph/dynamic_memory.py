"""动态记忆图（DynamicMemory）：建图流程的真正执行者。

分层结构：
    中层（实体关系图）  graph_builder.graph  —— NetworkX 有向图，节点是实体，边是关系
    原文层              chunk_storage       —— chunk_id → 完整 Chunk（含原文/会话时间）
    索引层              entity_name_to_index —— 实体名 → 序号，供上层事件层引用
    上层（二选一）      session_summaries.json / event_state_layer
                        —— 摘要模式或事件状态模式，由 config.upper_memory_mode 决定

从头建图（_build_graph）的完整步骤：
    1. 清理旧图/旧 Chunk；若为 event_state 模式则新建空事件层
    2. 保存原始 Chunk 到 chunk_storage
    3. 逐 Chunk 调 LLM 抽实体（+关系）：
         dialogue_input=True  → DialogueExtractor（联合抽取）
         dialogue_input=False → EntityExtractor（仅实体）
    4. 实体/关系去重并归一化名称
    5. 写入 NetworkX 图（GraphBuilder）
    6. 建立实体名索引
    7. 批量预计算实体/关系（以及摘要）的向量，写入图属性
    8. event_state 模式下，基于已建好的中层图和证据构建事件状态层

增量写入（add_single_session）在单个会话内重复上述 2~5 步，
但用增量合并接口，保留已有图内容。

持久化：
    save_graph 把 graph + chunk_storage + entity_name_to_index (+ event_state_layer)
    一起 pickle 到 base_dir/<index_name>.pkl。
    event_state 模式使用“临时文件 + 原子替换”，写入失败时旧图仍然可用。
"""

from typing import List, Dict, Any
import asyncio
import sys
import os
import pickle
import networkx as nx

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from init.logger import logger
from init.config import Config
from .graph_builder import GraphBuilder
from .entity_extractor import EntityExtractor
from .dialogue_extractor import DialogueExtractor
from .session_summarizer import SessionSummarizer
from .event_state_layer import EventStateLayer
from .event_candidate_schema import triple_id
from .event_state_writer import EventStateWriter
from utils.time_statistic import GraphBuildingTimeStatistic
from utils.cost_manager import GraphBuildingCostManager


class DynamicMemory:
    """持有图、原文和上层记忆，并负责它们的构建、更新与落盘。"""

    def __init__(self, config: Config, base_dir: str = "./results"):
        """初始化空图与各组件占位。

        LLM 相关的抽取器不在这里创建，而是等 GraphRAGCore 调用
        set_extractors() 后再实例化（避免 DynamicMemory 依赖 LLM 的具体构造方式）。
        """
        self.config = config
        self.base_dir = base_dir
        self.graph_builder = GraphBuilder(config)
        # 以下抽取器由 set_extractors() 注入。
        self.entity_extractor = None
        self.dialogue_extractor = None
        self.session_summarizer = None
        self.chunk_processor = None
        self.time_manager = GraphBuildingTimeStatistic()
        self.cost_manager = None
        self.chunk_storage = {}
        # 原始 Triple 的出现记录：chunk_id -> [{src, relation, tgt, ...}]。
        # DiGraph 同一对实体只能保留一条边，后写入的关系会覆盖先前的 relation_name；
        # 这份 append-only 记录专门给上层事件层提供可靠的关系证据。
        self.triple_storage: Dict[str, List[Dict[str, Any]]] = {}
        self.triple_storage_origin = "extracted"
        self.entity_name_to_index = {}
        # 新上层随图保存；旧图没有这个字段时恢复为空层。
        self.event_state_layer = EventStateLayer()
        self.event_state_writer = None

        logger.info("Dynamic Memory initialized")

    async def generate_session_summaries(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """为语料中的所有会话生成摘要（session_summary 模式）。

        Args:
            documents: 原始文档列表；同一 session_id 的多条 doc 会合并 context。

        Returns:
            摘要列表，并已写入 base_dir/session_summaries.json。

        会话聚合规则：
            以首次出现的 doc 确定 session_time；
            后续 doc 的 context 若尚未包含在已有 context 中，则追加，
            以保持同一会话内多轮对话的完整顺序。
        """
        logger.info("Starting session summary generation...")

        self.time_manager.start_summary_generation()
        cost_before_summary = self.cost_manager.get_costs()

        # ---- 1. 按 session_id 聚合上下文 ----
        sessions_map = {}
        for doc in documents:
            session_id = doc.get('session_id', 'unknown')
            if session_id not in sessions_map:
                sessions_map[session_id] = {
                    'session_id': session_id,
                    # 转成字符串，避免 datetime 无法写入 JSON。
                    'session_time': str(doc.get('session_time', 'unknown')),  # Convert to string to avoid serialization issues
                    'context': doc.get('context', '')
                }
            else:
                existing_context = sessions_map[session_id]['context']
                new_context = doc.get('context', '')
                if new_context and new_context not in existing_context:
                    sessions_map[session_id]['context'] += '\n\n' + new_context

        sessions = list(sessions_map.values())
        logger.info(f"Found {len(sessions)} unique sessions to summarize")

        # ---- 2. 批量调用 LLM 生成摘要 ----
        # Stage 1: Session summary generation - create progress bar with actual number of sessions
        from tqdm import tqdm
        summary_progress_bar = tqdm(total=len(sessions), desc="Stage 1: Session Summary Generation", unit="calls")

        logger.info(f"Stage 1: Processing {len(sessions)} session summaries")

        summaries = await self.session_summarizer.summarize_sessions(sessions, progress_bar=summary_progress_bar)
        summary_progress_bar.close()

        # ---- 3. 统计 token 消耗（用总用量差值计算，兼容并发） ----
        cost_after_summary = self.llm_manager.cost_manager.get_costs()
        summary_prompt_tokens = cost_after_summary.total_prompt_tokens - cost_before_summary.total_prompt_tokens
        summary_completion_tokens = cost_after_summary.total_completion_tokens - cost_before_summary.total_completion_tokens
        self.cost_manager.update_summary_generation_cost(summary_prompt_tokens, summary_completion_tokens, self.llm_manager.model)

        self.time_manager.end_summary_generation()

        # ---- 4. 单独保存，摘要不进入 pkl ----
        summary_path = os.path.join(self.base_dir, "session_summaries.json")
        self.session_summarizer.save_summaries(summaries, summary_path)

        logger.info(f"Generated and saved {len(summaries)} session summaries")
        return summaries

    def set_extractors(self, llm_manager, embedding_manager=None):
        """注入 LLM/Embedding，并创建各抽取器与成本统计器。

        由 GraphRAGCore.__init__ 在创建完 core 后调用。
        """
        self.entity_extractor = EntityExtractor(llm_manager)
        self.relationship_extractor = None
        data_type = getattr(self.config, 'data_type', 'LongmemEval')
        self.dialogue_extractor = DialogueExtractor(llm_manager, data_type=data_type)
        self.session_summarizer = SessionSummarizer(llm_manager)
        self.event_state_writer = EventStateWriter(llm_manager, embedding_manager)
        self.llm_manager = llm_manager
        self.embedding_manager = embedding_manager
        self.cost_manager = GraphBuildingCostManager(
            max_budget=llm_manager.cost_manager.max_budget
        )

        logger.info("Extractors initialized with LLM manager and cost tracking")

    def _linked_entity_index(self) -> Dict[str, int]:
        """上层只可引用仍存在于中层实体关系图中的实体。

        去重/增量合并后，entity_name_to_index 里可能残留已从图中删除的实体
        （例如孤立节点被清理）。事件层写入前必须过滤，否则会引用不存在的实体。
        """
        return {name: index for name, index in self.entity_name_to_index.items()
                if name in self.graph_builder.graph}

    def _record_triple_occurrences(self, relationships: List[Dict[str, Any]],
                                   chunks: List[Dict[str, Any]] = None) -> None:
        """把本次抽取的原始关系按 Chunk 追加保存。

        为什么不能只依赖 graph：
            NetworkX ``DiGraph`` 对同一对 ``(src, tgt)`` 只能保留一条边，
            后写入的关系会覆盖先前的 ``relation_name``。上层事件层需要知道
            “某次抽取在某个 Chunk 里到底抽到了哪些关系”，因此这里保存
            append-only 的出现记录，而不是最终图边。

        结构::

            triple_storage[chunk_id] = [
                {"src": ..., "relation": ..., "tgt": ...,
                 "chunk_id": ..., "session_id": ..., "session_time": ...,
                 "strength": ..., "weight": ..., "description": ...},
                ...
            ]

        去重键为 ``(src, relation, tgt)``，只限制在同一个 Chunk 内；
        不同 Chunk 的同一关系会分别保留，供时间断言和事件实例回溯。

        Args:
            relationships: 抽取器输出的关系列表，字段为 src/relation/tgt，
                           可带 chunk_id、chunk_ids、session_id、create_time 等。
            chunks:        本批 Chunk，用于补齐关系缺失的 session 元数据。
        """
        chunk_meta = {str(chunk.get("chunk_id", "")): chunk for chunk in (chunks or [])}
        for relationship in relationships or []:
            src = str(relationship.get("src") or "").strip()
            relation = str(relationship.get("relation") or "").strip()
            tgt = str(relationship.get("tgt") or "").strip()
            if not (src and relation and tgt):
                continue

            chunk_ids = relationship.get("chunk_ids") or []
            if isinstance(chunk_ids, (list, tuple, set)):
                chunk_ids = list(chunk_ids)
            else:
                chunk_ids = [chunk_ids]
            if relationship.get("chunk_id") is not None:
                chunk_ids.append(relationship.get("chunk_id"))

            for raw_chunk_id in {str(value) for value in chunk_ids
                                 if value is not None and str(value).strip()}:
                chunk = chunk_meta.get(raw_chunk_id, {})
                entries = self.triple_storage.setdefault(raw_chunk_id, [])
                if any(entry.get("src") == src and entry.get("relation") == relation
                       and entry.get("tgt") == tgt for entry in entries):
                    continue
                entry = {
                    "src": src,
                    "relation": relation,
                    "tgt": tgt,
                    "chunk_id": raw_chunk_id,
                    # create_time 是抽取器给关系标注的发生时间，不得冒充
                    # Chunk 所属会话的观察时间。
                    "session_id": str(chunk.get("session_id")
                                      or relationship.get("session_id") or ""),
                    "session_time": str(chunk.get("session_time")
                                        or relationship.get("session_time") or ""),
                }
                if relationship.get("create_time"):
                    entry["event_time"] = str(relationship["create_time"])
                if relationship.get("strength") is not None:
                    entry["strength"] = relationship["strength"]
                if relationship.get("weight") is not None:
                    entry["weight"] = relationship["weight"]
                if relationship.get("description"):
                    entry["description"] = str(relationship["description"])
                entry["triple_id"] = triple_id(entry)
                entries.append(entry)

    def _relationships_for_chunks(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """从逐 Chunk 原始记录提供上层输入，不依赖 DiGraph 或跨 Chunk 去重结果。"""
        return [ref for chunk in chunks
                for ref in self.triple_storage.get(str(chunk.get("chunk_id")), [])]

    def _backfill_triple_storage(self) -> None:
        """从旧 pkl 的图边和上层 fact 快照恢复能恢复的 Triple。

        仅用于没有 ``triple_storage`` 的旧文件，属于兼容性兜底：
        已被 DiGraph 覆盖的关系无法恢复。新建图会直接写入完整原始记录，
        不需要走这里。
        """
        graph = getattr(self.graph_builder, "graph", None)
        self.triple_storage_origin = "recovered_partial"
        if graph is not None:
            for src, tgt, edge in graph.edges(data=True):
                relation = edge.get("relation_name") or edge.get("relation")
                if not relation:
                    continue
                chunk_ids = edge.get("chunk_ids") or []
                if isinstance(chunk_ids, (list, tuple, set)):
                    chunk_ids = list(chunk_ids)
                else:
                    chunk_ids = [chunk_ids]
                if edge.get("chunk_id") is not None:
                    chunk_ids.append(edge.get("chunk_id"))
                for chunk_id in {str(value) for value in chunk_ids
                                 if value is not None and str(value).strip()}:
                    self._record_triple_occurrences([{
                        "src": str(src), "relation": str(relation), "tgt": str(tgt),
                        "chunk_id": chunk_id,
                        "session_id": edge.get("session_id"),
                        "session_time": edge.get("session_time"),
                    }])

        for fact in self.event_state_layer.facts.values():
            for ref in fact.get("triple_refs") or []:
                if not isinstance(ref, dict):
                    continue
                self._record_triple_occurrences([{
                    **ref,
                    "session_id": fact.get("session_id"),
                    "session_time": fact.get("observed_at"),
                }])

        if self.triple_storage:
            logger.info(f"Backfilled triple_storage for {len(self.triple_storage)} chunks")

    async def build_graph(self, chunks: List[Dict[str, Any]], force: bool = None, add: bool = None) -> None:
        """建图总入口：根据 force / add / 图是否存在选择模式。

        Args:
            chunks: 已切好的 Chunk（add 模式下可为空，表示只加载）。
            force:  是否强制重建；None 时读 config.graph.force。
            add:    是否增量追加；None 时读 config.graph.add。

        三种分支：
            add=True         先尝试加载旧图，再逐 Chunk 增量追加；
                             加载失败（首次运行）则从头构建；
            force 或图不存在 从头构建；
            其它             直接加载 pkl（本次 chunks 不会被写入）。
        """

        if force is None:
            force = getattr(self.config.graph, 'force', False)
        if add is None:
            add = getattr(self.config.graph, 'add', False)

        logger.info(f"Graph build mode: force={force}, add={add}")

        # graph_builder 始终存在，但 graph 可能尚未被替换成已加载的图。
        graph_exists = hasattr(self, 'graph_builder') and self.graph_builder.graph is not None

        if add:
            # Dynamic addition mode - always try to load existing graph first
            logger.info("🔄 Dynamic addition mode: loading existing graph and adding new chunks")
            try:
                await self._load_existing_graph()
                logger.info("✅ Successfully loaded existing graph")
            except FileNotFoundError:
                # 首次运行没有 pkl：先建一个空图，再走增量追加（等价于从头写）。
                logger.info("📝 No existing graph found, building from scratch")
                await self._build_graph(chunks)
                return
            await self._dynamic_add_chunks(chunks)
        elif force or not graph_exists:
            # Build from scratch
            mode_reason = "force rebuild" if force else "graph doesn't exist"
            logger.info(f"🏗️ Build from scratch mode: {mode_reason}")
            await self._build_graph(chunks)
        else:
            # Load existing mode - load from pkl file
            logger.info("📂 Load existing mode: loading graph from pkl file")
            await self._load_existing_graph()

    async def _manage_session_summaries(self, chunks: List[Dict[str, Any]], force: bool) -> None:
        """管理摘要文件：event_state 模式跳过；摘要开启时按需生成或复用。

        注意：当前主流程（_build_graph）并未调用本方法，
        摘要生成实际由 GraphRAGCore.insert 提前触发。此处保留为旧接口。
        """
        if getattr(self.config, 'upper_memory_mode', 'session_summary') == 'event_state':
            return
        if getattr(self.config.retriever, 'enable_summary', False):
            summaries_path = os.path.join(self.base_dir, 'session_summaries.json')
            if force and os.path.exists(summaries_path):
                # 强制重建时删除旧摘要，避免摘要与重建后的图不一致。
                logger.info(f"Removing existing session summaries file due to forced rebuild with summary enabled: {summaries_path}")
                os.remove(summaries_path)
            if force or not os.path.exists(summaries_path):
                logger.info("Generating session summaries at the start of the process")
                await self.generate_session_summaries(chunks)
            else:
                logger.info("Session summaries file already exists, skipping generation")
        else:
            logger.info("Summary generation is disabled, skipping session summaries management")

    async def _load_existing_graph(self) -> None:
        """从 base_dir/<index_name>.pkl 加载已有图；文件不存在时抛 FileNotFoundError。

        异常设计：FileNotFoundError 用于让 build_graph 的 add 分支区分
        “首次运行”与“加载出错”，其它异常继续向上抛。
        """
        try:
            pkl_filename = f"{self.config.index_name}.pkl"
            pkl_path = os.path.join(self.base_dir, pkl_filename)

            if not os.path.exists(pkl_path):
                logger.error(f"PKL file not found: {pkl_path}")
                raise FileNotFoundError(f"Graph file not found: {pkl_path}")

            logger.info(f"Loading graph from: {pkl_path}")
            self.load_graph(pkl_path)

            try:
                stats = self.get_graph_stats()
                logger.info(f"Loaded graph with {stats['num_nodes']} nodes and {stats['num_edges']} edges")
            except Exception as stats_error:
                # 统计失败不影响图本身可用（例如旧格式文件缺少某些属性）。
                logger.warning(f"Could not retrieve graph statistics after loading: {stats_error}")
                logger.info("Graph loaded successfully, but statistics unavailable")

        except Exception as e:
            logger.error(f"Failed to load existing graph: {e}")
            raise

    async def _build_graph(self, chunks: List[Dict[str, Any]]) -> None:
        """从头构建整张图（一次性写入的核心实现）。

        步骤细节见模块文档；这里额外记录成本统计口径：
            - 抽取阶段按“prompt token / completion token 各取一半”拆分实体与关系成本，
              因为 dialogue 模式一次调用同时完成两件事，无法精确区分；
            - 图构建与向量预计算的时间分别由 time_manager 记录。
        """
        logger.info(f"Starting to build graph from {len(chunks)} chunks")

        self.time_manager.start_total_graph_building()
        # 某些调用路径（如 add 模式首次加载失败）可能未初始化成本统计器，这里兜底。
        self.cost_manager = GraphBuildingCostManager() if self.cost_manager is None else self.cost_manager

        # ---- 0. 清理旧文件 ----
        # 清理旧文件（force rebuild）。event_state 模式保留旧 pkl：
        # 由后续 save_graph 的原子替换覆盖，失败时旧图仍可回滚使用。
        # Clean up existing files if they exist (force rebuild)
        pkl_filename = f"{self.config.index_name}.pkl"
        pkl_path = os.path.join(self.base_dir, pkl_filename)
        if (os.path.exists(pkl_path) and
                getattr(self.config, 'upper_memory_mode', 'session_summary') != 'event_state'):
            logger.info(f"Removing existing graph file: {pkl_path}")
            os.remove(pkl_path)

        # Also remove backup file if it exists
        backup_path = pkl_path + ".backup"
        if (os.path.exists(backup_path) and
                getattr(self.config, 'upper_memory_mode', 'session_summary') != 'event_state'):
            logger.info(f"Removing existing backup file: {backup_path}")
            os.remove(backup_path)

        # ---- 1. 重置内存中的图结构 ----
        logger.info("Clearing existing graph and chunk storage...")
        self.graph_builder.graph = nx.DiGraph()
        self.chunk_storage.clear()
        self.triple_storage.clear()
        self.triple_storage_origin = "extracted"
        self.event_state_layer = EventStateLayer()
        logger.info("✅ Cleared existing graph and chunk storage")

        # ---- 2. 保存原文（后续所有环节只引用 chunk_id） ----
        logger.info("Storing original chunk data...")
        for chunk in chunks:
            # Use unique chunk_id only
            chunk_id = str(chunk.get('chunk_id', 0))
            # Store the complete chunk dict including session_time and other metadata
            if chunk.get('text', ''):
                self.chunk_storage[chunk_id] = chunk
                logger.debug(f"Stored chunk {chunk_id}: {len(chunk.get('text', ''))} characters")
        logger.info(f"Stored {len(self.chunk_storage)} chunks in chunk_storage")

        # ---- 3. 实体/关系抽取 ----
        # Stage 2: Entity and relationship extraction - create progress bar with actual number of chunks
        dialogue_mode = getattr(self.config.chunk, 'dialogue_input', False)
        cost_before_add = self.llm_manager.cost_manager.get_costs()

        # Create progress bar with actual number of chunks (each chunk requires one LLM call)
        from tqdm import tqdm

        if dialogue_mode:
            logger.info("Extracting entities and relationships using dialogue mode...")
            cost_before_dialogue = self.llm_manager.cost_manager.get_costs()

            dialogue_progress_bar = tqdm(total=len(chunks), desc="Stage 2: Dialogue Entity & Relationship Extraction", unit="calls")
            logger.info(f"Stage 2: Processing {len(chunks)} chunks for entity and relationship extraction")

            self.time_manager.start_entity_extraction()
            # 一次调用同时抽实体和关系；返回结果扁平且未去重。
            entities, relationships = await self.dialogue_extractor.extract_from_chunks(chunks, progress_bar=dialogue_progress_bar)
            dialogue_progress_bar.close()
            self.time_manager.end_entity_extraction()
            # 联合抽取没有单独的关系抽取耗时，置 0 以免统计出现重复累计。
            self.time_manager.relationship_extraction_time = 0

            cost_after_dialogue = self.llm_manager.cost_manager.get_costs()
            dialogue_prompt_tokens = cost_after_dialogue.total_prompt_tokens - cost_before_dialogue.total_prompt_tokens
            dialogue_completion_tokens = cost_after_dialogue.total_completion_tokens - cost_before_dialogue.total_completion_tokens

            entity_tokens = dialogue_prompt_tokens // 2 + dialogue_completion_tokens // 2
            relation_tokens = dialogue_prompt_tokens - (dialogue_prompt_tokens // 2) + dialogue_completion_tokens - (dialogue_completion_tokens // 2)

            # 成本无法精确拆分：按一半实体、一半关系计入（仅用于预算统计，不影响行为）。
            self.cost_manager.update_entity_extraction_cost(dialogue_prompt_tokens // 2, dialogue_completion_tokens // 2, self.llm_manager.model)
            self.cost_manager.update_relationship_extraction_cost(
                dialogue_prompt_tokens - (dialogue_prompt_tokens // 2),
                dialogue_completion_tokens - (dialogue_completion_tokens // 2),
                self.llm_manager.model
            )

            # ---- 4a. 对话模式去重 ----
            # 实体先去重并得到“别名 → 规范名”映射，关系再用该映射修正两端。
            # Deduplicate using dialogue-specific methods
            entities, entity_mapping = self.dialogue_extractor.deduplicate_entities(
                entities, self.config.graph.entity_merge_threshold
            )
            # 先把关系两端的别名换成规范实体名，再保存原始出现记录；
            # 上层事件层必须能按实体索引找到关系端点。
            relationships = self.dialogue_extractor._update_relationship_entity_references(
                relationships, entity_mapping
            )
            self._record_triple_occurrences(relationships, chunks)
            relationships = self.dialogue_extractor.deduplicate_relationships(
                relationships, entity_mapping, self.config.graph.relationship_merge_threshold
            )

            logger.info(f"Dialogue mode - After deduplication: {len(entities)} entities, {len(relationships)} relationships")
        else:
            # Use traditional separate extraction
            logger.info("Extracting entities...")

            cost_before_entity = self.llm_manager.cost_manager.get_costs()

            progress_bar = tqdm(total=len(chunks), desc="Stage 2: Entity Extraction", unit="calls")
            logger.info(f"Stage 2: Processing {len(chunks)} chunks for entity extraction")

            self.time_manager.start_entity_extraction()
            entities = await self.entity_extractor.extract_from_chunks(chunks, progress_bar=progress_bar)
            progress_bar.close()
            self.time_manager.end_entity_extraction()

            cost_after_entity = self.llm_manager.cost_manager.get_costs()
            entity_prompt_tokens = cost_after_entity.total_prompt_tokens - cost_before_entity.total_prompt_tokens
            entity_completion_tokens = cost_after_entity.total_completion_tokens - cost_before_entity.total_completion_tokens
            self.cost_manager.update_entity_extraction_cost(entity_prompt_tokens, entity_completion_tokens, self.llm_manager.model)

            # ---- 4b. 非对话模式去重（仅实体） ----
            entities = self.entity_extractor.deduplicate_entities(
                entities, self.config.graph.entity_merge_threshold
            )

            relationships = []
            logger.info("Skipping relationship extraction (unused in dialogue mode)")

        # ---- 5. 生成三元组视图（供统计/后续扩展；图本身直接从 entities/relationships 构建） ----
        # Create triples
        triples = self._create_triples(entities, relationships, chunks)

        # ---- 6. 写入 NetworkX 图 ----
        # Build graph directly from deduplicated entities and relationships
        self.time_manager.start_graph_construction()
        self.graph_builder.build_from_entities_and_relationships(entities, relationships)
        self.time_manager.end_graph_construction()

        # ---- 7. 建立实体名 → 序号索引（事件层会引用这个序号） ----
        # Build entity index for query processing
        self.entity_name_to_index = {entity.get('entity', ''): idx for idx, entity in enumerate(entities)}

        # ---- 8. 预计算向量：查询阶段直接读图属性，避免在线重复编码 ----
        # Pre-compute and store embeddings for entities, relationships, and summaries
        await self._precompute_embeddings(entities, relationships)

        # ---- 9. 上层事件状态（event_state 模式） ----
        if getattr(self.config, 'upper_memory_mode', 'session_summary') == 'event_state':
            # 上层必须在中层实体索引和 Triple 建成后写入，才能记录可靠的证据引用。
            self.event_state_layer = await self.event_state_writer.write_chunks(
                self.event_state_layer, chunks, self._relationships_for_chunks(chunks),
                self._linked_entity_index())

        self.time_manager.end_total_graph_building()
        self._log_graph_building_summary()
        logger.info("Graph built successfully")

    def _log_graph_building_summary(self):
        """打印建图耗时与 token/费用统计。"""
        logger.info("=" * 80)
        logger.info("📊 GRAPH BUILDING SUMMARY")
        logger.info("=" * 80)
        time_summary = self.time_manager.get_graph_building_summary()
        logger.info(f"⏱️  TIME STATISTICS:")
        logger.info(f"   📄 Chunking Time: {time_summary['chunking_time']}s")
        logger.info(f"   🏷️  Entity and Relation Extraction Time: {time_summary['entity_extraction_time']}s")
        logger.info(f"   🕸️  Graph Construction Time: {time_summary['graph_construction_time']}s")
        logger.info(f"   📝 Summary Generation Time: {time_summary['summary_generation_time']}s")
        logger.info(f"   📊 Total Graph Building Time: {time_summary['total_graph_building_time']}s")

        if hasattr(self, 'cost_manager') and self.cost_manager:
            cost_summary = self.cost_manager.get_graph_building_summary()
            logger.info(f"💰 COST STATISTICS:")
            total_extraction_tokens = cost_summary['entity_extraction_tokens'] + cost_summary['relationship_extraction_tokens']
            logger.info(f"   🏷️  Entity and Relation Extraction Tokens: {total_extraction_tokens}")
            logger.info(f"   📝 Summary Generation Tokens: {cost_summary['summary_generation_tokens']}")
            logger.info(f"   📊 Total Graph Building Tokens: {cost_summary['total_graph_building_tokens']}")
            logger.info(f"   💵 Total Cost: ${cost_summary['total_cost_usd']}")

        logger.info("=" * 80)

    def _log_dynamic_add_summary(self):
        """打印增量写入的费用统计（耗时统计由调用链外部负责）。"""
        logger.info("=" * 80)
        logger.info("📊 DYNAMIC ADDITION SUMMARY")
        logger.info("=" * 80)

        if hasattr(self, 'cost_manager') and self.cost_manager:
            cost_summary = self.cost_manager.get_graph_building_summary()
            total_extraction_tokens = cost_summary['entity_extraction_tokens'] + cost_summary['relationship_extraction_tokens']
            logger.info(f"💰 COST STATISTICS:")
            logger.info(f"   🏷️  Entity and Relation Extraction Tokens: {total_extraction_tokens}")
            logger.info(f"   📝 Summary Update Tokens: {cost_summary['summary_generation_tokens']}")
            logger.info(f"   📊 Total Graph Building Tokens: {cost_summary['total_graph_building_tokens']}")
            logger.info(f"   💵 Total Cost: ${cost_summary['total_cost_usd']}")

        logger.info("=" * 80)

    async def _update_summary_for_session(self, session_data: Dict[str, Any]) -> Dict[str, Any]:
        """更新或创建单个会话的摘要（增量写入模式内部使用）。

        Args:
            session_data: 含 session_id、session_time、context。

        Returns:
            更新后的摘要 dict。

        行为：
            - 从 session_summaries.json 读全部旧摘要；
            - 找到同 session_id → ADDITION_PROMPT 增量更新；
            - 没找到 → SUMMARY_PROMPT 新建；
            - 最后把该会话写回列表并整体保存。
        """
        session_id = session_data.get('session_id', 'unknown')

        # Load existing summaries
        summary_path = os.path.join(self.base_dir, "session_summaries.json")
        if os.path.exists(summary_path):
            existing_summaries = self.session_summarizer.load_summaries(summary_path)
        else:
            existing_summaries = []

        # Check if summary for this session already exists
        existing_summary = None
        for summary in existing_summaries:
            if summary.get('session_id') == session_id:
                existing_summary = summary
                break

        # Update or create summary
        if existing_summary:
            logger.info(f"📝 Updating existing summary for session: {session_id} using ADDITION_PROMPT")
            # 摘要更新接口接收的是 chunk 形状的数据，这里把会话包装成同样字段。
            # Create a pseudo-chunk for update
            chunk_data = {
                'session_id': session_id,
                'session_time': session_data.get('session_time', 'unknown'),
                'text': session_data.get('context', '')
            }
            updated_summary = await self.session_summarizer.update_summary_with_chunk(chunk_data, existing_summary)
        else:
            logger.info(f"📝 Creating new summary for session: {session_id} using SUMMARY_PROMPT")
            updated_summary = await self.session_summarizer.summarize_session(session_data)

        # 用 session_id 做键覆盖旧值，再转回列表保存。
        # Save all summaries (merge with existing ones)
        all_summaries_dict = {s.get('session_id', ''): s for s in existing_summaries}
        all_summaries_dict[session_id] = updated_summary
        all_summaries = list(all_summaries_dict.values())
        self.session_summarizer.save_summaries(all_summaries, summary_path)

        return updated_summary

    async def add_single_session(self, session_corpus: List[Dict[str, Any]]) -> None:
        """增量写入一个完整会话（config.graph.add=True 时逐会话调用）。

        单个会话内的步骤：
            1. （session_summary 模式且启用摘要）更新/创建该会话摘要；
            2. 切块；
            3. 保存原文；
            4. 抽实体/关系；
            5. 去重；
            6. 用增量接口并入已有图；
            7. 更新实体索引；
            8. （event_state 模式）把本会话的 Chunk 写入事件层。

        注意：方法本身不落盘，保存由调用方 GraphRAGCore.insert 每会话后执行。
        """
        if not session_corpus:
            logger.warning("Empty session corpus, skipping")
            return

        # Get session info from first document
        session_id = session_corpus[0].get('session_id', 'unknown')
        session_time = session_corpus[0].get('session_time', 'unknown')

        logger.info("=" * 80)
        logger.info(f"🔄 Processing session: {session_id} (time: {session_time})")
        logger.info("=" * 80)

        # ---- Step 1: 摘要更新（两种模式互斥） ----
        # Step 1: Update or create summary for this session
        if (getattr(self.config, 'upper_memory_mode', 'session_summary') != 'event_state'
                and getattr(self.config.retriever, 'enable_summary', False)):
            cost_before_summary = self.llm_manager.cost_manager.get_costs()

            # 一个会话的多条 doc 的 context 用空行拼接后整体摘要。
            # Prepare session data for summarization
            session_data = {
                'session_id': session_id,
                'session_time': str(session_time),
                'context': '\n\n'.join([doc.get('context', '') for doc in session_corpus])
            }

            await self._update_summary_for_session(session_data)

            cost_after_summary = self.llm_manager.cost_manager.get_costs()
            summary_prompt_tokens = cost_after_summary.total_prompt_tokens - cost_before_summary.total_prompt_tokens
            summary_completion_tokens = cost_after_summary.total_completion_tokens - cost_before_summary.total_completion_tokens
            self.cost_manager.update_summary_generation_cost(
                summary_prompt_tokens,
                summary_completion_tokens,
                self.llm_manager.model
            )

        # ---- Step 2: 切块 ----
        # Step 2: Process documents into chunks
        chunks = self.chunk_processor.process_corpus(session_corpus)
        logger.info(f"Processed {len(chunks)} chunks from session {session_id}")

        # ---- Step 3: 保存原文 ----
        # Step 3: Store chunks
        for chunk in chunks:
            chunk_id = str(chunk.get('chunk_id', 0))
            if chunk.get('text', ''):
                self.chunk_storage[chunk_id] = chunk

        # ---- Step 4: 抽取 ----
        # Step 4: Extract entities and relationships
        cost_before_extract = self.llm_manager.cost_manager.get_costs()

        dialogue_mode = getattr(self.config.chunk, 'dialogue_input', False)
        if dialogue_mode:
            logger.info("Extracting entities and relationships using dialogue mode...")
            entities, relationships = await self.dialogue_extractor.extract_from_chunks(chunks)

            # ---- Step 5: 去重 ----
            # Step 5: Deduplicate
            entities, entity_mapping = self.dialogue_extractor.deduplicate_entities(
                entities, self.config.graph.entity_merge_threshold
            )
            relationships = self.dialogue_extractor._update_relationship_entity_references(
                relationships, entity_mapping
            )
            self._record_triple_occurrences(relationships, chunks)
            relationships = self.dialogue_extractor.deduplicate_relationships(
                relationships, entity_mapping, self.config.graph.relationship_merge_threshold
            )

            logger.info(f"Extracted and deduplicated: {len(entities)} entities, {len(relationships)} relationships")
        else:
            logger.info("Extracting entities...")
            entities = await self.entity_extractor.extract_from_chunks(chunks)
            entities = self.entity_extractor.deduplicate_entities(
                entities, self.config.graph.entity_merge_threshold
            )
            relationships = []
            logger.info(f"Extracted {len(entities)} unique entities")

        # 增量模式的成本同样无法精确拆分，按一半实体/一半关系记录。
        cost_after_extract = self.llm_manager.cost_manager.get_costs()
        extract_prompt_tokens = cost_after_extract.total_prompt_tokens - cost_before_extract.total_prompt_tokens
        extract_completion_tokens = cost_after_extract.total_completion_tokens - cost_before_extract.total_completion_tokens

        self.cost_manager.update_entity_extraction_cost(
            extract_prompt_tokens // 2,
            extract_completion_tokens // 2,
            self.llm_manager.model
        )
        self.cost_manager.update_relationship_extraction_cost(
            extract_prompt_tokens - (extract_prompt_tokens // 2),
            extract_completion_tokens - (extract_completion_tokens // 2),
            self.llm_manager.model
        )

        # ---- Step 6: 增量并入图（同名实体/边合并证据） ----
        # Step 6: Add to graph incrementally
        logger.info("Adding entities and relationships to graph...")
        self.graph_builder.add_entities_and_relationships_incrementally(entities, relationships)

        # ---- Step 7: 新实体追加到索引（已存在的保持原序号） ----
        # Step 7: Update entity index
        for entity in entities:
            entity_name = entity.get('entity', '')
            if entity_name and entity_name not in self.entity_name_to_index:
                self.entity_name_to_index[entity_name] = len(self.entity_name_to_index)

        # ---- Step 8: 事件层写入（在中层图更新之后） ----
        if getattr(self.config, 'upper_memory_mode', 'session_summary') == 'event_state':
            self.event_state_layer = await self.event_state_writer.write_chunks(
                self.event_state_layer, chunks, self._relationships_for_chunks(chunks),
                self._linked_entity_index())

        logger.info(f"✅ Session {session_id} processed successfully")
        logger.info("=" * 80)

    async def _dynamic_add_chunks(self, chunks: List[Dict[str, Any]]) -> None:
        """把新 Chunk 追加到已有图（按 Chunk 而非按会话的旧接口）。

        DEPRECATED: This method is kept for backward compatibility.
        New code should use add_single_session() for session-by-session processing.
        """
        logger.info(f"Dynamically adding {len(chunks)} new chunks to existing graph")

        logger.info("Storing new chunk data to chunk_storage...")
        for chunk in chunks:
            # Use unique chunk_id only
            chunk_id = str(chunk.get('chunk_id', 0))
            # Store the complete chunk dict including session_time and other metadata
            if chunk.get('text', ''):
                self.chunk_storage[chunk_id] = chunk
                logger.debug(f"Stored new chunk {chunk_id}: {len(chunk.get('text', ''))} characters")
        logger.info(f"Stored {len(chunks)} new chunks in chunk_storage")

        dialogue_mode = getattr(self.config.chunk, 'dialogue_input', False)
        cost_before_add = self.llm_manager.cost_manager.get_costs()

        if dialogue_mode:
            logger.info("Extracting entities and relationships from new chunks using dialogue mode...")
            entities, relationships = await self.dialogue_extractor.extract_from_chunks(chunks)

            entities, entity_mapping = self.dialogue_extractor.deduplicate_entities(
                entities, self.config.graph.entity_merge_threshold
            )
            relationships = self.dialogue_extractor._update_relationship_entity_references(
                relationships, entity_mapping
            )
            self._record_triple_occurrences(relationships, chunks)
            relationships = self.dialogue_extractor.deduplicate_relationships(
                relationships, entity_mapping, self.config.graph.relationship_merge_threshold
            )

            logger.info(f"Dialogue mode - Extracted {len(entities)} entities and {len(relationships)} relationships from new chunks")
        else:
            # Use traditional separate extraction
            logger.info("Extracting entities from new chunks...")
            entities = await self.entity_extractor.extract_from_chunks(chunks)
            entities = self.entity_extractor.deduplicate_entities(
                entities, self.config.graph.entity_merge_threshold
            )
            logger.info(f"Extracted {len(entities)} unique entities from new chunks")

            # Use traditional separate extraction (entity_extractor only)
            relationships = []
            logger.info("Skipping relationship extraction from new chunks (unused in dialogue mode)")

        # 与 _build_graph 一样先生成三元组视图，再增量写图。
        # Create triples
        triples = self._create_triples(entities, relationships, chunks)
        logger.info(f"Created {len(triples)} triples from new chunks")

        # Add new entities and relationships to existing graph incrementally
        logger.info("Adding new entities and relationships to existing graph incrementally...")
        self.graph_builder.add_entities_and_relationships_incrementally(entities, relationships)

        # 实体索引：新实体从当前长度开始编号，避免与已有编号冲突。
        # Update entity index with new entities
        start_idx = len(self.entity_name_to_index)
        for idx, entity in enumerate(entities):
            entity_name = entity.get('entity', '')
            if entity_name and entity_name not in self.entity_name_to_index:
                self.entity_name_to_index[entity_name] = start_idx + idx
        logger.info(f"Updated entity index, now has {len(self.entity_name_to_index)} entities")

        if getattr(self.config, 'upper_memory_mode', 'session_summary') == 'event_state':
            self.event_state_layer = await self.event_state_writer.write_chunks(
                self.event_state_layer, chunks, self._relationships_for_chunks(chunks),
                self._linked_entity_index())

        cost_after_add = self.llm_manager.cost_manager.get_costs()
        add_prompt_tokens = cost_after_add.total_prompt_tokens - cost_before_add.total_prompt_tokens
        add_completion_tokens = cost_after_add.total_completion_tokens - cost_before_add.total_completion_tokens

        self.cost_manager.update_entity_extraction_cost(add_prompt_tokens // 2, add_completion_tokens // 2, self.llm_manager.model)
        self.cost_manager.update_relationship_extraction_cost(
            add_prompt_tokens - (add_prompt_tokens // 2),
            add_completion_tokens - (add_completion_tokens // 2),
            self.llm_manager.model
        )
        self._log_dynamic_add_summary()
        logger.info("Dynamic addition completed")

    def _create_triples(self, entities: List[Dict[str, Any]],
                       relationships: List[Dict[str, Any]],
                       chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """把关系补齐两端实体类型，生成三元组视图。

        说明：该结果目前只用于日志/统计与潜在扩展；
        图构建直接使用 entities/relationships，并不依赖本方法的输出。
        chunk_id 直接取关系自身携带的来源。

        Args:
            entities:      去重后的实体列表。
            relationships: 去重后的关系列表。
            chunks:        当前批次 Chunk（本方法未使用，保留接口）。

        Returns:
            [{'src', 'tgt', 'src_type', 'tgt_type', 'relation', 'chunk_id'}, ...]
        """
        triples = []

        # Create entity to chunk mapping
        # 注意：entity_chunks 收集后未参与 triple 构造，是历史遗留代码。
        entity_chunks = {}
        for entity in entities:
            entity_name = entity.get('entity', '')
            chunk_id = entity.get('chunk_id', 0)
            if entity_name not in entity_chunks:
                entity_chunks[entity_name] = []
            entity_chunks[entity_name].append(chunk_id)

        for relationship in relationships:
            src = relationship.get('src', '')
            tgt = relationship.get('tgt', '')
            relation = relationship.get('relation', '')
            chunk_id = relationship.get('chunk_id', 0)

            # 三元组的端点类型来自实体表；找不到时为 unknown。
            # Find entity types
            src_type = self._get_entity_type(entities, src)
            tgt_type = self._get_entity_type(entities, tgt)

            triple = {
                'src': src,
                'tgt': tgt,
                'src_type': src_type,
                'tgt_type': tgt_type,
                'relation': relation,
                'chunk_id': chunk_id
            }
            triples.append(triple)

        return triples

    def _get_entity_type(self, entities: List[Dict[str, Any]], entity_name: str) -> str:
        """线性查找实体类型；找不到返回 'unknown'。"""
        for entity in entities:
            if entity.get('entity', '') == entity_name:
                return entity.get('type', 'unknown')
        return 'unknown'

    async def _precompute_embeddings(self, entities: List[Dict[str, Any]],
                                     relationships: List[Dict[str, Any]]) -> None:
        """
        Pre-compute embeddings for all entities, relationships, and summaries.
        Store them in the graph structure for fast retrieval during query time.

        向量用途：
            - 节点向量：查询时把问题实体与图中实体做相似匹配；
            - 边向量：三元组级别的语义匹配/重排；
            - 摘要向量：session_summary 模式下按语义召回相关会话。

        容错：整体失败只记录错误，不让建图失败——
        缺少向量时查询仍可退化为关键词/结构检索。
        """
        if not hasattr(self, 'embedding_manager') or not self.embedding_manager:
            logger.warning("⚠️  Embedding manager not available, skipping embedding precomputation")
            return

        import time
        start_time = time.time()

        try:
            await self._precompute_entity_embeddings(entities)
            await self._precompute_relationship_embeddings(relationships)
            if getattr(self.config, 'upper_memory_mode', 'session_summary') == 'session_summary':
                # event_state 模式没有摘要文件，跳过摘要向量。
                await self._precompute_summary_embeddings()
            elapsed = time.time() - start_time
            logger.info(f"Embedding precomputation completed in {elapsed:.2f}s")

        except Exception as e:
            logger.error(f"❌ Error during embedding precomputation: {e}")
            import traceback
            traceback.print_exc()

    async def _precompute_entity_embeddings(self, entities: List[Dict[str, Any]]) -> None:
        """批量计算实体向量，写入图节点属性 embedding。

        文本构造：名称 + 类型 + 描述（有描述时带上），
        这样相似度能同时利用名称和语义描述。
        若返回数量与输入不一致，整批放弃，避免把向量错配到别的实体。
        """
        if not entities:
            return

        # Prepare entity texts for embedding
        entity_texts = []
        entity_names = []
        for entity in entities:
            entity_name = entity.get('entity', '')
            entity_type = entity.get('type', 'unknown')
            description = entity.get('description', '')

            # Create entity text: combine name, type, and description
            if description:
                entity_text = f"{entity_name} ({entity_type}): {description}"
            else:
                entity_text = f"{entity_name} ({entity_type})"

            entity_texts.append(entity_text)
            entity_names.append(entity_name)

        # Get embeddings in batch (much faster than one by one)
        embeddings = await self.embedding_manager.get_embeddings(entity_texts)

        if not embeddings or len(embeddings) != len(entity_texts):
            logger.error(f"Failed to get embeddings for entities (got {len(embeddings) if embeddings else 0}, expected {len(entity_texts)})")
            return

        # 只写入图中真实存在的实体（去重/清理后可能有实体已不在图中）。
        # Store embeddings in graph nodes
        for entity_name, embedding in zip(entity_names, embeddings):
            if entity_name in self.graph_builder.graph.nodes:
                self.graph_builder.graph.nodes[entity_name]['embedding'] = embedding

    async def _precompute_relationship_embeddings(self, relationships: List[Dict[str, Any]]) -> None:
        """批量计算关系向量，写入对应边的 embedding 属性。

        文本构造：“源 - 关系 - 目标”，有描述时追加描述。
        以 (src, tgt) 定位边；重复关系在图中已合并，后面的向量会覆盖前面的，
        因此最终保留的是该边最后一次出现的语义表示。
        """
        if not relationships:
            return

        # Prepare triple texts for embedding
        triple_texts = []
        triple_keys = []

        for rel in relationships:
            src = rel.get('src', '')
            tgt = rel.get('tgt', '')
            relation = rel.get('relation', '')
            description = rel.get('description', '')

            if description:
                triple_text = f"{src} - {relation} - {tgt}: {description}"
            else:
                triple_text = f"{src} - {relation} - {tgt}"

            triple_texts.append(triple_text)
            triple_keys.append((src, tgt))

        embeddings = await self.embedding_manager.get_embeddings(triple_texts)

        if not embeddings or len(embeddings) != len(triple_texts):
            logger.error(f"Failed to get embeddings for relationships (got {len(embeddings) if embeddings else 0}, expected {len(triple_texts)})")
            return

        for (src, tgt), embedding in zip(triple_keys, embeddings):
            if self.graph_builder.graph.has_edge(src, tgt):
                self.graph_builder.graph.edges[src, tgt]['embedding'] = embedding

    async def _precompute_summary_embeddings(self) -> None:
        """为会话摘要计算向量并回写到 session_summaries.json。

        读取文件 → 批量编码 → 给每条摘要加 embedding 字段 → 覆盖保存。
        文件不存在或为空时直接返回（可能是未启用摘要）。
        """
        summary_path = os.path.join(self.base_dir, "session_summaries.json")

        if not os.path.exists(summary_path):
            return

        import json
        with open(summary_path, 'r', encoding='utf-8') as f:
            summaries = json.load(f)

        if not summaries:
            return

        summary_texts = []
        for summary in summaries:
            summary_text = summary.get('summary', '')
            session_id = summary.get('session_id', '')

            # 把 session_id 拼进文本，避免多条内容相近的摘要难以区分。
            full_text = f"Session {session_id}: {summary_text}"
            summary_texts.append(full_text)

        embeddings = await self.embedding_manager.get_embeddings(summary_texts)

        if not embeddings or len(embeddings) != len(summary_texts):
            logger.error(f"Failed to get embeddings for summaries (got {len(embeddings) if embeddings else 0}, expected {len(summary_texts)})")
            return

        for summary, embedding in zip(summaries, embeddings):
            summary['embedding'] = embedding

        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summaries, f, indent=2, ensure_ascii=False)

    def get_graph_stats(self) -> Dict[str, Any]:
        """返回图统计（节点/边/类型分布）。"""
        stats = self.graph_builder.get_graph_stats()
        return stats

    def get_entity_events(self, entity_name: str, attribute: str = None) -> List[Dict[str, Any]]:
        """从现有中层实体索引直接追溯其上层状态变化。"""
        if entity_name not in self.entity_name_to_index:
            return []
        return self.event_state_layer.events_for_entity(entity_name, attribute)

    def save_graph(self, path: str) -> None:
        """把图、原文、实体索引、原始 Triple 记录（以及事件层）序列化到 pkl。

        event_state 模式额外保存 event_state_layer；旧格式/普通模式只在
        事件层非空时附带保存，以兼容历史文件。

        原子写入（event_state 模式）：
            先写同目录临时文件并 fsync，再用 os.replace 原子替换目标文件。
            这样进程在写入中途崩溃时，旧 pkl 仍然完整可加载。
        """
        graph_data = {
            'graph': self.graph_builder.graph,
            'chunk_storage': self.chunk_storage,
            'entity_name_to_index': self.entity_name_to_index,
        }
        # 原始 Triple 出现记录与图边分开保存：图边是检索索引，
        # triple_storage 是上层可追溯的关系来源，不会被 DiGraph 覆盖。
        if self.triple_storage:
            graph_data['triple_storage'] = self.triple_storage
            graph_data['triple_storage_origin'] = self.triple_storage_origin
        if (getattr(self.config, 'upper_memory_mode', 'session_summary') == 'event_state'
                or self.event_state_layer.facts or self.event_state_layer.events):
            graph_data['event_state_layer'] = self.event_state_layer.to_dict()
        if getattr(self.config, 'upper_memory_mode', 'session_summary') == 'event_state':
            # 新模式先完整写入同目录临时文件，再原子替换；失败时旧图仍可加载。
            import tempfile
            directory = os.path.dirname(os.path.abspath(path))
            os.makedirs(directory, exist_ok=True)
            with tempfile.NamedTemporaryFile(mode='wb', dir=directory, prefix='.event_state_',
                                             suffix='.pkl', delete=False) as stream:
                temp_path = stream.name
                try:
                    pickle.dump(graph_data, stream)
                    stream.flush()
                    os.fsync(stream.fileno())
                except BaseException:
                    # 写入失败：清理临时文件后原样抛出，不触碰目标文件。
                    stream.close()
                    os.remove(temp_path)
                    raise
            try:
                os.replace(temp_path, path)
            except BaseException:
                os.remove(temp_path)
                raise
        else:
            with open(path, 'wb') as f:
                pickle.dump(graph_data, f)
        logger.info(f"Graph and chunk storage saved to {path}")

    def load_graph(self, path: str) -> None:
        """从 pkl 恢复图状态，兼容“只有图”的旧格式文件。

        新格式是 dict，包含 graph/chunk_storage/entity_name_to_index/event_state_layer；
        旧格式直接 pickled 了一个 NetworkX 图，此时原文与索引置空，事件层为空层。
        """
        with open(path, 'rb') as f:
            data = pickle.load(f)

            if isinstance(data, dict):
                self.graph_builder.graph = data.get('graph')
                self.chunk_storage = data.get('chunk_storage', {})
                self.entity_name_to_index = data.get('entity_name_to_index', {})
                self.triple_storage = data.get('triple_storage', {})
                self.triple_storage_origin = data.get(
                    'triple_storage_origin', 'unknown' if self.triple_storage else 'recovered_partial')
                self.event_state_layer = EventStateLayer(data.get('event_state_layer'))
                logger.info(f"Graph loaded from {path} with {len(self.chunk_storage)} chunks, "
                            f"{len(self.entity_name_to_index)} entities and "
                            f"{sum(len(v) for v in self.triple_storage.values())} triple occurrences")
            else:
                self.graph_builder.graph = data
                self.chunk_storage = {}
                self.entity_name_to_index = {}
                self.triple_storage = {}
                self.triple_storage_origin = "recovered_partial"
                self.event_state_layer = EventStateLayer()
                logger.info(f"Graph loaded from {path} (old format, no chunk storage or entity index)")

            # 旧 pkl 没有 triple_storage：从现存图边和上层 fact 快照做一次兜底恢复。
            # 已被 DiGraph 覆盖的关系无法找回，新建图应直接使用 _record_triple_occurrences。
            if not self.triple_storage:
                self._backfill_triple_storage()
        logger.info(f"Graph loaded from {path}")
