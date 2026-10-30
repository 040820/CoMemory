"""非对话模式的实体抽取器（仅抽实体，不抽关系）。

何时使用：
    config.chunk.dialogue_input = False 时，DynamicMemory 走这条路径。
    对话数据集默认 dialogue_input=True，实际使用的是 DialogueExtractor
    （一次 LLM 调用同时抽实体和关系），本类主要服务于普通文档/文本场景。

输出格式：
    每个实体是 dict，至少含 entity（实体名）、type（类型），
    并由本类补上 chunk_id、source_text（原文前 100 字）用于溯源。
"""

from typing import List, Dict, Any
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from init.logger import logger
from base.llm import LLMManager


class EntityExtractor:
    """基于 LLM 的实体抽取与去重。"""

    def __init__(self, llm_manager: LLMManager):
        """注入建图阶段使用的 LLM；本类不持有图或 Embedding。"""
        self.llm = llm_manager
        logger.info("Entity Extractor initialized")

    async def extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """从单段文本抽取实体。

        异常不向上抛，返回空列表：单个 Chunk 抽取失败不应中断整轮建图。
        """
        if not text:
            return []
        try:
            entities = await self.llm.extract_entities(text)
            logger.debug(f"Extracted {len(entities)} entities from text")
            return entities
        except Exception as e:
            logger.error(f"Failed to extract entities: {e}")
            return []

    async def extract_from_chunks(self, chunks: List[Dict[str, Any]], progress_bar=None) -> List[Dict[str, Any]]:
        """批量抽取多个 Chunk 的实体，并补齐来源信息。

        Args:
            chunks:       切块结果，需含 text 与 chunk_id。
            progress_bar: 可选的 tqdm 进度条；并发模式下由 LLMManager 按完成数更新，
                          串行模式下由本方法每处理完一个 Chunk 更新 1。

        Returns:
            展平后的实体列表（同一实体可能在多个 Chunk 中出现，此处不去重）。

        两条路径：
            - LLM 开启并发：batch_extract_entities 内部并发请求，返回值与 chunks 一一对应；
            - 未开启并发：逐个 await，失败时该 Chunk 记为空列表。
        """
        if not chunks:
            return []

        if hasattr(self.llm, 'enable_concurrent') and self.llm.enable_concurrent:
            logger.info(f"Extracting entities from {len(chunks)} chunks concurrently")
            texts = [chunk.get('text', '') for chunk in chunks]
            try:
                # 批量接口按输入顺序返回，每个元素是该 Chunk 的实体列表。
                # Pass progress_bar to batch_extract_entities so it updates as each request completes
                chunk_entities_list = await self.llm.batch_extract_entities(texts, progress_bar=progress_bar)
            except Exception as e:
                # 整批失败时用空列表占位，保证后续 zip 长度一致。
                logger.error(f"Failed to extract entities from chunks: {e}")
                chunk_entities_list = [[] for _ in chunks]
                # Update progress bar for failed requests
                if progress_bar:
                    progress_bar.update(len(chunks))

            all_entities = []
            for i, (chunk, chunk_entities) in enumerate(zip(chunks, chunk_entities_list)):
                for entity in chunk_entities:
                    # 来源信息在建图时用于把实体关联回原文 Chunk。
                    entity['chunk_id'] = chunk.get('chunk_id', 0)
                    entity['source_text'] = chunk.get('text', '')[:100] + '...' if len(chunk.get('text', '')) > 100 else chunk.get('text', '')
                all_entities.extend(chunk_entities)
        else:
            all_entities = []
            for chunk in chunks:
                text = chunk.get('text', '')
                try:
                    chunk_entities = await self.extract_entities(text)
                except Exception as e:
                    logger.error(f"Failed to extract entities from chunk {chunk.get('chunk_id')}: {e}")
                    chunk_entities = []
                finally:
                    # 无论成功失败都推进进度条，否则失败时进度会卡住。
                    if progress_bar:
                        progress_bar.update(1)

                for entity in chunk_entities:
                    entity['chunk_id'] = chunk.get('chunk_id', 0)
                    entity['source_text'] = text[:100] + '...' if len(text) > 100 else text

                all_entities.extend(chunk_entities)

        logger.info(f"Extracted {len(all_entities)} entities from {len(chunks)} chunks")
        return all_entities

    def deduplicate_entities(self, entities: List[Dict[str, Any]],
                           similarity_threshold: float = 0.85) -> List[Dict[str, Any]]:
        """按名称相似度去重（保留首次出现的实体）。

        判定方式：把实体名转小写后按空格切成词集合，计算 Jaccard 相似度；
        达到阈值即视为重复。中文实体没有空格，词集合通常只有一个元素，
        因此实际效果接近“完全相同才合并”——这是该实现的已知局限。

        注意：与 DialogueExtractor.deduplicate_entities 不同，
        本方法不返回 entity_mapping，也不合并 chunk/session 证据。
        """
        if not entities:
            return []

        unique_entities = []

        for entity in entities:
            entity_name = entity.get('entity', '').lower()
            is_duplicate = False

            # 与已保留的每个实体比较，命中第一个即停止（O(n²)）。
            for unique_entity in unique_entities:
                unique_name = unique_entity.get('entity', '').lower()
                similarity = self._calculate_similarity(entity_name, unique_name)

                if similarity >= similarity_threshold:
                    is_duplicate = True
                    break

            if not is_duplicate:
                unique_entities.append(entity)

        return unique_entities

    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Jaccard 相似度：词集合交集 / 并集。"""
        if not text1 or not text2:
            return 0.0

        set1 = set(text1.split())
        set2 = set(text2.split())

        intersection = len(set1 & set2)
        union = len(set1 | set2)

        return intersection / union if union > 0 else 0.0
