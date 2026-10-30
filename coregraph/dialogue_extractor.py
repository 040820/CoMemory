"""对话模式的实体 + 关系联合抽取器（建图阶段的主力）。

与 EntityExtractor 的区别：
    EntityExtractor         每个 Chunk 一次调用，只抽实体；
    DialogueExtractor       每个 Chunk 一次调用，同时抽实体和关系（本项目默认路径）。

LLM 需要输出的文本格式（由 prompt 约定，解析器按 '##' 分段）：
    ("entity"|实体名|实体类型|描述)
    ("relationship"|创建时间|会话ID|源实体|目标实体|关系名|关系强度)
    ##
    ...

数据流：
    chunks → 组装带 session_time/session_id 的 JSON 提示词 → LLM 批量生成
           → 解析出实体/关系 → 补 chunk_id/source_text 等来源信息
           → deduplicate_* 合并同名实体与重复关系 → 交给 GraphBuilder 建图。
"""

from typing import List, Dict, Any, Tuple
import re
import json
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from init.logger import logger
from base.llm import LLMManager
from prompt.entity_prompt import DIALOGUE_EXTRACTION_PROMPT, LOCOMO_EXTRACTION_PROMPT


class DialogueExtractor:
    """按数据集选择提示词，完成“对话 Chunk → 实体/关系”的抽取与合并。"""

    def __init__(self, llm_manager: LLMManager, data_type: str = "LongmemEval"):
        """Args:
            llm_manager: 建图阶段共用的 LLM（并发设置由它决定）。
            data_type:   "LOCOMO" 使用 LOCOMO 专用提示词，否则使用通用对话提示词。
        """
        self.llm = llm_manager
        self.data_type = data_type
        if data_type == "LOCOMO":
            self.extraction_prompt = LOCOMO_EXTRACTION_PROMPT
            logger.info("Dialogue Extractor initialized with LOCOMO prompt")
        else:
            self.extraction_prompt = DIALOGUE_EXTRACTION_PROMPT
            logger.info("Dialogue Extractor initialized with LongmemEval prompt")

    async def extract_entities_and_relationships(self, text: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """抽取单段文本的实体与关系（无会话元数据的简化入口）。

        主流程实际使用 extract_from_chunks；本方法保留给单条文本调用。
        这里把纯文本伪装成一个 JSON（session 信息为 unknown），
        以满足提示词要求的结构。
        """
        if not text:
            return [], []

        try:
            mock_json = {
                "session_time": "unknown",
                "text": text,
                "session_id": "unknown"
            }

            formatted_text = json.dumps(mock_json)
            logger.debug(f"Converted plain text to JSON format for prompt: {formatted_text[:100]}")
            prompt = self.extraction_prompt.format(text=formatted_text)
            response = await self.llm.generate(prompt)

            entities, relationships = self._parse_dialogue_response(response)
            logger.debug(f"Extracted {len(entities)} entities and {len(relationships)} relationships from dialogue")
            return entities, relationships

        except Exception as e:
            # 单条失败返回空结果，不中断整体建图。
            logger.error(f"Failed to extract entities and relationships from dialogue: {e}")
            logger.error(f"Input text that caused error: {text[:200]}")
            return [], []

    def _parse_dialogue_response(self, response: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """解析 LLM 输出：按 '##' 切分，逐段判断是实体还是关系。

        约定：实体段以 '("entity"|' 开头，关系段以 '("relationship"|' 开头；
        'END' 与空段忽略。无法解析的段只记警告并跳过，不影响其它段。
        """
        entities = []
        relationships = []

        if not response:
            return entities, relationships
        parts = response.split('##')

        for part in parts:
            part = part.strip()
            if not part or part == 'END':
                continue

            if part.startswith('("entity"|'):
                entity = self._parse_entity(part)
                if entity:
                    entities.append(entity)
            elif part.startswith('("relationship"|'):
                relationship = self._parse_relationship(part)
                if relationship:
                    relationships.append(relationship)

        return entities, relationships

    def _parse_entity(self, entity_str: str) -> Dict[str, Any]:
        """解析实体行：("entity"|名称|类型|描述)

        第 4 段（描述）可省略；字段不足 3 段时返回 None。
        """
        try:
            # Remove the outer parentheses and split by |
            content = entity_str.strip('()')
            parts = [part.strip('"') for part in content.split('|')]

            if len(parts) >= 3 and parts[0] == 'entity':
                return {
                    'entity': parts[1],
                    'type': parts[2],
                    'description': parts[3] if len(parts) > 3 else ''
                }
        except Exception as e:
            logger.warning(f"Failed to parse entity: {entity_str}, error: {e}")

        return None

    def _parse_relationship(self, relationship_str: str) -> Dict[str, Any]:
        """解析关系行：("relationship"|创建时间|会话ID|源实体|目标实体|关系名|强度)

        强度（strength）是 0~10 的整数，同时换算成 weight = strength / 10，
        供查询阶段做加权排序。字段不足 7 段时返回 None。
        """
        try:
            # Remove the outer parentheses and split by |
            content = relationship_str.strip('()')
            parts = [part.strip('"') for part in content.split('|')]

            if len(parts) >= 7 and parts[0] == 'relationship':
                return {
                    'create_time': parts[1],
                    'session_id': parts[2],  # Fixed: was 'origin', now 'session_id' to match prompt
                    'src': parts[3],
                    'tgt': parts[4],
                    'relation': parts[5],
                    'strength': int(parts[6]) if parts[6].isdigit() else 1,
                    'weight': float(parts[6]) / 10.0 if parts[6].isdigit() else 0.1  # Convert strength to weight
                }
        except Exception as e:
            logger.warning(f"Failed to parse relationship: {relationship_str}, error: {e}")

        return None

    async def extract_from_chunks(self, chunks: List[Dict[str, Any]], progress_bar=None) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """批量抽取多个对话 Chunk 的实体与关系。

        Args:
            chunks:       DialogChunkProcessor 产出的 Chunk，含 text/chunk_id/
                          session_id/session_time。
            progress_bar: 可选 tqdm；并发时由批量接口更新，串行时每个 Chunk 更新 1。

        Returns:
            (all_entities, all_relationships)：
            尚未去重，每个结果都带 chunk_id；串行路径还会补 session_time/session_id
            与 source_text。

        并发与串行两条路径的差异：
            - 并发路径只给实体补 chunk_id/source_text，关系的 session 信息
              依赖提示词输出或后续去重合并；
            - 串行路径额外为实体补 session_time/session_id，并为关系补 source_text、
              create_time/session_time/session_id 的兜底值。字段更完整。
        """
        if not chunks:
            return [], []

        if hasattr(self.llm, 'enable_concurrent') and self.llm.enable_concurrent:
            prompts = []
            chunk_metas = []

            for chunk in chunks:
                text = chunk.get('text', '')

                # 提示词要求输入是 JSON，并带上会话时间/ID 以便模型判断事件时序。
                # Get session metadata from chunk for better context
                session_time = chunk.get('session_time', 'unknown')
                session_id = chunk.get('session_id', 'unknown')

                # Create JSON structure with session metadata (matching prompt format)
                enhanced_json = {
                    "text": text,
                    "session_time": str(session_time),
                    "session_id": str(session_id)
                }

                # Use the enhanced JSON format
                formatted_text = json.dumps(enhanced_json)
                prompt = self.extraction_prompt.format(text=formatted_text)
                prompts.append(prompt)
                chunk_metas.append(chunk)

            try:
                # 批量接口返回顺序与 prompts 一致；进度条按完成数推进。
                # Pass progress_bar to batch_generate so it updates as each request completes
                responses = await self.llm.batch_generate(prompts, progress_bar=progress_bar)
            except Exception as e:
                logger.error(f"Failed to batch extract from chunks: {e}")
                responses = ["" for _ in prompts]
                # Update progress bar for failed requests
                if progress_bar:
                    progress_bar.update(len(prompts))

            all_entities = []
            all_relationships = []

            for i, (response, chunk) in enumerate(zip(responses, chunk_metas)):
                try:
                    chunk_entities, chunk_relationships = self._parse_dialogue_response(response)

                    # 来源信息：实体记录 Chunk 与原话片段，关系记录 Chunk。
                    # Add chunk metadata to entities
                    for entity in chunk_entities:
                        entity['chunk_id'] = chunk.get('chunk_id', 0)
                        entity['source_text'] = chunk.get('text', '')[:100] + '...' if len(chunk.get('text', '')) > 100 else chunk.get('text', '')

                    # Add chunk metadata to relationships
                    for relationship in chunk_relationships:
                        relationship['chunk_id'] = chunk.get('chunk_id', 0)

                    all_entities.extend(chunk_entities)
                    all_relationships.extend(chunk_relationships)
                except Exception as e:
                    # 单个 Chunk 的解析失败只影响它自己。
                    logger.error(f"Failed to parse response for chunk {i}: {e}")
                    continue
        else:
            all_entities = []
            all_relationships = []

            for chunk in chunks:
                text = chunk.get('text', '')

                # 与并发路径相同的 JSON 包装。
                session_time = chunk.get('session_time', 'unknown')
                session_id = chunk.get('session_id', 'unknown')
                enhanced_json = {
                    "text": text,
                    "session_time": str(session_time),
                    "session_id": str(session_id)
                }

                formatted_text = json.dumps(enhanced_json)
                try:
                    prompt = self.extraction_prompt.format(text=formatted_text)
                    response = await self.llm.generate(prompt)
                    chunk_entities, chunk_relationships = self._parse_dialogue_response(response)
                except Exception as e:
                    logger.error(f"Failed to extract from chunk: {e}")
                    chunk_entities, chunk_relationships = [], []
                finally:
                    # 成功失败都推进进度。
                    if progress_bar:
                        progress_bar.update(1)

                # 串行路径补齐实体来源与会话信息。
                # Add chunk metadata to entities
                for entity in chunk_entities:
                    entity['chunk_id'] = chunk.get('chunk_id', chunk.get('doc_id', 0))
                    entity['source_text'] = text[:100] + '...' if len(text) > 100 else text
                    entity['session_time'] = session_time
                    entity['session_id'] = session_id

                # 串行路径补齐关系来源；时间优先级：模型给出的具体时间 > 会话时间。
                # Add chunk metadata to relationships
                for relationship in chunk_relationships:
                    relationship['chunk_id'] = chunk.get('chunk_id', chunk.get('doc_id', 0))
                    relationship['source_text'] = text[:100] + '...' if len(text) > 100 else text
                    # Preserve create_time from LLM extraction, fallback to session_time from chunk
                    if 'create_time' not in relationship or not relationship['create_time']:
                        relationship['create_time'] = session_time
                    # Also set session_time for compatibility
                    if 'session_time' not in relationship or not relationship['session_time']:
                        relationship['session_time'] = relationship.get('create_time', session_time)
                    # Set session_id if not already present
                    if 'session_id' not in relationship or not relationship['session_id']:
                        relationship['session_id'] = session_id

                all_entities.extend(chunk_entities)
                all_relationships.extend(chunk_relationships)

        logger.info(f"Extracted {len(all_entities)} entities and {len(all_relationships)} relationships from {len(chunks)} dialogue chunks")
        return all_entities, all_relationships

    def deduplicate_entities(self, entities: List[Dict[str, Any]],
                           similarity_threshold: float = 0.85) -> Tuple[List[Dict[str, Any]], Dict[str, str]]:
        """合并同名/近名实体，并返回“原名称 → 规范名称”的映射。

        判定顺序：
            1. 名称（小写）完全相同且类型相同 → 直接合并；
            2. 名称 Jaccard 相似度 ≥ 阈值，且类型相同 → 合并；
        合并时把被合并实体的 session/chunk 证据追加到保留实体上。

        Returns:
            (unique_entities, entity_mapping)
            entity_mapping 用于随后修正关系两端的实体名，
            保证关系不会指向一个已被合并掉的别名。

        注意：相似度按空格分词计算，对中文实体基本等价于“完全相同才合并”。
        """
        if not entities:
            return [], {}

        unique_entities = []
        entity_mapping = {}

        for entity in entities:
            entity_name = entity.get('entity', '')
            entity_name_lower = entity_name.lower()
            entity_type = entity.get('type', '').lower()
            is_duplicate = False
            canonical_entity = None

            for unique_entity in unique_entities:
                unique_name = unique_entity.get('entity', '')
                unique_name_lower = unique_name.lower()
                unique_type = unique_entity.get('type', '').lower()
                if entity_name_lower == unique_name_lower and entity_type == unique_type:
                    # 大小写不同但名称、类型一致：视为同一实体。
                    is_duplicate = True
                    canonical_entity = unique_entity
                    self._merge_entity_sessions(unique_entity, entity)
                    break

                similarity = self._calculate_similarity(entity_name_lower, unique_name_lower)
                if similarity >= similarity_threshold and self._are_types_compatible(entity_type, unique_type):
                    is_duplicate = True
                    canonical_entity = unique_entity
                    self._merge_entity_sessions(unique_entity, entity)
                    break

            if is_duplicate:
                # Record the mapping from duplicate to canonical name
                canonical_name = canonical_entity.get('entity', '')
                entity_mapping[entity_name] = canonical_name  # Map original names
                logger.debug(f"Entity mapping: '{entity_name}' -> '{canonical_name}'")
            else:
                unique_entities.append(entity)
                # Entity is its own canonical name
                entity_mapping[entity_name] = entity_name

        return unique_entities, entity_mapping

    def deduplicate_relationships(self, relationships: List[Dict[str, Any]],
                                entity_mapping: Dict[str, str] = None,
                                similarity_threshold: float = 0.9) -> List[Dict[str, Any]]:

        """合并重复关系，并先把两端实体名换成规范名称。

        去重键：(src, tgt, relation) 全小写后完全相等。
        similarity_threshold 参数当前未使用（仅保留接口兼容）。

        合并时：强度取最大、weight 重算，并累积 chunk/session/time 证据。
        """
        if not relationships:
            return []
        if entity_mapping:
            # 先归一化实体名，否则 (A,B) 与 (a,b) 会被当成两条不同关系。
            relationships = self._update_relationship_entity_references(relationships, entity_mapping)

        unique_relationships = []

        for relationship in relationships:
            src = relationship.get('src', '').lower()
            tgt = relationship.get('tgt', '').lower()
            relation = relationship.get('relation', '').lower()
            is_duplicate = False

            for unique_relationship in unique_relationships:
                unique_src = unique_relationship.get('src', '').lower()
                unique_tgt = unique_relationship.get('tgt', '').lower()
                unique_relation = unique_relationship.get('relation', '').lower()

                # Exact match
                if src == unique_src and tgt == unique_tgt and relation == unique_relation:
                    is_duplicate = True
                    # 保留第一条关系对象，把后续证据并入其中。
                    # Merge session information and update strength
                    self._merge_relationship_sessions(unique_relationship, relationship)
                    break

            if not is_duplicate:
                unique_relationships.append(relationship)

        return unique_relationships

    def _update_relationship_entity_references(self, relationships: List[Dict[str, Any]],
                                             entity_mapping: Dict[str, str]) -> List[Dict[str, Any]]:
        """把关系两端的别名替换为去重后的规范实体名（返回浅拷贝列表）。"""
        updated_relationships = []
        updates_count = 0

        for relationship in relationships:
            updated_relationship = relationship.copy()

            # Update src entity reference
            src = relationship.get('src', '')
            if src in entity_mapping and entity_mapping[src] != src:
                updated_relationship['src'] = entity_mapping[src]
                updates_count += 1
                logger.debug(f"Updated src: '{src}' -> '{entity_mapping[src]}'")

            # Update tgt entity reference
            tgt = relationship.get('tgt', '')
            if tgt in entity_mapping and entity_mapping[tgt] != tgt:
                updated_relationship['tgt'] = entity_mapping[tgt]
                updates_count += 1
                logger.debug(f"Updated tgt: '{tgt}' -> '{entity_mapping[tgt]}'")

            updated_relationships.append(updated_relationship)

        return updated_relationships

    def _merge_entity_sessions(self, existing_entity: Dict[str, Any], new_entity: Dict[str, Any]):
        """把被合并实体的 session/chunk 证据累积到保留实体上。

        规则：
            - session_time/session_id/chunk_id 字段保留“最新一次”的值；
            - 同时维护 session_times/session_ids/chunk_ids 列表，保存全部历史来源。
        这让查询阶段既能定位最新证据，也能追溯该实体的所有出现位置。
        """
        if 'session_time' in new_entity:
            existing_sessions = existing_entity.get('session_times', [])
            if 'session_time' in existing_entity:
                existing_sessions.append(existing_entity['session_time'])
                existing_entity['session_times'] = list(set(existing_sessions + [new_entity['session_time']]))
            else:
                existing_entity['session_times'] = [new_entity['session_time']]
            existing_entity['session_time'] = new_entity['session_time']  # Keep latest

        if 'session_id' in new_entity:
            existing_sessions = existing_entity.get('session_ids', [])
            if 'session_id' in existing_entity:
                existing_sessions.append(existing_entity['session_id'])
                existing_entity['session_ids'] = list(set(existing_sessions + [new_entity['session_id']]))
            else:
                existing_entity['session_ids'] = [new_entity['session_id']]
            existing_entity['session_id'] = new_entity['session_id']  # Keep latest

        if 'chunk_id' in new_entity:
            existing_chunks = existing_entity.get('chunk_ids', [])
            if 'chunk_id' in existing_entity:
                existing_chunks.append(existing_entity['chunk_id'])
                existing_entity['chunk_ids'] = list(set(existing_chunks + [new_entity['chunk_id']]))
            else:
                existing_entity['chunk_ids'] = [new_entity['chunk_id']]
            existing_entity['chunk_id'] = new_entity['chunk_id']  # Keep latest

    def _merge_relationship_sessions(self, existing_relationship: Dict[str, Any], new_relationship: Dict[str, Any]):
        """把重复关系的证据合并到保留关系上。

        - strength 取最大值（越常出现越可信），并重算 weight；
        - session_time/session_id/chunk_id 指向最新来源；
        - *_times/*_ids/*_chunk_ids 列表累积全部来源，供图构建与查询使用。
        """
        # Update strength (take the higher value)
        existing_strength = existing_relationship.get('strength', 1)
        new_strength = new_relationship.get('strength', 1)
        existing_relationship['strength'] = max(existing_strength, new_strength)
        existing_relationship['weight'] = existing_relationship['strength'] / 10.0

        if 'session_time' in new_relationship and new_relationship['session_time']:
            existing_sessions = existing_relationship.get('session_times', [])
            if 'session_time' in existing_relationship and existing_relationship['session_time']:
                existing_sessions.append(existing_relationship['session_time'])
                existing_relationship['session_times'] = list(set(existing_sessions + [new_relationship['session_time']]))
            else:
                existing_relationship['session_times'] = [new_relationship['session_time']]
            existing_relationship['session_time'] = new_relationship['session_time']

        if 'session_id' in new_relationship:
            existing_sessions = existing_relationship.get('session_ids', [])
            if 'session_id' in existing_relationship:
                existing_sessions.append(existing_relationship['session_id'])
                existing_relationship['session_ids'] = list(set(existing_sessions + [new_relationship['session_id']]))
            else:
                existing_relationship['session_ids'] = [new_relationship['session_id']]
            existing_relationship['session_id'] = new_relationship['session_id']

        # Merge chunk IDs
        if 'chunk_id' in new_relationship:
            existing_chunks = existing_relationship.get('chunk_ids', [])
            if 'chunk_id' in existing_relationship:
                existing_chunks.append(existing_relationship['chunk_id'])
                existing_relationship['chunk_ids'] = list(set(existing_chunks + [new_relationship['chunk_id']]))
            else:
                existing_relationship['chunk_ids'] = [new_relationship['chunk_id']]
            existing_relationship['chunk_id'] = new_relationship['chunk_id']  # Keep latest

    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """词集合 Jaccard 相似度（空格分词）。"""
        if not text1 or not text2:
            return 0.0
        set1 = set(text1.split())
        set2 = set(text2.split())
        intersection = len(set1 & set2)
        union = len(set1 | set2)

        return intersection / union if union > 0 else 0.0

    def _are_types_compatible(self, type1: str, type2: str) -> bool:
        """类型兼容判定：当前要求完全相同，避免把 PERSON 与 ORGANIZATION 合并。"""
        return type1 == type2
