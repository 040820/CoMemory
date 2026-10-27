"""中层知识图构造器：event_state 使用逐来源多边，旧模式保留单边图。

数据结构约定（与 DynamicMemory 的调用方保持一致）：
    节点：节点名就是实体名（字符串），属性包含
          entity_name / entity_type / chunk_id / description，之后还会被写入 embedding。
    边：  src --relation_name--> tgt，新模式以稳定 Triple ID 区分平行边，属性包含
          chunk_id / session_id / session_time / description，
          增量合并时还会累积 chunk_ids / session_ids / session_times / strength / weight。

两个构建入口：
    build_from_entities_and_relationships        从头建图（一次性写入）
    add_entities_and_relationships_incrementally 增量并入已有图（add 模式）
新模式中不同 Chunk 的相同 Triple 也是不同边，同 Chunk 的重复写入幂等。
旧模式增量入口保留原有的实体及单边来源合并逻辑。
"""

from typing import List, Dict, Any
import networkx as nx
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from init.logger import logger
from init.config import Config
from .event_candidate_schema import triple_id


class GraphBuilder:
    """封装对 NetworkX 图的所有写操作。"""

    def __init__(self, config: Config):
        """初始化配置对应的有向图；实体节点不因平行关系而复制。"""
        self.config = config
        self.graph = self.new_graph()
        logger.info("Graph Builder initialized")

    def new_graph(self):
        """新记忆模式按 Triple 出现记录建多边；旧模式保留原图语义。"""
        if getattr(self.config, 'upper_memory_mode', 'session_summary') == 'event_state':
            return nx.MultiDiGraph(memory_graph_schema='triple_occurrence_v1')
        return nx.DiGraph()

    def _occurrence_metadata(self, relationship, relationship_data):
        """给新模式的边补齐逐来源元数据，不用发生时间替代观察时间。"""
        if not self.graph.is_multigraph():
            return relationship_data
        relationship_data['session_time'] = relationship.get('session_time', '')
        relationship_data['event_time'] = relationship.get('event_time') or relationship.get('create_time')
        for field in ('strength', 'weight', 'embedding'):
            if field in relationship:
                relationship_data[field] = relationship[field]
        return relationship_data

    def restore_graph(self, graph, occurrences, origin):
        """旧图仅在内存中适配；只有完整原始出现记录能补回平行关系。"""
        if graph is None:
            raise ValueError('记忆文件缺少 graph')
        event_mode = getattr(self.config, 'upper_memory_mode', 'session_summary') == 'event_state'
        if not event_mode:
            if graph.is_multigraph():
                raise ValueError('多关系记忆图需要 event_state 配置，不能静默压缩为 Baseline 图')
            self.graph = graph
            return
        if graph.is_multigraph():
            self.graph = graph
            return
        if origin == 'extracted' and occurrences:
            self.graph = self.new_graph()
            self.graph.add_nodes_from(graph.nodes(data=True))
            refs = [ref for entries in occurrences.values() for ref in entries]
            self.add_entities_and_relationships_incrementally([], refs)
            self.graph.graph['restored_from'] = 'extracted_triple_storage'
        else:
            # 丢失的旧关系不猜测；仅把旧文件中实际存在的边搬入多边容器。
            self.graph = nx.MultiDiGraph(graph)
            self.graph.graph['memory_graph_schema'] = 'legacy_partial'
            self.graph.graph['restored_from'] = 'legacy_edges_only'

    def add_entity(self, entity: Dict[str, Any]) -> None:
        """添加（或覆盖）一个实体节点。

        Args:
            entity: 必须含 entity_name；其余字段（entity_type/chunk_id/description）
                    会作为节点属性一并写入。

        注意：NetworkX 的 add_node 对已存在节点是“更新属性”，不是报错。
        增量路径里调用前通常已判断过是否为新实体。
        """
        entity_name = entity.get('entity_name', '')
        if entity_name:
            self.graph.add_node(entity_name, **entity)
            logger.debug(f"Added entity: {entity_name}")

    def add_relationship(self, relationship: Dict[str, Any]) -> None:
        """添加一条有向边。

        Args:
            relationship: 必须含 src_id / tgt_id / relation_name。
                          缺少任一字段则静默跳过（LLM 抽取可能返回不完整关系）。

        event_state 中使用稳定的逐 Chunk Triple ID 作边 key：同实体对的
        不同关系、不同 Chunk 的同一关系均独立保留；同源重复写入幂等。
        旧模式保持 DiGraph 行为。
        """
        src = relationship.get('src_id', '')
        tgt = relationship.get('tgt_id', '')
        relation = relationship.get('relation_name', '')

        if src and tgt and relation:
            if self.graph.is_multigraph():
                chunk_id = relationship.get('chunk_id')
                if chunk_id is None or str(chunk_id) == '':
                    raise ValueError('多关系记忆边必须携带 Chunk ID')
                ref = {'src': src, 'relation': relation, 'tgt': tgt,
                       'chunk_id': str(chunk_id)}
                key = triple_id(ref)
                data = dict(relationship, chunk_id=str(chunk_id), triple_id=key)
                if self.graph.has_edge(src, tgt, key):
                    # 同一来源只补充非空元数据，不累积其它 Chunk 的来源。
                    self.graph.edges[src, tgt, key].update({
                        name: value for name, value in data.items()
                        if value is not None and value != ''})
                else:
                    self.graph.add_edge(src, tgt, key=key, **data)
            else:
                self.graph.add_edge(src, tgt, **relationship)
            logger.debug(f"Added relationship: {src} -> {relation} -> {tgt}")

    def build_from_entities_and_relationships(self, entities: List[Dict[str, Any]],
                                             relationships: List[Dict[str, Any]]) -> None:
        """一次性建图：先写实体，再写关系，最后清理孤立节点。

        Args:
            entities:      抽取器输出的实体列表，字段为 entity/type/chunk_id/description。
            relationships: 抽取器输出的关系列表，字段为 src/tgt/relation/chunk_id/
                           session_id/create_time/description 等。

        字段映射：调用方使用 entity/src/tgt，图内部统一改名为
        entity_name/src_id/tgt_id，避免与 NetworkX 保留属性混淆。
        """
        logger.debug(f"Adding {len(entities)} entities to graph")
        for entity in entities:
            entity_data = {
                'entity_name': entity.get('entity', ''),
                'entity_type': entity.get('type', 'unknown'),
                'chunk_id': entity.get('chunk_id', ''),
                'description': entity.get('description', '')
            }
            self.add_entity(entity_data)

        logger.debug(f"Adding {len(relationships)} relationships to graph")
        for relationship in relationships:
            # 关系时间优先用 LLM 从原文识别出的 create_time，
            # 抽不到才退回该 Chunk 所属会话的时间。
            # Prefer create_time from LLM extraction, fallback to session_time from chunk
            session_time = relationship.get('create_time', '') or relationship.get('session_time', '')

            relationship_data = {
                'src_id': relationship.get('src', ''),
                'tgt_id': relationship.get('tgt', ''),
                'relation_name': relationship.get('relation', ''),
                'chunk_id': relationship.get('chunk_id', ''),
                'session_id': relationship.get('session_id', ''),
                'session_time': session_time,  # Use create_time if available
                'description': relationship.get('description', '')
            }
            self.add_relationship(self._occurrence_metadata(relationship, relationship_data))

        # 没有边的实体无法参与图检索，保留它们只会污染统计和 PPR。
        self._remove_isolated_nodes()

    def get_graph_stats(self) -> Dict[str, Any]:
        """返回节点数、边数以及按类型统计的分布（用于日志）。"""
        return {
            'num_nodes': len(self.graph.nodes),
            'num_edges': len(self.graph.edges),
            'node_types': self._get_node_type_counts(),
            'edge_types': self._get_edge_type_counts()
        }

    def _get_node_type_counts(self) -> Dict[str, int]:
        """统计各 entity_type 的节点数量。"""
        type_counts = {}
        for node, data in self.graph.nodes(data=True):
            node_type = data.get('entity_type', 'unknown')
            type_counts[node_type] = type_counts.get(node_type, 0) + 1
        return type_counts

    def _get_edge_type_counts(self) -> Dict[str, int]:
        """统计各 relation_name 的边数量。"""
        type_counts = {}
        for _, _, data in self.graph.edges(data=True):
            edge_type = data.get('relation_name', 'unknown')
            type_counts[edge_type] = type_counts.get(edge_type, 0) + 1
        return type_counts

    def _remove_isolated_nodes(self) -> None:
        """删除度数为 0 的节点（既无出边也无入边）。"""
        isolated_nodes = [node for node in self.graph.nodes() if self.graph.degree(node) == 0]

        if isolated_nodes:
            self.graph.remove_nodes_from(isolated_nodes)

    def add_entities_and_relationships_incrementally(self, entities: List[Dict[str, Any]],
                                                    relationships: List[Dict[str, Any]]) -> None:
        """把新一批抽取结果并入已有图（add 模式 / 逐会话写入）。

        与一次性建图的区别：
            - 同名实体不新建，而是合并属性；
            - 已存在的 (src, tgt) 边不新建，而是合并证据列表；
            - 每次结束都清理孤立节点，避免历史残留。

        实体按规范名称去重；event_state 关系按逐来源 Triple ID 去重。
        以下按 (src, tgt) 合并边的分支只用于原版单边图模式。
        """
        logger.info(f"🔄 Incrementally adding {len(entities)} entities and {len(relationships)} relationships")

        new_entities = 0
        merged_entities = 0
        new_relationships = 0

        for entity in entities:
            entity_name = entity.get('entity', '')
            if not entity_name:
                continue

            entity_data = {
                'entity_name': entity_name,
                'entity_type': entity.get('type', 'unknown'),
                'chunk_id': entity.get('chunk_id', ''),
                'description': entity.get('description', '')
            }

            if entity_name in self.graph.nodes:
                # 已存在：只补空字段，不覆盖已有描述；chunk_id 记录最新来源。
                existing_data = self.graph.nodes[entity_name]
                merged_data = self._merge_entity_attributes(existing_data, entity_data)
                self.graph.nodes[entity_name].update(merged_data)
                merged_entities += 1
                logger.debug(f"Merged entity: {entity_name}")
            else:
                self.add_entity(entity_data)
                new_entities += 1
                logger.debug(f"Added new entity: {entity_name}")

        for relationship in relationships:
            relationship_data = {
                'src_id': relationship.get('src', ''),
                'tgt_id': relationship.get('tgt', ''),
                'relation_name': relationship.get('relation', ''),
                'chunk_id': relationship.get('chunk_id', ''),
                'session_id': relationship.get('session_id', ''),
                'session_time': relationship.get('session_time', ''),
                'description': relationship.get('description', '')
            }

            src = relationship_data['src_id']
            tgt = relationship_data['tgt_id']

            if src and tgt and relationship_data['relation_name']:
                if self.graph.is_multigraph():
                    before = self.graph.number_of_edges()
                    self.add_relationship(self._occurrence_metadata(relationship, relationship_data))
                    new_relationships += self.graph.number_of_edges() - before
                elif self.graph.has_edge(src, tgt):
                    # 已存在同向边：累积 chunk/session/时间证据，并更新强度。
                    existing_edge_data = self.graph.edges[src, tgt]
                    self._merge_relationship_data(existing_edge_data, relationship_data)
                    logger.debug(f"Merged relationship: {src} -> {relationship_data['relation_name']} -> {tgt}")
                else:
                    self.add_relationship(relationship_data)
                    new_relationships += 1
                    logger.debug(f"Added new relationship: {src} -> {relationship_data['relation_name']} -> {tgt}")

        logger.info(f"✅ Incremental addition completed:")
        logger.info(f"   - New entities: {new_entities}")
        logger.info(f"   - Merged entities: {merged_entities}")
        logger.info(f"   - New relationships: {new_relationships}")
        logger.info(f"   - Total graph: {len(self.graph.nodes)} nodes, {len(self.graph.edges)} edges")

        self._remove_isolated_nodes()

        logger.info(f"After cleaning: {len(self.graph.nodes)} nodes and {len(self.graph.edges)} edges")

    def _merge_entity_attributes(self, existing: Dict[str, Any], new: Dict[str, Any]) -> Dict[str, Any]:
        """合并同名实体的属性。

        规则：
            - 新值非空且旧值为空 → 用新值补齐；
            - chunk_id 特判 → 即使旧值非空也更新为最新来源（保留最新证据定位）；
            - 其余情况保留旧值，避免后用描述覆盖先前的完整描述。
        """
        merged = existing.copy()

        for key, value in new.items():
            if value and (key not in merged or not merged[key]):
                merged[key] = value
            elif key == 'chunk_id' and value:
                existing_chunk = merged.get('chunk_id', '')
                if existing_chunk != value:
                    merged[key] = value

        return merged

    def _merge_relationship_data(self, existing: Dict[str, Any], new: Dict[str, Any]) -> None:
        """合并重复边的证据信息（原地修改 existing）。

        累积内容：
            chunk_ids/session_ids/session_times：该关系在不同来源中出现过的全部证据；
            chunk_id/session_id/session_time：指向最新一次来源（供旧查询逻辑使用）；
            strength：多次出现取最大值；weight = strength / 10。
        """
        # 强度取历史最大值：反复出现的关系更可信。
        existing_strength = existing.get('strength', 1)
        new_strength = new.get('strength', 1)
        existing['strength'] = max(existing_strength, new_strength)
        existing['weight'] = existing['strength'] / 10.0

        # ---- 累积 chunk 证据 ----
        existing_chunks = existing.get('chunk_ids', [])

        if 'chunk_id' in existing and existing['chunk_id']:
            if existing['chunk_id'] not in existing_chunks:
                existing_chunks.append(existing['chunk_id'])

        if 'chunk_ids' in new and new['chunk_ids']:
            existing_chunks.extend(new['chunk_ids'])
        elif 'chunk_id' in new and new['chunk_id']:
            existing_chunks.append(new['chunk_id'])

        existing['chunk_ids'] = list(set(existing_chunks))
        if 'chunk_id' in new:
            existing['chunk_id'] = new['chunk_id']

        # ---- 累积 session 证据 ----
        existing_sessions = existing.get('session_ids', [])

        if 'session_id' in existing and existing['session_id']:
            if existing['session_id'] not in existing_sessions:
                existing_sessions.append(existing['session_id'])

        if 'session_ids' in new and new['session_ids']:
            existing_sessions.extend(new['session_ids'])
        elif 'session_id' in new and new['session_id']:
            existing_sessions.append(new['session_id'])

        existing['session_ids'] = list(set(existing_sessions))
        if 'session_id' in new:
            existing['session_id'] = new['session_id']

        # ---- 累积时间证据 ----
        existing_times = existing.get('session_times', [])

        if 'session_time' in existing and existing['session_time']:
            if existing['session_time'] not in existing_times:
                existing_times.append(existing['session_time'])

        if 'session_times' in new and new['session_times']:
            existing_times.extend(new['session_times'])
        elif 'session_time' in new and new['session_time']:
            existing_times.append(new['session_time'])

        existing['session_times'] = list(set(existing_times))
        if 'session_time' in new and new['session_time']:
            existing['session_time'] = new['session_time']

        logger.debug(f"Merged relationship data: chunk_ids={existing.get('chunk_ids', [])}, session_ids={existing.get('session_ids', [])}, strength={existing.get('strength', 1)}")
