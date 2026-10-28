"""三层图的只读证据适配器：服务于 Judge 调度，不修改构图与原检索排序。

检索器只提供候选 ID，实际文本和事实必须从本图规范存储回查，避免把模型编造
或不同图的片段当作证据。事件分组/候选 Link 只用于定位来源，不当作有效性证明。
"""

from __future__ import annotations

from copy import deepcopy
from collections import defaultdict


class GraphEvidenceProvider:
    """注入已有语义检索和初始事实召回回调，按 Chunk 去重并保留来源路线。"""

    def __init__(self, graph, search_callback, seed_callback=None):
        if graph.get('event_state_layer', {}).get('schema') != 'event_index_v1':
            raise ValueError('Judge 图适配器当前要求精简 event_index_v1 格式')
        self.graph, self.search_callback, self.seed_callback = graph, search_callback, seed_callback
        self.chunks = {str(k): v for k, v in graph['chunk_storage'].items()}
        self.rows, self.by_chunk, self.sessions = {}, defaultdict(list), defaultdict(list)
        table = graph['fact_storage']
        for collection in ('observations', 'time_assertions', 'attribute_assertions', 'facts'):
            for fid, row in table.get(collection, {}).items():
                if fid in self.rows:
                    raise ValueError('中层 ID 重复')
                self.rows[fid] = row
                cid = self.source_chunk(row)
                self.by_chunk[cid].append(fid)
        # 插入顺序作为兜底；有明确的局部 Chunk 序号时优先用它，绝不对哈希 ID 做加减。
        for position, (cid, row) in enumerate(self.chunks.items()):
            session = row.get('session_id')
            if session is None or str(session).strip().lower() in ('', 'unknown', 'none'):
                continue
            order = row.get('source_chunk_id', row.get('chunk_id', cid))
            try:
                order = int(order)
            except (ValueError, TypeError):
                order = position
            self.sessions[str(session)].append((order, position, cid))
        self.neighbors = {}
        for members in self.sessions.values():
            ids = [row[2] for row in sorted(members)]
            for index, cid in enumerate(ids):
                self.neighbors[cid] = ids[max(0, index - 1):index] + ids[index + 1:index + 2]

    @staticmethod
    def source_chunk(row):
        """统一原始 Triple、来源观察和时间/属性断言的来源字段。"""
        return str(row.get('chunk_id', row.get('evidence_chunk_id', '')))

    @staticmethod
    def fact_view(fid, row):
        """仅提供短结构视图；不把向量、已存置信度或大审计记录填入 Judge。"""
        keys = ('entity_id', 'attribute', 'value', 'src', 'relation', 'tgt', 'predicate',
                'event_id', 'event_time', 'event_time_raw', 'event_time_range',
                'observed_at', 'scope', 'modality')
        view = {'fact_id': fid}
        for key in keys:
            if key in row:
                value = deepcopy(row[key])
                view[key] = value[:160] if isinstance(value, str) else value
        return view

    def record(self, chunk_id, route, preferred_ids=()):
        """从规范 Chunk 回查证据；未知 ID 不接受，连续引文留作窗口定位锚点。"""
        cid = str(chunk_id)
        row = self.chunks.get(cid)
        if not row or not row.get('text'):
            return None
        ids = list(dict.fromkeys([fid for fid in preferred_ids if fid in self.by_chunk[cid]] + self.by_chunk[cid]))[:8]
        quote = next((self.rows[fid].get('quote', self.rows[fid].get('evidence_quote')) for fid in ids
                      if self.rows[fid].get('quote', self.rows[fid].get('evidence_quote'))), None)
        return dict(chunk_id=cid, text=str(row['text']), session_id=row.get('session_id'),
                    session_time=row.get('session_time'), fact_ids=ids,
                    facts=[self.fact_view(fid, self.rows[fid]) for fid in ids],
                    anchor_quote=quote, retrieval_route=route)

    async def initial(self, task, top_k):
        """新事实原文固定入场；历史候选复用现有状态/事件召回，缺回调时用语义检索。"""
        records = []
        target = task.get('target_fact') or {}
        fid = target.get('fact_id', target.get('assertion_id'))
        if fid:
            if fid not in self.rows:
                raise ValueError('新事实 ID 不在当前图中')
            source = self.record(self.source_chunk(self.rows[fid]), 'new_fact_source', [fid])
            if source is None:
                raise ValueError('新事实缺少原始 Chunk')
            records.append(source)
        if self.seed_callback is not None and fid:
            hits = await self.seed_callback(fid, top_k)
            for old_id in hits:
                if old_id in self.rows:
                    source = self.record(self.source_chunk(self.rows[old_id]), 'historical_fact', [old_id])
                    if source:
                        records.append(source)
        else:
            records.extend(await self.search(task['query'], top_k))
        return records

    async def search(self, query, top_k):
        """只信检索结果的本图 Chunk ID；不直接采用外部返回的文本内容或相似度作为证据。"""
        result = await self.search_callback(query, top_k)
        hits = result.get('evidence', []) if isinstance(result, dict) else result
        records = []
        for hit in hits[:top_k]:
            record = self.record(hit['chunk_id'], 'structural_semantic')
            if record:
                records.append(record)
        return records

    async def expand(self, evidence, stage, top_k):
        """补查同组/候选关系端点来源，或同 Session 前后文；不召回全部同实体历史。"""
        seen = {r['chunk_id'] for r in evidence}
        candidates = []
        if stage == 'linked':
            seeds = {fid for r in evidence for fid in r.get('fact_ids', [])}
            ids = []
            upper = self.graph['event_state_layer']
            for group in upper['groups'].values():
                if seeds.intersection(group['fact_ids']):
                    ids.extend(group['fact_ids'])
            for link in upper['links']:
                if seeds.intersection((link['from_fact_id'], link['to_fact_id'])):
                    ids.extend((link['from_fact_id'], link['to_fact_id']))
            candidates = [self.source_chunk(self.rows[fid]) for fid in ids if fid in self.rows]
        elif stage == 'neighbors':
            candidates = [cid for source in evidence for cid in self.neighbors.get(source['chunk_id'], [])]
        else:
            raise ValueError('未知补查阶段')
        result = []
        for cid in dict.fromkeys(candidates):
            if cid not in seen:
                record = self.record(cid, stage)
                if record:
                    result.append(record)
                    if len(result) == top_k:
                        break
        return result
