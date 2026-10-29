"""证据支撑检索模块：精确索引、稠密余弦、有限 BFS 与 RRF 排名融合。

这里只寻找候选材料，不调用 LLM，不判支持/反驳，不修改输入图。输出中的
paths 是检索路径，rrf_score/cosine 是检索分数，均不能充当事实可信度。
"""

from collections import defaultdict, deque
from copy import deepcopy
from datetime import datetime, time, timezone
import json
import math

from ..graph_evidence_provider import GraphEvidenceProvider
from ..state_slot import canonical_attribute
from .config import EvidenceSupportConfig


def _date(value, *, cutoff=False):
    """时间可见性工具：只解析 ISO 时间；日期截止值包含当天，未知不猜测。"""
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        text = value.strip()
        parsed = datetime.fromisoformat(text.replace('Z', '+00:00'))
        if cutoff and len(text) == 10:
            parsed = datetime.combine(parsed.date(), time.max)
        return parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None else parsed.astimezone(timezone.utc)
    except ValueError:
        return None


def _slot(row):
    """结构化召回键：不含取值，才能召回不同值；作用域/模态保持分离。"""
    subject = row.get('entity_id') or row.get('src') or row.get('subject')
    attribute = row.get('attribute') or row.get('relation') or row.get('predicate')
    # 时间/属性断言可能只有 event_id，不能把不同事件归入“空主体”的共同状态槽。
    if not subject or not attribute:
        return None
    subject, attribute = str(subject).casefold(), str(attribute)
    try:
        attribute = canonical_attribute(attribute)
    except ValueError:
        attribute = attribute.casefold().strip()
    return (subject, attribute, json.dumps(row.get('scope') or {}, sort_keys=True, ensure_ascii=False),
            row.get('modality', 'actual'))


def _unit(values):
    """严格向量校验：空、零、NaN/Inf 不允许冒充有效余弦结果。"""
    vector = tuple(float(x) for x in values)
    norm = math.sqrt(sum(x * x for x in vector))
    if not vector or not math.isfinite(norm) or norm == 0:
        raise ValueError('Embedding 返回空、零或非有限向量')
    return tuple(x / norm for x in vector)


class EvidenceRetriever:
    """只读 Fact 取证器，Embedding 通过现有 get_embeddings 异步接口注入。"""

    def __init__(self, graph, embeddings=None, config=None):
        self.config = config or EvidenceSupportConfig()
        self.embeddings = embeddings
        self.provider = GraphEvidenceProvider(graph, None)
        self.rows, self.chunks = self.provider.rows, self.provider.chunks
        self.slots, self.events = defaultdict(list), defaultdict(list)
        self.adjacency = defaultdict(set)
        self._vectors = {}
        for fid, row in self.rows.items():
            if _slot(row) is not None:
                self.slots[_slot(row)].append(fid)
            if row.get('event_id'):
                self.events[row['event_id']].append(fid)
            # 事实引用原始 Triple 时也可定位其来源；引用不是语义蕴含边。
            for name in ('middle_fact_ids', 'event_middle_fact_ids', 'time_middle_fact_ids',
                         'source_middle_fact_ids'):
                for other in row.get(name, []) or []:
                    self._edge(fid, other, 'middle_reference')
        upper = graph['event_state_layer']
        self.groups_by_fact, self.groups = defaultdict(list), {}
        # 分组不预先展开成完全图，避免大组产生 O(n²) 条邻接记录。
        for gid, group in upper.get('groups', {}).items():
            members = list(dict.fromkeys(fid for fid in group.get('fact_ids', []) if fid in self.rows))
            self.groups[gid] = members
            for fid in members:
                self.groups_by_fact[fid].append(gid)
        for link in upper.get('links', []):
            self._edge(link.get('from_fact_id'), link.get('to_fact_id'), 'candidate_link')

    def _edge(self, left, right, kind):
        if left in self.rows and right in self.rows and left != right:
            self.adjacency[left].add((right, kind))
            self.adjacency[right].add((left, kind))

    def clear_cache(self):
        """更换模型后显式清缓存；缓存仅存在当前实例，不写入原图。"""
        self._vectors.clear()

    def _visible(self, fid, cutoff):
        """所有路径共享过滤：Fact 和来源 Chunk 都必须在截止时间内可见。"""
        row = self.rows[fid]
        chunk = self.chunks.get(self.provider.source_chunk(row))
        if not chunk or not chunk.get('text'):
            return False
        if cutoff is None:
            return True
        source_time = _date(chunk.get('session_time'))
        if source_time is None or source_time > cutoff:
            return False
        observed = row.get('observed_at') or row.get('session_time')
        return not observed or (_date(observed) is not None and _date(observed) <= cutoff)

    def _text(self, fid):
        row = self.rows[fid]
        keys = ('entity_id', 'attribute', 'value', 'src', 'relation', 'tgt', 'subject',
                'predicate', 'scope', 'modality', 'event_time', 'quote', 'evidence_quote')
        return json.dumps({k: row[k] for k in keys if k in row}, ensure_ascii=False,
                          sort_keys=True, default=str)[:self.config.max_embedding_chars]

    async def _encode(self, texts):
        """按文本缓存单位向量；编码不完整则明确失败，不填充零向量。"""
        missing = list(dict.fromkeys(t for t in texts if t not in self._vectors))
        for start in range(0, len(missing), self.config.embedding_batch_size):
            batch = missing[start:start + self.config.embedding_batch_size]
            values = await self.embeddings.get_embeddings(batch)
            if len(values) != len(batch):
                raise ValueError('Embedding 数量与输入不一致')
            validated = [_unit(v) for v in values]
            self._vectors.update(zip(batch, validated))
        return [self._vectors[t] for t in texts]

    async def retrieve(self, query, anchor_fact_ids=(), *, as_of=None, followups=(),
                       exclude_chunk_ids=(), semantic_query_budget=None):
        """返回候选 Fact、规范原文和审计；双端直接来源不参加排名淘汰。

        每个锚点独立建立结构/图/语义路线，避免要求材料同时接近两端而漏证。
        后续缺口查询保留原问题上下文；跨主体语义候选交给 Judge 核对身份。
        """
        if not isinstance(query, str) or not query.strip():
            raise ValueError('取证请求必须有非空 query')
        anchors = list(dict.fromkeys(anchor_fact_ids))
        if len(anchors) > 2:
            raise ValueError('首版最多接收两个明确 Fact 锚点')
        cutoff = _date(as_of, cutoff=True) if as_of is not None else None
        if as_of is not None and cutoff is None:
            raise ValueError('as_of 必须是 ISO 日期或时间')
        eligible = {fid for fid in self.rows if self._visible(fid, cutoff)}
        excluded = set(map(str, exclude_chunk_ids))
        # 补查要先排除已见来源再取 Top-k；已见节点仍留在 eligible 中供 BFS 中转。
        fresh = {fid for fid in eligible if self.provider.source_chunk(self.rows[fid]) not in excluded}
        if any(fid not in eligible for fid in anchors):
            raise ValueError('锚点不存在、缺少原文或不在本次时间范围内')
        config = self.config
        routes, paths, cosines = {}, defaultdict(list), defaultdict(dict)
        diagnostics = dict(eligible_facts=len(eligible), excluded_facts=len(self.rows) - len(eligible),
                           semantic_status='unavailable:no_embedding_manager', semantic_queries=0,
                           graph_nodes_visited=0, graph_truncated=False)
        queries = [('semantic:task', query)] + [('semantic:anchor:' + fid,
                    query + '\n锚点：' + self._text(fid)) for fid in anchors]
        if followups:
            queries = [('semantic:gap:' + str(i), query + '\n需要核查：' + gap)
                       for i, gap in enumerate(followups)]
        budget = config.max_semantic_queries if semantic_query_budget is None else max(0, semantic_query_budget)
        queries = queries[:budget]
        if self.embeddings is not None and queries and fresh:
            try:
                ids = sorted(fresh)
                vectors = await self._encode([self._text(fid) for fid in ids])
                query_vectors = await self._encode([text[:config.max_embedding_chars] for _, text in queries])
                temporary = {}
                for (route, _), vector in zip(queries, query_vectors):
                    scored = []
                    for fid, candidate in zip(ids, vectors):
                        if len(vector) != len(candidate):
                            raise ValueError('Embedding 维度不一致')
                        score = sum(a * b for a, b in zip(vector, candidate))
                        if score >= config.min_cosine and fid not in anchors:
                            scored.append((fid, score))
                    scored.sort(key=lambda pair: (-pair[1], pair[0]))
                    temporary[route] = scored[:config.semantic_k]
                for route, scored in temporary.items():
                    routes[route] = [fid for fid, _ in scored]
                    for fid, score in scored:
                        cosines[fid][route] = score
                diagnostics.update(semantic_status='ok', semantic_queries=len(queries))
            except Exception as error:
                diagnostics.update(semantic_status=f'unavailable:{type(error).__name__}:{error}',
                                   semantic_queries=len(queries))
        elif self.embeddings is not None:
            diagnostics['semantic_status'] = 'query_budget_exhausted' if not queries else 'empty_index'

        # 问答没有固定两端时，先语义召回，再从最佳 Fact 建立结构/图路线。
        # 这些种子只是搜索入口，不能被当作已确认答案或已确认同一事件。
        search_seeds = anchors or list(dict.fromkeys(fid for route, ids in routes.items()
            if route.startswith('semantic:') for fid in ids))[:config.max_question_seeds]
        diagnostics['graph_seed_fact_ids'] = search_seeds
        for fid in search_seeds:
            row = self.rows[fid]
            # 同事件优先，再查精确槽位；时间/值只用于描述，不据此裁决变化。
            ids = self.events.get(row.get('event_id'), []) + self.slots.get(_slot(row), [])
            routes['structured:' + fid] = sorted((set(ids) & fresh) - set(anchors))[:config.structured_k]

        # 多源、有限深度 BFS，保留每条路线的锚点与边类型；不经过不可见节点。
        queue = deque((seed, seed, [seed], []) for seed in search_seeds) if config.max_hops else deque()
        seen = {(seed, seed) for seed in search_seeds}
        while queue and diagnostics['graph_nodes_visited'] < config.max_graph_nodes:
            seed, current, path, kinds = queue.popleft()
            diagnostics['graph_nodes_visited'] += 1
            if len(path) - 1 >= config.max_hops:
                continue
            neighbors = set(self.adjacency[current])
            for gid in self.groups_by_fact.get(current, []):
                neighbors.update((other, 'event_group') for other in self.groups[gid] if other != current)
            # 同一目标可能同时有分组/引用/候选链接；先合并边类型，再扣邻居名额。
            by_neighbor = defaultdict(set)
            for other, kind in neighbors:
                if other in eligible:
                    by_neighbor[other].add(kind)
            neighbors = [(other, '+'.join(sorted(kinds))) for other, kinds in sorted(by_neighbor.items())]
            diagnostics['graph_truncated'] |= len(neighbors) > config.max_neighbors
            for other, kind in neighbors[:config.max_neighbors]:
                if (seed, other) in seen:
                    continue
                if len(seen) >= config.max_graph_nodes:
                    diagnostics['graph_truncated'] = True
                    continue
                seen.add((seed, other))
                new_path, new_kinds = path + [other], kinds + [kind]
                queue.append((seed, other, new_path, new_kinds))
                if other not in anchors and other in fresh:
                    routes.setdefault('graph:' + seed, []).append(other)
                    paths[other].append(dict(fact_ids=new_path, edge_types=new_kinds,
                                             interpretation='retrieval_path_not_proof'))
        diagnostics['graph_truncated'] |= bool(queue)


        # RRF 只合并排名，不把余弦值与规则分数相加，更不解释成支持概率。
        ranks = defaultdict(dict)
        for route, ids in routes.items():
            for rank, fid in enumerate(dict.fromkeys(ids), 1):
                ranks[fid][route] = rank
        scores = {fid: sum(1 / (config.rrf_k + rank) for rank in values.values())
                  for fid, values in ranks.items()}
        fused = sorted(scores, key=lambda fid: (-scores[fid], fid))
        reserved = list(dict.fromkeys(fid for ids in routes.values() for fid in ids[:config.reserve_per_route]))
        ranked = list(dict.fromkeys(anchors + reserved + fused))[:config.max_candidate_facts]
        candidates = [dict(fact_id=fid, chunk_id=self.provider.source_chunk(self.rows[fid]),
                           rrf_score=scores.get(fid), route_ranks=ranks.get(fid, {}),
                           cosine_scores=cosines.get(fid, {}), paths=paths.get(fid, []),
                           mandatory_source=fid in anchors, support_status='not_assessed') for fid in ranked]
        grouped = {}
        for item in candidates:
            cid = item['chunk_id']
            if cid not in excluded:
                grouped.setdefault(cid, []).append(item['fact_id'])
        records = []
        for cid, ids in list(grouped.items())[:config.max_chunks]:
            # 禁止 GraphEvidenceProvider.record 自动带入本 Chunk 的未来或未选中事实。
            ids = ids[:config.max_facts_per_chunk]
            source = self.chunks[cid]
            quote = next((self.rows[fid].get('quote') or self.rows[fid].get('evidence_quote') for fid in ids
                          if self.rows[fid].get('quote') or self.rows[fid].get('evidence_quote')), None)
            records.append(dict(chunk_id=cid, text=str(source['text']), session_id=source.get('session_id'),
                session_time=source.get('session_time'), fact_ids=ids,
                facts=[self.provider.fact_view(fid, self.rows[fid]) for fid in ids], anchor_quote=quote,
                retrieval_route='evidence_support', independence_group=str(source.get('source_id') or 'chunk:' + cid),
                independence_verified=False))
        diagnostics['dropped_chunk_ids'] = list(grouped)[config.max_chunks:]
        diagnostics['truncated_fact_ids_by_chunk'] = {
            cid: ids[config.max_facts_per_chunk:] for cid, ids in grouped.items()
            if len(ids) > config.max_facts_per_chunk}
        return dict(schema='evidence_retrieval_v1', query=query, anchor_fact_ids=anchors, as_of=as_of,
                    config=config.to_dict(), candidates=candidates, evidence=deepcopy(records),
                    routes=routes, semantic_query_texts={route: text[:config.max_embedding_chars]
                        for route, text in queries}, diagnostics=diagnostics)
