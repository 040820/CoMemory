"""冲突审查模块：从上层候选链接锁定两端，组织证据，生成可解释决策建议。

写入阶段的 conflict/correction/replacement 只是待审查标签，不是真实冲突或
更新许可。本模块不修改原始事实；置信度未校准、采纳政策未确定时，仅保存
决策建议和连续原文引用，禁止自动令旧事实失效。
"""
from copy import deepcopy
import json

from .evidence_judge import EvidenceJudge
from .graph_evidence_provider import GraphEvidenceProvider


CONFLICT_CANDIDATES = frozenset(('conflict', 'correction', 'replacement'))
PAIR_RELATIONS = ('duplicate', 'complement', 'coexists', 'conflict', 'correction',
                  'replacement', 'unrelated', 'uncertain')


def conflict_marker(relation):
    """服务于写入持久化：只给可能需要演化裁决的链接增加轻量待审查标记。

    普通并存/补充链接仍保持原结构；不因两端取值不同就制造 conflict。
    冲突、纠正和变化建议都需要进一步审查，所以共用同一个待审查队列。
    """
    if relation in CONFLICT_CANDIDATES:
        return {'review_kind': 'potential_conflict', 'review_status': 'pending'}
    return {}


class ConflictPairIndex:
    """上层定位模块：按新 Fact 找待审查链接，直接返回已绑定的旧/新两端。"""

    def __init__(self, graph):
        upper = graph.get('event_state_layer', {})
        if upper.get('schema') != 'event_index_v1':
            raise ValueError('冲突对索引需要 event_index_v1 图')
        self.observations = graph['fact_storage']['observations']
        self.by_new_fact = {}
        for link in upper.get('links', []):
            # 兼容没有 review_* 字段的旧版新图；候选标签本身已能表达待审查意图。
            if (link.get('relation') not in CONFLICT_CANDIDATES or
                    link.get('status') != 'candidate' or
                    link.get('review_status', 'pending') != 'pending'):
                continue
            old, new = link['from_fact_id'], link['to_fact_id']
            if old == new or old not in self.observations or new not in self.observations:
                raise ValueError(f"冲突候选端点无效：{link.get('link_id')}")
            pair = dict(link_id=link['link_id'], old_fact_id=old, new_fact_id=new,
                proposed_relation=link['relation'], review_status='pending',
                old_fact=dict(deepcopy(self.observations[old]), fact_id=old),
                new_fact=dict(deepcopy(self.observations[new]), fact_id=new))
            self.by_new_fact.setdefault(new, []).append(pair)

    def pending(self, new_fact_id=None):
        """返回独立副本；无链接时返回空，由写入召回负责补建，绝不虚构两端。"""
        return deepcopy(self.by_new_fact.get(new_fact_id, []) if new_fact_id else
                        [pair for pairs in self.by_new_fact.values() for pair in pairs])


class PairEvidenceProvider(GraphEvidenceProvider):
    """双端来源模块：初始证据直接取已知新旧 Fact，不再次泛搜同主体事实。

    后续的 linked/neighbor 扩展沿用三层图指针，只定位证据，不把图上的连通性
    当作事实支持或因果关系。两端来自同一 Chunk 时，合并引用 ID 后再去重。
    """

    def __init__(self, graph, search_callback, pair):
        super().__init__(graph, search_callback)
        self.pair = deepcopy(pair)

    async def initial(self, task, top_k):
        records = {}
        for fid in (self.pair['new_fact_id'], self.pair['old_fact_id']):
            if fid not in self.rows:
                raise ValueError('冲突对的事实不在当前图中')
            cid = self.source_chunk(self.rows[fid])
            previous = records.get(cid, {}).get('fact_ids', [])
            record = self.record(cid, 'conflict_pair_endpoint', [fid] + previous)
            if record is None:
                raise ValueError('冲突对缺少原始 Chunk')
            records[cid] = record
        return list(records.values())


class ConflictPairJudge(EvidenceJudge):
    """双端审查模块：判定具体一对事实的关系，防止退化为只检查新事实原文。"""

    def __init__(self, llm, pair, **kwargs):
        super().__init__(llm, **kwargs)
        self.pair = deepcopy(pair)

    def _extend_properties(self, properties):
        """独立入口的结构化协议：必须返回关系类型和本次固定的两端 ID。"""
        return dict(properties, relation={'type': 'string', 'enum': list(PAIR_RELATIONS)},
            old_fact_id={'type': 'string'}, new_fact_id={'type': 'string'})

    def _prompt(self, task, evidence):
        """将问题收敛到具体两端；上层 proposed_relation 不输入模型，减少标签暗示。"""
        task = deepcopy(task)
        task['query'] += (
            '\n本次只审查以下固定事实对：'
            + json.dumps({key: self.pair[key] for key in ('old_fact_id', 'new_fact_id')}, ensure_ascii=False)
            + '。必须核对双方原文，输出 relation、old_fact_id、new_fact_id。'
              'SUFFICIENT 必须同时引用两端 Fact 的原文，只引新事实不能说明新旧关系。'
              'replacement 表示现实状态变化；correction 表示明确纠正旧记录；'
              'coexists 表示不同对象、多值并存或不同时态可分别成立；'
              'conflict 表示确有不可同时成立的断言，但尚不能确定谁有效；'
              'complement 表示兼容补充；duplicate 表示重复；unrelated 表示无直接关系。'
              'CONFLICT 状态必须配 relation=conflict；SUFFICIENT 不得配 conflict/uncertain。'
              '若两端、作用域或时态尚未核对，应 NEED_MORE 或 DEFER。'
              'reason 说明两端关系依据，不要只复述新事实。不得凭来源身份或新旧时间单独决定真伪。')
        return super()._prompt(task, evidence)

    async def assess(self, task, evidence):
        """证据门控模块：端点一致、双方引用、比较条件通过，才生成关系决策建议。"""
        output = await super().assess(task, evidence)
        if output['status'] in ('SUFFICIENT', 'CONFLICT'):
            ids = {c['fact_id'] for c in output.get('citations', [])}
            endpoints = {self.pair['old_fact_id'], self.pair['new_fact_id']}
            aligned = all(output.get(key) == self.pair[key] for key in ('old_fact_id', 'new_fact_id'))
            relation = output.get('relation')
            valid_relation = relation in PAIR_RELATIONS and (
                relation == 'conflict' if output['status'] == 'CONFLICT'
                else relation not in ('conflict', 'uncertain'))
            if not aligned or not endpoints.issubset(ids) or not valid_relation:
                output['status'] = 'NEED_MORE' if output.get('followup_queries') else 'DEFER'
                output['model_reason'] = output.get('reason')
                output['reason'] = '双端关系校验未通过：端点、双方原文引用或关系状态不一致'
                output['missing_evidence'] = list(output.get('missing_evidence', [])) + [
                    '需要固定新旧两端的合法原文引用，以及与状态一致的关系判定']
        accepted = output['status'] == 'SUFFICIENT'
        action = {'duplicate': 'RETAIN', 'coexists': 'RETAIN', 'unrelated': 'RETAIN',
                  'complement': 'REFINE', 'replacement': 'REPLACE', 'correction': 'CORRECT'}
        output['decision'] = dict(link_id=self.pair['link_id'],
            old_fact_id=self.pair['old_fact_id'], new_fact_id=self.pair['new_fact_id'],
            relation=output.get('relation', 'uncertain'),
            proposed_action=action.get(output.get('relation'), 'DEFER') if accepted else 'DEFER',
            reason=output.get('reason'), evidence=deepcopy(output.get('citations', [])),
            update_applied=False, validation_status='not_verified',
            confidence=None, decision_status='proposal' if accepted else 'deferred')
        return output
