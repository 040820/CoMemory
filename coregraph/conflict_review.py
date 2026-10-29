"""冲突审查模块：从上层候选链接锁定两端，组织证据，生成可解释决策建议。

写入阶段的 conflict/correction/replacement 只是待审查标签，不是真实冲突或
更新许可。本模块不修改原始事实；置信度未校准、采纳政策未确定时，仅保存
决策建议和连续原文引用，禁止自动令旧事实失效。
"""
from copy import deepcopy
from hashlib import sha256
import json

from .evidence_judge import EvidenceJudge
from .graph_evidence_provider import GraphEvidenceProvider


CONFLICT_CANDIDATES = frozenset(('conflict', 'correction', 'replacement'))
PAIR_RELATIONS = ('duplicate', 'complement', 'coexists', 'conflict', 'correction',
                  'replacement', 'unrelated', 'uncertain')


def explicit_fact_pair(graph, old_fact_id, new_fact_id):
    """固定事实对实验入口：可直接指定两端，不要求它们曾被标成冲突。

    只创建本次审查描述，不向上层补写虚构链接。这样可评估并存、重复以及漏连
    情况，避免测试集只包含曾被模型判成冲突的样本而引入选择偏差。
    """
    observations = graph['fact_storage']['observations']
    if old_fact_id == new_fact_id or any(fid not in observations for fid in (old_fact_id, new_fact_id)):
        raise ValueError('固定事实对必须是两个存在的不同 Observation ID')
    pair_id = 'pair-' + sha256(json.dumps([old_fact_id, new_fact_id]).encode()).hexdigest()[:20]
    return dict(link_id=pair_id, old_fact_id=old_fact_id, new_fact_id=new_fact_id,
        old_fact=dict(deepcopy(observations[old_fact_id]), fact_id=old_fact_id),
        new_fact=dict(deepcopy(observations[new_fact_id]), fact_id=new_fact_id),
        selection_mode='explicit_pair', proposed_relation=None, review_status='pending')


def vllm_compact_json_options(options):
    """模型传输适配模块：在显式启用时把同一 Schema 交给 vLLM 的紧凑解码接口。

    response_format 与独立 structured_outputs 参数不能在当前服务中混合设置。
    本函数不改变 Schema 内容、不修补模型文本，也不放松引用校验；仅适用于已
    确认支持该扩展的 vLLM 服务，其他 OpenAI 兼容后端默认不启用。
    """
    result = dict(options)
    response_format = result.pop('response_format')
    extra = dict(result.get('extra_body') or {})
    extra['structured_outputs'] = dict(json=response_format['json_schema']['schema'],
                                       disable_any_whitespace=True)
    result['extra_body'] = extra
    return result


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
        """从上层 links 建立“新 Fact -> 待审查旧 Fact”索引。

        只接受 relation 属于 conflict/correction/replacement、status=candidate
        且 review_status=pending 的链接；并集/补充链接不进入冲突审查队列。
        """
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

    def __init__(self, graph, search_callback, pair, initial_context_k=0):
        """固定双端事实并可选预取少量上层关联来源。

        initial_context_k 只为“先取证再判断”服务，不能替代 Judge 的原文核对；
        限制在 0–10，避免在审查开始时就把大量无关邻居塞进上下文。
        """
        super().__init__(graph, search_callback)
        self.pair = deepcopy(pair)
        if not isinstance(initial_context_k, int) or not 0 <= initial_context_k <= 10:
            raise ValueError('初始关联上下文数量必须在 0 到 10 之间')
        self.initial_context_k = initial_context_k

    async def initial(self, task, top_k):
        """返回新旧两端的直接来源；两端同 Chunk 时合并为一个来源但保留两个 Fact ID。"""
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
        sources = list(records.values())
        if self.initial_context_k:
            # 服务于证据先取后判：可选地预取一轮上层相邻来源，避免模型只看日期
            # 差异就提前终止。只按已有边定位，不把关联标签当作事实或参考答案。
            sources.extend(await self.expand(sources, 'linked', self.initial_context_k))
        return sources


class ConflictPairJudge(EvidenceJudge):
    """双端审查模块：判定具体一对事实的关系，防止退化为只检查新事实原文。"""

    def __init__(self, llm, pair, review_goal='relation', max_response_repairs=0, **kwargs):
        """固定一对事实，并限制输出协议修复次数。

        review_goal：
            relation         判断两端是什么关系；
            evolution_reason 进一步要求解释“旧记录从一开始就错”还是“现实后来改变”。
        max_response_repairs 只允许 0 或 1：格式/引用失败可以修一次，语义证据不足
        不应通过反复重试来“碰”一个通过结果。
        """
        super().__init__(llm, **kwargs)
        if review_goal not in ('relation', 'evolution_reason'):
            raise ValueError('审查目标必须为 relation 或 evolution_reason')
        self.pair = deepcopy(pair)
        self.review_goal = review_goal
        if max_response_repairs not in (0, 1):
            raise ValueError('每对事实只允许 0 或 1 次输出协议修复')
        self.max_response_repairs, self.response_repairs_used = max_response_repairs, 0

    def _extend_properties(self, properties):
        """独立入口的结构化协议：必须返回关系类型和本次固定的两端 ID。"""
        properties = deepcopy(properties)
        # 服务于小模型的稳定输出：限制冗长解释及无限引用数组，不放宽原文核对。
        properties['reason']['maxLength'] = 300
        properties['citations']['maxItems'] = 6
        properties['followup_queries']['maxItems'] = 2
        properties['missing_evidence']['maxItems'] = 3
        return dict(properties, relation={'type': 'string', 'enum': list(PAIR_RELATIONS)},
            old_fact_id={'type': 'string'}, new_fact_id={'type': 'string'})

    def _prompt(self, task, evidence):
        """将问题收敛到具体两端；上层 proposed_relation 不输入模型，减少标签暗示。"""
        task = deepcopy(task)
        if self.review_goal == 'evolution_reason':
            # 服务于问题目标约束：兼容并存与“为何变化”是不同问题。该目标对所有
            # 对照样本一致，不包含参考答案；不允许用并存绕过缺失的原因证据。
            task['query'] += ('\n本次目标是解释差异原因：明确区分旧记录当时就错了，还是当时正确而后来真实改变。'
                              '只有有依据的 correction/replacement 可标 SUFFICIENT。'
                              '并存、无关、重复不能回答这个原因问题；原因没有依据应 relation=uncertain，'
                              '状态 NEED_MORE 或 DEFER，不得凭日期排序推定发生了变化。')
        # 服务于结构化输出稳定性：服务端 JSON Schema 约束解码，并不意味着模型
        # 已看见完整字段约定。显式展示完整结构，避免漏掉 checks/citations 后停滞。
        template = dict(status=['SUFFICIENT', 'NEED_MORE', 'CONFLICT', 'DEFER'], reason='依据实际原文填写判断理由',
            missing_evidence=[], followup_queries=[],
            checks={key: ['pass', 'fail', 'unknown', 'not_applicable']
                    for key in ('identity', 'scope', 'modality', 'granularity', 'time')},
            citations=[self._citation_template()],
            relation=list(PAIR_RELATIONS), old_fact_id=self.pair['old_fact_id'], new_fact_id=self.pair['new_fact_id'])
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
              '按原文区分：明确指出原记录错误属于 correction；真实情况后来改变属于 replacement；'
              '先判断旧说法在原记录时是否正确：当时正确、后来实际改变，必须归 replacement；'
              '只有旧说法从一开始就记错、误读或不真实，才归 correction。'
              '把当前值更新为新值这个操作本身不叫 correction。'
              '已确认纠正后不应仍标成未解决 conflict。不能只比较结构化值而忽略原文中的纠正说明。'
              '同属性同值且无新增限定属于 duplicate，不因记录日期不同就标 complement。'
              'complement 限于同一对象/作用域增加兼容细节；不同对象或不同项目下分别成立归 coexists。'
              'time 检查的是事件/状态比较的时序是否清楚，不是要求新旧日期值相等。'
              '若任务要求解释差异原因，必须引用区分原因的具体陈述，不能仅凭两个值不同猜原因。'
              'reason 说明两端关系依据，不要只复述新事实。不得凭来源身份或新旧时间单独决定真伪。'
              '\n完整字段规范如下；status、relation、checks中的列表表示允许的选项，输出时各选一个字符串。'
              '不是默认答案，必须依证据填写判断与引用；已核对清楚的检查项填pass，未核对清楚才填unknown：'
              + json.dumps(template, ensure_ascii=False, separators=(',', ':')))
        return super()._prompt(task, evidence)

    async def _assess_once(self, task, evidence):
        """证据门控模块：端点一致、双方引用、比较条件通过，才生成关系决策建议。"""
        output = await super().assess(task, evidence)
        # 基础 Judge 通过后，再做双端一致性校验：端点 ID 正确、双方都被引用、
        # 关系类型与状态不矛盾。任何一项失败都要降级，不能保留“看似通过”的结果。
        if output['status'] in ('SUFFICIENT', 'CONFLICT'):
            ids = {c['fact_id'] for c in output.get('citations', [])}
            endpoints = {self.pair['old_fact_id'], self.pair['new_fact_id']}
            aligned = all(output.get(key) == self.pair[key] for key in ('old_fact_id', 'new_fact_id'))
            relation = output.get('relation')
            # CONFLICT 必须对应 relation=conflict；其他通过状态不能配 conflict/uncertain。
            valid_relation = relation in PAIR_RELATIONS and (
                relation == 'conflict' if output['status'] == 'CONFLICT'
                else relation not in ('conflict', 'uncertain'))
            if not aligned or not endpoints.issubset(ids) or not valid_relation:
                output['pair_validation_errors'] = (
                    ([] if aligned else ['wrong_endpoint_ids']) +
                    ([] if endpoints.issubset(ids) else ['missing_endpoint_citations']) +
                    ([] if valid_relation else ['inconsistent_relation_status']))
                output['status'] = 'NEED_MORE' if output.get('followup_queries') else 'DEFER'
                output['model_reason'] = output.get('reason')
                output['reason'] = '双端关系校验未通过：端点、双方原文引用或关系状态不一致'
                output.update(evaluation_status='technical_error', failure_kind='pair_binding_invalid')
                output['missing_evidence'] = list(output.get('missing_evidence', [])) + [
                    '需要固定新旧两端的合法原文引用，以及与状态一致的关系判定']
        # evolution_reason 目标下，duplicate/coexists/unrelated 都没有回答“为什么变化”，
        # 即使模型判 SUFFICIENT 也只能降级，避免用兼容关系绕过原因证据。
        if (self.review_goal == 'evolution_reason' and output['status'] == 'SUFFICIENT'
                and output.get('relation') not in ('correction', 'replacement')):
            output['model_reason'] = output.get('reason')
            output['status'] = 'NEED_MORE' if output.get('followup_queries') else 'DEFER'
            output['reason'] = '当前关系没有解释差异源于旧记录错误还是现实情况改变'
            output['missing_evidence'] = list(output.get('missing_evidence', [])) + ['缺少区分演化原因的原文依据']
        # 只有 SUFFICIENT 才生成非 DEFER 的动作建议；所有建议都不执行更新。
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
        output['decision']['review_goal'] = self.review_goal
        return output

    async def assess(self, task, evidence):
        """有界协议修复：只修复格式/引用错误；语义证据不足不当作技术故障重试。

        一对事实整个审查过程至多增加一次模型请求，不是每轮额外一次。修复仍需
        通过同样的端点和原文核验；保留首个失败响应，不能用重试次数当作置信度。
        """
        output = await self._assess_once(task, evidence)
        # 只有“格式/引用协议”失败才允许修复；语义上的不确定、证据不足、
        # 预算耗尽都不是技术故障，不能靠重试碰出通过结果。
        repairable = (output.get('stop_reason') == 'judge_failed' or any(
            error in ('wrong_endpoint_ids', 'missing_endpoint_citations')
            for error in output.get('pair_validation_errors', [])))
        if repairable and self.response_repairs_used < self.max_response_repairs:
            # 保留修复前的完整输出，审计时能区分“首次失败”和“修复后成功”。
            previous = {key: deepcopy(output.get(key)) for key in (
                'status', 'reason', 'metadata', 'raw_response', 'pair_validation_errors')}
            self.response_repairs_used += 1
            repaired_task = deepcopy(task)
            repaired_task['query'] += (
                '\nOUTPUT_PROTOCOL_REPAIR：上次输出未通过格式或引用校验。'
                + str(output.get('reason')) + '；必须给出固定两端的合法引用：'
                + self.pair['old_fact_id'] + ' 与 ' + self.pair['new_fact_id']
                + '。核对已展示的text后重新输出完整结构；不允许补造引文或降低证据要求。')
            output = await self._assess_once(repaired_task, evidence)
            output['response_repair_history'] = [previous]
        output['response_repairs_used'] = self.response_repairs_used
        return output
