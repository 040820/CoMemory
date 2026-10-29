"""共用证据审查服务：按缺口补查，输出可追溯角色，供更新与问答消费。

服务分层：
    EvidenceRetriever  只读取证：直接回查、结构化索引、语义、BFS、RRF；
    _EvidenceRoles     Judge 协议扩展：把每条证据绑定到具体目标断言和连续引文；
    EvidenceSupport    有界调度：按缺口补查，限制轮数/来源/语义查询预算，
                       最后只提出关系/动作建议，不执行任何更新。

安全边界：
    - 本服务只提出建议，不调用旧图的覆盖或失效方法；
    - 继承现有 Judge 以复用身份、时间、原文和引用校验；
    - 角色语义仍是模型判断，引用校验只证明“引文存在且绑定正确”，
      不证明 quote 蕴含 claim，也不证明现实事实为真。
"""

from copy import deepcopy
import asyncio
import json

from ..conflict_review import ConflictPairJudge, explicit_fact_pair
from ..evidence_judge import EvidenceJudge, shown_quote_span
from .config import EvidenceSupportConfig
from .retriever import EvidenceRetriever


# 证据角色枚举；新增角色必须同步更新 _prompt、Schema 和下游消费方。
ROLES = ('supports', 'refutes', 'explains_change', 'background', 'uncertain')


class _EvidenceRoles:
    """Judge 协议扩展：每条证据角色绑定具体目标断言和合法原文引用。"""

    _compact_evidence = True

    def _generation_options(self):
        """服务于新证据 Judge：取消人为输出 Token 上限，避免完整 JSON 被截断。

        显式传 None 穿透底层旧默认值；LLMManager 会彻底省略请求 max_tokens。
        物理上下文和模型服务自身的生成配置仍然有效，length 检查继续保留。
        """
        return dict(max_tokens=None, temperature=0.0)

    def _extend_properties(self, properties):
        """在基础 Judge Schema 上增加 evidence_roles 数组。

        每条角色必须携带 fact_id/chunk_id/quote 三元组和目标断言，
        上限 6 条，避免模型用大量泛泛的“支持”淹没真正的关键证据。
        """
        properties = super()._extend_properties(properties)
        fields = {name: {'type': 'string'} for name in ('quote_id', 'target_fact_id', 'claim')}
        if getattr(self, '_available_quote_ids', None):
            fields['quote_id']['enum'] = self._available_quote_ids
        targets = getattr(self, 'role_targets', None)
        fields['target_fact_id']['enum'] = sorted(targets or (
            {q['fact_id'] for q in self._available_quote_options} | {'question'}))
        fields['role'] = {'type': 'string', 'enum': list(ROLES)}
        properties['evidence_roles'] = {'type': 'array', 'maxItems': 6, 'items': {
            'type': 'object', 'properties': fields, 'required': list(fields), 'additionalProperties': False}}
        # 限制理由和引用数量，防止输出被低信息量内容占满。
        properties['reason']['maxLength'] = 300
        # 服务于证据协议：模型只输出已展示片段编号，不再自由生成原文或 Chunk ID。
        properties['citations'] = {'type': 'array', 'maxItems': 6, 'items': {'type': 'object',
            'properties': {'quote_id': deepcopy(fields['quote_id'])}, 'required': ['quote_id'], 'additionalProperties': False}}
        if getattr(self, 'pair', None):
            # 两端分别约束枚举，比自由列表 minItems=2 更严格：不能两次选新端点。
            choices = {}
            for side in ('old', 'new'):
                fid = self.pair[side + '_fact_id']
                ids = [q['quote_id'] for q in self._available_quote_options if q['fact_id'] == fid]
                choices[side + '_quote_id'] = dict(type='string', enum=ids)
                properties[side + '_fact_id'] = dict(type='string', enum=[fid])
            properties['endpoint_citations'] = dict(type='object', properties=choices,
                required=list(choices), additionalProperties=False)
        return properties

    def _citation_template(self):
        """供双端 Judge 的提示词模板使用，与实际 Schema 保持一致。"""
        return dict(quote_id='从已展示 quote_options 中选择实际编号')

    def _citation_instruction(self):
        """说明片段编号协议；来源、原文和位置全部由程序绑定。"""
        return ('citations 每项只含 quote_id，必须选择已展示 quote_options 的编号。'
                '严禁编造编号或重新抄写引文。fact_id/chunk_id/quote/span 由程序回填。')

    def _resolve_citations(self, output, shown):
        """服务于证据支撑模块：白名单解析 ID，旧响应只允许走原有严格校验。

        解析不做语义扩写；同一个编号唯一对应一个事实、Chunk 和连续原文位置。
        旧输出兼容仅用于历史响应/固定测试，不接受模糊匹配或省略号。
        """
        inventory = {q['quote_id']: q for row in shown for q in row.get('quote_options', [])}
        original_citations = deepcopy(output.get('citations', []))
        for field in ('citations', 'evidence_roles'):
            values = output.get(field, [])
            if not isinstance(values, list):
                raise ValueError('引用及角色必须为数组')
            for value in values:
                if not isinstance(value, dict):
                    raise ValueError('引用及角色项必须为对象')
                if 'quote_id' in value:
                    selected = inventory.get(value['quote_id'])
                    if selected is None:
                        raise ValueError('证据片段编号不在实际展示白名单中')
                    if any(k in value and value[k] != selected[k] for k in ('fact_id', 'chunk_id', 'quote')):
                        raise ValueError('片段编号与输出来源不一致')
                    value.update(deepcopy(selected))
        if getattr(self, 'pair', None):
            endpoints = output.get('endpoint_citations')
            if endpoints is None:
                # 历史自由引文响应没有新字段，仍交由原来的双端/原文校验，不能放宽。
                legacy = bool(original_citations) and all(isinstance(c, dict) and
                    'quote_id' not in c and all(k in c for k in ('chunk_id', 'fact_id', 'quote'))
                    for c in original_citations)
                if legacy:
                    output['endpoint_citations'] = None
            elif isinstance(endpoints, dict):
                for side in ('old', 'new'):
                    selected = inventory.get(endpoints.get(side + '_quote_id'))
                    if selected is None or selected['fact_id'] != self.pair[side + '_fact_id']:
                        raise ValueError('双端片段没有绑定对应的旧/新 Fact')
                    output['citations'].append(deepcopy(selected))
            else:
                raise ValueError('双端片段选择格式无效')
        # 角色中的片段编号本身也是模型显式引用，不要求在另一数组重复抄写。
        # 这里只合并已经白名单校验的编号；旧自由引文角色不能借此绕过引用校验。
        for role in output.get('evidence_roles', []):
            if 'quote_id' in role:
                output['citations'].append(deepcopy(inventory[role['quote_id']]))
        unique = {}
        for citation in output.get('citations', []):
            unique.setdefault((citation.get('fact_id'), citation.get('chunk_id'), citation.get('quote')), citation)
        output['citations'] = list(unique.values())
        output['model_citations'] = original_citations
        return output

    def _prompt(self, task, evidence):
        """在基础 Judge 提示词后追加“证据角色”协议。"""
        task = deepcopy(task)
        if getattr(self, 'pair', None):
            keys = ('fact_id', 'entity_id', 'attribute', 'value', 'scope', 'modality', 'event_time', 'observed_at')
            task['comparison'] = {side: {key: self.pair[side + '_fact'].get(key) for key in keys}
                                  for side in ('old', 'new')}
            task['query'] += ('\n先比较 comparison.old 和 comparison.new；不要只验证新事实。'
                              'endpoint_citations 必须分别选择 old_quote_id 与 new_quote_id，'
                              '各自来自对应 Fact 的 quote_options。citations 可补充额外片段。')
        task['query'] += (
            '\nEVIDENCE_SUPPORT：evidence_roles 将每条已引用证据绑定到具体待核查断言。'
            '每条 evidence_roles 使用 quote_id 绑定 citations 中同一个片段编号；'
            'claim 是你正在核查的具体命题，不是检索问题的复述。'
            'target_fact_id 指该证据作用的目标 Fact；问答使用 question；'
            '审查关系本身时可使用 relation，但须在 claim 明确写出新旧两端及关系。'
            'role 取 supports/refutes/explains_change/background/uncertain。'
            '检索路径不证明支持，候选关联不证明因果；跨 Chunk 重复转述不能当独立来源。'
            '主动核对反证，不能只找支持新说法的材料；未发现反证不表示不存在反证。'
            '充分判断至少有一条非背景、非不确定的角色；证据不足说明具体缺口。'
            '这是模型的局部语义审查，不能输出已验证真值概率。')
        prompt, shown, context = super()._prompt(task, evidence)
        # 当前展示白名单同时进入 JSON Schema 枚举，约束解码阶段就阻止编造编号。
        self._available_quote_options = [q for row in shown for q in row.get('quote_options', [])]
        self._available_quote_ids = list(dict.fromkeys(q['quote_id'] for q in self._available_quote_options))
        return prompt, shown, context

    async def assess(self, task, evidence):
        """先走基础 Judge，再机械校验 evidence_roles 的绑定关系。

        基础 Judge 已经验证了 citations 的 chunk/quote/fact 绑定；这里额外要求
        每个角色都指向一条已引用证据、一个合法目标断言和一条具体命题。
        角色语义仍是模型判断：引用校验只证明“这条原文存在且绑定正确”，
        不证明 quote 蕴含 claim。
        """
        output = await super().assess(task, evidence)
        try:
            # citations 是“角色可以绑定的唯一来源集合”，不能引用未展示材料。
            citations = {(c['fact_id'], c['chunk_id'], c['quote']) for c in output.get('citations', [])}
            # shown 来自本次实际展示的证据；问答模式额外允许目标为 question。
            shown = {fid for r in output.get('shown_evidence', []) for fid in r['fact_ids']}
            targets = getattr(self, 'role_targets', None) or shown | {'question'}
            roles = output.get('evidence_roles', [])
            if not isinstance(roles, list) or len(roles) > 6:
                raise ValueError('证据角色必须为最多六项的数组')
            for role in roles:
                # 任一字段不合法都拒绝整组角色；不保留“部分有效”的角色制造假证据。
                if (not isinstance(role, dict) or role.get('role') not in ROLES or
                        role.get('target_fact_id') not in targets or
                        not isinstance(role.get('claim'), str) or not role['claim'].strip() or
                        (role.get('fact_id'), role.get('chunk_id'), role.get('quote')) not in citations):
                    raise ValueError('证据角色没有绑定合法引用、目标断言或具体命题')
            # SUFFICIENT/CONFLICT 至少要有支持、反驳或变化解释中的一种；
            # 只有 background/uncertain 不能支撑“证据充分”。
            if output['status'] in ('SUFFICIENT', 'CONFLICT'):
                if not any(role['role'] in ('supports', 'refutes', 'explains_change') for role in roles):
                    raise ValueError('充分性判断缺少明确的支持、反驳或变化解释角色')
            output['role_verification'] = ('citation_validated_semantics_model_assessed'
                                           if roles else 'not_assessed')
        except (KeyError, TypeError, ValueError) as error:
            # 角色校验失败时整次判断降为 DEFER，并保留被拒绝的角色供审计；
            # 不能把无效角色悄悄丢弃后继续声称 SUFFICIENT。
            output.update(status='DEFER', stop_reason='evidence_role_invalid',
                          role_validation_error=str(error), role_verification='failed',
                          evaluation_status='technical_error', failure_kind='evidence_role_invalid')
            output['rejected_evidence_roles'] = output.pop('evidence_roles', [])
            output['evidence_roles'] = []
            if output.get('decision'):
                output['decision'].update(proposed_action='DEFER', decision_status='deferred')
        return output


class QuestionEvidenceJudge(_EvidenceRoles, EvidenceJudge):
    """问答审查器：只核对现有材料是否足以支持回答，不生成最终答案。"""

    def _prompt(self, task, evidence):
        """服务于问答协议：Schema 约束解码不会自动把字段含义提供给模型。"""
        task = deepcopy(task)
        # 候选集合及 RRF 结果不变；只调整有限输入预算的展示顺序。
        # 显式断言来源仍固定优先，再按原问题语义名次排，图扩展材料随后展示。
        evidence = sorted(evidence, key=lambda row: (not bool(row.get('mandatory_fact_ids')),
            row.get('task_semantic_rank') if row.get('task_semantic_rank') is not None else float('inf')))
        task['query'] += (
            '\n问答字段说明：reason 要写具体证据理由，不能填状态词或字段名。'
            'citations=[{quote_id:实际片段编号}]；evidence_roles 每项包含 quote_id、'
            'target_fact_id、claim、role。评价回答断言时 target_fact_id=question，'
            'claim 写具体答案命题。SUFFICIENT 必须有合法引用及支持角色；'
            'checks 已核对的项填 pass，确有缺口才 unknown。'
            'NEED_MORE 必须说明具体缺失内容，并可给具体 followup_queries。'
            'missing_evidence/followup_queries 是自然语言内容数组，不要填字段名占位符。'
            '日期推导可引用原始相对时间及该来源 session_time，不能把其他活动的时间套用。')
        return super()._prompt(task, evidence)

    async def assess(self, task, evidence):
        """技术协议失败最多修复一次；语义不足不靠重复请求碰出通过结果。"""
        output = await super().assess(task, evidence)
        used = getattr(self, 'response_repairs_used', 0)
        if used < 1 and output.get('evaluation_status') == 'technical_error' and output.get('failure_kind') not in (
                'transport_error', 'transport_or_input_error', 'truncated_output', 'input_budget_or_source_binding'):
            previous = deepcopy(output)
            repaired = deepcopy(task)
            repaired['query'] += ('\nOUTPUT_PROTOCOL_REPAIR：上次输出没有满足引用/角色协议：'
                                  + str(output.get('reason')) + '。只依据同一批证据重新输出，'
                                  '无法判断就明确 NEED_MORE/DEFER，不得编造引文或默认通过。')
            output = await super().assess(repaired, evidence)
            output['response_repair_history'] = [previous]
            self.response_repairs_used = used + 1
        output['response_repairs_used'] = getattr(self, 'response_repairs_used', 0)
        return output


class RelationEvidenceJudge(_EvidenceRoles, ConflictPairJudge):
    """关系审查器：保留旧/新双端引用要求，另外记录证据对哪个断言起作用。"""


class EvidenceSupport:
    """共用接口：review_pair、review_question、answer_question；输入图只读。"""

    def __init__(self, graph, embeddings=None, llm=None, config=None):
        """注入只读图、可选 Embedding/LLM 和实验参数。

        graph 不会被修改；embeddings 只用于语义召回，llm 只用于 Judge。
        两者都可以为 None：没有 LLM 时只返回取证报告，状态 NOT_EVALUATED。
        """
        self.graph, self.llm = graph, llm
        self.config = config or EvidenceSupportConfig()
        self.retriever = EvidenceRetriever(graph, embeddings, self.config)

    async def review_pair(self, old_fact_id, new_fact_id, query='核查新旧事实的实际关系及其证据。', *, as_of=None):
        """更新侧入口：固定两个已保存 Observation，提出关系建议，绝不执行更新。

        两个 Fact 必须已经存在；不要求它们曾被标成 conflict，也不预置关系。
        新事实作为第一锚点，旧事实作为第二锚点，保证双方来源都固定保留。
        """
        pair = explicit_fact_pair(self.graph, old_fact_id, new_fact_id)
        # 没有 LLM 时仍走同一取证流程，只是不产生 Judge 轮次。
        judge = None if self.llm is None else RelationEvidenceJudge(
            self.llm, pair, max_prompt_bytes=self.config.max_prompt_bytes, max_response_repairs=1)
        if judge is not None:
            # 关系审查的角色只能指向新旧两端，防止模型把角色绑到无关背景事实。
            judge.role_targets = {old_fact_id, new_fact_id, 'relation'}
        return await self._run(query, [new_fact_id, old_fact_id], 'memory_relation', judge, as_of, pair)

    async def review_question(self, question, *, claim_ids=(), as_of=None):
        """问答侧入口：候选断言可不提供，不能把标准答案输入取证过程。"""
        from .query import question_seed_fact
        anchors = list(claim_ids)
        trace = dict(method='explicit_claim_ids', selected_ids=anchors)
        if not anchors:
            anchors, trace = await question_seed_fact(question, self, as_of)
        judge = None if self.llm is None else QuestionEvidenceJudge(
            self.llm, max_prompt_bytes=self.config.max_prompt_bytes)
        report = await self._run(question, anchors, 'question', judge, as_of)
        report['query_seed_trace'] = trace
        return report

    async def _run(self, query, anchors, mode, judge, as_of, pair=None):
        """有界控制器：每轮仅补查明确缺口，去重查询、限制来源和模型次数。

        预算：
            max_rounds             Judge 最多轮数；
            max_chunks             累计不同来源上限；
            max_semantic_queries   整次审查的语义查询上限（失败也扣减）；
            max_followups_per_round 每轮最多接纳的新缺口查询数。
        停止条件：
            非 NEED_MORE、来源预算满、没有新缺口、没有新增来源、轮数耗尽。
        """
        config = self.config
        # pool 按 Chunk 去重；retrievals/rounds 保留每轮完整审计。
        pool, retrievals, rounds = {}, [], []
        used, followups, semantic_left = set(), [], config.max_semantic_queries
        target = pair['new_fact'] if pair else (self.retriever.rows.get(anchors[0]) if anchors else None)
        task = dict(query=query, mode=mode, target_fact=target)

        def finish(status, reason):
            """统一收尾：非 SUFFICIENT 时把关系建议降为 DEFER，绝不自动采纳。"""
            last = rounds[-1] if rounds else {}
            decision = deepcopy(last.get('decision'))
            if decision and status != 'SUFFICIENT':
                # 模型可能给出了 replacement/correction，但证据不足时只能暂缓；
                # 这里改的是本次建议副本，不修改图或历史记录。
                decision.update(proposed_action='DEFER', decision_status='deferred')
            from .confidence import confidence_shadow
            assessment = last.get('evaluation_status', 'not_evaluated' if not rounds else 'semantic_uncertain')
            shadow = confidence_shadow(last, retrievals, pair, mode)
            return dict(schema='evidence_support_v1', task=dict(mode=mode, query=query,
                anchor_fact_ids=anchors, as_of=as_of), config=config.to_dict(), status=status,
                stop_reason=reason, evidence=list(pool.values()), rounds=rounds,
                retrieval_rounds=retrievals, evidence_roles=last.get('evidence_roles', []),
                missing_evidence=last.get('missing_evidence', []), checks=last.get('checks', {}),
                decision_proposal=decision, update_applied=False, confidence=shadow,
                # 调度轮次与模型调用尝试数分开；协议修复计入调用，纯输入失败不计。
                judge_calls=sum(int(r.get('model_call_attempted', False)) +
                    sum(int(h.get('model_call_attempted', h.get('raw_response') is not None))
                        for h in r.get('response_repair_history', [])) for r in rounds),
                evaluation_status=assessment, failure_kind=last.get('failure_kind'),
                validation_status='not_verified', independence_verified=False,
                semantic_queries_used=config.max_semantic_queries - semantic_left)

        for index in range(config.max_rounds):
            # exclude_chunk_ids=pool：补查优先找“尚未展示”的来源，但已见节点仍可作 BFS 中转。
            result = await self.retriever.retrieve(query, anchors, as_of=as_of, followups=followups,
                exclude_chunk_ids=pool, semantic_query_budget=semantic_left)
            semantic_left -= result['diagnostics']['semantic_queries']
            retrievals.append(result)
            gained = 0
            for record in result['evidence']:
                # 只有新增 Chunk 才算“有进展”；超总预算的来源记录在 diagnostics。
                if record['chunk_id'] not in pool and len(pool) < config.max_chunks:
                    pool[record['chunk_id']] = record
                    gained += 1
                elif record['chunk_id'] not in pool:
                    result['diagnostics'].setdefault('dropped_by_total_chunk_budget', []).append(record['chunk_id'])
            # 第一轮允许没有新增（初始证据来自 pool 为空）；后续轮次没有新增即停止。
            if index and not gained:
                return finish('DEFER', 'no_new_evidence')
            if judge is None:
                return finish('NOT_EVALUATED', 'retrieval_only')
            if not pool:
                return finish('DEFER', 'no_initial_evidence')
            output = await judge.assess(task, list(pool.values()))
            rounds.append(dict(round=index + 1, **output))
            if output['status'] != 'NEED_MORE':
                return finish(output['status'], output.get('stop_reason', 'judge_' + output['status'].lower()))
            if len(pool) >= config.max_chunks:
                return finish('DEFER', 'evidence_budget_exhausted')
            # 只接纳本轮新出现的缺口查询，并限制每轮数量；重复问题直接过滤。
            followups = [gap for gap in output.get('followup_queries', [])
                         if gap not in used][:config.max_followups_per_round]
            used.update(followups)
            if not followups:
                return finish('DEFER', 'no_new_gap_query')
        return finish('DEFER', 'judge_budget_exhausted')

    async def answer_question(self, question, answer_llm, *, claim_ids=(), as_of=None):
        """下游回答适配：仅使用本次 Judge 实际展示的证据，并核对答案引用。

        机械校验只证明引用存在，不证明生成答案的每句话都被蕴含；该语义质量
        仍需独立评估。非充分、未评估或引用失败时不输出冒充确定的答案。
        """
        # 只有本次审查 SUFFICIENT 才调用回答模型；其余状态保留审查报告并返回空答案。
        report = await self.review_question(question, claim_ids=claim_ids, as_of=as_of)
        report.update(answer=None, answer_status='deferred')
        if report['status'] != 'SUFFICIENT':
            return report
        # 回答只能看到最后一轮实际展示的证据；不能偷偷使用未展示的历史。
        shown = report['rounds'][-1].get('shown_evidence', [])
        inventory = {q['quote_id']: q for row in shown for q in row.get('quote_options', [])}
        if not inventory:
            report.update(answer_status='rejected', answer_validation_error='审查没有可供回答引用的片段')
            return report
        # 回答与 Judge 共享同一片段编号协议，避免最后一步又自由抄写原文而串来源。
        fields = {'quote_id': {'type': 'string', 'enum': list(inventory)}}
        schema = {'type': 'json_schema', 'json_schema': {'name': 'grounded_memory_answer', 'strict': True,
            'schema': {'type': 'object', 'properties': {'answer': {'type': 'string', 'minLength': 1},
                'citations': {'type': 'array', 'minItems': 1, 'maxItems': 6, 'items': {'type': 'object',
                    'properties': fields, 'required': list(fields), 'additionalProperties': False}}},
                'required': ['answer', 'citations'], 'additionalProperties': False}}}
        # 回答提示词明确禁止把 candidate 边、较新说法或检索分数当事实真值；
        # 同时要求引用必须来自本次展示的证据。
        prompt = ('只依据提供的证据回答问题，不能把 candidate 边或较新说法当作真值。'
                  '只返回 JSON：answer 为回答文本，citations=[{quote_id:实际片段编号}]。'
                  '至少引用一条已展示 quote_options 的片段编号；原文和来源由程序回填，不能抄写或编造。'
                  '不能补造事实、来源可靠性或变化原因。\n' + json.dumps(dict(question=question,
                  evidence=shown, evidence_roles=report['evidence_roles']), ensure_ascii=False))
        metadata = {}
        try:
            # 回答调用与审查调用分开；失败时保留审查报告，不把失败伪装成空答案。
            raw = await answer_llm.generate(prompt, max_tokens=self.config.max_answer_tokens,
                temperature=0.0, response_format=schema, metadata_sink=metadata)
        except asyncio.CancelledError:
            raise
        except Exception as error:
            # 回答失败不丢弃审查报告；保留 prompt/metadata 供复现。
            report.update(answer_status='failed', answer_trace=dict(prompt=prompt,
                metadata=metadata, error=f'{type(error).__name__}:{error}'))
            return report
        report['answer_trace'] = dict(prompt=prompt, raw_response=raw, metadata=metadata)
        try:
            # 截断、空答案、引用不在展示来源中，都拒绝输出答案；不能把不完整回答当成功。
            if metadata.get('finish_reason') == 'length':
                raise ValueError('回答输出被截断')
            answer = json.loads(raw)
            # 用本次实际展示的 Chunk 建立白名单；回答只能引用这些来源。
            sources = {r['chunk_id']: r for r in shown}
            if (not isinstance(answer, dict) or not isinstance(answer.get('answer'), str)
                    or not answer['answer'].strip() or not isinstance(answer.get('citations'), list)
                    or not 1 <= len(answer['citations']) <= 6):
                raise ValueError('回答或原文引用为空')
            for citation in answer['citations']:
                if isinstance(citation, dict) and 'quote_id' in citation:
                    selected = inventory.get(citation['quote_id'])
                    if selected is None or any(k in citation and citation[k] != selected[k]
                                              for k in ('fact_id', 'chunk_id', 'quote')):
                        raise ValueError('回答片段编号未知或来源不一致')
                    citation.update(deepcopy(selected))
                if not isinstance(citation, dict) or any(not isinstance(citation.get(k), str)
                        for k in ('chunk_id', 'fact_id', 'quote')):
                    raise ValueError('回答引用字段必须是字符串')
                row = sources.get(citation['chunk_id'])
                # 引用必须同时满足：来源在本次展示中、Fact 属于该来源、quote 是连续原文。
                if (row is None or citation['fact_id'] not in row['fact_ids'] or
                        shown_quote_span(row, citation['quote']) is None):
                    raise ValueError('回答引用不在本次审查展示的来源中')
            # source_grounded 只表示引用可机械核对；answer_semantics 仍需独立评估。
            report.update(answer=answer['answer'], answer_citations=answer['citations'],
                          answer_status='source_grounded', answer_semantics='not_independently_verified')
        except (KeyError, TypeError, ValueError) as error:
            # 引用失败时保留审查报告，只把 answer 标为 rejected；不让非法答案外泄。
            report.update(answer_status='rejected', answer_validation_error=str(error))
        return report
