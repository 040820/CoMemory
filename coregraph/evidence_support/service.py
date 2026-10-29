"""共用证据审查服务：按缺口补查，输出可追溯角色，供更新与问答消费。

本服务只提出关系/动作建议，不调用旧图的覆盖或失效方法。继承现有 Judge
以复用身份、时间和原文校验，新增角色绑定校验；角色语义仍是模型判断。
"""

from copy import deepcopy
import asyncio
import json

from ..conflict_review import ConflictPairJudge, explicit_fact_pair
from ..evidence_judge import EvidenceJudge
from .config import EvidenceSupportConfig
from .retriever import EvidenceRetriever


ROLES = ('supports', 'refutes', 'explains_change', 'background', 'uncertain')


class _EvidenceRoles:
    """Judge 协议扩展：每条证据角色绑定具体目标断言和合法原文引用。"""

    def _generation_options(self):
        """服务于新证据 Judge：取消人为输出 Token 上限，避免完整 JSON 被截断。

        显式传 None 穿透底层旧默认值；LLMManager 会彻底省略请求 max_tokens。
        物理上下文和模型服务自身的生成配置仍然有效，length 检查继续保留。
        """
        return dict(max_tokens=None, temperature=0.0)

    def _extend_properties(self, properties):
        properties = super()._extend_properties(properties)
        fields = {name: {'type': 'string'} for name in
                  ('fact_id', 'chunk_id', 'quote', 'target_fact_id', 'claim')}
        fields['role'] = {'type': 'string', 'enum': list(ROLES)}
        properties['evidence_roles'] = {'type': 'array', 'maxItems': 6, 'items': {
            'type': 'object', 'properties': fields, 'required': list(fields), 'additionalProperties': False}}
        properties['reason']['maxLength'] = 300
        properties['citations']['maxItems'] = 6
        return properties

    def _prompt(self, task, evidence):
        task = deepcopy(task)
        task['query'] += (
            '\nEVIDENCE_SUPPORT：evidence_roles 将每条已引用证据绑定到具体待核查断言。'
            'fact_id/chunk_id/quote 必须与 citations 的一条完全一致；claim 是你正在核查的具体命题，'
            '不是检索问题的复述。target_fact_id 指该证据作用的目标 Fact；问答可使用 question。'
            'role 取 supports/refutes/explains_change/background/uncertain。'
            '检索路径不证明支持，候选关联不证明因果；跨 Chunk 重复转述不能当独立来源。'
            '主动核对反证，不能只找支持新说法的材料；未发现反证不表示不存在反证。'
            '充分判断至少有一条非背景、非不确定的角色；证据不足说明具体缺口。'
            '这是模型的局部语义审查，不能输出已验证真值概率。')
        return super()._prompt(task, evidence)

    async def assess(self, task, evidence):
        output = await super().assess(task, evidence)
        try:
            citations = {(c['fact_id'], c['chunk_id'], c['quote']) for c in output.get('citations', [])}
            shown = {fid for r in output.get('shown_evidence', []) for fid in r['fact_ids']}
            targets = getattr(self, 'role_targets', None) or shown | {'question'}
            roles = output.get('evidence_roles', [])
            if not isinstance(roles, list) or len(roles) > 6:
                raise ValueError('证据角色必须为最多六项的数组')
            for role in roles:
                if (not isinstance(role, dict) or role.get('role') not in ROLES or
                        role.get('target_fact_id') not in targets or
                        not isinstance(role.get('claim'), str) or not role['claim'].strip() or
                        (role.get('fact_id'), role.get('chunk_id'), role.get('quote')) not in citations):
                    raise ValueError('证据角色没有绑定合法引用、目标断言或具体命题')
            if output['status'] in ('SUFFICIENT', 'CONFLICT'):
                if not any(role['role'] in ('supports', 'refutes', 'explains_change') for role in roles):
                    raise ValueError('充分性判断缺少明确的支持、反驳或变化解释角色')
            output['role_verification'] = ('citation_validated_semantics_model_assessed'
                                           if roles else 'not_assessed')
        except (KeyError, TypeError, ValueError) as error:
            output.update(status='DEFER', stop_reason='evidence_role_invalid',
                          role_validation_error=str(error), role_verification='failed')
            output['rejected_evidence_roles'] = output.pop('evidence_roles', [])
            output['evidence_roles'] = []
            if output.get('decision'):
                output['decision'].update(proposed_action='DEFER', decision_status='deferred')
        return output


class QuestionEvidenceJudge(_EvidenceRoles, EvidenceJudge):
    """问答审查器：只核对现有材料是否足以支持回答，不生成最终答案。"""


class RelationEvidenceJudge(_EvidenceRoles, ConflictPairJudge):
    """关系审查器：保留旧/新双端引用要求，另外记录证据对哪个断言起作用。"""


class EvidenceSupport:
    """共用接口：review_pair、review_question、answer_question；输入图只读。"""

    def __init__(self, graph, embeddings=None, llm=None, config=None):
        self.graph, self.llm = graph, llm
        self.config = config or EvidenceSupportConfig()
        self.retriever = EvidenceRetriever(graph, embeddings, self.config)

    async def review_pair(self, old_fact_id, new_fact_id, query='核查新旧事实的实际关系及其证据。', *, as_of=None):
        """更新侧入口：固定两个已保存 Observation，提出关系建议，绝不执行更新。"""
        pair = explicit_fact_pair(self.graph, old_fact_id, new_fact_id)
        judge = None if self.llm is None else RelationEvidenceJudge(
            self.llm, pair, max_prompt_bytes=self.config.max_prompt_bytes)
        if judge is not None:
            judge.role_targets = {old_fact_id, new_fact_id}
        return await self._run(query, [new_fact_id, old_fact_id], 'memory_relation', judge, as_of, pair)

    async def review_question(self, question, *, claim_ids=(), as_of=None):
        """问答侧入口：候选断言可不提供，不能把标准答案输入取证过程。"""
        judge = None if self.llm is None else QuestionEvidenceJudge(
            self.llm, max_prompt_bytes=self.config.max_prompt_bytes)
        return await self._run(question, list(claim_ids), 'question', judge, as_of)

    async def _run(self, query, anchors, mode, judge, as_of, pair=None):
        """有界控制器：每轮仅补查明确缺口，去重查询、限制来源和模型次数。"""
        config = self.config
        pool, retrievals, rounds = {}, [], []
        used, followups, semantic_left = set(), [], config.max_semantic_queries
        target = pair['new_fact'] if pair else None
        task = dict(query=query, mode=mode, target_fact=target)

        def finish(status, reason):
            last = rounds[-1] if rounds else {}
            decision = deepcopy(last.get('decision'))
            if decision and status != 'SUFFICIENT':
                decision.update(proposed_action='DEFER', decision_status='deferred')
            return dict(schema='evidence_support_v1', task=dict(mode=mode, query=query,
                anchor_fact_ids=anchors, as_of=as_of), config=config.to_dict(), status=status,
                stop_reason=reason, evidence=list(pool.values()), rounds=rounds,
                retrieval_rounds=retrievals, evidence_roles=last.get('evidence_roles', []),
                missing_evidence=last.get('missing_evidence', []), checks=last.get('checks', {}),
                decision_proposal=decision, update_applied=False, confidence=None,
                validation_status='not_verified', independence_verified=False,
                semantic_queries_used=config.max_semantic_queries - semantic_left)

        for index in range(config.max_rounds):
            result = await self.retriever.retrieve(query, anchors, as_of=as_of, followups=followups,
                exclude_chunk_ids=pool, semantic_query_budget=semantic_left)
            semantic_left -= result['diagnostics']['semantic_queries']
            retrievals.append(result)
            gained = 0
            for record in result['evidence']:
                if record['chunk_id'] not in pool and len(pool) < config.max_chunks:
                    pool[record['chunk_id']] = record
                    gained += 1
                elif record['chunk_id'] not in pool:
                    result['diagnostics'].setdefault('dropped_by_total_chunk_budget', []).append(record['chunk_id'])
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
        report = await self.review_question(question, claim_ids=claim_ids, as_of=as_of)
        report.update(answer=None, answer_status='deferred')
        if report['status'] != 'SUFFICIENT':
            return report
        shown = report['rounds'][-1].get('shown_evidence', [])
        fields = {k: {'type': 'string'} for k in ('fact_id', 'chunk_id', 'quote')}
        schema = {'type': 'json_schema', 'json_schema': {'name': 'grounded_memory_answer', 'strict': True,
            'schema': {'type': 'object', 'properties': {'answer': {'type': 'string'},
                'citations': {'type': 'array', 'maxItems': 6, 'items': {'type': 'object',
                    'properties': fields, 'required': list(fields), 'additionalProperties': False}}},
                'required': ['answer', 'citations'], 'additionalProperties': False}}}
        prompt = ('只依据提供的证据回答问题，不能把 candidate 边或较新说法当作真值。'
                  '返回 answer 与 citations；每条引用包含 fact_id、chunk_id 和原文连续 quote。'
                  '不能补造事实、来源可靠性或变化原因。\n' + json.dumps(dict(question=question,
                  evidence=shown, evidence_roles=report['evidence_roles']), ensure_ascii=False))
        metadata = {}
        try:
            raw = await answer_llm.generate(prompt, max_tokens=self.config.max_answer_tokens,
                temperature=0.0, response_format=schema, metadata_sink=metadata)
        except asyncio.CancelledError:
            raise
        except Exception as error:
            report.update(answer_status='failed', answer_trace=dict(prompt=prompt,
                metadata=metadata, error=f'{type(error).__name__}:{error}'))
            return report
        report['answer_trace'] = dict(prompt=prompt, raw_response=raw, metadata=metadata)
        try:
            if metadata.get('finish_reason') == 'length':
                raise ValueError('回答输出被截断')
            answer = json.loads(raw)
            sources = {r['chunk_id']: r for r in shown}
            if (not isinstance(answer, dict) or not isinstance(answer.get('answer'), str)
                    or not answer['answer'].strip() or not isinstance(answer.get('citations'), list)
                    or not 1 <= len(answer['citations']) <= 6):
                raise ValueError('回答或原文引用为空')
            for citation in answer['citations']:
                if not isinstance(citation, dict) or any(not isinstance(citation.get(k), str)
                        for k in ('chunk_id', 'fact_id', 'quote')):
                    raise ValueError('回答引用字段必须是字符串')
                row = sources.get(citation['chunk_id'])
                if (row is None or citation['fact_id'] not in row['fact_ids'] or
                        not citation['quote'].strip() or citation['quote'] not in row['text']):
                    raise ValueError('回答引用不在本次审查展示的来源中')
            report.update(answer=answer['answer'], answer_citations=answer['citations'],
                          answer_status='source_grounded', answer_semantics='not_independently_verified')
        except (KeyError, TypeError, ValueError) as error:
            report.update(answer_status='rejected', answer_validation_error=str(error))
        return report
