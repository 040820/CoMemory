"""开放领域状态语义归一化；原始事实不改写，另存用于比较的语义表示。

为什么需要这一层：
    原始抽取得到的属性名、作用域和值表达可能非常自由，例如
    “current employer / 现在在哪工作 / 公司是 X”。如果直接按字符串比较，
    同一状态会被拆成多个槽，关系判断也就无法比较。本模块让 LLM 只做
    “把当前事实映射到可比较的语义表示”，不决定覆盖或纠正。

输入输出边界：
    输入 = 当前候选 + 少量已验证历史事实（不是主体的全部记忆）；
    输出 = 原 proposal 的副本，额外带 semantic_state / semantic_normalization。
    原始 attribute/value/scope/modality 保留在 source_* 字段，任何失败都只写
    semantic_normalization.status=uncertain/not_applied，不修改原始事实。

与下游的关系：
    归一化只对齐维度、作用域、模态和值；REPLACE/CORRECT 仍必须经过
    EventStateWriter 的独立证据/时间/基数门控，本模块没有更新权限。
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

from .state_slot import canonical_attribute, normalize_modality, normalize_scope


def _json_object(response: str) -> dict[str, Any]:
    """解析模型输出；兼容 ```json 代码块，但拒绝非对象结果。"""
    clean = response.strip()
    if clean.startswith('```'):
        # 模型偶尔把 JSON 包在 markdown 代码块里；只取第一行之后的完整块。
        clean = clean.split('\n', 1)[1].rsplit('```', 1)[0].strip()
    result = json.loads(clean)
    if not isinstance(result, dict):
        raise ValueError('归一化结果必须是对象')
    return result


class StateSemanticNormalizer:
    """属性名不设领域白名单，复用已由证据支持的历史槽描述。"""

    def __init__(self, llm: Any):
        """注入归一化模型；本类不持有图，也不负责写入。"""
        # 只依赖一个可 await 的 LLM 管理器；不在此处初始化模型或图。
        self.llm = llm

    async def normalize(self, proposal: dict[str, Any], hits: list[dict[str, Any]]) -> dict[str, Any]:
        """把当前状态候选对齐到可比较的维度/作用域/值；失败时返回 uncertain。

        Args:
            proposal: 当前候选事实，至少含 fact_type/entity_id/attribute/value/
                      scope/modality/quote/observed_at/event_time/triple_refs。
            hits:     双路召回得到的历史事实，每条形如 {'fact': {...}}。

        Returns:
            proposal 的副本。成功时带 semantic_state.validated=True；
            不适用时 status=not_applied；异常时 status=uncertain 并保留原因。
        """
        # 活动事件由 event_state_writer 的 event 分支处理；本归一化只服务持续状态。
        if proposal.get('fact_type') == 'event':
            return proposal
        # 只复用已经通过验证的历史语义槽。旧协议数据没有 semantic_state.validated，
        # 必须在独立重建流程里迁移，不能在这里临时改写旧 Fact。
        old = {hit['fact']['fact_id']: hit['fact'] for hit in hits
               if (hit['fact'].get('semantic_state') or {}).get('validated') is True}
        # 只把比较所需的字段送给模型；引文和 Triple 截断是输入预算，
        # 不是对已存储原文的截断。
        views = [{key: fact.get(key) for key in (
            'fact_id', 'attribute', 'value', 'scope', 'modality', 'semantic_state',
            'quote', 'observed_at', 'event_time', 'triple_refs')}
            for fact in old.values()]
        for view in views:
            view['quote'] = str(view.get('quote') or '')[:1000]
            view['triple_refs'] = list(view.get('triple_refs') or [])[:6]
            # 历史槽只暴露可比较字段，避免把评价历史、调试字段带进提示词。
            view['semantic_state'] = {key: view['semantic_state'].get(key) for key in (
                'dimension', 'value', 'cardinality', 'scope', 'factual_mode')}
        # 输出 Schema 只允许有限的归一化结论：是否适用、维度/作用域/值、
        # 与哪个历史槽匹配、证据引文和理由。不包含任何“执行更新”的字段。
        properties = {
            'applicable': {'type': 'boolean'},
            'dimension': {'type': 'string'},
            'scope': {'type': 'object', 'additionalProperties': {'type': 'string'}},
            'value': {'type': ['string', 'null']},
            'cardinality': {'type': 'string', 'enum': ['single', 'set', 'unknown']},
            'factual_mode': {'type': 'string', 'enum': ['actual', 'planned', 'hypothetical', 'unknown']},
            'intent': {'type': 'string', 'enum': ['assertion', 'state_change', 'correction', 'unknown']},
            'matched_fact_id': {'type': ['string', 'null'], 'enum': [None, *old]},
            'same_dimension': {'type': 'boolean'},
            'same_target_scope': {'type': 'boolean'},
            'value_relation': {'type': 'string', 'enum': ['equivalent', 'opposite', 'different', 'unknown']},
            'evidence_quote': {'type': 'string'},
            'reason': {'type': 'string'},
        }
        schema = {'type': 'json_schema', 'json_schema': {
            'name': 'state_semantic_normalization', 'strict': True, 'schema': {
                'type': 'object', 'properties': properties,
                'required': list(properties), 'additionalProperties': False}}}
        # 提示词刻意把“归一化”和“是否覆盖”分开：模型只描述可比较表示，
        # 任何 REPLACE/CORRECT 仍由 Writer 的独立门控决定。
        prompt = (
            'STATE_SEMANTIC_NORMALIZATION\n'
            '将当前事实转成可比较的状态表示，领域与属性名开放。只做归一化，不决定覆盖。'
            'dimension 为稳定英文 snake_case；scope 是作用对象和限定范围的键值。'
            '同一主体不同对象、项目或关系角色不能合槽；会变化的属性值不能错误地放进 scope。'
            '每个 scope 值都必须逐字出现在当前候选引文中；无依据不得补全。'
            'value 是归一化语义值，可为同义表达的统一标签；原始 value/quote 由程序另行保留。'
            '单个对象的同一状态维度有明确互斥取值时才用 single；多值集合用 set，拿不准用 unknown。'
            '正反表达必须指向同一状态维度和同一对象，不能靠 not/no 等词直接生成相反值。'
            '事实层面的否定（某状态现在不成立）仍属于 actual，否定进入状态值；'
            '计划、假设、愿望、反问、不确定说法不能变成已确认的 actual 状态。'
            '普通历史活动不得变成持续状态。不是可确定的状态时 applicable=false。'
            '若与候选历史事实同一维度且同一作用范围，选择 matched_fact_id，'
            '复用其维度、scope 字段名和语义值命名体系，并明确 same_dimension/same_target_scope；'
            'value_relation 描述与所选旧值的语义关系：同义用 equivalent，正反用 opposite，'
            '不同取值用 different，不确定用 unknown；无历史匹配时填 unknown。'
            '仅名称相似、共享主体、共享话题不够。不能为复用旧槽而遗漏原文中的范围。'
            '例如同一设备的“已开启/已经打开”可统一值，“已开启/已关闭”是不同值；'
            '另一台设备不能合并。不同项目的同类角色也不合并。'
            '当前状态后来变化用 state_change；只有原文明确说旧记录错误才用 correction；'
            '仅作陈述用 assertion。不知道就 unknown。'
            'evidence_quote 必须是当前候选引文的非空连续子串，reason 简述依据。'
            '只输出符合 Schema 的 JSON。\n'
            f'历史候选：{json.dumps(views, ensure_ascii=False)}\n'
            f'当前候选：{json.dumps({key: proposal.get(key) for key in ("entity_id", "attribute", "value", "scope", "modality", "quote", "triple_refs", "observed_at", "event_time")}, ensure_ascii=False)}'
        )
        try:
            response = await self.llm.generate(prompt, max_tokens=768, response_format=schema)
            data = _json_object(response)
            evidence = data.get('evidence_quote')
            # applicable=false 是合法的“放弃归一化”，不是技术失败：
            # 保留原 proposal，只记录原因，让下游按未归一化状态处理。
            if data.get('applicable') is not True:
                return dict(proposal, semantic_normalization={
                    'status': 'not_applied', 'reason': str(data.get('reason') or '未确认可归一化状态')})
            # 引文必须来自当前候选本身，不能拿历史事实或模型复述当依据。
            if not isinstance(evidence, str) or not evidence.strip() or evidence not in proposal['quote']:
                raise ValueError('状态归一化缺少当前候选的连续引文')
            # 维度名统一成稳定的短英文标识；scope 的每个值都必须在当前引文中出现。
            dimension = canonical_attribute(data.get('dimension'))
            scope = normalize_scope(data.get('scope'), proposal['quote'])
            if len(scope) > 8 or any(len(value) > 256 for value in scope.values()):
                raise ValueError('状态作用域超过大小限制')
            # 基数决定同一维度是单值还是多值集合；unknown 不允许进入已验证槽。
            cardinality = data.get('cardinality')
            if cardinality not in {'single', 'set'}:
                raise ValueError('状态基数不明确')
            mode = normalize_modality(data.get('factual_mode'))
            # 否定必须进入值（例如 not_employed），不能把模态改成 negated 绕过值比较。
            if mode == 'negated':
                raise ValueError('状态否定必须表示在语义值中')
            # 计划/假设不能被模型“升级”为已经发生；这是防止时间语义漂移的关键门。
            if proposal['modality'] in {'planned', 'hypothetical'} and mode != proposal['modality']:
                raise ValueError('计划或假设不能被升级为已发生状态')
            value = data.get('value')
            if value is not None and (not isinstance(value, str) or not value.strip() or len(value) > 256):
                raise ValueError('状态语义值必须为非空文本')
            # 只否认旧事实、没有给出新值时，不能凭空生成一个确定的新值。
            if proposal['value'] is None and value is not None:
                raise ValueError('只否认旧事实而没有新值时，不能生成确定的新语义值')
            if value is None and not (proposal['kind'] == 'correction' and proposal['value'] is None):
                raise ValueError('状态候选缺少语义值')
            matched_id = data.get('matched_fact_id')
            value_relation = data.get('value_relation')
            if value_relation not in {'equivalent', 'opposite', 'different', 'unknown'}:
                raise ValueError('语义值关系无效')
            # 模型声称匹配某个历史槽时，程序必须再次核对维度、作用域、模态和基数；
            # 不能仅凭模型说 same_dimension=true 就复用旧槽。
            if matched_id is not None:
                previous = old.get(matched_id)
                if (previous is None or data.get('same_dimension') is not True or
                        data.get('same_target_scope') is not True):
                    raise ValueError('历史状态槽未获得同维度同作用域确认')
                previous_scope = previous.get('scope') or {}
                previous_values = sorted(str(v).casefold() for v in previous_scope.values())
                current_values = sorted(str(v).casefold() for v in scope.values())
                if previous_values != current_values or previous.get('modality') != mode:
                    raise ValueError('历史状态槽与当前作用对象、范围或模态不同')
                if previous['semantic_state']['cardinality'] != cardinality:
                    raise ValueError('历史状态槽与当前基数不一致')
                # 复用历史槽的维度名，保证后续槽位键稳定。
                dimension = previous['attribute']
                if value_relation == 'equivalent':
                    # 同义对齐不能把“未知新值”填成旧值；那会制造模型没有看到的证据。
                    if value is None and previous['semantic_state']['value'] is not None:
                        raise ValueError('未知新值不能被同义对齐填成旧值')
                    value = previous['semantic_state']['value']
                elif value_relation in {'opposite', 'different'} and value is not None:
                    if value.casefold().strip() == str(previous['semantic_state']['value']).casefold().strip():
                        raise ValueError('不同语义值不能仍使用相同标签')
                # key 名可同义，但所有范围值仍须在新证据中出现；不得任意省略范围。
                scope = dict(previous_scope)
            intent = data.get('intent')
            if intent not in {'assertion', 'state_change', 'correction', 'unknown'}:
                raise ValueError('状态陈述意图无效')
            # 原始字段原样保留在 source_*，归一化结果只覆盖用于比较的字段。
            projected = dict(proposal)
            projected.update(source_attribute=proposal.get('source_attribute', proposal['attribute']),
                             source_value=proposal['value'], source_scope=dict(proposal['scope']),
                             source_modality=proposal['modality'], attribute=dimension,
                             scope=scope, modality=mode)
            if intent == 'state_change':
                # 状态后来变化：仍按新 state 记录，是否替换由 Writer 门控决定。
                projected['kind'] = 'state'
                projected['target_fact_id'] = None
            elif intent == 'correction':
                # 只有原文明确说旧记录错误时才是 correction；仍不直接覆盖旧事实。
                projected['kind'] = 'correction'
            # semantic_state 是“可比较表示”，不是对现实真值的断言；
            # validated 只表示通过了本模块的结构/引文检查，不表示关系已成立。
            projected['semantic_state'] = {
                'validated': True, 'dimension': dimension, 'value': value,
                'cardinality': cardinality, 'scope': scope, 'factual_mode': mode,
                'matched_fact_id': matched_id, 'evidence_quote': evidence,
                'value_relation': value_relation,
                'intent': intent, 'reason': str(data.get('reason') or '')[:500]}
            projected['semantic_normalization'] = {'status': 'applied', **projected['semantic_state']}
            return projected
        except asyncio.CancelledError:
            # 取消必须向上传播，不能降级成 uncertain 继续写入。
            raise
        except Exception as error:
            # 任何解析/校验失败都保守返回原 proposal，只附加 uncertain 原因；
            # 不因归一化失败而拒收已通过原文/Triple 校验的新事实。
            return dict(proposal, semantic_normalization={
                'status': 'uncertain', 'reason': f'{type(error).__name__}: {error}'})
