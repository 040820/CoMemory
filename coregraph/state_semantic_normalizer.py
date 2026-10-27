"""开放领域状态语义归一化；原始事实不改写，另存用于比较的语义表示。

新候选只携带当前原文和少量历史事实，不把主体的全部记忆送入 LLM。
该阶段只对齐维度、作用域、模态和值；REPLACE/CORRECT 仍必须经过独立门控。
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

from .state_slot import canonical_attribute, normalize_modality, normalize_scope


def _json_object(response: str) -> dict[str, Any]:
    clean = response.strip()
    if clean.startswith('```'):
        clean = clean.split('\n', 1)[1].rsplit('```', 1)[0].strip()
    result = json.loads(clean)
    if not isinstance(result, dict):
        raise ValueError('归一化结果必须是对象')
    return result


class StateSemanticNormalizer:
    """属性名不设领域白名单，复用已由证据支持的历史槽描述。"""

    def __init__(self, llm: Any):
        self.llm = llm

    async def normalize(self, proposal: dict[str, Any], hits: list[dict[str, Any]]) -> dict[str, Any]:
        if proposal.get('fact_type') == 'event':
            return proposal
        # 仅复用已有归一化槽；旧协议数据需独立重建，不能临时篡改旧 Fact。
        old = {hit['fact']['fact_id']: hit['fact'] for hit in hits
               if (hit['fact'].get('semantic_state') or {}).get('validated') is True}
        views = [{key: fact.get(key) for key in (
            'fact_id', 'attribute', 'value', 'scope', 'modality', 'semantic_state',
            'quote', 'observed_at', 'event_time', 'triple_refs')}
            for fact in old.values()]
        for view in views:
            view['quote'] = str(view.get('quote') or '')[:1000]
            view['triple_refs'] = list(view.get('triple_refs') or [])[:6]
            view['semantic_state'] = {key: view['semantic_state'].get(key) for key in (
                'dimension', 'value', 'cardinality', 'scope', 'factual_mode')}
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
            if data.get('applicable') is not True:
                return dict(proposal, semantic_normalization={
                    'status': 'not_applied', 'reason': str(data.get('reason') or '未确认可归一化状态')})
            if not isinstance(evidence, str) or not evidence.strip() or evidence not in proposal['quote']:
                raise ValueError('状态归一化缺少当前候选的连续引文')
            dimension = canonical_attribute(data.get('dimension'))
            scope = normalize_scope(data.get('scope'), proposal['quote'])
            if len(scope) > 8 or any(len(value) > 256 for value in scope.values()):
                raise ValueError('状态作用域超过大小限制')
            cardinality = data.get('cardinality')
            if cardinality not in {'single', 'set'}:
                raise ValueError('状态基数不明确')
            mode = normalize_modality(data.get('factual_mode'))
            if mode == 'negated':
                raise ValueError('状态否定必须表示在语义值中')
            if proposal['modality'] in {'planned', 'hypothetical'} and mode != proposal['modality']:
                raise ValueError('计划或假设不能被升级为已发生状态')
            value = data.get('value')
            if value is not None and (not isinstance(value, str) or not value.strip() or len(value) > 256):
                raise ValueError('状态语义值必须为非空文本')
            if proposal['value'] is None and value is not None:
                raise ValueError('只否认旧事实而没有新值时，不能生成确定的新语义值')
            if value is None and not (proposal['kind'] == 'correction' and proposal['value'] is None):
                raise ValueError('状态候选缺少语义值')
            matched_id = data.get('matched_fact_id')
            value_relation = data.get('value_relation')
            if value_relation not in {'equivalent', 'opposite', 'different', 'unknown'}:
                raise ValueError('语义值关系无效')
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
                dimension = previous['attribute']
                if value_relation == 'equivalent':
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
            projected = dict(proposal)
            projected.update(source_attribute=proposal.get('source_attribute', proposal['attribute']),
                             source_value=proposal['value'], source_scope=dict(proposal['scope']),
                             source_modality=proposal['modality'], attribute=dimension,
                             scope=scope, modality=mode)
            if intent == 'state_change':
                projected['kind'] = 'state'
                projected['target_fact_id'] = None
            elif intent == 'correction':
                projected['kind'] = 'correction'
            projected['semantic_state'] = {
                'validated': True, 'dimension': dimension, 'value': value,
                'cardinality': cardinality, 'scope': scope, 'factual_mode': mode,
                'matched_fact_id': matched_id, 'evidence_quote': evidence,
                'value_relation': value_relation,
                'intent': intent, 'reason': str(data.get('reason') or '')[:500]}
            projected['semantic_normalization'] = {'status': 'applied', **projected['semantic_state']}
            return projected
        except asyncio.CancelledError:
            raise
        except Exception as error:
            return dict(proposal, semantic_normalization={
                'status': 'uncertain', 'reason': f'{type(error).__name__}: {error}'})
