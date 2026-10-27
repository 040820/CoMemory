"""Event–State 写入模块：逐对判断新旧事实关系，不直接执行状态更新。

每次输入一对有原文支持的事实，输出关系建议及证据。分类器没有存储权限，
纠正和替换必须经过 Writer 的独立证据/时间门控，检索得分不是证据置信度。
"""

from __future__ import annotations

import asyncio
import json
from typing import Any


RELATIONS = ('duplicate', 'complement', 'coexists', 'conflict',
             'correction', 'replacement', 'unrelated', 'uncertain')


class StateFactRelationJudge:
    """服务于双路召回后的关系判断；不读取整个主体历史，也不修改图。"""

    def __init__(self, llm: Any):
        self.llm = llm

    async def judge(self, old: dict[str, Any], new: dict[str, Any]) -> dict[str, Any]:
        """返回关系 JSON；模型异常、解析失败或引文不匹配一律降为 uncertain。

        引文和 Triple 仅在提示词中限制长度，原始事实及证据不会被截断存储。
        """
        def view(fact):
            result = {key: fact.get(key) for key in (
                'fact_id', 'entity_id', 'attribute', 'value', 'source_attribute',
                'source_value', 'scope', 'modality', 'semantic_state',
                'observed_at', 'event_time', 'status')}
            result['quote'] = str(fact.get('quote') or '')[:1800]
            result['triple_refs'] = (fact.get('triple_refs') or [])[:8]
            return result

        properties = {
            'relation': {'type': 'string', 'enum': list(RELATIONS)},
            'same_slot': {'type': 'boolean'},
            'evidence_quote': {'type': 'string'}, 'reason': {'type': 'string'},
        }
        schema = {'type': 'json_schema', 'json_schema': {
            'name': 'state_fact_relation', 'strict': True, 'schema': {
                'type': 'object', 'properties': properties,
                'required': list(properties), 'additionalProperties': False}}}
        prompt = (
            'STATE_FACT_RELATION\n判断新旧事实实际关系，仅输出指定 JSON。'
            'duplicate=同一事实重复；complement=兼容补充；coexists=独立事实并存；'
            'conflict=同槽互斥但无法确定谁有效；correction=明确指出旧记录错误；'
            'replacement=现实状态后来变化；unrelated=无关；uncertain=证据不足。'
            '同主体、同谓词、不同值、较晚时间均不足以证明替换。'
            '多值集合、不同活动、不同项目角色不能自动替换。'
            'same_slot 须核对主体、维度、作用对象和模态，不得照抄语义标签。'
            'evidence_quote 必须是新事实 quote 中非空连续原文；无法给出则 uncertain。'
            '本次只给关系建议，不执行更新。\n'
            f'旧事实：{json.dumps(view(old), ensure_ascii=False)}\n'
            f'新事实：{json.dumps(view(new), ensure_ascii=False)}'
        )
        try:
            response = await self.llm.generate(prompt, max_tokens=512, response_format=schema)
            clean = response.strip()
            if clean.startswith('```'):
                clean = clean.split('\n', 1)[1].rsplit('```', 1)[0].strip()
            result = json.loads(clean)
            if not isinstance(result, dict) or result.get('relation') not in RELATIONS:
                raise ValueError('关系枚举无效')
            quote = result.get('evidence_quote')
            if not isinstance(result.get('same_slot'), bool):
                raise ValueError('same_slot 不是布尔值')
            if not isinstance(quote, str) or not quote.strip() or quote not in new['quote']:
                raise ValueError('关系判断缺少当前事实原文依据')
            return result
        except asyncio.CancelledError:
            raise
        except Exception as error:
            return {'relation': 'uncertain', 'same_slot': False, 'evidence_quote': '',
                    'reason': f'关系判断失败：{type(error).__name__}: {error}'}
