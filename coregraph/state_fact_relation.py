"""Event–State 写入模块：逐对判断新旧事实关系，不直接执行状态更新。

每次输入一对有原文支持的事实，输出关系建议及证据。分类器没有存储权限，
当前 append_only Writer 只保存候选关联，不执行纠正或替换；检索得分不是证据置信度。

引文校验只容忍格式差异：成对包裹引号、引文前多余的冒号以及换行/连续空格
会被还原为新事实 quote 中的连续原文；改写、跨事实引用或空引文仍然降为
uncertain。失败路径保留模型原始响应与原始字段，便于区分格式失败和真正不确定。
"""

from __future__ import annotations

import asyncio
import json
import re
from typing import Any

### 关系代存
RELATIONS = ('duplicate', 'complement', 'coexists', 'conflict',
             'correction', 'replacement', 'unrelated', 'uncertain')

_QUOTE_PAIRS = (("'", "'"), ('"', '"'), ('`', '`'), ('“', '”'), ('‘', '’'),
                ('「', '」'), ('『', '』'))


def _strip_wrapping_quotes(text: str) -> str:
    """服务于引文格式容错：只去掉成对包裹引号，不改写正文内容。"""
    value = text.strip()
    changed = True
    while changed and len(value) >= 2:
        changed = False
        for left, right in _QUOTE_PAIRS:
            if value[0] == left and value[-1] == right:
                value = value[1:-1].strip()
                changed = True
                break
    return value


def _find_quote_span(value: str, source: str) -> str | None:
    """按 token 序列定位原文片段，允许 token 之间使用任意空白。

    边界断言避免把 "he said" 匹配到 "the said" 这样的词内片段。
    """
    if not value:
        return None
    pattern = r'(?<!\w)' + r'\s+'.join(re.escape(token) for token in value.split()) + r'(?!\w)'
    match = re.search(pattern, source)
    return match.group(0) if match else None


def _match_new_quote(candidate: Any, source: str) -> tuple[str, str] | None:
    """在新事实 quote 中定位连续原文；返回 (原文片段, 匹配方式)。

    只处理格式差异：外层引号、引文前冒号、换行或连续空格。不接受同义改写、
    来自旧事实的引文或只凭关键词拼出的片段，避免把宽松匹配当成证据。
    """
    if not isinstance(candidate, str) or not source:
        return None
    raw = candidate.strip()
    normalized = _strip_wrapping_quotes(raw)
    attempts = [(raw, 'exact')]
    if normalized != raw:
        attempts.append((normalized, 'stripped_quotes'))
    if normalized.startswith(':'):
        attempts.append((normalized[1:].strip(), 'stripped_prefix'))
    for value, method in attempts:
        span = _find_quote_span(value, source)
        if span is None:
            continue
        if method == 'exact':
            return span, 'exact' if span == value else 'whitespace_normalized'
        return span, method if span == value else method + '+whitespace'
    return None


class StateFactRelationJudge:
    """服务于双路召回后的关系判断；不读取整个主体历史，也不修改图。"""

    def __init__(self, llm: Any, max_response_repairs: int = 1):
        """注入关系判断模型；本类不持有图或写入权限。"""
        self.llm = llm
        if max_response_repairs not in (0, 1):
            raise ValueError('写入关系协议最多修复一次')
        self.max_response_repairs = max_response_repairs

    async def judge(self, old: dict[str, Any], new: dict[str, Any]) -> dict[str, Any]:
        """返回关系 JSON；模型异常、解析失败或引文不匹配一律降为 uncertain。

        引文和 Triple 仅在提示词中限制长度，原始事实及证据不会被截断存储。
        """
        def view(fact):
            """把事实压成提示词视图；只保留判断所需语义字段。"""
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
            # 服务于写入期关系模块：模型只选择双端编号，原文由程序回填。
            # 不再让模型抄写 evidence_quote，避免抄旧句、加引号或同义改写。
            'evidence_refs': {'type': 'array', 'maxItems': 2,
                              'items': {'type': 'string', 'enum': ['old', 'new']}},
            'reason': {'type': 'string'},
        }

        schema = {'type': 'json_schema', 'json_schema': {
            'name': 'state_fact_relation', 'strict': True, 'schema': {
                'type': 'object', 'properties': properties,
                'required': list(properties), 'additionalProperties': False}}}

        prompt = (
            'STATE_FACT_RELATION\n提出新旧事实的候选关系，仅输出指定 JSON。'
            'duplicate=同一事实重复；complement=兼容补充；coexists=独立事实并存；'
            'conflict=同槽互斥但无法确定谁有效；correction=明确指出旧记录错误；'
            'replacement=现实状态后来变化；unrelated=无关；uncertain=证据不足。'
            '同主体、同谓词、不同值、较晚时间均不足以证明替换。'
            '多值集合、不同活动、不同项目角色不能自动替换。'
            'same_slot 须核对主体、维度、作用对象和模态，不得照抄语义标签。'
            '证据编号 old 对应旧事实 quote，new 对应新事实 quote。'
            'evidence_refs 只选择编号；确定关系必须同时选择 old 和 new。'
            '不要加引号、冒号、省略号或改写原句；无需直接逐字复制原句到输出。'
            '证据不足填 uncertain，不得为了选齐编号而猜关系。'
            '新旧内容是原文断言，不代表已确认的现实真值。'
            '本次没有完整证据链，关系仅为可能线索，不确认因果、有效性或事件身份，'
            '不执行覆盖、失效或更新。\n'
            f'旧事实：{json.dumps(view(old), ensure_ascii=False)}\n'
            f'新事实：{json.dumps(view(new), ensure_ascii=False)}'
        )

        history = []
        for attempt in range(self.max_response_repairs + 1):
            response, result, metadata, failure_kind = None, None, {}, 'transport_error'
            try:
                response = await self.llm.generate(prompt, max_tokens=None, temperature=0.0,
                                                   response_format=schema, metadata_sink=metadata)
                if metadata.get('error_type'):
                    raise RuntimeError('模型传输失败：' + str(metadata['error_type']))
                failure_kind = 'invalid_protocol'
                if metadata.get('finish_reason') == 'length':
                    failure_kind = 'truncated_output'
                    raise ValueError('服务端生成截断；当前代码未设置输出 Token 上限')
                clean = response.strip()
                if clean.startswith('```'):
                    clean = clean.split('\n', 1)[1].rsplit('```', 1)[0].strip()
                result = json.loads(clean)
                if not isinstance(result, dict) or result.get('relation') not in RELATIONS:
                    raise ValueError('关系枚举无效')
                if not isinstance(result.get('same_slot'), bool):
                    raise ValueError('same_slot 不是布尔值')
                if 'evidence_refs' in result:
                    refs = result['evidence_refs']
                    if (not isinstance(refs, list) or len(refs) > 2 or
                            any(ref not in ('old', 'new') for ref in refs) or
                            len(set(refs)) != len(refs) or
                            result['relation'] != 'uncertain' and set(refs) != {'old', 'new'}):
                        raise ValueError('确定关系必须引用 old/new 两端，不能伪造或遗漏证据编号')
                    evidence = []
                    for side in refs:
                        fact = old if side == 'old' else new
                        quote = view(fact)['quote']
                        if not quote.strip():
                            raise ValueError('已选择的事实没有原文引文')
                        evidence.append(dict(side=side, fact_id=fact.get('fact_id'),
                            chunk_id=str(fact.get('chunk_id', '')), quote=quote))
                    result.update(evidence=evidence, evidence_quote=view(new)['quote'] if 'new' in refs else '',
                                  quote_match='exact' if refs else None)
                    result.update(evaluation_status='semantic_uncertain' if result['relation'] == 'uncertain'
                                  else 'model_assessed', failure_kind=None, metadata=metadata,
                                  response_repair_history=history, protocol='source_selection_v2')
                    return result
                # 服务于旧实验/固定响应兼容：旧协议仍严格匹配当前原文，不放宽为旧引文。
                quote = result.get('evidence_quote')
                matched = _match_new_quote(quote, str(new.get('quote') or ''))
                if matched is None:
                    raise ValueError('关系判断缺少当前事实原文依据')
                # 存回新事实 quote 中的真实片段，避免把模型添加的引号带入后续门控。
                result['evidence_quote'], result['quote_match'] = matched
                result['model_evidence_quote'] = quote
                result.update(evaluation_status='semantic_uncertain' if result['relation'] == 'uncertain'
                              else 'model_assessed', failure_kind=None, metadata=metadata,
                              response_repair_history=history, protocol='legacy_quote_v1')
                return result
            except asyncio.CancelledError:
                raise
            except Exception as error:
                raw = None if response is None else str(response)
                failed = {'relation': 'uncertain', 'same_slot': False, 'evidence_quote': '',
                        'reason': f'关系判断失败：{type(error).__name__}: {error}',
                        'evaluation_status': 'technical_error', 'failure_kind': failure_kind,
                        'metadata': metadata,
                        'validation_error': str(error), 'quote_match': None,
                        'model_relation': result.get('relation') if isinstance(result, dict) else None,
                        'model_evidence_quote': result.get('evidence_quote') if isinstance(result, dict) else None,
                        'model_same_slot': result.get('same_slot') if isinstance(result, dict) else None,
                        'raw_response': raw[:4000] if raw else raw,
                        'raw_response_truncated': bool(raw and len(raw) > 4000)}
                # 只对协议错误进行一次修复；网络错误和服务端截断不靠重复请求掩盖。
                if attempt < self.max_response_repairs and failure_kind == 'invalid_protocol':
                    history.append(failed)
                    prompt += '\n协议修复：上次输出无效。只选 evidence_refs=["old","new"] 或证据不足时 uncertain；保持原语义任务。'
                    continue
                failed['response_repair_history'] = history
                return failed
