"""分项置信度模块：服务于三层记忆的来源评价与候选关系解释，不执行记忆更新。

Fact/Assertion 保存抽取分数与评价 ID；独立记录保存证据支持方向和关系评价。
未评估为 null；模型分数明确标记未校准，不能解释为事实正确的概率。本模块不
融合多证据、不确认完整证据链，也不因任何阈值覆盖、撤销或合并历史事实。
只依赖标准库，便于在构图完成后对副本运行，不干扰原版 Baseline 或冻结实验。
"""

from __future__ import annotations

import asyncio
from copy import deepcopy
from datetime import datetime, timezone
from hashlib import sha256
import json
import math
from typing import Any

METHOD = 'llm_source_assessment_v1'
RELATION_CHECKS = ('same_subject', 'same_event_or_slot', 'temporal_compatible',
                   'relation_supported', 'counterevidence_review')
CHECK_STATES = ('pass', 'fail', 'unknown', 'not_applicable')


class AssessmentOutputError(ValueError):
    """服务于失败审计：模型输出无效时仍携带原始响应与截断/API 元数据。"""

    def __init__(self, message, trace):
        """保留原始响应与截断/API 元数据，便于定位模型输出问题。"""
        super().__init__(message)
        self.trace = trace


def _hash(value: Any) -> str:
    """服务于稳定引用与失效检测：内容相同则哈希相同，不使用运行时随机 ID。"""
    return sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, default=str).encode()).hexdigest()


def middle_rows(graph: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """服务于统一评价：枚举中层各类记录，但不把语义断言写成实体图边。"""
    table = graph['fact_storage']
    result = {}
    for name in ('facts', 'observations', 'time_assertions', 'attribute_assertions'):
        for fid, row in table.get(name, {}).items():
            if fid in result:
                raise ValueError(f'中层记录 ID 重复：{fid}')
            result[fid] = row
    return result


def _claim(row):
    """只选取断言语义字段，排除向量、置信度自身及可变审计字段，防止循环失效。"""
    keys = ('fact_id', 'assertion_id', 'src', 'relation', 'tgt', 'entity_id', 'attribute', 'predicate',
            'value', 'source_attribute', 'source_value', 'source_scope', 'source_modality',
            'scope', 'modality', 'semantic_state', 'event_id', 'event_time', 'event_time_raw',
            'event_time_range', 'time_basis', 'observed_at', 'session_time', 'fact_type')
    return {key: deepcopy(row[key]) for key in keys if key in row}


def _source(graph, row, chunks=None):
    """定位原始 Chunk 和连续引文；同 Chunk 的多条派生证据共享独立性分组。

    大 Chunk 仅提供引文附近有限窗口，明确记录截断。模型引用必须在实际提供
    的窗口内，不能凭未展示历史得出结论。原始 Chunk 本身不被截断或修改。
    """
    cid = str(row.get('chunk_id', row.get('evidence_chunk_id', '')))
    if chunks is None:
        chunks = {str(k): v for k, v in graph['chunk_storage'].items()}
    chunk = chunks.get(cid, {})
    text = str(chunk.get('text') or '')
    quote = row.get('quote', row.get('evidence_quote'))
    valid = bool(text) and (quote is None or isinstance(quote, str) and bool(quote.strip()) and quote in text)
    position = text.find(quote) if valid and quote else 0
    start = max(0, position - 2400) if len(text) > 6000 else 0
    window = text[start:start + 6000]
    return dict(chunk_id=cid, text=window, anchor_quote=quote,
                session_time=str(chunk.get('session_time') or ''),
                source_valid=valid and (quote is None or quote in window),
                source_digest=_hash(text), window_start=start, truncated=len(window) < len(text))


def _context(graph, row, rows=None):
    """服务于时间/属性断言评价：有限展开同组来源事实，区分本次事件和其他事件。"""
    eid = row.get('event_id')
    group = graph.get('event_state_layer', {}).get('groups', {}).get(eid, {})
    if not group:
        return []
    if rows is None:
        rows = middle_rows(graph)
    return [_claim(rows[fid]) for fid in group.get('fact_ids', [])[:8] if fid in rows]


def _ensure(table, eid, base, fingerprint):
    """输入变化时令旧分数失效，保留旧评价历史；重复注册不会制造重复记录。"""
    current = table.get(eid)
    if current and current.get('input_digest') == fingerprint:
        return current
    history = []
    status = 'not_evaluated'
    if current:
        history = list(current.get('history', [])) + [{k: deepcopy(v) for k, v in current.items() if k != 'history'}]
        status = 'stale'
    record = dict(evaluation_id=eid, score=None, status=status, method=None,
                  calibration='uncalibrated', validation_status='not_verified', input_digest=fingerprint,
                  **base)
    if history:
        record['history'] = history
    table[eid] = record
    return record


def register_confidence(graph: dict[str, Any]) -> dict[str, Any]:
    """服务于持久化模块：注册三类评价引用，初始化未知分数，不调用 LLM。

    精简上层只增加 evaluation_id。具体评价独立保存在 confidence_storage，
    同一 Chunk 的派生事实不被当作多份独立证据，原始事实和关系状态保持原样。
    """
    if graph.get('event_state_layer', {}).get('schema') != 'event_index_v1':
        raise ValueError('分项置信度当前仅支持 event_index_v1，旧文件请显式迁移副本')
    storage = graph.setdefault('confidence_storage', dict(version=1, fact_evaluations={},
                                 evidence_links={}, relation_evaluations={}))
    if storage.get('version') != 1:
        raise ValueError('不支持的置信度记录版本')
    rows = middle_rows(graph)
    chunks = {str(k): v for k, v in graph['chunk_storage'].items()}
    evidence_by_fact = {}
    for evidence in storage['evidence_links'].values():
        evidence_by_fact.setdefault(evidence['fact_id'], []).append(evidence)
    fact_inputs = {}
    for fid, row in rows.items():
        source = _source(graph, row, chunks)
        fingerprint = _hash([_claim(row), source, _context(graph, row, rows)])
        fact_inputs[fid] = fingerprint
        eid = 'X' + _hash(fid)[:24]
        assessment = _ensure(storage['fact_evaluations'], eid, dict(fact_id=fid,
            score_semantics='extraction_fidelity', reason=None), fingerprint)
        row['extraction_evaluation_id'] = eid
        row['extraction_confidence'] = assessment['score']
        evidence_id = 'S' + _hash([fid, source['chunk_id']])[:24]
        # 服务于来源变更保护：旧 Chunk 评分不能被复用为新 Chunk 的支持证据。
        # 仅失效自动建立的来源锚点；未来额外检索的证据可用其他 link_type 独立保存。
        for previous in evidence_by_fact.get(fid, []):
            if (previous.get('link_type', 'source_anchor') == 'source_anchor' and
                    previous['chunk_id'] != source['chunk_id'] and previous.get('is_current_source', True)):
                old = {k: deepcopy(v) for k, v in previous.items() if k != 'history'}
                previous.setdefault('history', []).append(old)
                previous.update(score=None, status='stale', role='unassessed', span=None,
                                is_current_source=False, reason='原始来源已变更，旧评价不可用于当前断言')
        evidence = _ensure(storage['evidence_links'], evidence_id, dict(evidence_id=evidence_id,
            fact_id=fid, chunk_id=source['chunk_id'], role='unassessed',
            score_semantics='confidence_in_support_direction', span=None,
            independence_group='chunk:' + source['chunk_id'], reason=None,
            link_type='source_anchor', is_current_source=True), fingerprint)
        evidence['is_current_source'] = True
    for link in graph['event_state_layer']['links']:
        old, new = link['from_fact_id'], link['to_fact_id']
        if old not in rows or new not in rows:
            raise ValueError('关系评价存在悬空 Fact 引用')
        eid = 'R' + _hash(link['link_id'])[:24]
        _ensure(storage['relation_evaluations'], eid, dict(link_id=link['link_id'],
            from_fact_id=old, to_fact_id=new, proposed_relation=link['relation'],
            verdict='not_evaluated', scope='pair_local', reason=None,
            score_semantics='candidate_relation_support', evidence_quotes=[],
            checks={key: 'unknown' for key in RELATION_CHECKS}),
            _hash([fact_inputs[old], fact_inputs[new], link['relation']]))
        link['evaluation_id'] = eid
    return storage


def _score(value):
    """服务于模型接口校验：拒绝布尔值、NaN、无穷大和越界分数，不静默截断。"""
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError('模型分数必须是有限的 0–1 数值')
    return float(value)


class MemoryConfidenceEvaluator:
    """有限来源评价器：只评估指定断言/候选关系，所有结果仍需后续验证与校准。"""

    def __init__(self, llm):
        """注入评价模型；本类不持有图，所有读取通过调用方传入的 graph 完成。"""
        self.llm = llm

    async def _request(self, prompt, properties):
        """调用现有 LLMManager 的结构化输出，保存原始输出及 finish_reason 供追溯。"""
        metadata = {}
        schema = {'type': 'json_schema', 'json_schema': {'name': 'memory_confidence', 'strict': True,
            'schema': {'type': 'object', 'properties': properties, 'required': list(properties),
                       'additionalProperties': False}}}
        raw = None
        try:
            raw = await self.llm.generate(prompt + '\n仅输出紧凑 JSON，不要缩进、空行或填充空格。'
                'reason 用一句话简要说明，最多80个汉字。', max_tokens=1600, temperature=0.0,
                                          response_format=schema, metadata_sink=metadata)
            if metadata.get('finish_reason') == 'length':
                raise ValueError('模型输出达到 token 上限，评价不完整')
            result = json.loads(raw)
            if not isinstance(result, dict) or any(k not in result for k in properties):
                raise ValueError('模型评价结构不完整')
            return result, dict(raw_response=raw, metadata=metadata)
        except asyncio.CancelledError:
            raise
        except Exception as error:
            raise AssessmentOutputError(str(error), dict(raw_response=raw, metadata=metadata)) from error

    def _stamp(self, record):
        """服务于评价版本追溯：记录模型/方法/时间，明确未做概率校准或人工验证。"""
        if record['status'] in ('assessed', 'failed', 'cancelled'):
            previous = {k: deepcopy(v) for k, v in record.items() if k != 'history'}
            record.setdefault('history', []).append(previous)
        # 重试失败不能残留上一轮“支持”方向、引文或高分；旧结果只保留在历史中。
        for key in ('trace', 'model_score', 'model_verdict', 'quote'):
            record.pop(key, None)
        if 'role' in record:
            record.update(role='unassessed', span=None)
        if 'checks' in record:
            record.update(checks={k: 'unknown' for k in RELATION_CHECKS},
                          evidence_quotes=[], verdict='not_evaluated')
        record.update(method=METHOD, model=getattr(self.llm, 'model', 'unknown'),
                      evaluated_at=datetime.now(timezone.utc).isoformat(),
                      calibration='uncalibrated', validation_status='not_verified',
                      score=None, status='evaluating', reason=None)

    async def assess_fact(self, graph, fact_id):
        """分别评价抽取忠实度与证据支持方向；原话被准确复述不意味着现实真值成立。"""
        storage = register_confidence(graph)
        row = middle_rows(graph)[fact_id]
        record = storage['fact_evaluations'][row['extraction_evaluation_id']]
        source = _source(graph, row)
        evidence_id = 'S' + _hash([fact_id, source['chunk_id']])[:24]
        evidence = storage['evidence_links'][evidence_id]
        self._stamp(record)
        self._stamp(evidence)
        trace = {}
        try:
            if not source['source_valid']:
                raise ValueError('原始 Chunk 缺失或锚点引文不是连续原文')
            result, trace = await self._request(
                'MEMORY_CONFIDENCE_FACT\n评价抽取是否忠实于原文，不能评价现实真值。'
                '独立判断所给证据对断言是 supports/refutes/insufficient。'
                'extraction_score 为抽取准确性的未校准估计；evidence_score 为支持方向判断的未校准估计。'
                '检查主体、谓词、否定、模态、时间及作用域；准确引用错误陈述仍可抽取正确。'
                'evidence_quote 必须逐字引用提供的 text 中非空连续片段。不得补写未展示上下文。\n'
                + json.dumps(dict(claim=_claim(row), event_context=_context(graph, row), evidence=source), ensure_ascii=False),
                dict(extraction_score={'type': 'number'}, evidence_score={'type': 'number'},
                     evidence_role={'type': 'string', 'enum': ['supports', 'refutes', 'insufficient']},
                     evidence_quote={'type': 'string'}, reason={'type': 'string'}))
            extraction, support = _score(result['extraction_score']), _score(result['evidence_score'])
            quote, role = result['evidence_quote'], result['evidence_role']
            if role not in ('supports', 'refutes', 'insufficient') or not isinstance(quote, str) or not quote.strip() or quote not in source['text']:
                raise ValueError('证据方向无效或模型引文不在提供的原文中')
            record.update(score=extraction, status='assessed', reason=result['reason'], trace=trace)
            start = source['window_start'] + source['text'].index(quote)
            evidence.update(score=support, role=role, status='assessed', reason=result['reason'],
                            span=[start, start + len(quote)], quote=quote, trace=deepcopy(trace))
        except asyncio.CancelledError:
            for item in (record, evidence):
                item.update(score=None, status='cancelled', reason='评价已取消')
            row['extraction_confidence'] = None
            raise
        except Exception as error:
            for item in (record, evidence):
                item.update(score=None, status='failed', reason=f'{type(error).__name__}: {error}',
                            trace=trace or getattr(error, 'trace', {}))
        row['extraction_confidence'] = record['score']
        return record

    async def assess_relation(self, graph, link_id):
        """只检查一对事实的局部证据，不确认关系、不改旧事实，不声称反证已穷尽。

        纠正/替换需要同主体、同一事件或状态槽、时间可比较和关系有依据。缺少
        必要条件时，即使模型给高分也降为 uncertain；原始模型建议另行保留。
        """
        storage = register_confidence(graph)
        link = next(link for link in graph['event_state_layer']['links'] if link['link_id'] == link_id)
        record = storage['relation_evaluations'][link['evaluation_id']]
        rows = middle_rows(graph)
        old, new = rows[link['from_fact_id']], rows[link['to_fact_id']]
        sources = [_source(graph, old), _source(graph, new)]
        self._stamp(record)
        trace = {}
        try:
            if not all(s['source_valid'] for s in sources):
                raise ValueError('关系端点缺少有效原文')
            result, trace = await self._request(
                'MEMORY_CONFIDENCE_RELATION\n检查候选关系的局部依据，不确认完整证据链或执行任何更新。'
                '新旧值不同、新时间较晚、同主体、相似行为都不能单独证明替换。'
                '同一次事件的纠正必须核对事件指代；不同时间的同类活动不能强行合并。'
                'same_event_or_slot=同一次事件或同一可比较持续状态；temporal_compatible=时间关系适合所提关系。'
                'counterevidence_review 只能评价所给局部反证，未展示历史不能声称全部排除。'
                'score 为未校准关系估计，verdict 仅支持候选/不确定/反对候选。'
                'evidence_quotes 必须引用提供的 chunk_id 和连续原文。不得照抄已有分数作为依据。\n'
                + json.dumps(dict(proposed_relation=link['relation'], old=_claim(old), new=_claim(new),
                    old_context=_context(graph, old), new_context=_context(graph, new), sources=sources), ensure_ascii=False),
                dict(score={'type': 'number'}, verdict={'type': 'string',
                    'enum': ['supported_candidate', 'uncertain', 'rejected_candidate']}, reason={'type': 'string'},
                    checks={'type': 'object', 'properties': {k: {'type': 'string', 'enum': list(CHECK_STATES)} for k in RELATION_CHECKS},
                            'required': list(RELATION_CHECKS), 'additionalProperties': False},
                    evidence_quotes={'type': 'array', 'items': {'type': 'object',
                        'properties': {'chunk_id': {'type': 'string'}, 'quote': {'type': 'string'}},
                        'required': ['chunk_id', 'quote'], 'additionalProperties': False}}))
            score = _score(result['score'])
            checks = result['checks']
            if any(checks.get(k) not in CHECK_STATES for k in RELATION_CHECKS):
                raise ValueError('关系检查项缺失或无效')
            quotes = result['evidence_quotes']
            if not quotes or any(not isinstance(q.get('quote'), str) or not q['quote'].strip() or
                    not any(str(q.get('chunk_id')) == s['chunk_id'] and q['quote'] in s['text'] for s in sources) for q in quotes):
                raise ValueError('关系证据引用缺失或不属于所给来源')
            verdict = result['verdict']
            if verdict not in ('supported_candidate', 'uncertain', 'rejected_candidate'):
                raise ValueError('关系评价结果无效')
            required = ['relation_supported']
            if link['relation'] in ('correction', 'replacement', 'conflict', 'duplicate'):
                required += ['same_subject', 'same_event_or_slot']
            if link['relation'] == 'replacement':
                required += ['temporal_compatible']
            if verdict == 'supported_candidate' and any(checks[k] != 'pass' for k in required):
                verdict = 'uncertain'
            if (verdict == 'supported_candidate' and link['relation'] == 'correction' and
                    checks['temporal_compatible'] not in ('pass', 'not_applicable')):
                # 明确不涉及时间的名字等纠正可记为不适用；时间未知或冲突不能采纳高分。
                verdict = 'uncertain'
            record.update(score=score if verdict != 'uncertain' else None, model_score=score,
                status='assessed', verdict=verdict, model_verdict=result['verdict'], reason=result['reason'],
                checks=checks, evidence_quotes=quotes, trace=trace, scope='pair_local',
                counterevidence_scope='provided_sources_only')
        except asyncio.CancelledError:
            record.update(score=None, status='cancelled', verdict='uncertain', reason='评价已取消')
            raise
        except Exception as error:
            record.update(score=None, status='failed', verdict='uncertain', reason=f'{type(error).__name__}: {error}',
                          trace=trace or getattr(error, 'trace', {}))
        return record
