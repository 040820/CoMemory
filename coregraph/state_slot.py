"""状态槽位及属性更新语义；没有明确互斥依据时采取保守处理。

槽位键 = 主体 + 规范属性 + 作用域 + 模态。这个模块只做“能否比较/是否同槽”
的确定性判断，不决定现实状态是否改变：
    - 单值属性才可能进入替换门控，开放属性默认按集合处理；
    - 作用域值必须能在当前引文中找到，不能凭空补全；
    - 时间/模态不兼容时直接拦截，不靠“较新”推定替换。
"""

from __future__ import annotations

import json
import re
from datetime import date
from typing import Any, Mapping


# 只有语义上通常同时存在一个当前值的属性，才允许进入替换证据门控。
# 其余开放属性默认按集合处理；新值可以并存，绝不靠时间顺序自动替换。
_SINGLE_ATTRIBUTES = {
    "employer", "current_employer", "current_job", "current_role",
    "residence", "current_residence", "primary_residence",
    "relationship_status", "marital_status",
}

# 只合并语义足够明确的命名；“相关于”“喜欢”等开放关系不能据此变成单值状态。
_ATTRIBUTE_ALIASES = {
    "works_at": "employer", "works_for": "employer",
    "employed_by": "employer", "current_employer": "employer",
    "lives_in": "residence", "resides_in": "residence",
}


def canonical_attribute(attribute: str) -> str:
    """把模型的属性名称统一为稳定的 snake_case。

    规则：小写、非字母数字转下划线、去掉首尾下划线；长度限制 2–64。
    这样 "Current Employer" 和 "current_employer" 会落到同一个槽名，
    但不会把任意自然语言短语硬塞成属性名。
    """
    if not isinstance(attribute, str):
        raise ValueError("属性必须是字符串")
    result = re.sub(r"[^a-z0-9]+", "_", attribute.casefold()).strip("_")
    if not re.fullmatch(r"[a-z][a-z0-9_]{1,63}", result):
        raise ValueError("状态属性必须是短英文标识")
    return _ATTRIBUTE_ALIASES.get(result, result)


def cardinality_for(attribute: str, fact: Mapping[str, Any] | None = None) -> str:
    """优先采用有原文来源的开放语义槽基数；旧协议仍使用保守默认。

    已验证的 semantic_state.cardinality 来自归一化阶段；没有它时只对
    已知单值属性（雇主/住处/婚姻状态）给 single，其余一律 set。
    """
    semantic = (fact or {}).get('semantic_state') or {}
    if semantic.get('validated') is True and semantic.get('cardinality') in {'single', 'set'}:
        return semantic['cardinality']
    return "single" if canonical_attribute(attribute) in _SINGLE_ATTRIBUTES else "set"


def comparison_value(fact: Mapping[str, Any]) -> Any:
    """比较语义值，原始 value 仍保留为原文中的对象或短语。

    归一化成功时用 semantic_state.value 比较同义表达；否则退回原始 value。
    """
    semantic = fact.get('semantic_state') or {}
    return semantic.get('value') if semantic.get('validated') is True else fact.get('value')


def equivalent_value(old: Mapping[str, Any], new: Mapping[str, Any]) -> bool:
    """按比较语义值判断是否等价；大小写和空白差异不算新事实。"""
    def normalized(value):
        """统一大小写与空白；None 保持 None，避免与字符串 "None" 混淆。"""
        # None 保持 None，避免 "None" 字符串和真正的空值被混为一谈。
        return ' '.join(str(value).casefold().split()) if value is not None else None
    return normalized(comparison_value(old)) == normalized(comparison_value(new))


def normalize_scope(raw: Mapping[str, Any] | None, quote: str) -> dict[str, str]:
    """规范化作用域，并要求每个作用域值能在本条原始引文中定位。"""
    if raw is None:
        return {}
    if not isinstance(raw, Mapping):
        raise ValueError("scope 必须是对象")
    scope: dict[str, str] = {}
    for key, value in raw.items():
        name = canonical_attribute(key)
        if not isinstance(value, str) or not value.strip():
            raise ValueError("scope 的值必须是非空文本")
        normalized = " ".join(value.split())
        if normalized.casefold() not in quote.casefold():
            raise ValueError("scope 的值没有当前引文依据")
        scope[name] = normalized
    return dict(sorted(scope.items()))


def normalize_modality(raw: str | None) -> str:
    """模态分开建槽；计划、假设和否定不能覆盖已发生的事实。"""
    modality = str(raw or "").casefold().strip()
    if modality not in {"actual", "planned", "hypothetical", "negated"}:
        raise ValueError("modality 必须是 actual/planned/hypothetical/negated")
    return modality


def slot_key(fact: Mapping[str, Any]) -> str:
    """槽位键包含主体、属性、作用域和模态；作用域顺序不影响结果。"""
    scope = fact.get("scope") or {}
    if not isinstance(scope, Mapping):
        raise ValueError("scope 必须是对象")
    parts = [str(fact["entity_id"]), canonical_attribute(str(fact["attribute"])),
             sorted((str(key).casefold(), str(value).casefold()) for key, value in scope.items()),
             normalize_modality(fact.get("modality"))]
    return json.dumps(parts, ensure_ascii=False, separators=(",", ":"))


def preliminary_operation(old: Mapping[str, Any], new: Mapping[str, Any]) -> str:
    """先用可验证的槽位与基数规则排除误替换；GATE 才交给独立证据判断。

    返回值：
        ADD    不同槽或集合属性 -> 新事实直接追加，不涉及旧状态失效；
        NOOP   同槽同值 -> 重复，不新增；
        REFINE 同槽且新值是旧值的明确细化 -> 补充，不替换；
        GATE   同槽不同值且无法用规则排除 -> 交给 Writer 的证据/时间门控。
    """
    # 不同主体/属性/作用域/模态：不是同一个状态槽，直接按新增处理。
    if slot_key(old) != slot_key(new):
        return "ADD"
    previous = " ".join(str(comparison_value(old) or "").casefold().split())
    incoming = " ".join(str(comparison_value(new) or "").casefold().split())
    if previous == incoming:
        return "NOOP"
    # 多值集合（爱好、宠物、参与活动等）允许并存，不进入替换门控。
    if cardinality_for(str(new["attribute"]), new) == "set":
        return "ADD"
    if (old.get('semantic_state') or new.get('semantic_state')):
        # “已开启/未开启”等不同状态不能因文字包含关系被当作 REFINE。
        return "GATE"
    # 只有旧值作为独立词出现在新值中时，才把它当作明确细化；
    # 用词边界避免 "art" 命中 "start" 这类子串误判。
    if previous and re.search(rf"(?<!\w){re.escape(previous)}(?!\w)", incoming):
        return "REFINE"
    return "GATE"


def _date_in(value: Any) -> date | None:
    """只使用可解析的明确日期；相对时间由后续原文门控判断。"""
    match = re.search(r"\d{4}-\d{2}-\d{2}", str(value or ""))
    if not match:
        return None
    try:
        return date.fromisoformat(match.group())
    except ValueError:
        return None


def temporal_gate(old: Mapping[str, Any], new: Mapping[str, Any]) -> tuple[bool, str]:
    """排除时间倒置和模态混用；通过只表示无冲突，不单独证明发生了替换。

    比较顺序：
        1. 模态必须一致：计划/假设/否定的新事实不能覆盖已发生的旧事实；
        2. 观察时间不能倒退；
        3. 事件时间不能倒退；
        4. 用“事件时间优先、观察时间兜底”的有效时间再比较一次。
    返回 (是否通过, 说明)；说明会写入审计，便于解释为什么拦截/放行。
    """
    if normalize_modality(old.get("modality")) != normalize_modality(new.get("modality")):
        return False, "计划、假设、否定与已发生事实不能直接互相覆盖"
    old_observed = _date_in(old.get("observed_at"))
    new_observed = _date_in(new.get("observed_at"))
    old_event = _date_in(old.get("event_time"))
    new_event = _date_in(new.get("event_time"))
    if old_observed and new_observed and new_observed < old_observed:
        return False, "新会话的观察时间早于旧事实"
    if old_event and new_event and new_event < old_event:
        return False, "新事实描述的是更早发生的事件"
    # 旧事实若给出了事件时间，应以事件发生时间比较；观察时间只是缺省边界。
    # 例如旧事件一月发生、三月才被记录，新事件二月发生，不能因二月早于三月误拦截。
    old_effective = old_event or old_observed
    if new_event and old_effective and new_event < old_effective:
        return False, "新会话回忆的事件早于旧状态的有效时间"
    # 通过不代表替换成立：仍需 Writer 的原文/关系门控确认。
    return True, "未发现明确时间冲突；仍需原文支持状态转换"
