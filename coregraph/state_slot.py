"""状态槽位及属性更新语义；没有明确互斥依据时采取保守处理。"""

from __future__ import annotations

import json
import re
from typing import Any, Mapping


# 只有语义上通常同时存在一个当前值的属性，才允许进入替换证据门控。
# 其余开放属性默认按集合处理；新值可以并存，绝不靠时间顺序自动替换。
_SINGLE_ATTRIBUTES = {
    "employer", "current_employer", "current_job", "current_role",
    "residence", "current_residence", "primary_residence",
    "relationship_status", "marital_status",
}


def canonical_attribute(attribute: str) -> str:
    """把模型的属性名称统一为稳定的 snake_case。"""
    if not isinstance(attribute, str):
        raise ValueError("属性必须是字符串")
    result = re.sub(r"[^a-z0-9]+", "_", attribute.casefold()).strip("_")
    if not re.fullmatch(r"[a-z][a-z0-9_]{1,63}", result):
        raise ValueError("状态属性必须是短英文标识")
    return result


def cardinality_for(attribute: str) -> str:
    """返回 single/set；未知属性默认 set，避免误把并存事实写成变化。"""
    return "single" if canonical_attribute(attribute) in _SINGLE_ATTRIBUTES else "set"


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
    """先用可验证的槽位与基数规则排除误替换；GATE 才交给独立证据判断。"""
    if slot_key(old) != slot_key(new):
        return "ADD"
    previous = " ".join(str(old.get("value") or "").casefold().split())
    incoming = " ".join(str(new.get("value") or "").casefold().split())
    if previous == incoming:
        return "NOOP"
    if cardinality_for(str(new["attribute"])) == "set":
        return "ADD"
    if previous and re.search(rf"(?<!\w){re.escape(previous)}(?!\w)", incoming):
        return "REFINE"
    return "GATE"
