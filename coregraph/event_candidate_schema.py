"""事件候选的结构化输出协议；来源 ID 由程序生成，模型只能引用。"""

from hashlib import sha256
from typing import Any


def triple_id(ref: dict[str, Any]) -> str:
    """相同 Chunk 中同一 Triple 获得稳定 ID，不受列表顺序影响。"""
    parts = [str(ref.get(key, "")) for key in ("chunk_id", "src", "relation", "tgt")]
    return "T" + sha256("\x1f".join(parts).encode("utf-8")).hexdigest()[:20]


def extraction_format(names: list[str], refs: list[dict[str, Any]]) -> dict[str, Any]:
    """将字段类型和当前来源集合交给 vLLM 的 JSON Schema 解码器约束。"""
    properties = {
        "entity": {"type": "string", "enum": names},
        "attribute": {"type": "string"},
        "value": {"type": ["string", "null"]},
        "kind": {"type": "string", "enum": ["state", "correction"]},
        "fact_type": {"type": "string", "enum": ["state", "event"]},
        "modality": {"type": "string", "enum": ["actual", "planned", "hypothetical", "negated"]},
        "quote": {"type": "string", "minLength": 1},
        "scope": {"type": "object", "additionalProperties": {"type": "string"}},
        "event_time": {"type": ["string", "null"]},
        "event_attributes": {"type": "object", "additionalProperties": {"type": "string"}},
        # 服务器当前 grammar 后端不支持 uniqueItems，去重由 resolve_indices 完成。
        "triple_ids": {"type": "array", "items": {"type": "string"}},
        "target_fact_id": {"type": ["string", "null"]},
    }
    if refs:
        properties["triple_ids"]["items"]["enum"] = [triple_id(ref) for ref in refs]
    else:
        properties["triple_ids"]["maxItems"] = 0
    schema = {"type": "object", "additionalProperties": False,
              "required": ["facts"], "properties": {"facts": {
                  "type": "array", "maxItems": 4, "items": {
                      "type": "object", "additionalProperties": False,
                      "properties": properties, "required": list(properties)}}}}
    return {"type": "json_schema", "json_schema": {
        "name": "event_candidates", "strict": True, "schema": schema}}


def resolve_indices(item: dict[str, Any], refs: list[dict[str, Any]]) -> list[int]:
    """新协议用 ID；旧整数/对象协议只在精确来源核对后兼容。"""
    if "triple_ids" in item:
        ids = item["triple_ids"]
        lookup = {triple_id(ref): index for index, ref in enumerate(refs)}
        if not isinstance(ids, list) or any(not isinstance(key, str) or key not in lookup for key in ids):
            raise ValueError("Triple ID 不属于当前 Chunk")
        return list(dict.fromkeys(lookup[key] for key in ids))
    legacy = item.get("triple_indices", [])
    if not isinstance(legacy, list):
        raise ValueError("Triple 引用格式错误")
    indices = []
    for value in legacy:
        if type(value) is int:
            if value < 0 or value >= len(refs):
                raise ValueError("Triple 引用越界")
            index = value
        elif isinstance(value, dict):
            matches = [i for i, ref in enumerate(refs) if all(
                str(value.get(key, "")) == str(ref.get(key, ""))
                for key in ("src", "relation", "tgt", "chunk_id"))]
            if len(matches) != 1:
                raise ValueError("Triple 对象不属于当前 Chunk 的原始来源")
            index = matches[0]
        else:
            raise ValueError("Triple 引用格式错误")
        if index not in indices:
            indices.append(index)
    return indices
