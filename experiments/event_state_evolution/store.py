"""独立 SQLite 状态索引：原图只读，状态事实和演化关系单独持久化。"""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path
import sqlite3
from typing import Any

from .decision import Decision
from .evidence import EvidenceCandidate


SCHEMA = """
PRAGMA foreign_keys = ON;
CREATE TABLE IF NOT EXISTS entities (
  entity_id TEXT PRIMARY KEY, graph_name TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS facts (
  fact_id TEXT PRIMARY KEY, source_key TEXT NOT NULL,
  entity_id TEXT NOT NULL REFERENCES entities(entity_id), attribute TEXT NOT NULL,
  value TEXT, triple_subject TEXT NOT NULL, triple_predicate TEXT NOT NULL,
  triple_object TEXT NOT NULL, chunk_id TEXT NOT NULL, session_id TEXT,
  observed_at TEXT, event_time TEXT, evidence_quote TEXT NOT NULL,
  kind TEXT NOT NULL, detection TEXT, event_group_id TEXT, target_fact_id TEXT,
  position INTEGER NOT NULL, status TEXT NOT NULL,
  UNIQUE(source_key, entity_id, attribute)
);
CREATE INDEX IF NOT EXISTS facts_entity_attribute ON facts(entity_id, attribute, position);
CREATE INDEX IF NOT EXISTS facts_chunk ON facts(chunk_id);
CREATE TABLE IF NOT EXISTS current_states (
  entity_id TEXT NOT NULL, attribute TEXT NOT NULL, fact_id TEXT NOT NULL,
  value TEXT NOT NULL, PRIMARY KEY (entity_id, attribute),
  FOREIGN KEY (fact_id) REFERENCES facts(fact_id)
);
CREATE TABLE IF NOT EXISTS changes (
  change_id TEXT PRIMARY KEY, entity_id TEXT NOT NULL, attribute TEXT NOT NULL,
  old_fact_id TEXT NOT NULL, new_fact_id TEXT NOT NULL,
  old_value TEXT NOT NULL, new_value TEXT NOT NULL, event_time TEXT,
  observed_at TEXT, detection TEXT NOT NULL, event_group_id TEXT,
  status TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS changes_entity_attribute ON changes(entity_id, attribute, observed_at);
CREATE INDEX IF NOT EXISTS changes_old_fact ON changes(old_fact_id);
CREATE INDEX IF NOT EXISTS changes_new_fact ON changes(new_fact_id);
CREATE TABLE IF NOT EXISTS evidence_links (
  change_id TEXT NOT NULL, fact_id TEXT NOT NULL, chunk_id TEXT NOT NULL,
  PRIMARY KEY (change_id, fact_id)
);
CREATE INDEX IF NOT EXISTS links_fact ON evidence_links(fact_id);
CREATE INDEX IF NOT EXISTS links_chunk ON evidence_links(chunk_id);
CREATE TABLE IF NOT EXISTS review_queue (
  source_key TEXT PRIMARY KEY, chunk_id TEXT NOT NULL, reason TEXT NOT NULL
);
"""


class StateStore:
    """所有状态更新都在单个事务中完成；重跑相同来源不会重复写入。"""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.path)
        self.db.row_factory = sqlite3.Row
        self.db.executescript(SCHEMA)

    def close(self) -> None:
        self.db.close()

    @staticmethod
    def _dict(row: sqlite3.Row | None) -> dict[str, Any] | None:
        return dict(row) if row is not None else None

    def _rebuild(self, entity_id: str, attribute: str) -> None:
        """纠正后按有效证据重算此实体属性，保留被撤销变化的审计记录。"""
        self.db.execute("UPDATE changes SET status='revoked' WHERE entity_id=? AND attribute=?",
                        (entity_id, attribute))
        facts = self.db.execute(
            "SELECT * FROM facts WHERE entity_id=? AND attribute=? AND status='active' "
            "AND value IS NOT NULL ORDER BY position, rowid", (entity_id, attribute)).fetchall()
        previous = None
        for fact in facts:
            if previous is not None and previous["value"] != fact["value"]:
                key = f'{previous["fact_id"]}\x1f{fact["fact_id"]}'
                change_id = sha256(key.encode("utf-8")).hexdigest()[:24]
                self.db.execute(
                    "INSERT INTO changes (change_id,entity_id,attribute,old_fact_id,new_fact_id,"
                    "old_value,new_value,event_time,observed_at,detection,event_group_id,status) "
                    "VALUES (?,?,?,?,?,?,?,?,?,?,?,'active') ON CONFLICT(change_id) DO UPDATE SET status='active'",
                    (change_id, entity_id, attribute, previous["fact_id"], fact["fact_id"],
                     previous["value"], fact["value"], fact["event_time"], fact["observed_at"],
                     fact["detection"] or "inferred", fact["event_group_id"]))
                for linked in (previous, fact):
                    self.db.execute(
                        "INSERT OR IGNORE INTO evidence_links(change_id,fact_id,chunk_id) VALUES (?,?,?)",
                        (change_id, linked["fact_id"], linked["chunk_id"]))
            previous = fact
        if previous is None:
            self.db.execute("DELETE FROM current_states WHERE entity_id=? AND attribute=?",
                            (entity_id, attribute))
        else:
            self.db.execute(
                "INSERT INTO current_states(entity_id,attribute,fact_id,value) VALUES (?,?,?,?) "
                "ON CONFLICT(entity_id,attribute) DO UPDATE SET fact_id=excluded.fact_id,value=excluded.value",
                (entity_id, attribute, previous["fact_id"], previous["value"]))

    def apply(self, candidate: EvidenceCandidate, decision: Decision) -> str | None:
        """写入一个经校验的决定，失败时 SQLite 自动回滚整个事务。"""
        if decision.kind == "ignore":
            return None
        with self.db:
            if decision.kind == "review":
                self.db.execute("INSERT OR IGNORE INTO review_queue VALUES (?,?,?)",
                                (candidate.source_key, candidate.chunk_id, decision.reason or "待人工核查"))
                return None
            if (not decision.entity_id or decision.entity_id not in candidate.entity_names or
                    not decision.attribute or not decision.triple or not decision.evidence_quote or
                    decision.evidence_quote not in candidate.text):
                raise ValueError("状态事实缺少可核对的实体、三元组或原文")
            existing = self.db.execute(
                "SELECT fact_id FROM facts WHERE source_key=? AND entity_id=? AND attribute=?",
                (candidate.source_key, decision.entity_id, decision.attribute)).fetchone()
            if existing:
                return str(existing["fact_id"])
            target = None
            if decision.kind == "correction":
                target = self.db.execute(
                    "SELECT * FROM facts WHERE fact_id=? AND entity_id=? AND attribute=? AND status='active'",
                    (decision.corrected_fact_id, decision.entity_id, decision.attribute)).fetchone()
                if target is None:
                    raise ValueError("纠正目标不存在或已失效")
            self.db.execute("INSERT OR IGNORE INTO entities VALUES (?,?)",
                            (decision.entity_id, decision.entity_id))
            next_position = self.db.execute("SELECT COALESCE(MAX(position),0)+1 FROM facts").fetchone()[0]
            position = target["position"] if target is not None else next_position
            fact_id = sha256((candidate.source_key + "\x1f" + decision.entity_id + "\x1f" +
                              decision.attribute).encode("utf-8")).hexdigest()[:24]
            status = "retrospective" if decision.kind == "retrospective" else "active"
            if decision.kind == "correction" and decision.value is None:
                status = "audit"
            self.db.execute(
                "INSERT INTO facts VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (fact_id, candidate.source_key, decision.entity_id, decision.attribute,
                 decision.value, *decision.triple, candidate.chunk_id, candidate.session_id,
                 candidate.observed_at, decision.event_time, decision.evidence_quote,
                 decision.kind, decision.detection, decision.event_group_id,
                 decision.corrected_fact_id, position, status))
            if target is not None:
                self.db.execute("UPDATE facts SET status='corrected' WHERE fact_id=?", (target["fact_id"],))
            if decision.kind != "retrospective":
                self._rebuild(decision.entity_id, decision.attribute)
            return fact_id

    def get_current(self, entity_id: str, attribute: str) -> dict[str, Any] | None:
        row = self.db.execute(
            "SELECT c.*,f.chunk_id,f.observed_at,f.evidence_quote,f.triple_subject,"
            "f.triple_predicate,f.triple_object FROM current_states c JOIN facts f ON f.fact_id=c.fact_id "
            "WHERE c.entity_id=? AND c.attribute=?", (entity_id, attribute)).fetchone()
        return self._dict(row)

    def get_history(self, entity_id: str, attribute: str) -> dict[str, list[dict[str, Any]]]:
        facts = self.db.execute("SELECT * FROM facts WHERE entity_id=? AND attribute=? ORDER BY position,rowid",
                                (entity_id, attribute)).fetchall()
        changes = self.db.execute("SELECT * FROM changes WHERE entity_id=? AND attribute=? ORDER BY observed_at",
                                  (entity_id, attribute)).fetchall()
        return {"facts": [dict(row) for row in facts], "changes": [dict(row) for row in changes]}

    def get_by_fact(self, fact_id: str) -> list[dict[str, Any]]:
        rows = self.db.execute("SELECT * FROM facts WHERE fact_id=?", (fact_id,)).fetchall()
        return [dict(row) for row in rows]

    def get_by_chunk(self, chunk_id: str) -> list[dict[str, Any]]:
        rows = self.db.execute("SELECT * FROM facts WHERE chunk_id=?", (chunk_id,)).fetchall()
        return [dict(row) for row in rows]

    def review_count(self) -> int:
        return int(self.db.execute("SELECT COUNT(*) FROM review_queue").fetchone()[0])

    def has_source(self, source_key: str, attribute: str) -> bool:
        """断点续跑时跳过已有的该属性事实。"""
        row = self.db.execute("SELECT 1 FROM facts WHERE source_key=? AND attribute=? LIMIT 1",
                              (source_key, attribute)).fetchone()
        return row is not None

    def has_matching_fact(self, chunk_id: str, entity_id: str, attribute: str,
                          value: str | None) -> bool:
        """同一 Chunk 的原文候选和图边候选可能指向同一事实。"""
        row = self.db.execute(
            "SELECT 1 FROM facts WHERE chunk_id=? AND entity_id=? AND attribute=? "
            "AND value IS ? AND status='active' LIMIT 1",
            (chunk_id, entity_id, attribute, value)).fetchone()
        return row is not None

    def context(self, entity_names: tuple[str, ...]) -> dict[str, dict[str, Any]]:
        """提供 LLM 可核对的当前值与历史事实 ID，不让其猜测旧状态。"""
        result: dict[str, dict[str, Any]] = {}
        for name in entity_names:
            attrs = self.db.execute("SELECT DISTINCT attribute FROM facts WHERE entity_id=?", (name,))
            for attr_row in attrs:
                attr = attr_row["attribute"]
                history = self.get_history(name, attr)
                result.setdefault(name, {})[attr] = {
                    "value": (self.get_current(name, attr) or {}).get("value"),
                    "fact_id": (self.get_current(name, attr) or {}).get("fact_id"),
                    "history": [{"fact_id": row["fact_id"], "value": row["value"],
                                 "status": row["status"]} for row in history["facts"]],
                }
        return result
