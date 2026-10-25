import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import networkx as nx

from experiments.event_state_evolution.build_index import index_memory
from experiments.event_state_evolution.retrieve import retrieve_with_state
from experiments.event_state_evolution.store import StateStore


class RuleLLM:
    """只代替外部模型服务；索引、事务和查询逻辑均使用真实代码。"""

    async def generate(self, prompt):
        if "当前雇主 employer" not in prompt or "Alice" not in prompt:
            return '{"kind":"ignore"}'
        if "Alice works at B Co." in prompt:
            return ('{"kind":"baseline","entity":"Alice","attribute":"employer",'
                    '"value":"B Co","evidence_quote":"Alice works at B Co.",'
                    '"triple":["Alice","works at","B Co"]}')
        if "Alice now works at C Co." in prompt:
            return ('{"kind":"change","entity":"Alice","attribute":"employer",'
                    '"value":"C Co","evidence_quote":"Alice now works at C Co.",'
                    '"triple":["Alice","works at","C Co"],"detection":"inferred"}')
        return '{"kind":"ignore"}'


class FakeRetriever:
    async def retrieve(self, query, top_k=None):
        return {"query": query, "query_analysis": {"entities": ["Alice"]},
                "evidence": [{"chunk_id": "1", "text": "Alice works at B Co."}]}


class WorkflowTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.store = StateStore(Path(self.directory.name) / "state.sqlite")
        self.addCleanup(self.store.close)
        graph = nx.DiGraph()
        graph.add_edge("Alice", "B Co", relation_name="works at", chunk_id="1")
        graph.add_edge("Alice", "C Co", relation_name="works at", chunk_id="2")
        self.memory = SimpleNamespace(graph_builder=SimpleNamespace(graph=graph), chunk_storage={
            "1": {"text": "Alice works at B Co.", "session_id": "D1", "session_time": "2023-01-01"},
            "2": {"text": "Alice now works at C Co.", "session_id": "D2", "session_time": "2023-02-01"},
        })

    async def test_index_replay_and_current_retrieval(self):
        counts = await index_memory(self.memory, self.store, RuleLLM(), "sample")
        self.assertEqual(counts["written"], 2)
        await index_memory(self.memory, self.store, RuleLLM(), "sample")
        self.assertEqual(len(self.store.get_history("Alice", "employer")["facts"]), 2)
        off = await retrieve_with_state("Where does Alice work?", FakeRetriever(), self.store,
                                        top_k=5, mode="off")
        self.assertEqual(off, await FakeRetriever().retrieve("Where does Alice work?", top_k=5))
        current = await retrieve_with_state("Where does Alice work?", FakeRetriever(), self.store,
                                            top_k=5, mode="current", entity_id="Alice",
                                            attribute="employer")
        self.assertEqual([row["value"] for row in current["state_evidence"]], ["C Co"])
        self.assertEqual(current["state_evidence"][0]["chunk_id"], "2")


if __name__ == "__main__":
    unittest.main()
