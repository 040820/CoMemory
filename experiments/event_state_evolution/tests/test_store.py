import tempfile
import unittest
from pathlib import Path

from experiments.event_state_evolution.decision import Decision
from experiments.event_state_evolution.evidence import EvidenceCandidate
from experiments.event_state_evolution.store import StateStore


def evidence(key, chunk, date, value):
    return EvidenceCandidate(key, chunk, "D" + chunk, date, f"Alice works at {value}.",
                             "Alice", "works at", value, ("Alice", value))


def decision(kind, value, target=None, group=None):
    return Decision(kind=kind, entity_id="Alice", attribute="employer", value=value,
                    evidence_quote=f"Alice works at {value}." if value else "Alice corrected this.",
                    triple=("Alice", "works at", value or "B Co"),
                    detection="inferred" if kind == "change" else None,
                    corrected_fact_id=target, event_group_id=group)


class StoreTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.store = StateStore(Path(self.directory.name) / "state.sqlite")
        self.addCleanup(self.store.close)

    def test_baseline_change_and_idempotency(self):
        first = self.store.apply(evidence("one", "1", "2023-01-01", "B Co"),
                                 decision("baseline", "B Co"))
        self.assertEqual(self.store.get_history("Alice", "employer")["changes"], [])
        second = self.store.apply(evidence("two", "2", "2023-02-01", "C Co"),
                                  decision("change", "C Co"))
        self.assertEqual(self.store.apply(evidence("two", "2", "2023-02-01", "C Co"),
                                          decision("change", "C Co")), second)
        self.assertEqual(self.store.get_current("Alice", "employer")["value"], "C Co")
        changes = self.store.get_history("Alice", "employer")["changes"]
        self.assertEqual([(row["old_value"], row["new_value"], row["status"])
                          for row in changes], [("B Co", "C Co", "active")])
        self.assertEqual({row["chunk_id"] for row in self.store.get_by_fact(first)}, {"1"})
        self.assertTrue(self.store.get_by_fact(first)[0]["change_ids"])
        self.assertEqual(self.store.get_by_chunk("2")[0]["change_ids"],
                         self.store.get_by_fact(first)[0]["change_ids"])

    def test_historical_correction_preserves_later_current(self):
        first = self.store.apply(evidence("one", "1", "2023-01-01", "B Co"),
                                 decision("baseline", "B Co"))
        self.store.apply(evidence("two", "2", "2023-02-01", "C Co"), decision("change", "C Co"))
        self.store.apply(evidence("three", "3", "2023-03-01", "D Co"), decision("change", "D Co"))
        self.store.apply(evidence("four", "4", "2023-04-01", "A Co"),
                         decision("correction", "A Co", target=first))
        self.assertEqual(self.store.get_current("Alice", "employer")["value"], "D Co")
        changes = self.store.get_history("Alice", "employer")["changes"]
        self.assertIn(("B Co", "C Co", "revoked"),
                      [(r["old_value"], r["new_value"], r["status"]) for r in changes])
        self.assertIn(("A Co", "C Co", "active"),
                      [(r["old_value"], r["new_value"], r["status"]) for r in changes])

    def test_review_does_not_change_state(self):
        row = evidence("bad", "1", "2023-01-01", "B Co")
        self.assertIsNone(self.store.apply(row, Decision(kind="review", reason="ambiguous")))
        self.assertIsNone(self.store.get_current("Alice", "employer"))
        self.assertEqual(self.store.review_count(), 1)

    def test_same_chunk_can_change_two_attributes_in_one_event_group(self):
        self.store.apply(evidence("old1", "1", "2023-01-01", "B Co"),
                         decision("baseline", "B Co"))
        title_old = Decision(kind="baseline", entity_id="Alice", attribute="job_title",
                             value="Clerk", evidence_quote="Alice is Clerk.",
                             triple=("Alice", "is", "Clerk"))
        title_new = Decision(kind="change", entity_id="Alice", attribute="job_title",
                             value="Manager", evidence_quote="Alice is Manager.",
                             triple=("Alice", "is", "Manager"), detection="explicit",
                             event_group_id="promotion-1")
        old_title_evidence = EvidenceCandidate("old2", "1", "D1", "2023-01-01",
                                               "Alice is Clerk.", "Alice", "is", "Clerk",
                                               ("Alice", "Clerk"))
        self.store.apply(old_title_evidence, title_old)
        shared = EvidenceCandidate("shared", "2", "D2", "2023-02-01",
                                   "Alice works at C Co. Alice is Manager.", "Alice", "works at",
                                   "C Co", ("Alice", "C Co", "Manager"))
        self.store.apply(shared, decision("change", "C Co", group="promotion-1"))
        self.store.apply(shared, title_new)
        self.assertEqual(self.store.get_current("Alice", "job_title")["value"], "Manager")
        self.assertEqual(len(self.store.get_by_chunk("2")), 2)
        self.assertEqual(self.store.get_history("Alice", "employer")["changes"][-1]["event_group_id"],
                         "promotion-1")

    def test_failed_correction_rolls_back(self):
        first = self.store.apply(evidence("one", "1", "2023-01-01", "B Co"),
                                 decision("baseline", "B Co"))
        with self.assertRaises(ValueError):
            self.store.apply(evidence("bad", "2", "2023-02-01", "A Co"),
                             decision("correction", "A Co", target="missing"))
        self.assertEqual(self.store.get_current("Alice", "employer")["fact_id"], first)
        self.assertEqual(len(self.store.get_history("Alice", "employer")["facts"]), 1)

    def test_correcting_latest_value_without_replacement_leaves_current_unknown(self):
        self.store.apply(evidence("one", "1", "2023-01-01", "B Co"),
                         decision("baseline", "B Co"))
        latest = self.store.apply(evidence("two", "2", "2023-02-01", "C Co"),
                                  decision("change", "C Co"))
        correction = EvidenceCandidate("correction", "3", "D3", "2023-03-01",
                                       "Alice corrected this.", "Alice", "corrected", "this",
                                       ("Alice",))
        self.store.apply(correction, decision("correction", None, target=latest))
        self.assertIsNone(self.store.get_current("Alice", "employer"))
        self.assertIn(("B Co", "C Co", "revoked"),
                      [(r["old_value"], r["new_value"], r["status"])
                       for r in self.store.get_history("Alice", "employer")["changes"]])


if __name__ == "__main__":
    unittest.main()
