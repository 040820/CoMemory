import unittest

from dataset.locomo import extract_evidence_prefix


class LocomoConversionTests(unittest.TestCase):
    def test_evidence_keeps_multidigit_session_id(self):
        self.assertEqual(
            extract_evidence_prefix(["D1:3", "D10:8", "D10:14"]),
            ["D1", "D10"],
        )


if __name__ == "__main__":
    unittest.main()
