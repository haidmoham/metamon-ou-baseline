import unittest
from ou_baseline.metrics import wilson, latency_summary, require_legal, summarize

class MetricsTests(unittest.TestCase):
    def test_empty(self):
        self.assertIsNone(wilson(0, 0))
        self.assertIsNone(latency_summary([])["median_ms"])
    def test_wilson(self):
        low, high = wilson(5, 10)
        self.assertAlmostEqual(low, 0.236593, places=5)
        self.assertAlmostEqual(high, 0.763407, places=5)
    def test_invalid_counts(self):
        with self.assertRaises(ValueError): wilson(2, 1)
    def test_latency(self):
        self.assertEqual(latency_summary(range(1, 21))["p95_ms"], 19)
    def test_mask(self):
        require_legal(2, [1, 2])
        with self.assertRaises(RuntimeError): require_legal(3, [1, 2])
        with self.assertRaises(RuntimeError): require_legal(0, [])
    def test_incomplete_not_loss(self):
        summary = summarize([{"opponent": "r", "outcome": "incomplete"}], [])
        self.assertEqual(summary["opponents"]["r"]["losses"], 0)
        self.assertIsNone(summary["opponents"]["r"]["win_rate"])
    def test_ties(self):
        summary = summarize([{"opponent": "r", "outcome": "tie"}], [])
        self.assertEqual(summary["opponents"]["r"]["completed"], 1)
        self.assertEqual(summary["opponents"]["r"]["ties"], 1)

if __name__ == "__main__": unittest.main()
