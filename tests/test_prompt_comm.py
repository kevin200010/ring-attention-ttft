"""RAG prompt builder and communication model (no model download needed)."""

import unittest

from ring_ttft.comm import kv_bytes, ring_traffic
from ring_ttft.rag_prompt import NEEDLE, QUESTION, build_rag_text


def words(s):
    return len(s.split())


class TestRagPrompt(unittest.TestCase):
    def test_reaches_target_length(self):
        rag = build_rag_text(words, 5000)
        self.assertGreaterEqual(words(rag.text), 4800)

    def test_needle_and_question_present_once(self):
        rag = build_rag_text(words, 3000)
        self.assertEqual(rag.text.count(NEEDLE), 1)
        self.assertTrue(rag.text.rstrip().endswith(QUESTION))

    def test_needle_depth(self):
        for depth in (0.0, 0.5, 1.0):
            rag = build_rag_text(words, 4000, needle_depth=depth)
            pos = rag.text.index(NEEDLE) / len(rag.text)
            self.assertAlmostEqual(pos, depth, delta=0.06)

    def test_deterministic(self):
        self.assertEqual(build_rag_text(words, 2000, seed=3).text, build_rag_text(words, 2000, seed=3).text)
        self.assertNotEqual(build_rag_text(words, 2000, seed=3).text, build_rag_text(words, 2000, seed=4).text)

    def test_bad_depth(self):
        with self.assertRaises(ValueError):
            build_rag_text(words, 1000, needle_depth=1.5)


class TestComm(unittest.TestCase):
    def test_kv_bytes(self):
        # Qwen3-4B: 36 layers, 8 KV heads, head dim 128, fp16 -> 147,456 bytes per token
        self.assertEqual(kv_bytes(1, 8, 128, 36), 147_456)

    def test_ring_traffic(self):
        t = ring_traffic(32768, 8192, 8, 128, 36, link_gbps=5.0)
        self.assertEqual(t.bytes_received_per_device, kv_bytes(24576, 8, 128, 36))
        self.assertAlmostEqual(t.seconds, t.bytes_received_per_device / 5e9)
        self.assertEqual(t.exposed(10.0), 0.0)
        self.assertGreater(t.exposed(0.1), 0.0)


if __name__ == "__main__":
    unittest.main()
