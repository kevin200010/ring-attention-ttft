"""Ring Attention must equal plain causal attention for every ring size and layout."""

import unittest

import mlx.core as mx

from ring_ttft.attention import (
    block_mask,
    causal_work,
    device_work,
    reference_attention,
    ring_attention,
    segment_len,
    segments,
)


def rand_qkv(L, H=8, Hkv=2, D=32, seed=0, dtype=mx.float32):
    mx.random.seed(seed)
    return tuple(mx.random.normal((1, h, L, D)).astype(dtype) for h in (H, Hkv, Hkv))


class TestLayouts(unittest.TestCase):
    def test_every_position_owned_exactly_once(self):
        for layout in ("contiguous", "zigzag"):
            for L in (1, 7, 100, 1001):
                for n in (1, 2, 3, 4, 8):
                    owned = sorted(p for segs in segments(L, n, layout) for a, b in segs for p in range(a, b))
                    self.assertEqual(owned, list(range(L)), f"{layout} L={L} n={n}")

    def test_zigzag_balances_causal_work(self):
        L, n = 4096, 8
        cont = device_work(L, n, "contiguous")
        zig = device_work(L, n, "zigzag")
        self.assertGreater(max(cont) / min(cont), 10)       # last device does ~15x the first
        self.assertLess(max(zig) / min(zig), 1.01)          # zigzag is nearly even
        self.assertEqual(sum(cont), sum(zig))               # same total work

    def test_device_work_matches_brute_force(self):
        for layout in ("contiguous", "zigzag"):
            segs = segments(53, 3, layout)
            brute = [causal_work(s, [(0, 53)]) for s in segs]
            self.assertEqual(brute, device_work(53, 3, layout))

    def test_segment_len(self):
        self.assertEqual(segment_len([(0, 3), (10, 12)]), 5)

    def test_bad_layout(self):
        with self.assertRaises(ValueError):
            segments(10, 2, "spiral")


class TestBlockMask(unittest.TestCase):
    def test_future_block_skipped(self):
        skip, _ = block_mask((0, 10), (10, 20))
        self.assertTrue(skip)

    def test_past_block_unmasked(self):
        skip, mask = block_mask((10, 20), (0, 10))
        self.assertFalse(skip)
        self.assertIsNone(mask)

    def test_diagonal_block_masked(self):
        skip, mask = block_mask((0, 4), (0, 4))
        self.assertFalse(skip)
        self.assertEqual(mask.sum().item(), 10)  # lower triangle of 4x4


class TestRingAttention(unittest.TestCase):
    """Exactness on CPU, where fp32 matmul is full precision (ring matches to ~5e-7).

    MLX's Metal fp32 matmul uses reduced internal precision (plain a @ b differs
    from CPU by ~1e-2 at 256 dims), so GPU checks live in TestRingAttentionGPU.
    """

    def setUp(self):
        self._dev = mx.default_device()
        mx.set_default_device(mx.cpu)

    def tearDown(self):
        mx.set_default_device(self._dev)

    def assert_close(self, a, b, tol=2e-5):
        err = mx.abs(a.astype(mx.float32) - b.astype(mx.float32)).max().item()
        self.assertLess(err, tol)

    def test_matches_reference_all_sizes_and_layouts(self):
        q, k, v = rand_qkv(257)  # odd length: uneven chunks
        ref = reference_attention(q, k, v, 32**-0.5)
        for layout in ("contiguous", "zigzag"):
            for n in (1, 2, 3, 4, 8):
                with self.subTest(layout=layout, n=n):
                    self.assert_close(ring_attention(q, k, v, 32**-0.5, n, layout), ref)

    def test_small_query_tiles(self):
        q, k, v = rand_qkv(300)
        ref = reference_attention(q, k, v, 32**-0.5)
        self.assert_close(ring_attention(q, k, v, 32**-0.5, 4, "zigzag", q_tile=17), ref)

    def test_more_devices_than_tokens(self):
        q, k, v = rand_qkv(5)
        ref = reference_attention(q, k, v, 32**-0.5)
        self.assert_close(ring_attention(q, k, v, 32**-0.5, 8, "zigzag"), ref)

    def test_multi_head_attention_no_gqa(self):
        q, k, v = rand_qkv(128, H=4, Hkv=4)
        ref = reference_attention(q, k, v, 32**-0.5)
        self.assert_close(ring_attention(q, k, v, 32**-0.5, 4), ref)

    def test_bfloat16_inputs(self):
        q, k, v = rand_qkv(256, dtype=mx.bfloat16)
        ref = reference_attention(q, k, v, 32**-0.5)
        out = ring_attention(q, k, v, 32**-0.5, 4, "zigzag")
        self.assertEqual(out.dtype, mx.bfloat16)
        self.assert_close(out, ref, tol=2e-2)

    def test_stats(self):
        q, k, v = rand_qkv(256)
        _, stats = ring_attention(q, k, v, 32**-0.5, 4, "contiguous", return_stats=True)
        self.assertEqual(stats.blocks_per_device, [1, 2, 3, 4])  # device i sees i+1 shards
        self.assertEqual(stats.skipped_per_device, [3, 2, 1, 0])
        self.assertGreater(stats.imbalance, 1.5)
        _, zs = ring_attention(q, k, v, 32**-0.5, 4, "zigzag", return_stats=True)
        self.assertLess(zs.imbalance, 1.05)


class TestRingAttentionGPU(unittest.TestCase):
    """On the GPU, ring vs reference is limited by Metal matmul precision, not the algorithm."""

    def err(self, a, b):
        return mx.abs(a.astype(mx.float32) - b.astype(mx.float32)).max().item()

    def test_gpu_ring_error_same_as_single_device(self):
        q, k, v = rand_qkv(257)
        ref = reference_attention(q, k, v, 32**-0.5)
        base = self.err(ring_attention(q, k, v, 32**-0.5, 1), ref)  # no ring at all
        for n in (2, 4, 8):
            self.assertLess(self.err(ring_attention(q, k, v, 32**-0.5, n, "zigzag"), ref), max(2 * base, 1e-4))

    def test_matches_fused_kernel(self):
        q, k, v = rand_qkv(512, D=64)
        fused = mx.fast.scaled_dot_product_attention(q, k, v, scale=64**-0.5, mask="causal")
        self.assertLess(self.err(ring_attention(q, k, v, 64**-0.5, 4, "zigzag"), fused), 5e-3)


if __name__ == "__main__":
    unittest.main()
