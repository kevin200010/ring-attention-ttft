"""Launch real multi-process rings on this machine and check the output."""

import os
import re
import shutil
import subprocess
import sys
import unittest
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
LAUNCH = shutil.which("mlx.launch") or str(Path(sys.executable).with_name("mlx.launch"))


@unittest.skipUnless(Path(LAUNCH).exists(), "mlx.launch not found")
class TestDistributedRing(unittest.TestCase):
    def run_ring(self, n, layout, seq_len=1000):
        env = dict(os.environ, PYTHONPATH=str(SRC))
        cmd = [LAUNCH, "-n", str(n), "--backend", "ring", "--python", sys.executable,
               "-m", "ring_ttft.dist_bench", "--seq-len", str(seq_len), "--heads", "8", "--kv-heads", "2",
               "--head-dim", "64", "--layout", layout, "--dtype", "float32", "--repeats", "1"]
        if n == 1:
            cmd = [sys.executable, "-m", "ring_ttft.dist_bench"] + cmd[cmd.index("--seq-len"):]
        out = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=180)
        self.assertEqual(out.returncode, 0, out.stdout + out.stderr)
        m = re.search(r"max abs error vs fused attention: ([0-9.e+-]+)", out.stdout)
        self.assertIsNotNone(m, out.stdout + out.stderr)
        return float(m.group(1))

    def test_single_process(self):
        self.assertLess(self.run_ring(1, "zigzag"), 5e-3)

    def test_two_ranks(self):
        for layout in ("contiguous", "zigzag"):
            with self.subTest(layout=layout):
                self.assertLess(self.run_ring(2, layout), 5e-3)

    def test_three_ranks_odd_ring(self):
        self.assertLess(self.run_ring(3, "zigzag", seq_len=999), 5e-3)


if __name__ == "__main__":
    unittest.main()
