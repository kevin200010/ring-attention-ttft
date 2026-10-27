"""Experiment 3: real distributed Ring Attention (attention layer) over mlx.distributed.

One process per device. On one Mac (shared GPU) it checks correctness and
measures real send/recv cost; across Macs it measures the real speedup.

    # 1 process baseline
    ring-ttft-dist --seq-len 16384
    # 2 local processes
    mlx.launch -n 2 --backend ring --python $(which python) -m ring_ttft.dist_bench --seq-len 16384
    # 2 Macs: see docs/multi-mac.md
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import mlx.core as mx

from .attention import segments
from .distributed import local_shard, ring_attention_distributed
from .envinfo import env_info

DTYPES = {"float32": mx.float32, "bfloat16": mx.bfloat16, "float16": mx.float16}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seq-len", type=int, default=16384)
    ap.add_argument("--heads", type=int, default=32)
    ap.add_argument("--kv-heads", type=int, default=8)
    ap.add_argument("--head-dim", type=int, default=128)
    ap.add_argument("--layout", default="zigzag", choices=["contiguous", "zigzag"])
    ap.add_argument("--dtype", default="bfloat16", choices=list(DTYPES))
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--no-check", action="store_true", help="skip the correctness check")
    ap.add_argument("--out", default=None, help="JSON path (written by rank 0)")
    args = ap.parse_args(argv)

    group = mx.distributed.init()
    rank, n = group.rank(), group.size()
    dt = DTYPES[args.dtype]
    L, scale = args.seq_len, args.head_dim**-0.5

    # Every rank builds the same tensors from the same seed, then keeps only its shard.
    mx.random.seed(0)
    q = mx.random.normal((1, args.heads, L, args.head_dim)).astype(dt)
    k = mx.random.normal((1, args.kv_heads, L, args.head_dim)).astype(dt)
    v = mx.random.normal((1, args.kv_heads, L, args.head_dim)).astype(dt)
    all_segs = segments(L, n, args.layout)
    my = all_segs[rank]
    q_loc, k_loc, v_loc = (local_shard(x, my) for x in (q, k, v))
    mx.eval(q_loc, k_loc, v_loc)

    ring_attention_distributed(q_loc, k_loc, v_loc, all_segs, scale, group)  # warm-up
    walls, timings = [], None
    for _ in range(args.repeats):
        mx.eval(mx.distributed.all_sum(mx.array(0.0), group=group))  # barrier
        t = time.perf_counter()
        out, timings = ring_attention_distributed(q_loc, k_loc, v_loc, all_segs, scale, group)
        mx.eval(out)
        walls.append(time.perf_counter() - t)
    wall = min(walls)
    wall_max = mx.distributed.all_max(mx.array(wall), group=group).item()  # slowest rank sets the time

    err = float("nan")
    if not args.no_check:
        errs = []
        off = 0
        for a, b in my:  # reference: fused causal kernel over all keys up to this segment
            ref = mx.fast.scaled_dot_product_attention(q[..., a:b, :], k[..., :b, :], v[..., :b, :],
                                                       scale=scale, mask="causal")
            got = out[..., off:off + (b - a), :]
            errs.append(mx.abs(got.astype(mx.float32) - ref.astype(mx.float32)).max())
            off += b - a
        err = mx.distributed.all_max(mx.max(mx.stack(errs)), group=group).item()

    comm = mx.distributed.all_max(mx.array(sum(timings.comm_s)), group=group).item()
    compute = mx.distributed.all_max(mx.array(sum(timings.compute_s)), group=group).item()
    if rank == 0:
        print(f"ranks={n} layout={args.layout} seq={L} heads={args.heads}/{args.kv_heads} dtype={args.dtype}")
        print(f"  attention time (slowest rank): {wall_max * 1000:.1f} ms  "
              f"[compute {compute * 1000:.1f} ms, send/recv {comm * 1000:.1f} ms]")
        print(f"  K/V sent per rank: {timings.bytes_sent / 1e6:.1f} MB over {n - 1} hops")
        if not args.no_check:
            print(f"  max abs error vs fused attention: {err:.2e}  ({'OK' if err < 5e-2 else 'CHECK'})")
        if args.out:
            p = Path(args.out)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps({"experiment": "distributed_attention", "env": env_info(), "ranks": n,
                                     "layout": args.layout, "seq_len": L, "heads": args.heads,
                                     "kv_heads": args.kv_heads, "head_dim": args.head_dim, "dtype": args.dtype,
                                     "wall_s": wall_max, "compute_s": compute, "comm_s": comm,
                                     "bytes_sent_per_rank": timings.bytes_sent, "max_abs_err": err}, indent=2))


if __name__ == "__main__":
    main()
