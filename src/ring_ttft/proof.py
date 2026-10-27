"""Experiment 2: measured multi-device TTFT (per-device timing on one GPU).

    ring-ttft-proof --lengths 8192 16384 32768 --devices 1 2 4 8
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import mlx.core as mx
from mlx_lm import load

from .comm import LINKS
from .envinfo import env_info
from .parallel import parallel_prefill
from .prefill import prefill
from .rag_prompt import build_prompt_ids


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="mlx-community/Qwen3-4B-4bit")
    ap.add_argument("--lengths", type=int, nargs="+", default=[8192, 16384, 32768])
    ap.add_argument("--devices", type=int, nargs="+", default=[1, 2, 4, 8])
    ap.add_argument("--layouts", nargs="+", default=["contiguous", "zigzag"], choices=["contiguous", "zigzag"])
    ap.add_argument("--link", default="thunderbolt5", choices=list(LINKS) + ["custom"])
    ap.add_argument("--link-gbps", type=float, help="override link bandwidth in GB/s")
    ap.add_argument("--out", default="results/parallel_emulation.json")
    args = ap.parse_args(argv)
    gbps = args.link_gbps or LINKS.get(args.link, 5.0)

    model, tok = load(args.model)
    warm, _ = build_prompt_ids(tok, 1024)
    parallel_prefill(model, warm, 2, "zigzag", gbps)
    prefill(model, warm, "full")

    runs = []
    for L in args.lengths:
        ids, _ = build_prompt_ids(tok, L)
        mx.clear_cache()
        t = time.perf_counter()
        _, ref, _ = prefill(model, ids, "full")
        t_real = time.perf_counter() - t
        _, chunked, _ = prefill(model, ids, "chunked")
        noise = mx.abs(chunked - ref).max().item()
        print(f"\n=== RAG prompt: {ids.size} tokens ===")
        print(f"  real one-pass TTFT: {t_real:.2f}s   bf16 noise (chunked vs full): {noise:.3f}")
        print(f"  {'layout':<11}{'dev':>4}{'TTFT':>9}{'no-overlap':>12}{'speedup':>9}{'balance':>9}"
              f"{'xfer':>7}{'KV/dev':>8}{'logit Δ':>9}  first")
        base = None
        for layout in args.layouts:
            for n in args.devices:
                if n == 1 and base is not None:
                    continue
                mx.clear_cache()
                logits, r = parallel_prefill(model, ids, n, layout, gbps)
                base = base or r.ttft_overlap_s
                diff = mx.abs(logits - ref).max().item()
                same = mx.argmax(logits).item() == mx.argmax(ref).item()
                first = tok.decode([mx.argmax(logits).item()])
                print(f"  {layout if n > 1 else 'single':<11}{n:>4}{r.ttft_overlap_s:>8.2f}s{r.ttft_no_overlap_s:>11.2f}s"
                      f"{base / r.ttft_overlap_s:>8.2f}x{r.balance:>9.2f}{r.transfer_s:>6.2f}s"
                      f"{r.kv_per_device_gb:>6.2f}GB{diff:>9.3f}  {first!r} {'ok' if same else 'DIFF'}")
                d = r.to_dict()
                d.update(speedup=base / r.ttft_overlap_s, logit_diff_vs_full=diff, same_first_token=same,
                         real_one_pass_ttft_s=t_real, bf16_noise=noise)
                runs.append(d)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"experiment": "parallel_emulation", "env": env_info(), "model": args.model,
                               "link": args.link, "link_gbps": gbps, "runs": runs}, indent=2))
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
