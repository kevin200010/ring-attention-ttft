"""Experiment 1: chunked vs full vs ring prefill TTFT on one Mac.

    ring-ttft-bench --lengths 4096 8192 16384 32768 --devices 4
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mlx.core as mx
from mlx_lm import load

from .attention import reference_attention, ring_attention
from .envinfo import env_info
from .prefill import MODES, measure_ttft, prefill
from .rag_prompt import NEEDLE_ANSWER, build_prompt_ids


def exactness_check(n_dev: int, layout: str) -> float:
    """fp32, attention only: ring vs plain causal attention."""
    mx.random.seed(0)
    q, k, v = (mx.random.normal((1, h, 2048, 128)) for h in (32, 8, 8))
    ref = reference_attention(q, k, v, 128**-0.5)
    got = ring_attention(q, k, v, 128**-0.5, n_dev, layout)
    return mx.abs(got - ref).max().item()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="mlx-community/Qwen3-4B-4bit")
    ap.add_argument("--lengths", type=int, nargs="+", default=[4096, 8192, 16384, 32768])
    ap.add_argument("--devices", type=int, default=4, help="simulated ring size for ring mode")
    ap.add_argument("--layout", default="contiguous", choices=["contiguous", "zigzag"])
    ap.add_argument("--modes", nargs="+", default=list(MODES), choices=MODES)
    ap.add_argument("--needle-depth", type=float, default=0.5)
    ap.add_argument("--out", default="results/bench_modes.json")
    args = ap.parse_args(argv)

    err = exactness_check(args.devices, args.layout)
    print(f"Attention exactness (fp32, 2048 tokens, {args.devices} devices, {args.layout}): max abs error {err:.2e}")

    print(f"Loading {args.model} ...")
    model, tok = load(args.model)
    warm, _ = build_prompt_ids(tok, 1024)
    for m in args.modes:
        prefill(model, warm, m, args.devices, args.layout)

    runs = []
    for L in args.lengths:
        ids, rag = build_prompt_ids(tok, L, needle_depth=args.needle_depth)
        print(f"\n=== {ids.size} tokens, {rag.n_docs} documents, needle at depth {rag.needle_depth:.0%} ===")
        logits = {}
        for mode in args.modes:
            res, logits[mode] = measure_ttft(model, tok, ids, mode, NEEDLE_ANSWER, args.devices, args.layout)
            print(f"  {mode:8s} TTFT {res.ttft_s:7.2f}s  {res.tokens_per_s:7.0f} tok/s  peak {res.peak_gb:5.2f} GB  "
                  f"needle {'found' if res.needle_found else 'MISSED'}  -> {res.answer[:30]!r}")
            runs.append(res.to_dict())
        if "full" in logits:
            ref = logits["full"]
            for mode, lg in logits.items():
                if mode != "full":
                    d = mx.abs(lg - ref).max().item()
                    same = mx.argmax(lg).item() == mx.argmax(ref).item()
                    print(f"  {mode} vs full: max logit diff {d:.3f}, same first token: {same}")
                    runs[-len(logits) + list(logits).index(mode)]["logit_diff_vs_full"] = d

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"experiment": "bench_modes", "env": env_info(), "model": args.model,
                               "devices": args.devices, "layout": args.layout,
                               "exactness_fp32_max_err": err, "runs": runs}, indent=2))
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
