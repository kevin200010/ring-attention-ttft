"""
Proof-of-speedup harness for Ring Attention TTFT on one Mac.

In a real ring, N devices work at the same time, so prefill time is set by the
SLOWEST device, plus any K/V transfer time that compute can't hide.
This script runs every device's real share of the work (its chunk of tokens
through every layer: projections, attention against the K/V it would receive,
MLP) separately on the GPU and times it with the fused attention kernel.

  measured parallel TTFT = sum over layers of (slowest device's time per layer)
                           + K/V transfer time that compute can't hide

devices=1 is plain single-device prefill in the same code, which is the baseline.
The output logits are checked against the model's normal forward pass.

Layouts:
  contiguous - device i gets chunk i (later devices have more causal work)
  zigzag     - 2N chunks, device i gets chunks i and 2N-1-i (balanced work)
"""

import argparse
import time

import mlx.core as mx
from mlx_lm import load

from ring_ttft_bench import build_prompt, prefill


def owned_chunks(n_dev, layout):
    if layout == "contiguous":
        return n_dev, [[i] for i in range(n_dev)]
    return 2 * n_dev, [[i, 2 * n_dev - 1 - i] for i in range(n_dev)]


def timed(fn):
    t = time.perf_counter()
    out = fn()
    mx.eval(out)
    return out, time.perf_counter() - t


def parallel_prefill(model, ids, n_dev, layout, link_gbps):
    inner = model.model
    L = ids.size
    n_chunks, owners = owned_chunks(n_dev, layout)
    bounds = [round(c * L / n_chunks) for c in range(n_chunks + 1)]
    sl = [slice(bounds[c], bounds[c + 1]) for c in range(n_chunks)]

    h = {}
    t_embed = []
    for chunks in owners:
        out, t = timed(lambda: [inner.embed_tokens(ids[None, sl[c]]) for c in chunks])
        for c, x in zip(chunks, out):
            h[c] = x
        t_embed.append(t)

    critical = max(t_embed)
    critical_overlap = critical
    total_work = sum(t_embed)
    comm_total = 0.0

    for layer in inner.layers:
        a = layer.self_attn
        k, v, q = {}, {}, {}
        t_a = []
        for chunks in owners:  # phase A: each device projects its own tokens to Q/K/V
            def proj():
                res = []
                for c in chunks:
                    x = layer.input_layernorm(h[c])
                    B, n, _ = x.shape
                    qq = a.q_norm(a.q_proj(x).reshape(B, n, a.n_heads, -1)).transpose(0, 2, 1, 3)
                    kk = a.k_norm(a.k_proj(x).reshape(B, n, a.n_kv_heads, -1)).transpose(0, 2, 1, 3)
                    vv = a.v_proj(x).reshape(B, n, a.n_kv_heads, -1).transpose(0, 2, 1, 3)
                    off = bounds[c]
                    res.append((a.rope(qq, offset=off), a.rope(kk, offset=off), vv))
                return res
            out, t = timed(proj)
            for c, (qq, kk, vv) in zip(chunks, out):
                q[c], k[c], v[c] = qq, kk, vv
            t_a.append(t)

        t_b, t_attn, new_h = [], [], {}
        for chunks in owners:  # phase B: attention over received K/V blocks, then MLP
            def attn():
                res = []
                for c in chunks:
                    # K/V blocks 0..c: everything this query chunk may see (causal).
                    # A ring device gets these one block at a time; the fused kernel over
                    # the same blocks gives the same result and FLOPs.
                    K = mx.concatenate([k[j] for j in range(c + 1)], axis=2)
                    V = mx.concatenate([v[j] for j in range(c + 1)], axis=2)
                    o = mx.fast.scaled_dot_product_attention(q[c], K, V, scale=a.scale, mask="causal")
                    res.append(o)
                return res
            outs, ta = timed(attn)

            def rest():
                res = []
                for c, o in zip(chunks, outs):
                    B, _, n, _ = o.shape
                    x = h[c] + a.o_proj(o.transpose(0, 2, 1, 3).reshape(B, n, -1))
                    res.append(x + layer.mlp(layer.post_attention_layernorm(x)))
                return res
            hs, tr = timed(rest)
            for c, x in zip(chunks, hs):
                new_h[c] = x
            t_attn.append(ta)
            t_b.append(ta + tr)
        h = new_h

        # Each device receives every K/V block it doesn't own as the ring rotates.
        hd = k[0].shape[-1]
        own_max = max(sum(bounds[c + 1] - bounds[c] for c in ch) for ch in owners)
        comm = 0.0 if n_dev == 1 else (L - own_max) * 2 * a.n_kv_heads * hd * 2 / (link_gbps * 1e9)
        comm_total += comm

        step = max(t_a) + max(t_b)
        critical += step + comm                                     # transfer not hidden at all
        critical_overlap += max(t_a) + max(max(t_b), comm)          # transfer overlapped with attention
        total_work += sum(t_a) + sum(t_b)

    last = h[n_chunks - 1][:, -1:]
    logits = model.model.embed_tokens.as_linear(inner.norm(last)) if model.args.tie_word_embeddings else model.lm_head(inner.norm(last))
    logits = logits[0, -1].astype(mx.float32)
    mx.eval(logits)
    return logits, critical_overlap, critical, total_work, comm_total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mlx-community/Qwen3-4B-4bit")
    ap.add_argument("--lengths", type=int, nargs="+", default=[8192, 16384, 32768])
    ap.add_argument("--devices", type=int, nargs="+", default=[1, 2, 4, 8])
    ap.add_argument("--layouts", nargs="+", default=["contiguous", "zigzag"])
    ap.add_argument("--link-gbps", type=float, default=5.0, help="GB/s between devices (Thunderbolt 5 ~5)")
    args = ap.parse_args()

    model, tok = load(args.model)
    warm = build_prompt(tok, 1024)
    parallel_prefill(model, warm, 2, "zigzag", args.link_gbps)
    prefill(model, warm, "chunked")

    for L in args.lengths:
        ids = build_prompt(tok, L)
        print(f"\n=== RAG prompt: {ids.size} tokens ===")
        mx.clear_cache()
        t = time.perf_counter()
        _, ref, _ = prefill(model, ids, "full")
        t_full = time.perf_counter() - t
        _, chunked_logits, _ = prefill(model, ids, "chunked")
        noise = mx.abs(chunked_logits - ref).max().item()
        print(f"  real one-pass TTFT (normal mlx-lm forward): {t_full:.2f}s   [bf16 noise level: {noise:.3f}]")
        print(f"  {'layout':<11}{'devices':>7}{'TTFT overlap':>14}{'TTFT no-overlap':>17}{'speedup':>9}"
              f"{'K/V xfer':>10}{'logit diff':>12}{'1st token':>11}")
        base = None
        for layout in args.layouts:
            for n in args.devices:
                if n == 1 and layout != args.layouts[0]:
                    continue
                mx.clear_cache()
                logits, t_ov, t_no, work, comm = parallel_prefill(model, ids, n, layout, args.link_gbps)
                if n == 1:
                    base = t_ov
                diff = mx.abs(logits - ref).max().item()
                first = tok.decode([mx.argmax(logits).item()])
                same = mx.argmax(logits).item() == mx.argmax(ref).item()
                print(f"  {layout if n > 1 else 'single':<11}{n:>7}{t_ov:>13.2f}s{t_no:>16.2f}s{base / t_ov:>8.2f}x"
                      f"{comm:>9.2f}s{diff:>12.3f}{(repr(first) + (' ok' if same else ' DIFF')):>11}")


if __name__ == "__main__":
    main()
