"""
Ring Attention TTFT benchmark for long-context RAG on Apple Silicon (MLX).

Builds a synthetic RAG prompt (many retrieved "documents" plus one hidden fact),
then measures time-to-first-token (TTFT) and peak memory for three prefill modes:

  chunked  - normal mlx-lm prefill (2048-token steps, fused attention kernel)
  full     - whole prompt in one pass, fused attention kernel
  ring     - whole prompt in one pass, attention computed as Ring Attention
             across N *simulated* devices (K/V blocks rotate, online softmax)

On one Mac the ring mode runs every simulated device one after another, so it
can't be faster. It shows the method is exact, and the script projects what
TTFT would be on N real devices.

Usage:
  .venv/bin/python ring_ttft_bench.py --lengths 4096 8192 16384 32768 --devices 4
"""

import argparse
import math
import random
import time

import mlx.core as mx
from mlx_lm import load
from mlx_lm.models import qwen3, llama
from mlx_lm.models.cache import make_prompt_cache

# ---------------------------------------------------------------- ring attention

RING = {"on": False, "devices": 4, "q_tile": 1024}
_original_sdpa = qwen3.scaled_dot_product_attention


def ring_attention(q, k, v, scale, n_dev, q_tile):
    """Exact causal attention, computed the way Ring Attention does it.

    The sequence is split into n_dev contiguous blocks. "Device" i keeps query
    block i and, at ring step s, holds K/V block (i - s) mod n_dev. Partial
    results are merged with a running max / running sum (online softmax).
    """
    B, H, L, D = q.shape
    Hkv = k.shape[1]
    rep = H // Hkv  # grouped-query attention: several query heads share one K/V head
    q = q.reshape(B, Hkv, rep, L, D)
    k = mx.expand_dims(k, 2)
    v = mx.expand_dims(v, 2)
    blk = math.ceil(L / n_dev)
    outs = []

    for i in range(n_dev):  # each simulated device
        q0, q1 = i * blk, min((i + 1) * blk, L)
        if q0 >= L:
            break
        for t0 in range(q0, q1, q_tile):  # tile queries to bound memory, as a flash kernel would
            t1 = min(t0 + q_tile, q1)
            qt = (q[..., t0:t1, :] * scale).astype(mx.float32)
            m = mx.full(qt.shape[:-1] + (1,), -mx.inf, dtype=mx.float32)
            l = mx.zeros_like(m)
            o = mx.zeros(qt.shape, dtype=mx.float32)

            for s in range(n_dev):  # ring steps: K/V block j arrives at device i
                j = (i - s) % n_dev
                if j > i:
                    continue  # causal: this block is entirely in the future, device idles
                k0, k1 = j * blk, min((j + 1) * blk, L)
                kj = k[..., k0:k1, :].astype(mx.float32)
                vj = v[..., k0:k1, :].astype(mx.float32)
                scores = qt @ kj.swapaxes(-1, -2)
                if j == i:  # diagonal block needs the causal mask
                    rows = mx.arange(t0, t1)[:, None]
                    cols = mx.arange(k0, k1)[None, :]
                    scores = mx.where(cols <= rows, scores, -mx.inf)
                m_new = mx.maximum(m, scores.max(axis=-1, keepdims=True))
                p = mx.exp(scores - m_new)
                corr = mx.exp(m - m_new)
                l = l * corr + p.sum(axis=-1, keepdims=True)
                o = o * corr + p @ vj
                m = m_new

            out = (o / l).astype(v.dtype)
            mx.eval(out)
            outs.append(out)

    out = mx.concatenate(outs, axis=-2)
    return out.reshape(B, H, L, D)


def patched_sdpa(queries, keys, values, cache, scale, mask, sinks=None):
    # Use ring attention only for a full prefill from an empty cache.
    if RING["on"] and queries.shape[2] > 1 and queries.shape[2] == keys.shape[2]:
        return ring_attention(queries, keys, values, scale, RING["devices"], RING["q_tile"])
    return _original_sdpa(queries, keys, values, cache=cache, scale=scale, mask=mask, sinks=sinks)


qwen3.scaled_dot_product_attention = patched_sdpa
llama.scaled_dot_product_attention = patched_sdpa

# ---------------------------------------------------------------- RAG prompt

CITIES = ["Lisbon", "Osaka", "Denver", "Nairobi", "Oslo", "Lima", "Hanoi", "Perth", "Quebec", "Tbilisi"]
ITEMS = ["copper wire", "solar panels", "coffee beans", "steel beams", "lithium cells", "wool", "rice", "glass"]
NEEDLE = "Internal memo: the launch code name for Project Atlas is CRIMSON-OTTER-7731."
QUESTION = "Using only the documents above, what is the launch code name for Project Atlas? Answer with the code name only."


def make_doc(rng, n):
    lines = [f"[Document {n}] Source: logistics report #{rng.randint(1000, 9999)}."]
    for _ in range(rng.randint(4, 8)):
        lines.append(
            f"The {rng.choice(CITIES)} warehouse received {rng.randint(10, 9000)} units of "
            f"{rng.choice(ITEMS)} on day {rng.randint(1, 365)}, handled by team {rng.randint(1, 80)}."
        )
    return " ".join(lines)


def build_prompt(tok, target_tokens, seed=0):
    rng = random.Random(seed)
    docs, n, total = [], 0, 0
    while total < target_tokens - 120:
        d = make_doc(rng, n)
        docs.append(d)
        total += len(tok.encode(d)) + 1
        n += 1
    docs.insert(len(docs) // 2, NEEDLE)  # hide the fact in the middle ("lost in the middle" test)
    context = "\n".join(docs)
    msgs = [{"role": "user", "content": f"Retrieved documents:\n{context}\n\n{QUESTION}"}]
    try:
        text = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False, enable_thinking=False)
    except TypeError:
        text = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    return mx.array(tok.encode(text))

# ---------------------------------------------------------------- runs


def prefill(model, ids, mode):
    cache = make_prompt_cache(model)
    RING["on"] = mode == "ring"
    if mode == "chunked":
        step = 2048
        for s in range(0, ids.size - 1, step):
            e = min(s + step, ids.size - 1)
            model(ids[None, s:e], cache=cache)
            mx.eval([c.state for c in cache])
        logits = model(ids[None, -1:], cache=cache)
    else:
        logits = model(ids[None], cache=cache)
    logits = logits[0, -1].astype(mx.float32)
    first = mx.argmax(logits)
    mx.eval(first, logits)
    RING["on"] = False
    return first, logits, cache


def decode(model, tok, first, cache, n=16):
    toks, y = [first.item()], first
    for _ in range(n - 1):
        y = mx.argmax(model(y.reshape(1, 1), cache=cache)[0, -1])
        toks.append(y.item())
        if toks[-1] in tok.eos_token_ids:
            break
    return tok.decode(toks).strip()


def run(model, tok, ids, mode):
    mx.clear_cache()
    mx.reset_peak_memory()
    t = time.perf_counter()
    first, logits, cache = prefill(model, ids, mode)
    ttft = time.perf_counter() - t
    peak = mx.get_peak_memory() / 1e9
    answer = decode(model, tok, first, cache)
    return ttft, peak, logits, answer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mlx-community/Qwen3-4B-4bit")
    ap.add_argument("--lengths", type=int, nargs="+", default=[4096, 8192, 16384, 32768])
    ap.add_argument("--devices", type=int, default=4, help="simulated ring size")
    ap.add_argument("--link-gbps", type=float, default=5.0, help="assumed GB/s between real devices (TB5 ~5)")
    args = ap.parse_args()
    RING["devices"] = args.devices

    print(f"Loading {args.model} ...")
    model, tok = load(args.model)
    cfg = model.args
    n_layers, n_kv, hd = cfg.num_hidden_layers, cfg.num_key_value_heads, getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads

    # Exactness check in fp32, attention only (no model rounding in the way)
    q, k, v = (mx.random.normal((1, h, 4096, 128)) for h in (32, 8, 8))
    ref = mx.fast.scaled_dot_product_attention(q, k, v, scale=128**-0.5, mask="causal")
    got = ring_attention(q, k, v, 128**-0.5, args.devices, RING["q_tile"])
    print(f"Attention exactness (fp32, 4096 tokens, {args.devices} devices): "
          f"max abs error {mx.abs(got - ref).max().item():.2e}")

    print("Warming up ...")
    warm = build_prompt(tok, 1024)
    for m in ("chunked", "full", "ring"):
        prefill(model, warm, m)

    N = args.devices
    rows = []
    for L in args.lengths:
        ids = build_prompt(tok, L)
        print(f"\n=== prompt: {ids.size} tokens ===")
        res = {}
        for mode in ("chunked", "full", "ring"):
            ttft, peak, logits, ans = run(model, tok, ids, mode)
            res[mode] = (ttft, peak, logits, ans)
            ok = "CRIMSON-OTTER-7731" in ans
            print(f"  {mode:8s} TTFT {ttft:7.2f}s  {ids.size / ttft:7.0f} tok/s  peak {peak:5.2f} GB  "
                  f"needle {'found' if ok else 'MISSED'}  -> {ans[:40]!r}")

        ref = res["full"][2]
        diff = mx.abs(res["ring"][2] - ref).max().item()
        noise = mx.abs(res["chunked"][2] - ref).max().item()
        same = mx.argmax(res["ring"][2]).item() == mx.argmax(ref).item()
        print(f"  ring vs full: max logit diff {diff:.4f} (chunked vs full, i.e. normal bf16 noise: {noise:.4f}), "
              f"same first token: {same}")

        # Projection for N real devices (estimate, not a measurement)
        t_full = res["full"][0]
        balanced = t_full / N                      # zig-zag / striped layout spreads causal work evenly
        contiguous = t_full * 2 / (N + 1)          # busiest device does N of the N(N+1)/2 blocks
        kv_bytes = 2 * n_layers * n_kv * ids.size * hd * 2  # K+V, fp16
        comm = kv_bytes * (N - 1) / N / (args.link_gbps * 1e9)
        print(f"  projected on {N} real devices: compute {balanced:.2f}-{contiguous:.2f}s, "
              f"K/V traffic per device {kv_bytes * (N - 1) / N / 1e9:.2f} GB ≈ {comm:.2f}s at {args.link_gbps} GB/s "
              f"(hidden only if ≤ compute); KV cache per device {kv_bytes / N / 1e9:.2f} GB vs {kv_bytes / 1e9:.2f} GB")
        rows.append((ids.size, res, diff, same, balanced, contiguous, comm))

    print("\nSummary (TTFT seconds)")
    print(f"{'tokens':>8} {'chunked':>8} {'full':>8} {'ring-sim':>9} {'proj ' + str(N) + 'dev':>10} {'comm':>6} {'logit diff':>11}")
    for n, res, diff, same, bal, con, comm in rows:
        proj = max(bal, comm)
        print(f"{n:>8} {res['chunked'][0]:>8.2f} {res['full'][0]:>8.2f} {res['ring'][0]:>9.2f} {proj:>10.2f} {comm:>6.2f} {diff:>11.4f}")


if __name__ == "__main__":
    main()
