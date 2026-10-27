"""Measure multi-device Ring Attention TTFT on one machine.

In a real ring all N devices work at once, so prefill time is set by the
slowest device in each layer plus any K/V transfer time compute can't hide.
This module runs each device's real share of the work (its tokens through
every layer: Q/K/V projections, attention against the K/V it receives, MLP)
separately on the GPU and times it:

    parallel TTFT = embed + sum over layers of (max_d proj_d + max(max_d rest_d, transfer))

with a "no-overlap" variant that adds the transfer time instead of overlapping it.
Attention uses the fused kernel over the K/V blocks a device receives, which
gives the same result and FLOPs as merging them one block at a time.

Supports Llama/Qwen-style mlx-lm models (self_attn.{q,k,v,o}_proj, optional
q_norm/k_norm, rope; mlp; input/post_attention layernorms).
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field

import mlx.core as mx

from .attention import segment_len, segments
from .comm import ring_traffic


def _timed(fn):
    t = time.perf_counter()
    out = fn()
    mx.eval(out)
    return out, time.perf_counter() - t


@dataclass
class ParallelResult:
    n_dev: int
    layout: str
    tokens: int
    ttft_overlap_s: float
    ttft_no_overlap_s: float
    total_work_s: float
    transfer_s: float
    kv_per_device_gb: float
    kv_total_gb: float
    per_device_s: list[float] = field(default_factory=list)

    @property
    def balance(self) -> float:
        """Average device time / slowest device time (1.0 = perfect)."""
        return (sum(self.per_device_s) / len(self.per_device_s)) / max(self.per_device_s)

    def to_dict(self):
        d = asdict(self)
        d["balance"] = self.balance
        return d


def _project(layer, x, offset):
    a = layer.self_attn
    xn = layer.input_layernorm(x)
    B, n, _ = xn.shape
    q = a.q_proj(xn).reshape(B, n, a.n_heads, -1)
    k = a.k_proj(xn).reshape(B, n, a.n_kv_heads, -1)
    v = a.v_proj(xn).reshape(B, n, a.n_kv_heads, -1)
    if hasattr(a, "q_norm"):
        q, k = a.q_norm(q), a.k_norm(k)
    q, k, v = (t.transpose(0, 2, 1, 3) for t in (q, k, v))
    return a.rope(q, offset=offset), a.rope(k, offset=offset), v


def parallel_prefill(model, ids: mx.array, n_dev: int, layout: str = "zigzag",
                     link_gbps: float = 5.0):
    """Returns (last_token_logits_fp32, ParallelResult)."""
    inner = model.model
    L = ids.size
    dev_segs = segments(L, n_dev, layout)
    chunks = sorted(s for segs in dev_segs for s in segs)  # all segments in position order
    chunk_idx = {s: i for i, s in enumerate(chunks)}
    owners = [[chunk_idx[s] for s in segs] for segs in dev_segs]

    h, t_embed = {}, []
    for own in owners:
        out, t = _timed(lambda: [inner.embed_tokens(ids[None, chunks[c][0]:chunks[c][1]]) for c in own])
        h.update(zip(own, out))
        t_embed.append(t)

    per_dev = list(t_embed)
    overlap = no_overlap = max(t_embed)
    total = sum(t_embed)
    a0 = inner.layers[0].self_attn
    head_dim = None
    own_max = max(segment_len(s) for s in dev_segs)
    transfer_total = 0.0

    for layer in inner.layers:
        a = layer.self_attn
        q, k, v, t_proj = {}, {}, {}, []
        for own in owners:  # phase 1: each device projects its own tokens
            out, t = _timed(lambda: [_project(layer, h[c], chunks[c][0]) for c in own])
            for c, (qq, kk, vv) in zip(own, out):
                q[c], k[c], v[c] = qq, kk, vv
            t_proj.append(t)
        head_dim = head_dim or k[0].shape[-1]

        t_rest, new_h = [], {}
        for own in owners:  # phase 2: attention over received K/V, then MLP
            def work():
                res = []
                for c in own:
                    K = mx.concatenate([k[j] for j in range(c + 1)], axis=2)
                    V = mx.concatenate([v[j] for j in range(c + 1)], axis=2)
                    o = mx.fast.scaled_dot_product_attention(q[c], K, V, scale=a.scale, mask="causal")
                    B, _, n, _ = o.shape
                    x = h[c] + a.o_proj(o.transpose(0, 2, 1, 3).reshape(B, n, -1))
                    res.append(x + layer.mlp(layer.post_attention_layernorm(x)))
                return res
            out, t = _timed(work)
            new_h.update(zip(own, out))
            t_rest.append(t)
        h = new_h

        xfer = 0.0 if n_dev == 1 else ring_traffic(L, own_max, a.n_kv_heads, head_dim, 1, link_gbps).seconds
        transfer_total += xfer
        overlap += max(t_proj) + max(max(t_rest), xfer)
        no_overlap += max(t_proj) + max(t_rest) + xfer
        total += sum(t_proj) + sum(t_rest)
        for d in range(n_dev):
            per_dev[d] += t_proj[d] + t_rest[d]

    last = inner.norm(h[len(chunks) - 1][:, -1:])
    if getattr(model.args, "tie_word_embeddings", False):
        logits = inner.embed_tokens.as_linear(last)
    else:
        logits = model.lm_head(last)
    logits = logits[0, -1].astype(mx.float32)
    mx.eval(logits)

    n_layers = len(inner.layers)
    kv_total = 2 * L * a0.n_kv_heads * head_dim * n_layers * 2 / 1e9
    res = ParallelResult(n_dev, layout, L, overlap, no_overlap, total, transfer_total,
                         kv_total * own_max / L, kv_total, per_dev)
    return logits, res
