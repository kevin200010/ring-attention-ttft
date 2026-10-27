"""Ring Attention core: sequence layouts, online-softmax merging, and a
single-process simulation of an N-device ring.

Shapes follow mlx-lm: queries (B, H, L, D), keys/values (B, Hkv, L, D), with
grouped-query attention when H > Hkv.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import mlx.core as mx

LAYOUTS = ("contiguous", "zigzag")

Segment = tuple[int, int]  # [start, end) token positions


# --------------------------------------------------------------------------- layouts


def segments(seq_len: int, n_dev: int, layout: str = "contiguous") -> list[list[Segment]]:
    """Which token positions each device owns.

    contiguous: N equal chunks, device i gets chunk i. With causal attention the
                last device does the most work (it attends to everything before it).
    zigzag:     2N chunks, device i gets chunks i and 2N-1-i, pairing a cheap
                early chunk with an expensive late one so every device does
                about the same amount of work.
    """
    if n_dev < 1:
        raise ValueError("n_dev must be >= 1")
    if layout == "contiguous":
        n_chunks, owners = n_dev, [[i] for i in range(n_dev)]
    elif layout == "zigzag":
        n_chunks, owners = 2 * n_dev, [[i, 2 * n_dev - 1 - i] for i in range(n_dev)]
    else:
        raise ValueError(f"unknown layout {layout!r}, expected one of {LAYOUTS}")
    bounds = [round(c * seq_len / n_chunks) for c in range(n_chunks + 1)]
    return [
        [(bounds[c], bounds[c + 1]) for c in sorted(own) if bounds[c + 1] > bounds[c]]
        for own in owners
    ]


def segment_len(segs: list[Segment]) -> int:
    return sum(b - a for a, b in segs)


def causal_work(q_segs: list[Segment], k_segs: list[Segment]) -> int:
    """Number of (query, key) pairs a causal mask leaves visible."""
    total = 0
    for qa, qb in q_segs:
        for ka, kb in k_segs:
            for qpos in range(qa, qb):
                total += max(0, min(kb, qpos + 1) - ka)
    return total


def device_work(seq_len: int, n_dev: int, layout: str) -> list[int]:
    """Visible attention pairs per device (cheap closed form, used for balance stats)."""
    segs = segments(seq_len, n_dev, layout)
    work = []
    for own in segs:
        w = 0
        for qa, qb in own:
            # query at position p sees p + 1 keys
            w += (qb * (qb + 1) - qa * (qa + 1)) // 2
        work.append(w)
    return work


# --------------------------------------------------------------------------- online softmax


@dataclass
class OnlineSoftmax:
    """Running state for one query tile: unnormalized output, running max, running sum."""

    o: mx.array
    m: mx.array
    l: mx.array
    blocks: int = field(default=0)

    @classmethod
    def empty(cls, q: mx.array, v_dim: int) -> "OnlineSoftmax":
        shape = q.shape[:-1]
        return cls(
            o=mx.zeros(shape + (v_dim,), dtype=mx.float32),
            m=mx.full(shape + (1,), -mx.inf, dtype=mx.float32),
            l=mx.zeros(shape + (1,), dtype=mx.float32),
        )

    def update(self, q_scaled: mx.array, k: mx.array, v: mx.array, mask: mx.array | None) -> None:
        scores = q_scaled @ k.swapaxes(-1, -2)
        if mask is not None:
            scores = mx.where(mask, scores, -mx.inf)
        m_new = mx.maximum(self.m, scores.max(axis=-1, keepdims=True))
        # Rows that have seen only masked keys keep m = -inf; guard the exp.
        safe_m = mx.where(mx.isinf(m_new), mx.zeros_like(m_new), m_new)
        p = mx.exp(scores - safe_m)
        corr = mx.exp(self.m - safe_m)
        self.l = self.l * corr + p.sum(axis=-1, keepdims=True)
        self.o = self.o * corr + p @ v
        self.m = m_new
        self.blocks += 1

    def finalize(self, dtype) -> mx.array:
        return (self.o / self.l).astype(dtype)


def key_tiles(ka: int, kb: int, k_tile: int) -> list[Segment]:
    """Split a key segment into tiles of at most k_tile keys (bounds score-block memory)."""
    return [(a, min(a + k_tile, kb)) for a in range(ka, kb, k_tile)]


def block_mask(q_seg: Segment, k_seg: Segment) -> tuple[bool, mx.array | None]:
    """(skip, mask) for a causal block. skip=True when every key is in the future."""
    qa, qb = q_seg
    ka, kb = k_seg
    if ka > qb - 1:
        return True, None  # whole block is in the future
    if kb - 1 <= qa:
        return False, None  # whole block is in the past: no mask needed
    rows = mx.arange(qa, qb)[:, None]
    cols = mx.arange(ka, kb)[None, :]
    return False, cols <= rows


# --------------------------------------------------------------------------- reference


def reference_attention(q: mx.array, k: mx.array, v: mx.array, scale: float) -> mx.array:
    """Plain causal attention in fp32 (materializes the full score matrix)."""
    B, H, L, D = q.shape
    rep = H // k.shape[1]
    kf = mx.repeat(k, rep, axis=1).astype(mx.float32)
    vf = mx.repeat(v, rep, axis=1).astype(mx.float32)
    scores = (q.astype(mx.float32) * scale) @ kf.swapaxes(-1, -2)
    mask = mx.arange(L)[None, :] <= mx.arange(L)[:, None]
    scores = mx.where(mask, scores, -mx.inf)
    return (mx.softmax(scores, axis=-1) @ vf).astype(q.dtype)


# --------------------------------------------------------------------------- simulated ring


@dataclass
class RingStats:
    n_dev: int
    layout: str
    work_per_device: list[int]
    blocks_per_device: list[int]
    skipped_per_device: list[int]

    @property
    def imbalance(self) -> float:
        """Busiest device's work / average work (1.0 = perfectly balanced)."""
        avg = sum(self.work_per_device) / len(self.work_per_device)
        return max(self.work_per_device) / avg if avg else 1.0


def _split_heads(q, k, v):
    B, H, L, D = q.shape
    Hkv = k.shape[1]
    if H % Hkv:
        raise ValueError(f"query heads {H} not divisible by kv heads {Hkv}")
    rep = H // Hkv
    return q.reshape(B, Hkv, rep, L, D), mx.expand_dims(k, 2), mx.expand_dims(v, 2)


def ring_attention(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    scale: float,
    n_dev: int = 4,
    layout: str = "contiguous",
    q_tile: int = 1024,
    return_stats: bool = False,
    k_tile: int = 4096,
):
    """Exact causal attention computed the way an N-device ring computes it.

    Device d keeps the queries for its segments. At ring step s it holds the
    K/V shard of device (d - s) mod N, scores its queries against that shard,
    and folds the result into a running softmax. After N steps every device
    has seen every shard. Here the devices run one after another in one
    process; see ring_ttft.distributed for the multi-process version.
    """
    B, H, L, D = q.shape
    qg, kg, vg = _split_heads(q, k, v)
    segs = segments(L, n_dev, layout)
    pieces: list[tuple[int, mx.array]] = []
    blocks, skipped = [0] * n_dev, [0] * n_dev

    for d in range(n_dev):
        for qa, qb in segs[d]:
            for t0 in range(qa, qb, q_tile):
                t1 = min(t0 + q_tile, qb)
                q_scaled = qg[..., t0:t1, :].astype(mx.float32) * scale
                state = OnlineSoftmax.empty(q_scaled, v.shape[-1])
                for s in range(n_dev):
                    src = (d - s) % n_dev  # whose K/V shard is here at step s
                    for ka, kb in segs[src]:
                        skip, mask = block_mask((t0, t1), (ka, kb))
                        if skip:
                            skipped[d] += 1
                            continue
                        for a, b in key_tiles(ka, kb, k_tile):
                            skip, mask = block_mask((t0, t1), (a, b))
                            if not skip:
                                state.update(
                                    q_scaled,
                                    kg[..., a:b, :].astype(mx.float32),
                                    vg[..., a:b, :].astype(mx.float32),
                                    mask,
                                )
                        mx.eval(state.o, state.m, state.l)
                        blocks[d] += 1
                out = state.finalize(q.dtype)
                mx.eval(out)
                pieces.append((t0, out))

    pieces.sort(key=lambda p: p[0])
    out = mx.concatenate([p for _, p in pieces], axis=-2).reshape(B, H, L, D)
    if return_stats:
        stats = RingStats(n_dev, layout, device_work(L, n_dev, layout), blocks, skipped)
        return out, stats
    return out
