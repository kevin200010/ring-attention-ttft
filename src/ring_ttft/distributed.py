"""Ring Attention across real processes or machines with mlx.distributed.

Each rank holds Q/K/V only for its own segments. K/V shards travel around
the ring with point-to-point send/recv; each rank merges every shard it sees
into a running softmax. Launch with `mlx.launch` (see docs/multi-mac.md).

On one Mac, several ranks share the same GPU, so this checks correctness and
measures real communication cost, but won't show a speedup. Across Macs it
does both.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field

import mlx.core as mx

from .attention import OnlineSoftmax, Segment, block_mask, key_tiles, segment_len


@dataclass
class DistTimings:
    compute_s: list[float] = field(default_factory=list)
    comm_s: list[float] = field(default_factory=list)
    bytes_sent: int = 0

    @property
    def total_s(self) -> float:
        return sum(self.compute_s) + sum(self.comm_s)

    def to_dict(self):
        d = asdict(self)
        d["total_s"] = self.total_s
        return d


def _exchange(x: mx.array, rank: int, n: int, in_shape, group) -> mx.array:
    """Send x to rank+1 and receive the previous rank's shard, deadlock-free."""
    dst, src = (rank + 1) % n, (rank - 1) % n
    send_first = (rank % 2 == 0) if n % 2 == 0 else (rank == 0)
    if send_first:
        mx.eval(mx.distributed.send(x, dst, group=group))
        y = mx.distributed.recv(in_shape, x.dtype, src, group=group)
        mx.eval(y)
    else:
        y = mx.distributed.recv(in_shape, x.dtype, src, group=group)
        mx.eval(y)
        mx.eval(mx.distributed.send(x, dst, group=group))
    return y


def local_shard(x: mx.array, segs: list[Segment]) -> mx.array:
    """Concatenate this rank's segments along the sequence axis."""
    return mx.concatenate([x[..., a:b, :] for a, b in segs], axis=-2)


def ring_attention_distributed(q_loc, k_loc, v_loc, all_segs: list[list[Segment]], scale: float,
                               group=None, q_tile: int = 1024, k_tile: int = 4096):
    """Causal Ring Attention for this rank's queries.

    q_loc: (B, H, Lr, D) and k_loc/v_loc: (B, Hkv, Lr, D) for this rank's
    segments, concatenated in position order. all_segs: every rank's
    segments, from ring_ttft.attention.segments(). Returns (out_loc, DistTimings).
    """
    group = group or mx.distributed.init()
    rank, n = group.rank(), group.size()
    my_segs = all_segs[rank]
    B, H, _, D = q_loc.shape
    Hkv = k_loc.shape[1]
    rep = H // Hkv
    qg = q_loc.reshape(B, Hkv, rep, -1, D)

    tiles, off = [], 0  # (global start, global end, local offset)
    for a, b in my_segs:
        for t0 in range(a, b, q_tile):
            t1 = min(t0 + q_tile, b)
            tiles.append((t0, t1, off + (t0 - a)))
        off += b - a
    states = [OnlineSoftmax.empty(qg[..., lo:lo + (t1 - t0), :], v_loc.shape[-1]) for t0, t1, lo in tiles]
    q_scaled = [qg[..., lo:lo + (t1 - t0), :].astype(mx.float32) * scale for t0, t1, lo in tiles]

    kv = mx.stack([k_loc, v_loc])  # one message per hop
    src = rank
    tm = DistTimings()
    for step in range(n):
        t = time.perf_counter()
        k_sh = mx.expand_dims(kv[0], 2).astype(mx.float32)
        v_sh = mx.expand_dims(kv[1], 2).astype(mx.float32)
        for (t0, t1, _), st, qs in zip(tiles, states, q_scaled):
            # Evaluate one query tile at a time so only one score block is ever in memory.
            sh_off = 0
            for ka, kb in all_segs[src]:
                for kt, (a, b) in enumerate(key_tiles(ka, kb, k_tile)):
                    skip, mask = block_mask((t0, t1), (a, b))
                    if not skip:
                        lo = sh_off + (a - ka)
                        st.update(qs, k_sh[..., lo:lo + (b - a), :], v_sh[..., lo:lo + (b - a), :], mask)
                sh_off += kb - ka
            mx.eval(st.o, st.m, st.l)
        tm.compute_s.append(time.perf_counter() - t)

        if step < n - 1:
            nxt = (src - 1) % n  # the previous rank currently holds this shard
            in_shape = (2, B, Hkv, segment_len(all_segs[nxt]), D)
            tm.bytes_sent += kv.nbytes
            t = time.perf_counter()
            kv = _exchange(kv, rank, n, in_shape, group)
            tm.comm_s.append(time.perf_counter() - t)
            src = nxt

    out = mx.concatenate([s.finalize(q_loc.dtype) for s in states], axis=-2)
    return out.reshape(B, H, -1, D), tm
