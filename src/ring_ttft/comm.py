"""Simple model of K/V traffic in a ring."""

from __future__ import annotations

from dataclasses import dataclass

# Rough sustained bandwidths in GB/s, for projections only.
LINKS = {
    "thunderbolt4": 2.5,
    "thunderbolt5": 5.0,
    "10gbe": 1.1,
    "nvlink4": 450.0,
}


def kv_bytes(tokens: int, n_kv_heads: int, head_dim: int, n_layers: int = 1, bytes_per_el: int = 2) -> int:
    """Bytes of K plus V for `tokens` tokens."""
    return 2 * tokens * n_kv_heads * head_dim * n_layers * bytes_per_el


@dataclass
class RingTraffic:
    bytes_received_per_device: int
    seconds: float

    def exposed(self, compute_seconds: float) -> float:
        """Transfer time left over once it overlaps with compute."""
        return max(0.0, self.seconds - compute_seconds)


def ring_traffic(seq_len: int, own_tokens: int, n_kv_heads: int, head_dim: int,
                 n_layers: int, link_gbps: float) -> RingTraffic:
    """Each device receives every K/V shard it doesn't own as the ring rotates."""
    b = kv_bytes(seq_len - own_tokens, n_kv_heads, head_dim, n_layers)
    return RingTraffic(b, b / (link_gbps * 1e9) if link_gbps > 0 else 0.0)
