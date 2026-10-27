"""Prefill modes for a real mlx-lm model, and TTFT measurement.

  chunked - mlx-lm's normal prefill: fixed-size steps, fused attention kernel
  full    - whole prompt in one forward pass, fused attention kernel
  ring    - whole prompt in one pass, attention replaced by the simulated ring
"""

from __future__ import annotations

import sys
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass

import mlx.core as mx
from mlx_lm.models.cache import make_prompt_cache

from .attention import ring_attention

MODES = ("chunked", "full", "ring")


@contextmanager
def ring_attention_enabled(n_dev: int, layout: str = "contiguous", q_tile: int = 1024):
    """Swap mlx-lm's attention for Ring Attention during a full prefill.

    mlx-lm model files import `scaled_dot_product_attention` by name, so the
    function is replaced in every loaded model module. Decode steps (one query)
    and chunked prefill (queries shorter than keys) use the original kernel.
    """
    originals = {
        name: mod.scaled_dot_product_attention
        for name, mod in list(sys.modules.items())
        if name.startswith("mlx_lm.models.") and hasattr(mod, "scaled_dot_product_attention")
    }

    def make_patched(orig):
        def patched(queries, keys, values, cache, scale, mask, sinks=None, **kw):
            if sinks is None and queries.shape[2] > 1 and queries.shape[2] == keys.shape[2]:
                return ring_attention(queries, keys, values, scale, n_dev, layout, q_tile)
            return orig(queries, keys, values, cache=cache, scale=scale, mask=mask, sinks=sinks, **kw)
        return patched

    for name, orig in originals.items():
        sys.modules[name].scaled_dot_product_attention = make_patched(orig)
    try:
        yield
    finally:
        for name, orig in originals.items():
            sys.modules[name].scaled_dot_product_attention = orig


def prefill(model, ids: mx.array, mode: str, n_dev: int = 4, layout: str = "contiguous",
            step: int = 2048):
    """Run prefill and return (first_token, last_logits_fp32, cache)."""
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    cache = make_prompt_cache(model)
    if mode == "chunked":
        for s in range(0, ids.size - 1, step):
            model(ids[None, s:min(s + step, ids.size - 1)], cache=cache)
            mx.eval([c.state for c in cache])
        logits = model(ids[None, -1:], cache=cache)
    elif mode == "full":
        logits = model(ids[None], cache=cache)
    else:
        with ring_attention_enabled(n_dev, layout):
            logits = model(ids[None], cache=cache)
    logits = logits[0, -1].astype(mx.float32)
    first = mx.argmax(logits)
    mx.eval(first, logits)
    return first, logits, cache


def greedy_decode(model, tokenizer, first: mx.array, cache, max_tokens: int = 16) -> str:
    toks, y = [first.item()], first
    eos = set(getattr(tokenizer, "eos_token_ids", []) or [tokenizer.eos_token_id])
    for _ in range(max_tokens - 1):
        if toks[-1] in eos:
            break
        y = mx.argmax(model(y.reshape(1, 1), cache=cache)[0, -1])
        toks.append(y.item())
    return tokenizer.decode(toks).strip()


@dataclass
class TTFTResult:
    mode: str
    tokens: int
    ttft_s: float
    tokens_per_s: float
    peak_gb: float
    answer: str
    needle_found: bool
    n_dev: int | None = None
    layout: str | None = None

    def to_dict(self):
        return asdict(self)


def measure_ttft(model, tokenizer, ids, mode: str, needle: str, n_dev: int = 4,
                 layout: str = "contiguous"):
    """TTFT = wall clock from start of prefill until the first token is computed."""
    mx.clear_cache()
    mx.reset_peak_memory()
    t0 = time.perf_counter()
    first, logits, cache = prefill(model, ids, mode, n_dev, layout)
    ttft = time.perf_counter() - t0
    peak = mx.get_peak_memory() / 1e9
    answer = greedy_decode(model, tokenizer, first, cache)
    res = TTFTResult(mode, ids.size, ttft, ids.size / ttft, peak, answer, needle in answer,
                     n_dev if mode == "ring" else None, layout if mode == "ring" else None)
    return res, logits
