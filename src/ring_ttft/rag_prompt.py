"""Synthetic long-context RAG prompts with a hidden fact ("needle")."""

from __future__ import annotations

import random
from dataclasses import dataclass

CITIES = ["Lisbon", "Osaka", "Denver", "Nairobi", "Oslo", "Lima", "Hanoi", "Perth", "Quebec", "Tbilisi"]
ITEMS = ["copper wire", "solar panels", "coffee beans", "steel beams", "lithium cells", "wool", "rice", "glass"]
NEEDLE_ANSWER = "CRIMSON-OTTER-7731"
NEEDLE = f"Internal memo: the launch code name for Project Atlas is {NEEDLE_ANSWER}."
QUESTION = ("Using only the documents above, what is the launch code name for Project Atlas? "
            "Answer with the code name only.")


@dataclass
class RagPrompt:
    text: str
    n_docs: int
    needle_index: int
    needle_depth: float


def make_doc(rng: random.Random, n: int) -> str:
    lines = [f"[Document {n}] Source: logistics report #{rng.randint(1000, 9999)}."]
    for _ in range(rng.randint(4, 8)):
        lines.append(
            f"The {rng.choice(CITIES)} warehouse received {rng.randint(10, 9000)} units of "
            f"{rng.choice(ITEMS)} on day {rng.randint(1, 365)}, handled by team {rng.randint(1, 80)}."
        )
    return " ".join(lines)


def build_rag_text(count_tokens, target_tokens: int, seed: int = 0, needle_depth: float = 0.5) -> RagPrompt:
    """Retrieved documents with the needle placed at `needle_depth` (0 = first, 1 = last).

    count_tokens: callable str -> int, so this works with any tokenizer.
    """
    if not 0.0 <= needle_depth <= 1.0:
        raise ValueError("needle_depth must be in [0, 1]")
    rng = random.Random(seed)
    docs, total = [], 0
    while total < target_tokens - 120:
        d = make_doc(rng, len(docs))
        docs.append(d)
        total += count_tokens(d) + 1
    idx = round(needle_depth * len(docs))
    docs.insert(idx, NEEDLE)
    context = "\n".join(docs)
    return RagPrompt(f"Retrieved documents:\n{context}\n\n{QUESTION}", len(docs) - 1, idx, needle_depth)


def build_prompt_ids(tokenizer, target_tokens: int, seed: int = 0, needle_depth: float = 0.5):
    """Token ids for a chat-formatted RAG prompt (thinking off for Qwen3-style templates)."""
    import mlx.core as mx

    rag = build_rag_text(lambda s: len(tokenizer.encode(s)), target_tokens, seed, needle_depth)
    msgs = [{"role": "user", "content": rag.text}]
    try:
        text = tokenizer.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False,
                                             enable_thinking=False)
    except TypeError:
        text = tokenizer.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    return mx.array(tokenizer.encode(text)), rag
