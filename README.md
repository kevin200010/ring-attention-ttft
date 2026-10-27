# Ring Attention for long-context RAG: does it cut time-to-first-token?

An experiment on a single Apple Silicon Mac (M5 Pro, 24 GB) with a real LLM
(Qwen3-4B, 4-bit, MLX) and long RAG prompts (4k–32k tokens).

**Short answer:** Ring Attention cut TTFT by **3.8× on 4 devices and 7.3× on 8**
for a 32k-token RAG prompt, and the output was the same as normal prefill. On a single machine it
makes TTFT **worse**. The gain only comes from splitting the sequence across devices.

| 32k-token RAG prompt | TTFT | Speedup |
|---|---|---|
| 1 device (normal prefill) | 27.9 s | 1.0× |
| 2 devices, zigzag | 15.2 s | 1.8× |
| 4 devices, zigzag | 7.4 s | 3.8× |
| 8 devices, zigzag | 3.8 s | 7.3× |

> These multi-device numbers come from timing each device's real work on the
> Mac's GPU, not from a live multi-machine cluster. See [Method](#method) and [Limitations](#limitations).

---

## What is Ring Attention?

Ring Attention ([Liu, Zaharia & Abbeel, 2023](https://arxiv.org/abs/2310.01889)) computes
**exact** attention over very long sequences by splitting the sequence across N
devices arranged in a ring:

1. Each device keeps the queries (Q) for its chunk of tokens.
2. Key/value (K/V) blocks rotate around the ring, one hop per step.
3. Each device merges partial results with an online softmax (running max and running sum), as FlashAttention does.
4. After N steps, every device has the exact attention for its chunk.

While a device computes on one block, it is already sending the next, so communication
time is hidden behind compute. Memory per device is about 1/N of the sequence, so
the maximum context length grows with the number of devices.

TTFT for RAG is mostly **prefill**: the model reading the whole prompt. Attention
cost grows with the square of prompt length, and that is the part Ring Attention spreads across devices.

## Experiments

### 1. `ring_ttft_bench.py`: three prefill modes on one Mac

Builds a synthetic RAG prompt (hundreds of retrieved "documents" with one hidden
fact in the middle, `CRIMSON-OTTER-7731`), then measures TTFT, peak memory and
whether the model finds the hidden fact:

- **chunked**: normal mlx-lm prefill (2,048-token steps, fused attention kernel)
- **full**: the whole prompt in one forward pass
- **ring**: Ring Attention across N simulated devices, run one after another on the GPU

| Tokens | Chunked | Full | Ring (4 simulated, sequential) | Peak mem (chunked / full) |
|---|---|---|---|---|
| 4,086 | 1.56 s | 1.73 s | 3.95 s | 3.9 / 4.9 GB |
| 8,315 | 3.60 s | 3.99 s | 12.63 s | 4.3 / 7.0 GB |
| 16,416 | 9.21 s | 9.68 s | 37.95 s | 5.5 / 11.3 GB |
| 32,598 | 28.43 s | 30.22 s | 159.11 s | 7.9 / 19.7 GB |

Takeaways:
- All modes found the hidden fact and produced the same first token. Ring's logits differ from
  full prefill by about as much as chunked prefill does (bf16 rounding noise).
- Prefill throughput falls from ~2,600 to ~1,150 tokens/s between 4k and 32k: the quadratic attention cost shows up.
- On one machine, Ring Attention is slower. The simulated devices run one after another, and the
  ring loop is plain MLX operations, not a fused kernel. Memory doesn't drop either, because everything shares the same RAM.

Full output: [`results/single_mac_modes.txt`](results/single_mac_modes.txt)

### 2. `ring_parallel_proof.py`: measuring the multi-device speedup

In a real ring, all devices work at the same time, so TTFT is set by the **slowest
device**, plus any transfer time compute can't hide. This script runs each device's real
share of the work through all 36 layers (projections, attention against the K/V blocks it
receives using the fused kernel, MLP) and times it on the GPU:

```
parallel TTFT = sum over layers of (slowest device's time per layer)
              + K/V transfer time that compute can't hide (Thunderbolt 5, ~5 GB/s)
```

Running it with 1 device gives the baseline in the same code. It measured 27.9 s against
30.9 s for the real model at 32k, so the harness tracks reality. The output logits are checked against the
normal forward pass on every run.

TTFT in seconds (overlap = transfers hidden behind compute):

| Tokens | 1 dev | 2 dev zigzag | 4 dev zigzag | 8 dev zigzag | 8 dev contiguous |
|---|---|---|---|---|---|
| 8,315 | 3.75 | 1.92 (2.0×) | 1.04 (3.6×) | 0.59 (6.3×) | 0.65 (5.8×) |
| 16,416 | 9.82 | 4.95 (2.0×) | 2.56 (3.8×) | 1.36 (7.2×) | 1.71 (5.8×) |
| 32,598 | 27.92 | 15.22 (1.8×) | 7.42 (3.8×) | 3.83 (7.3×) | 5.63 (5.0×) |

Takeaways:
- **The output was the same in every run:** max logit difference 0.125, below bf16 rounding noise (~0.45), same first token.
- **Load balance matters.** With causal attention, the device holding the last chunk does the most work.
  Contiguous chunks reach only 5.0× on 8 devices. The **zigzag** layout (device *i* gets chunks *i* and *2N−1−i*)
  spreads the work evenly and reaches **7.3×**.
- **Communication isn't the bottleneck here.** About 0.8 s of K/V transfer per device at 32k over Thunderbolt-class links. Even with
  zero overlap, 4 devices at 32k come in at 8.1 s vs 27.9 s on one.
- **Memory per device falls to 1/N:** 1.2 GB of KV cache per device instead of 4.8 GB at 32k.

Full output: [`results/parallel_emulation.txt`](results/parallel_emulation.txt)

## Method

- Model: `mlx-community/Qwen3-4B-4bit` via `mlx-lm`, greedy decoding.
- Prompt: synthetic RAG context (logistics "documents") with one hidden fact inserted in the
  middle and a question at the end, in the model's chat template.
- TTFT = wall-clock time from the start of prefill to the first output token.
- Ring Attention correctness: an fp32 attention-only check against `mx.fast.scaled_dot_product_attention`,
  plus a logit comparison on the full model.
- Each device's attention is computed with the fused kernel over the K/V blocks it would receive.
  This matches the FLOPs and the result of the block-by-block online-softmax merge.

## Limitations

- **This is an emulation, not a live cluster.** Synchronization overhead, network jitter and
  uneven hardware on real machines would reduce the gains.
- **Each device needs its own copy of the model** (+2.3 GB per device here).
- **Thunderbolt is far slower than the GPU links** (NVLink, TPU ICI) Ring Attention was designed for. Real
  multi-Mac results should fall between the "overlap" and "no-overlap" columns in the raw output, at best.
- **It doesn't apply to hosted APIs.** Claude, OpenAI and other providers choose their own prefill parallelism.
  For API TTFT, the levers are prompt caching, fewer tokens, lower reasoning effort and smaller models.

## Reproduce

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

# single-Mac modes (downloads ~2.3 GB model on first run)
.venv/bin/python ring_ttft_bench.py --lengths 4096 8192 16384 32768 --devices 4

# measured multi-device speedup
.venv/bin/python ring_parallel_proof.py --lengths 8192 16384 32768 --devices 1 2 4 8
```

Requires an Apple Silicon Mac. Around 20 GB of free memory is needed for the 32k one-pass runs.

## Next steps

- [ ] Live 2-Mac run over Thunderbolt with MLX's distributed ring backend (`mlx.launch`)
- [ ] Multi-GPU cloud run with PyTorch `context_parallel` / `ring-flash-attention`
- [ ] Fused Metal kernel that returns the log-sum-exp, for a real blockwise ring loop
- [ ] Longer contexts (64k–128k) with RoPE scaling

## License

MIT
