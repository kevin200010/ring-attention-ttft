# Methodology

## Setup

| | |
|---|---|
| Hardware | Apple M5 Pro, 24 GB unified memory |
| Software | macOS, Python 3.14, MLX 0.32.4, mlx-lm 0.32.2 |
| Model | `mlx-community/Qwen3-4B-4bit` (36 layers, 32 query heads, 8 KV heads, head dim 128) |
| Decoding | Greedy, thinking disabled in the chat template |

## The RAG prompt

`rag_prompt.py` builds a realistic retrieval context: hundreds of short
"logistics report" documents with random cities, items, quantities and dates,
which makes them hard to compress or skim. One fact, the **needle**, is inserted
at a chosen depth (default: the middle, the hardest spot for many models):

> Internal memo: the launch code name for Project Atlas is CRIMSON-OTTER-7731.

The question at the end asks for the code name. Each mode's answer is checked
for the exact string. The prompt is deterministic for a given seed.

## What "TTFT" means here

Wall-clock time from the start of prefill until the first output token has been
computed (`mx.eval` on the argmax). Model loading, tokenization and network time
are excluded. Every mode is warmed up once on a short prompt first, and the MLX
memory cache is cleared before each run.

## Experiment 1: prefill modes on one Mac (`ring-ttft-bench`)

| Mode | What runs |
|---|---|
| chunked | mlx-lm's default prefill: 2,048-token steps, each attending to the growing KV cache with the fused kernel |
| full | the whole prompt in one forward pass with the fused kernel |
| ring | one pass with mlx-lm's attention swapped for `ring_attention` (N simulated devices run one after another) |

The ring mode is expected to be slower: it does the same total work as `full`
but one device at a time, with plain MLX operations instead of a fused kernel.
It shows that a ring-structured computation gives the same answer as normal
attention.

**Correctness signals:** the needle found in the answer, the same first token, and the max
absolute difference of final-token logits versus `full`. The chunked-vs-full
difference is reported as the "normal bf16 noise" baseline.

## Experiment 2: measured multi-device TTFT (`ring-ttft-proof`)

The question is what TTFT would be if N devices did this work at the same time. For each layer:

1. **Projection phase:** each device computes Q/K/V for its own tokens (`_project`).
2. **Attention + MLP phase:** each device attends with its queries over every K/V
   block it is allowed to see (received around the ring), then runs o_proj and the MLP.

Each device's phase is evaluated and timed **on its own** on the GPU. The ring
waits for the slowest device in each phase, so:

```
TTFT_overlap    = embed + Σ_layers [ max_d proj_d + max( max_d rest_d , transfer ) ]
TTFT_no_overlap = embed + Σ_layers [ max_d proj_d + max_d rest_d + transfer ]
transfer        = (K/V bytes a device receives per layer) / link bandwidth
```

The attention in phase 2 uses `mx.fast.scaled_dot_product_attention` over the
concatenated visible blocks. That matches the FLOPs and the output of the
block-by-block online-softmax merge, and uses the same fused kernel a real
implementation would use.

**Checks:**
- With N = 1 the harness reproduces real TTFT (27.9 s vs 30.9 s at 32k), so the
  timing method doesn't flatter itself.
- Final logits match the model's normal forward pass in every run (max diff
  0.125, below the 0.44–0.53 bf16 noise level), with the same first token.

## Experiment 3: real distributed attention (`ring-ttft-dist`)

One process per device, launched with `mlx.launch --backend ring`. Each rank
creates the same random Q/K/V (seeded), keeps only its shard, and runs
`ring_attention_distributed`: compute on the K/V it holds, then send it to rank+1
and receive from rank−1 (with ordering chosen to avoid deadlock), N−1 times.
Output is checked against the fused kernel on each rank's slice.

On one Mac all ranks share one GPU, so this experiment checks correctness and
measures **actual send/recv cost**. A speedup needs separate machines (see
[multi-mac.md](multi-mac.md)).

Local results (16k tokens, 32/8 heads, bf16, zigzag):

| Ranks | Attention time (slowest rank) | of which send/recv | K/V sent per rank | Max error vs fused |
|---|---|---|---|---|
| 1 | 805 ms | 0 ms | 0 MB | 1.6e-2 (bf16) |
| 2 | 832 ms | 23 ms | 34 MB | 1.6e-2 |
| 4 | 753 ms | 66 ms | 50 MB | 1.6e-2 |

## Precision note

MLX's Metal backend computes fp32 matrix multiplies with reduced internal
precision: a plain 256×256 `a @ b` differs from the CPU result by ~1e-2. So on the
GPU, ring attention and plain attention differ by ~1e-3 even in fp32, and **a
single-device run shows the same difference**. On CPU they agree to ~5e-7. The
unit tests check exactness on CPU and check on the GPU that the ring adds no error
beyond the single-device baseline.

## Threats to validity

- **Emulation, not a cluster.** Experiment 2 assumes devices start each layer
  together and adds nothing for synchronization or jitter.
- **Same-GPU timing of small pieces.** Very small per-device chunks (8 devices at
  8k = 1k tokens each) use the GPU less efficiently, which shows up as the lower
  8k efficiency (79%). On real hardware each device would have the same
  inefficiency, so this effect is realistic, not an artifact.
- **Bandwidth model.** 5 GB/s is an optimistic Thunderbolt 5 figure; 10 GbE is ~1.1 GB/s.
  Use `--link` / `--link-gbps` to test other values. At 32k and 4 devices,
  transfer only becomes the bottleneck below ~0.5 GB/s.
- **Single model and machine.** Larger models do more compute per token, which hides
  communication better. Smaller memory budgets favor ring attention more.
