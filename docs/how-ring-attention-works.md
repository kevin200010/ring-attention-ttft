# How Ring Attention works (and how this repo implements it)

## The problem: prefill is quadratic

Time-to-first-token (TTFT) for a RAG request is dominated by **prefill**: one
forward pass over the entire prompt before the first output token appears.
Inside every layer, attention compares each token with every earlier token, so
its cost grows with the **square** of prompt length. The MLP and projections
grow only linearly.

You can see this in the single-Mac results: prefill throughput falls from
~2,600 tokens/s at 4k tokens to ~1,150 tokens/s at 32k.

Memory is the second problem. One device has to hold the K/V cache and
activations for the whole sequence: 4.8 GB of KV cache alone for Qwen3-4B at 32k
tokens.

## The idea

Split the sequence across N devices. Each device:

- keeps the **queries** for its own tokens, which never move
- starts with the **keys/values** for its own tokens
- passes K/V to the next device in a ring, one hop per step

```
step 0:  dev0 [KV0]   dev1 [KV1]   dev2 [KV2]   dev3 [KV3]
step 1:  dev0 [KV3]   dev1 [KV0]   dev2 [KV1]   dev3 [KV2]
step 2:  dev0 [KV2]   dev1 [KV3]   dev2 [KV0]   dev3 [KV1]
step 3:  dev0 [KV1]   dev1 [KV2]   dev2 [KV3]   dev3 [KV0]
           ↑ after N steps every device has seen every K/V block
```

While a device computes on the block it holds, it is already sending that block
on and receiving the next one. If compute per block takes longer than
transferring it, communication costs nothing extra.

## Exactness: the online softmax

Softmax needs a sum over **all** keys, but each device sees keys one block at a
time. The fix (the same one FlashAttention uses) is a running state per query row:

```
m  = running max of scores           (for numerical stability)
l  = running sum of exp(score - m)
o  = running sum of exp(score - m) · v

for each new block of scores S with values V:
    m_new = max(m, rowmax(S))
    l = l · exp(m - m_new) + rowsum(exp(S - m_new))
    o = o · exp(m - m_new) + exp(S - m_new) @ V
    m = m_new

output = o / l
```

The order of blocks doesn't matter, and the result equals standard attention up
to floating-point rounding. In this repo it's `OnlineSoftmax` in
[`attention.py`](../src/ring_ttft/attention.py). On CPU, ring and plain attention
agree to ~5e-7 for every ring size. See the tests.

## Causal masking and load balance

A decoder model uses a causal mask: token *p* attends only to tokens ≤ *p*. Each
(query block, key block) pair falls into one of three cases (`block_mask()`):

| Key block vs query block | Action |
|---|---|
| entirely in the future | **skip**: nothing to compute |
| entirely in the past | full block, no mask |
| overlapping (diagonal) | apply the triangular mask |

With **contiguous** chunks, device 0 holds the first tokens, so it skips almost
everything and sits idle. The last device holds the last tokens and must attend
to everything. With 8 devices the last one does ~15× the work of the first, and
the whole ring waits for it.

**Zigzag** splits the sequence into 2N chunks and gives device *i* chunks *i* and
*2N−1−i*, pairing a cheap early chunk with an expensive late one:

```
contiguous (N=4):   [ d0 ][ d1 ][ d2 ][ d3 ]
zigzag     (N=4):   [d0][d1][d2][d3][d3][d2][d1][d0]
```

Every device then does the same amount of work (to within 1%, see
`test_zigzag_balances_causal_work`). In the measured results, zigzag raises the
8-device speedup at 32k from **5.0× to 7.3×**.

## The three implementations in this repo

| Module | What it is | Used for |
|---|---|---|
| `attention.ring_attention` | All N devices simulated one after another in one process | Correctness; patching into a real model (`prefill.py`) |
| `parallel.parallel_prefill` | Each device's real work through the full model, timed separately | Measuring multi-device TTFT on one machine |
| `distributed.ring_attention_distributed` | One process per device, K/V passed with `mx.distributed.send/recv` | Running across real Macs |

## Further reading

- Liu, Zaharia, Abbeel. *Ring Attention with Blockwise Transformers for Near-Infinite Context* (2023). arXiv:2310.01889
- Brandon et al. *Striped Attention* (2023), another causal load-balancing layout. arXiv:2311.09431
- Dao et al. *FlashAttention* (2022): the online softmax and tiling. arXiv:2205.14135
- Jacobs et al. *DeepSpeed-Ulysses* (2023): sequence parallelism using all-to-all over attention heads. arXiv:2309.14509
