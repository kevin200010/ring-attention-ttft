# Running across multiple Macs

`ring-ttft-dist` runs real Ring Attention with one process per machine,
using MLX's `ring` backend (TCP over Ethernet or a Thunderbolt bridge).

## 1. Connect the Macs

Best: a direct **Thunderbolt cable** between them. macOS creates a
*Thunderbolt Bridge* network interface. Give each Mac a static IP on it
(System Settings → Network → Thunderbolt Bridge), for example `10.0.0.1` and `10.0.0.2`.
Ethernet or Wi-Fi also work, just more slowly.

## 2. Same environment on every Mac

```bash
git clone https://github.com/kevin200010/ring-attention-ttft
cd ring-attention-ttft
python3 -m venv .venv && .venv/bin/pip install -e .
```

Keep the repo at the **same absolute path** on each machine, and turn on
passwordless SSH from the launching Mac to the others (System Settings →
General → Sharing → Remote Login, then `ssh-copy-id`).

## 3. Hostfile

`hosts.json`:

```json
[
  {"ssh": "mac-a.local", "ips": ["10.0.0.1"]},
  {"ssh": "mac-b.local", "ips": ["10.0.0.2"]}
]
```

## 4. Run

```bash
# baseline on one machine
.venv/bin/ring-ttft-dist --seq-len 32768 --out results/dist_1mac.json

# two machines
.venv/bin/mlx.launch --backend ring --hostfile hosts.json \
    --python "$PWD/.venv/bin/python" -m ring_ttft.dist_bench \
    --seq-len 32768 --layout zigzag --out results/dist_2mac.json
```

Rank 0 prints the slowest rank's attention time, split into compute and
send/recv, plus the error versus the fused kernel. Compare the 1-Mac and 2-Mac
`wall_s`: with zigzag, the expected attention speedup is close to 2× when
send/recv time is small next to compute.

## Testing locally first

You can rehearse on one Mac with several processes (they share the GPU, so expect
no speedup, only correctness checks and real communication timing):

```bash
.venv/bin/mlx.launch -n 4 --backend ring --python "$PWD/.venv/bin/python" \
    -m ring_ttft.dist_bench --seq-len 16384
```

## Limits of the current distributed code

- It covers the **attention layer only**, not the whole model. Full-model
  distributed prefill would also shard the projections and MLP by token, which is
  straightforward because they're per-token, but isn't implemented yet.
- Communication and compute run one after the other, not overlapped. The printed
  send/recv time shows the upper bound of what overlapping could hide.
