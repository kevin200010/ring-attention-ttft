# Results

| File | Experiment | Produced by |
|---|---|---|
| `bench_modes.json` | 1: chunked vs full vs ring prefill | `ring-ttft-bench` |
| `parallel_emulation.json` | 2: measured multi-device TTFT | `ring-ttft-proof` |
| `distributed_local_{1rank,2ranks,4ranks}.json` | 3: real multi-process attention on one Mac | `ring-ttft-dist` |
| `tables.md` | Markdown tables generated from the JSON | `ring-ttft-report` |
| `legacy_*.txt` | Original console logs for experiments 1 and 2 | first version of the scripts |

`bench_modes.json` and `parallel_emulation.json` were converted from the
`legacy_*.txt` logs. Those runs used the first version of the scripts, which used
the same algorithms now in `src/ring_ttft/`. Running `ring-ttft-bench` or
`ring-ttft-proof` overwrites them with fresh measurements.

Every JSON file includes an `env` block (chip, memory, OS, Python, MLX versions).
