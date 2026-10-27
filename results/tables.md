### Single Mac: prefill modes
| Tokens | chunked (mlx-lm) | full (one pass) | ring (4 sim.) | Peak memory (chunked / full) | Needle found |
|---|---|---|---|---|---|
| 4,086 | 1.56 s | 1.73 s | 3.95 s | 3.9 / 4.9 GB | yes, all modes |
| 8,315 | 3.60 s | 3.99 s | 12.63 s | 4.3 / 7.0 GB | yes, all modes |
| 16,416 | 9.21 s | 9.68 s | 37.95 s | 5.5 / 11.3 GB | yes, all modes |
| 32,598 | 28.43 s | 30.22 s | 159.11 s | 7.9 / 19.6 GB | yes, all modes |

### Measured multi-device TTFT
| Tokens | Layout | 1 dev | 2 dev | 4 dev | 8 dev |
|---|---|---|---|---|---|
| 8,315 | zigzag | 3.75 s | 1.92 s (1.9x) | 1.04 s (3.6x) | 0.59 s (6.3x) |
| 8,315 | contiguous | 3.75 s | 2.13 s (1.8x) | 1.19 s (3.2x) | 0.65 s (5.8x) |
| 16,416 | zigzag | 9.82 s | 4.95 s (2.0x) | 2.56 s (3.8x) | 1.36 s (7.2x) |
| 16,416 | contiguous | 9.82 s | 6.00 s (1.6x) | 3.20 s (3.1x) | 1.71 s (5.8x) |
| 32,598 | zigzag | 27.92 s | 15.22 s (1.8x) | 7.42 s (3.8x) | 3.83 s (7.3x) |
| 32,598 | contiguous | 27.92 s | 18.19 s (1.5x) | 10.10 s (2.8x) | 5.63 s (5.0x) |
