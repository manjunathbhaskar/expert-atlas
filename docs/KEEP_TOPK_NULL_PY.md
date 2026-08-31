# Keep-top-K: random-keep-set control (directional null)

## Limitations (read first)

- Only 8 random draws per fraction (each draw is a full 7B forward-pass eval), far below the project's >=200-permutation standard. This is a DIRECTIONAL control, not a calibrated p-value.
- One model, one seed, one domain pair (`python` vs `history`), 15 held-out split-B prompts per domain. 
- Random sets are uniform over all 1,024 expert slots; layers left with fewer than top_k kept experts are unrestricted (same policy as the fair probe) and per-condition skipped-layer counts are in the JSON.

Baseline loss: target 2.0867, control 2.7021 nats.

## Results

| keep frac | n kept | selected loss (target) | random mean | random min..max | # random <= selected |
|---|---|---|---|---|---|
| 10% | 155/1024 | 6.9283 | 9.8971 | 9.0496..10.3977 | 0/8 |
| 20% | 246/1024 | 4.4804 | 9.3515 | 8.5961..9.9620 | 0/8 |
| 30% | 334/1024 | 3.4783 | 8.1352 | 7.5430..9.3316 | 0/8 |
| 40% | 422/1024 | 2.8029 | 7.4476 | 6.1951..8.7857 | 0/8 |
| 50% | 515/1024 | 2.4887 | 5.9025 | 3.9131..6.6460 | 0/8 |

## Reading this

- If the selected (lift + hot-core) set beats every random draw, the atlas ranking carries real signal for conditional compute even though the absolute loss cost is high.
- If random draws match or beat it, the lift ranking adds nothing usable at these fractions -- a negative result to report as such.
