# Keep-top-K: random-keep-set control (directional null)

## Limitations (read first)

- Only 8 random draws per fraction (each draw is a full 7B forward-pass eval), far below the project's >=200-permutation standard. This is a DIRECTIONAL control, not a calibrated p-value.
- One model, one seed, one domain pair (`medicine` vs `cooking`), 15 held-out split-B prompts per domain. 
- Random sets are uniform over all 1,024 expert slots; layers left with fewer than top_k kept experts are unrestricted (same policy as the fair probe) and per-condition skipped-layer counts are in the JSON.

Baseline loss: target 2.7836, control 2.8341 nats.

## Results

| keep frac | n kept | selected loss (target) | random mean | random min..max | # random <= selected |
|---|---|---|---|---|---|
| 40% | 501/1024 | 4.3105 | 5.8752 | 4.7573..7.8270 | 0/8 |
| 50% | 593/1024 | 3.6366 | 5.0253 | 3.6296..6.1402 | 1/8 |
| 60% | 685/1024 | 3.1650 | 4.1854 | 3.5766..5.2463 | 0/8 |
| 70% | 774/1024 | 3.1384 | 3.8278 | 3.3728..4.6055 | 0/8 |

## Reading this

- If the selected (lift + hot-core) set beats every random draw, the atlas ranking carries real signal for conditional compute even though the absolute loss cost is high.
- If random draws match or beat it, the lift ranking adds nothing usable at these fractions -- a negative result to report as such.
