# Keep-top-K: random-keep-set control (directional null)

## Limitations (read first)

- Only 8 random draws per fraction (each draw is a full 7B forward-pass eval), far below the project's >=200-permutation standard. This is a DIRECTIONAL control, not a calibrated p-value.
- One model, one seed, one domain pair (`medicine` vs `cooking`), 15 held-out split-B prompts per domain. 
- Random sets are uniform over all 1,024 expert slots; layers left with fewer than top_k kept experts are unrestricted (same policy as the fair probe) and per-condition skipped-layer counts are in the JSON.

Baseline loss: target 2.7836, control 2.8341 nats.

## Results

| keep frac | n kept | selected loss (target) | random mean | random min..max | # random <= selected |
|---|---|---|---|---|---|
| 10% | 202/1024 | 7.0326 | 10.2821 | 9.8714..10.6951 | 0/8 |
| 20% | 305/1024 | 6.9828 | 8.5840 | 7.6667..9.0023 | 0/8 |
| 30% | 403/1024 | 5.5234 | 7.3764 | 6.7231..8.7121 | 0/8 |

## Reading this

- If the selected (lift + hot-core) set beats every random draw, the atlas ranking carries real signal for conditional compute even though the absolute loss cost is high.
- If random draws match or beat it, the lift ranking adds nothing usable at these fractions -- a negative result to report as such.
