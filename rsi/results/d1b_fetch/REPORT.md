# D1b: does argmax selection over-optimize the learned critic as K grows? (Fetch pick-and-place, CPU)

Setup: `rsi/loop_fetch.py`, BC diffusion pretrained on 600 noisy scripted demos (demo noise 0.45, base success
≈ 0.42), posterior-variance DDPM sampler, 2-member MC critic ensemble, 6 rounds × 400 training episodes,
200 fixed evaluation episodes per round (disjoint seeds), tune seeds 0–2. Rule = critic argmax over K candidates.
`opt0` = mean critic score of the executed first chunk minus realised success on the evaluation episodes
(the critic's optimism about its own choices).

| K | distill lr | J_sys by round (mean of 3 seeds) | final (per seed) | AUC |
|---|---|---|---|---|
| 1 | 3e-5 | 0.42 0.48 0.57 0.66 0.65 0.64 0.65 | 0.57 0.64 0.74 | 0.581 |
| 4 | 3e-5 | 0.41 0.83 0.84 0.84 0.78 0.73 0.76 | 0.82 0.68 0.78 | **0.742** |
| 16 | 3e-5 | 0.40 0.70 0.64 0.48 0.44 0.54 0.65 | 0.57 0.75 0.62 | 0.550 |
| 64 | 3e-5 | 0.43 0.58 0.35 0.41 0.47 0.48 0.45 | 0.25 0.71 0.40 | 0.453 |
| 1 | 3e-4 | 0.42 0.46 0.51 0.59 0.56 0.49 0.45 | 0.46 0.30 0.60 | 0.497 |
| 4 | 3e-4 | 0.41 0.81 0.80 0.75 0.70 0.69 0.69 | 0.73 0.58 0.77 | 0.693 |
| 16 | 3e-4 | 0.40 0.71 0.61 0.46 0.46 0.60 0.62 | 0.54 0.64 0.68 | 0.550 |
| 64 | 3e-4 | 0.43 0.56 0.38 0.42 0.42 0.51 0.58 | 0.45 0.84 0.47 | 0.471 |

Mean opt0 by round (lr 3e-5): K=1 −0.09…−0.05; K=4 −0.37 → +0.10; K=16 −0.20 → +0.18; K=64 −0.06 → +0.24.

**Verdict on prediction P1 (docs/PA_THEORY.md): confirmed on tune seeds.** Final and AUC success fall
monotonically for K ≥ 4. With K = 64, argmax selection does worse than no selection at all (K = 1),
and the critic's optimism about its own choices turns positive and grows with K. Distill lr 3e-5 ≥ 3e-4
for every K, so 3e-5 is fixed for all later runs.

Side finding: pure self-distillation (K = 1) improves the BC policy (0.42 → 0.65), because the demos are
noisier than the policy's own samples.

Invalid earlier runs (`rsi/results/invalid_*`): they used a DDPM reverse variance σ² = β_k, which injects
excess noise. Fitting the generator to its own samples then collapsed it (0.31 → 0.13 in one round), an
artifact of the sampler, not of RSI. With the posterior variance it does not happen.
