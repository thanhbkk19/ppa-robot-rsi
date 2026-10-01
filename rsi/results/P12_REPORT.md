# P12: extrapolation across rounds (ExPO, idea C) on argmax K = 64, filtered init (take-off, 3 seeds, 10 rounds) — verdict: fails
Metric: training-episode object lift p95 per round (the frontier), plus stability and retention.

| α | lift r5 | lift r10 | mean lift r6–10 | best-round lift | collapses (> 3 cm drop between rounds) | round-to-round sd | table-goal success (cumulative) | target J (final) |
|---|---|---|---|---|---|---|---|---|
| 0 (control) | 0.096 | 0.087 | 0.083 | 0.116 | 3 | 0.029 | 0.40 / 0.45 / 0.47 | 0.018 |
| 0.5 | 0.079 | 0.105 | 0.101 | 0.125 | 3 | 0.040 | 0.36 / 0.44 / 0.27 | 0.030 |
| 1.0 | 0.046 | 0.077 | 0.063 | 0.107 | 5 | 0.041 | 0.21 / 0.20 / 0.29 | 0.018 |

P12 needed r5 ≥ control + 0.02 and r10 ≥ control + 0.03 without worse retention. α = 0.5 is slower early
(−0.017 at r5), only +0.018 later, and forgets more. α = 1.0 is worse and unstable.

Mechanism:
- ExPO assumes the round's update θ_r − θ_{r−1} points along the objective. Here each update is the sum of
  three parts:
  - the improvement direction (lift for in-air goals);
  - forgetting, because the curriculum concentrates the data at the frontier;
  - noise from the critic and the finite data.
- Extrapolation amplifies all three. It increases variance (round-to-round sd 0.029 → 0.041, more
  collapses) and forgetting more than it speeds up progress.
- This is the same law as the in-distribution K-inversion and the step-size results: the effective step
  must match the signal-to-noise ratio of the improvement direction.
  - Critic argmax over many candidates works for take-off because the critic's extrapolated direction is
    consistent (M1: Spearman +0.37 with lifting).
  - CFG fails because its direction has the wrong sign (M1: −0.08), so no step size helps.
  - ExPO fails because it enlarges the step along a low-SNR update.

What now limits take-off is **stability and retention**, not speed:
- the frontier oscillates (3 collapses even in the control);
- table-goal success decays as the frontier moves;
- goals beyond the frontier are out of distribution (M2).
