# P13: stability of take-off (argmax K = 64, filtered init, curriculum + HER, 10 rounds, seeds 0–2)
| replay | gate | train lift r5 / r10 | eval lift p95 r10 | collapses | J_easy r0 → r10 | J_full r10 | J_target r10 | best J_full | gate accept |
|---|---|---|---|---|---|---|---|---|---|
| no | no (control) | 0.078 / 0.125 | 0.146 | 2 | 0.90 → 0.44 | 0.180 | 0.062 | 0.237 | – |
| **yes** | no | 0.079 / 0.106 | 0.147 | 1 | 0.90 → **0.63** | **0.287** | **0.085** | **0.297** | – |
| no | yes | 0.059 / 0.083 | 0.078 | 0 | 0.90 → 0.32 | 0.117 | 0.018 | 0.233 | 0.63 |
| yes | yes | 0.067 / 0.084 | 0.086 | 0 | 0.90 → 0.56 | 0.193 | 0.058 | 0.250 | 0.77 |

Verdicts (tune seeds):
- **P13a (replay: J_easy ≥ control + 0.15, collapses ≤ control): holds.** J_easy +0.19; collapses 1 vs 2.
  J_full also rises (+0.107).
- **P13b (gate: ≤ 1 collapse, final lift ≥ control − 0.01): fails.** There are 0 collapses, but the lift is
  −0.042.
- **P13c (replay + gate: J_full ≥ control + 0.05): fails.** The gain is +0.013; replay alone gives +0.107.

Mechanism (this round's training success per 2.5 cm height bin, mean of 3 seeds, bins 0–15 cm):

| | round 5 | round 10 |
|---|---|---|
| control | 0.35 0.26 0.21 0.21 0.17 0.06 | 0.46 0.46 0.34 0.27 0.11 0.18 |
| replay | 0.68 0.76 0.69 0.46 0.17 0.06 | 0.77 0.72 0.65 0.44 0.27 0.12 |

1. **Forgetting is the main loss, and replay removes most of it.** Without replay the whole competence
   profile sinks while the frontier moves: the low bins fall to 0.3–0.5. With replay, the bins below the
   frontier stay at 0.65–0.77, and the frontier bins are at least as good.
2. **The gate statistic measures the cause directly.** The paired difference new − old on fresh full-range
   episodes is negative in most rounds without replay (≈ −0.03 to −0.04; 63% accepted). So the round-only
   distillation update is, on average, a regression over the task range: progress at the frontier is paid
   for by forgetting. With replay the update is neutral to positive (mean ≈ +0.01; 77% accepted).
3. **Why the gate does not help.**
   - It can only reject a bad update, not produce a good one.
   - It spends half of the episode budget (200 training + 200 gate episodes), which slows the frontier.
   - It removes the collapses (0 vs 2), but the frontier progress lost to the halved training budget costs
     more than the collapses did.
4. Consistent with the step-size law: replay changes the update's composition (less forgetting, less noise),
   which raises the SNR of each step. That is the variable that has mattered throughout.

Novelty note: replay is a standard technique (accumulate vs replace, continual learning). The contribution is
the measurement:
- the paired gate statistic identifies forgetting as the cause of the take-off instability;
- replay alone raises full-range success from 0.18 to 0.29.

Next by protocol: held-out seeds 3–7 for replay vs control (the first hypothesis of this study that holds on
tune seeds).
