# S1 CPU preview (Amendment R): seeds 10–19, 10 rounds, CPU container

Configs frozen from P20, so nothing was tuned. 30 runs (control, B3, D2 × 10 fresh seeds), all completed, none
failed (`scale/results/runs.csv`).

## Verdict: both pre-registered checks hold
| check | criterion | result |
|---|---|---|
| S1a: D2 vs control | J_full ≥ +0.05 with CI > 0, J_target ≥ +0.03, J_easy ≥ −0.05 | **holds**: J_full +0.223 [+0.122, +0.324], J_target +0.216, J_easy +0.224; all 10/10 seeds positive on all three |
| S1c: D2 vs B3 (the filter's own contribution) | J_full CI excludes 0 | **holds**: +0.141 [+0.030, +0.252], 8/10 seeds |

## Tables (round 10, paired 95% t-intervals over seeds)
| comparison | J_target | J_easy | J_full |
|---|---|---|---|
| D2 vs control | +0.216 [+0.105, +0.327] | +0.224 [+0.110, +0.338] | +0.223 [+0.122, +0.324] |
| D2 vs B3 | +0.141 [+0.031, +0.251] | +0.214 [+0.134, +0.294] | +0.141 [+0.030, +0.252] |
| B3 vs control | +0.075 [+0.002, +0.148] | +0.010 [−0.062, +0.082] | +0.082 [+0.029, +0.135] |

Absolute values (J_target / J_easy / J_full):

| method | J_target | J_easy | J_full |
|---|---|---|---|
| D2 | 0.336 | 0.788 | 0.473 |
| B3 | 0.195 | 0.574 | 0.332 |
| control | 0.119 | 0.564 | 0.250 |

## Reading
- The P20 effect replicates on 10 new seeds with almost the same size: J_full +0.223 here vs +0.216 on seeds 5–9.
- The filter's own contribution, not significant at 5 seeds (+0.102, CI [−0.022, +0.226]), is now significant:
  +0.141, CI [+0.030, +0.252].
- Balancing + larger step (B3) helps the frontier but not easy goals. The filter is what also lifts easy goals
  (+0.21 over B3), as the coupling mechanism predicts.

## Limits
- This is a preview of S1, not additional evidence on top of it: same seeds and code.
- Still untested and needing the GPU host: 20 rounds (persistence, S1b), 20 seeds, and 4× data / 2× width (S2).
- One task. The push pilot (P22) is negative, so the claim is scoped to loops whose critic selection already
  works.
