# P14: generator plasticity and sampling temperature (take-off, replay base, seeds 0–2, 10 rounds)
| config | J_easy | J_full | J_target | train lift r10 | eval lift p95 | collapses | generator goal slope (M5) | generator mean up at h = 0.2 |
|---|---|---|---|---|---|---|---|---|
| control (lr 3e-5, τ = 1) | 0.63 | 0.287 | 0.085 | 0.106 | 0.147 | 1 | +0.011 | +0.002 |
| **lr 3e-4, τ = 1** | **0.75** | **0.370** | **0.205** | 0.119 | 0.155 | 0 | +0.010 | **+0.031** |
| lr 3e-4, τ = 1.5 | 0.03 | 0.00 | 0.00 | 0.000 | – | – | – | – |
| lr 3e-5, τ = 1.5 (2 of 3 seeds; the third was stopped as uninformative) | 0.03 | 0.00 | 0.00 | 0.000 | – | – | – | – |

Verdicts:
- **P14a (plasticity): outcome met, predicted mechanism refuted.**
  - J_full +0.083 (criterion +0.05) and J_target +0.12.
  - The predicted mechanism (goal slope ≥ 3 × control) did not occur: the slope is unchanged (+0.010 vs
    +0.011).
  - What changed is a goal-independent upward shift of the generator (mean up +0.002 → +0.031). The tail
    picks reach further, and grasping and easy goals also improve (J_easy +0.12).
  - Because the conjunctive prediction failed, P14a is not counted as confirmed. A new outcome-only
    hypothesis goes to held-out (below).
- **P14b (temperature 1.5): fails, and the failure is informative.** Success on table goals is already
  0.07–0.09 at round 0, against 0.90 at τ = 1. Isotropic sampling noise destroys the precision-critical
  sub-skill (the grasp) before the critic can help, and distilling those noisy executions keeps the
  generator broken; this is the same mechanism as the early excess-variance-sampler artifact. Coverage has
  to be widened only along the task-relevant direction (here, vertical), not in every action dimension.
- **P14c:** not met (it collapses with τ = 1.5).

Remaining limit:
- The generator still does not use the goal height.
- At the frontier, the per-decision reach is set by a goal-blind spread (p90 ≈ 0.09–0.10).
- Next principled step: directed exploration. Widen or shift candidates along the critic's action
  gradient ∇_a Q̂ (value-guided denoising / classifier guidance) instead of isotropically.
