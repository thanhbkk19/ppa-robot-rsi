# Pushing the take-off frontier: P14 held-out and P15 (directed exploration)

## P14 held-out (distil lr 3e-4 vs 3e-5, replay base, seeds 3–7) — fails
| metric | lr 3e-4 | lr 3e-5 | paired diff [95% CI] |
|---|---|---|---|
| J_full | 0.344 | 0.368 | −0.024 [−0.119, +0.071] |
| J_target | 0.223 | 0.174 | +0.049 [−0.046, +0.144] |
| J_easy | 0.580 | 0.702 | −0.122 [−0.339, +0.095] |
| eval lift p95 | 0.173 | 0.140 | **+0.033 [+0.005, +0.061]** |

More plasticity does push the frontier higher (the lift interval excludes 0), but it trades it for
forgetting of easy goals, even with replay. The tune-seed gain on J_full (+0.083) did not replicate.

## P15 directed exploration along ∇_a Q̂ (on the lr 3e-4 base, seeds 0–2)
| config | J_easy | J_full | J_target | train lift r10 | eval lift p95 | collapses | generator goal slope | generator p90 up at h = 0.2 |
|---|---|---|---|---|---|---|---|---|
| base (lr 3e-4) | 0.75 | 0.370 | 0.205 | 0.119 | 0.155 | 0 | +0.010 | ≈ 0.09–0.10 |
| η = 0.1 | 0.60 | 0.337 | 0.237 | 0.157 | **0.217** | 0 | **+0.022** | 0.12–0.16 |
| η = 0.3 | 0.50 | 0.207 | 0.085 | 0.147 | 0.203 | 6 | +0.029 | 0.16 |

Verdicts:
- **P15a (mechanism: goal slope ≥ 2 × base): holds** (η = 0.1: +0.022 vs +0.010; η = 0.3: +0.029).
  Distilling the gradient-moved chunks makes the generator goal-conditioned for the first time, and its
  vertical reach widens.
- **P15b (J_target ≥ base + 0.05 and J_easy ≥ base − 0.05): fails.** η = 0.1 gives J_target +0.032 but
  J_easy −0.15. η = 0.3 takes too large a step: 6 collapses, and J_target drops.

## Pattern across P13–P15
Every intervention that moves the frontier up also costs easy-goal (table) competence:

| intervention | effect on the frontier | cost on easy goals |
|---|---|---|
| more plasticity | eval lift +3.3 cm (held-out) | J_easy −0.12 |
| gradient-directed exploration | eval lift +6 cm | J_easy −0.15 |
| replay | protects easy goals | does not move the frontier |

Mechanism:
- A single goal-conditioned generator still couples the goals: shifting its vertical behaviour up also
  shifts it at table goals (generator mean up at h = 0 rises with the frontier).
- Goal conditioning improves (slope +0.010 → +0.022), but it remains far too weak to separate "lift high
  for high goals" from "stay low for table goals".
- The step-size law again: η = 0.3 moves too far along the critic gradient and destabilises.

Next principled direction: break the goal coupling. Options:
- make the generator's goal dependence easy to learn (goal-relative input, e.g. goal − object);
- protect easy goals explicitly (goal-stratified replay weights);
- learn a residual "lift" policy conditioned on the goal-height gap, so improvement at high goals cannot
  leak to table goals.
