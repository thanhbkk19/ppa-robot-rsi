# P20: outcome-consistent distillation for self-improving goal-conditioned policies (2 Oct 2026)

## Problem
Testbed: Fetch pick-and-place take-off, CPU MuJoCo.
- Demonstrations have table goals only. Target goals are 0.10–0.30 m in the air.
- The base policy has 0.00 success on them.

Loop: sample K = 64 chunks from a diffusion policy → select with an MC critic → distil, with frontier
curriculum, hindsight relabelling (HER), replay and filtered-BC init. This is the held-out-confirmed P13 base.

Every earlier intervention that raised in-air success also lowered table-goal success (P14, P15, P17, P18/19).

## Diagnosis (pre-registered mechanism probes)
- **M8 (goal slope).** The generator ignores the goal height in every config: the vertical command differs
  by ≈ 0.01 between goal 0.30 m and goal 0 at the same state. Any lifting it learns is applied to table goals too.
- **Cause (Proposition 6, `docs/PA_THEORY.md`).** The distillation set holds each episode's selected actions
  under the commanded goal, *and* (HER) the same actions under the goal actually reached.
  - At the frontier most episodes fail (training success 0–20% for goals ≥ 10 cm).
  - So the same table-ending actions are labelled both "for goal 0.2 m" and "for the table goal".
  - The fitted conditional p(a | s, g) collapses towards the goal-marginal exactly where the frontier is.

## Method
Outcome-consistent distillation (`distill_filter="success"`):
- Keep commanded-goal rows only from successful episodes.
- Keep hindsight rows from all episodes.
- Every distilled row then pairs an action sequence with a goal it actually reached. This is the GCSL data
  condition, and the outcome filter of STaR, ReST and RAFT in LLM self-training.
- The critic still learns from all data, and selection is unchanged.

Combined with goal-height-balanced distillation weights and a larger distillation step (lr 3e-4), the
B3 config from P17, this gives D2.

## Results: tune seeds 0–2, round 10 (selection of D2 over D1 on J_full)
| config | J_target (10–30 cm) | J_easy (0–5 cm) | J_full (0–30 cm) |
|---|---|---|---|
| control (replay, K = 64, = best fixed K) | 0.085 | 0.630 | 0.287 |
| B3 = balanced + lr 3e-4 | 0.188 | 0.593 | 0.357 |
| D1 = filter, lr 3e-5 | 0.097 | 0.680 | 0.263 |
| **D2 = filter + balanced + lr 3e-4** | **0.497** | **0.857** | **0.630** |

## Held-out seeds 5–9: pre-registered test, single shot
| D2 vs | J_target | J_easy | J_full |
|---|---|---|---|
| control | **+0.226** [+0.021, +0.431] | **+0.152** [+0.066, +0.238] | **+0.216** [+0.090, +0.342] |
| B3 (same loop, no filter) | +0.103 [−0.172, +0.378] | +0.104 [+0.002, +0.206] | +0.102 [−0.022, +0.226] |

- Brackets are paired 95% t-intervals over 5 seeds.
- Absolute held-out values: D2 0.450 / 0.840 / 0.594; control 0.224 / 0.688 / 0.378.
- **Verdict: P20 confirmed.** All three pre-registered criteria hold and every CI against the control excludes 0.

## Mechanism check (tune-seed models, deployment-only probes)
| models | goal slope | gen up at h = 0 / 0.2 | generator alone (K = 1), in-air success |
|---|---|---|---|
| control | +0.011 | −0.005 / +0.002 | 0.00 |
| B3 | −0.009 | +0.031 / +0.024 | — |
| **D2** | **+0.141** | +0.038 / **+0.136** | **0.47** |

- The generator becomes goal-conditioned, with a slope 13× the control's.
- Improvement is carried by the generator itself, not only by critic selection. Every earlier config had
  0.00 in-air success at K = 1.
- Larger distillation steps then stop leaking to table goals, which is why lr 3e-4 helps only with the filter.

## Also learned on the way (pre-registered, see `rsi/PREREG.md` Amendments M–O)
- **FAS v1 (P18) fails.** This is a per-goal-bin bandit over the selection pressure K, motivated by Corollary 5
  (bound-optimal K grows with the squared critic SNR of the context). Its progress-proxy reward was gamed:
  K = 2 places the object accurately on the table and earns as much "progress" as lifting (M7).
- **FAS v2 (P19).** Success reward, large-K default where no success has been seen.
  - With balanced + qgrad: holds on tune seeds. On held-out it meets the point criteria, but the J_full CI
    includes 0, so it is not confirmed.
  - Its easy-goal protection is reliable held-out: J_easy +0.228, CI [+0.121, +0.335] vs the same loop
    without it.
- **Fixed K ∈ {4, 16} (P18b).** Protects easy goals but never takes off. The trade-off is monotone in K.
- The K-inversion persists in D2 (K = 4 beats K = 64 on both easy and in-air goals at deployment). Per-goal
  selection pressure on top of D2 is the obvious next test (P21, not yet run).

## Caveats
- 5 held-out seeds. The D2 vs B3 attribution is significant only for J_easy.
- One task, a CPU testbed, and 10 rounds of 400 episodes.
- The method-development search used about 15 configs on tune seeds 0–2 across P13–P20. The held-out test of
  D2 was run once. The comparator received the K and distil-lr tuning of earlier milestones.
- Seeds 3 and 4 were used for a mechanism diagnosis (M6), so held-out tests from P18 on use seeds 5–9.

## Addendum (3 Oct 2026): second task (push by direction, P22) — negative
- On FetchPush with front-only demos, back-goal targets and the same loop and hyper-parameters, every method
  degrades from round 0, and D2 = control.
- Cause: the base loop does not self-improve there. The critic has almost no ranking signal (deployment success
  is flat in K, opt0 is +0.17 to +0.54), and distilling its selections erodes the generator.
- Scope of the P20 claim: outcome-consistent distillation removes the frontier ↔ easy-goal coupling in loops
  whose critic selection already pushes the frontier. It does not substitute for a working selector.
