# Pre-registration: held-out evaluation of pessimistic (χ²) amplification on Fetch pick-and-place

Written before any run on held-out seeds 3–7. Theory and predictions: `docs/PA_THEORY.md`.
Testbed and protocol: `rsi/loop_fetch.py`, with defaults fixed in D1b and kept unchanged for every run:
- demo noise 0.45, 600 demos
- distill lr 3e-5, ρ = 0.1
- 6 rounds × 400 training episodes
- 200 evaluation episodes per round, from evaluation seeds that are disjoint from training seeds

## Configs (selected on tune seeds 0–2 only)
Each family gets the same tuning budget: 3 configs at K = 64, selected by the higher mean final J_sys.
Argmax has no extra parameter; its budget is K ∈ {4, 16, 64} (D1b).
- chi2: β ∈ {0.02, 0.05, 0.1} → **β = 0.05** (0.910; 0.1 gave 0.905, 0.02 gave 0.823)
- softmax: temp ∈ {0.01, 0.03, 0.1} → filled in from `rsi/results/tune_fetch` before the held-out launch
- lcb: κ ∈ {1, 2, 4} → filled in from `rsi/results/tune_fetch` before the held-out launch

## Held-out grid (seeds 3, 4, 5, 6, 7)
- rules {argmax, chi2, softmax, lcb} × K ∈ {4, 16, 64}
- K = 1 (no selection)
- K = 256 for argmax and chi2 (scaling check)

The selected parameter is used unchanged for every K. Theory says β does not depend on K, so this also
tests that claim.

## Predictions and decision rule (final J_sys, paired by seed, 95% t-intervals)
- **P1 (argmax inversion):** mean final J of argmax at K = 64 is at least 0.10 below argmax at K = 4.
- **P2 (χ² monotone in K):** χ² final J at K = 64 ≥ χ² at K = 4 − 0.03, and χ² at K = 256 ≥ χ² at K = 64 − 0.03.
- **P3 (χ² beats tuned argmax):** χ² at K = 64 minus argmax at its best K (best by held-out mean, which favours
  argmax) ≥ +0.05, with the 95% interval of the paired difference above 0.
- **P4 (χ² vs other pessimism):** report χ² minus softmax and χ² minus LCB, each at its own best K. Claim
  "better" only if the interval excludes 0; otherwise report "comparable".

**GO for the RTX 5090 scale-up** (Robomimic Square, then Transport) if P1, P2 and P3 hold.
If P3 fails, the method is not better than tuning K, and we go back to Phase 2.

## Amendment A (written before any hard-regime run): K = 2 matches χ² in the easy regime; test the coverage-limited regime
Tune-seed facts that change the plan (easy regime: demo noise 0.45, base success ≈ 0.42):
- Argmax with K = 2 reaches 0.92 final, the same as χ² at K = 64 (0.91). K = 3 reaches 0.85.
- The χ² trust region at δ = 1.3 (the step size of argmax-of-4) gives 0.78 at K = 64, close to argmax-of-4
  (0.76). At δ = 0.5 it gives 0.89.
- Reading: in this regime the step size (χ² of the selection) decides the outcome, and a small K is enough
  coverage. P3 as written ("beats tuned argmax") is not expected to hold here, and we do not claim it.

The theory (`docs/PA_THEORY.md`, corollary to Lemma 2) predicts that decoupling coverage from step size helps
only when coverage is the bottleneck, i.e. good chunks are rare under μ.

**Hard regime:** demo noise 0.9 (base success ≈ 0.18). Everything else is unchanged and nothing is re-tuned:
β = 0.05, δ = 0.5 and softmax temp = 0.3 are carried over from the easy regime.

**P6 (pre-registered):** on seeds 0–2, then held-out seeds 3–7, AUC of χ² (β = 0.05, K = 64) ≥ AUC of
argmax at its best K ∈ {2, 4, 16, 64} + 0.05.
- If P6 fails: soft large-K selection has no advantage over small-K argmax on Fetch. The contribution then
  reduces to the K-inversion diagnosis plus "match the step size", and we go back to Phase 2.

## Amendment B (written before any lcbopt run): per-state certified-bound selection
Diagnosis from the ground-truth probe (`rsi/results/probe_fetch`, argmax K = 4, round 1, decision 4):
- Candidates are nearly equivalent in true value: spread 0.05, at the Monte-Carlo noise level 0.08.
- The critic predicts more spread than exists (0.086); Spearman correlation with the true values is 0.34.
- The gain over a random pick is −0.006, i.e. none.
- So selection matters only at a few states, and a fixed step selects on noise everywhere else.

Method (`ppa/select.py:lcb_opt_weights`, rule `lcbopt`): at each state, maximise Lemma 1's certified lower
bound w·q̄ − z·ε(s)·‖w‖₂ over the simplex.
- ε(s) is the mean disagreement of a 5-member bootstrap ensemble (members are fit on Poisson(1)-resampled
  episodes).
- The solution is w ∝ (q̄ − λ)₊ with ‖(q̄ − λ)₊‖₂ = z·ε(s). It is argmax where one candidate beats the rest by
  more than the noise, and near-uniform where the candidates are within the noise.

Tuning (seeds 0–2, easy regime, K = 64): z ∈ {0.5, 1, 2}. Baselines use the same 5-member bootstrap critic:
argmax at K ∈ {2, 4} (its own tuning budget).

**P7:** on held-out seeds 3–7, in both the easy and the hard regime, AUC of lcbopt (best z, K = 64) ≥ AUC of
argmax (same critic, best K ∈ {2, 4}) + 0.03, with the paired 95% interval above 0 in at least one regime.
- **Bug fix (before any valid lcbopt result):** the first implementation penalised z·ε·‖w‖₂ instead of
  z·ε·sqrt(χ²(w‖u)) = z·ε·sqrt(K‖w‖² − 1). At K = 64 that is a √K-times-weaker penalty, so it behaved like
  argmax (selection χ² 8–30). Those runs are kept under `rsi/results/invalid_lcbopt_missing_sqrtK`.
  The corrected solution: w ∝ (q̄ − λ)₊ with sd_k((q̄ − λ)₊) = z·ε(s), and uniform weights when
  sd_k(q̄) ≤ z·ε(s). The tuning grid is unchanged: z ∈ {0.5, 1, 2}.

## Amendment C (written before any RB run): Rao–Blackwellised distillation
Measured bottleneck of the best baseline (argmax K = 2): the generator does not absorb the system's
improvement. In the hard regime, J_gen at round r+1 is 0.08–0.10 below J_sys at round r, and J_gen plateaus
near 0.7.

Method (`rb=True`):
- After the critic refit, re-weight **all K candidates** of every visited state with the selection rule
  under the new critic.
- Distil 4 candidates per state drawn from those weights, stratified by state.
- The target distribution is the same as distilling the executed action. By Rao–Blackwell the variance is
  lower by up to K_eff, and only soft rules over many candidates benefit.

The distillation compute is unchanged (same steps and batch). Control: argmax K = 2 with `rb=True`
(re-selection with the new critic; there is no variance reduction for one-hot weights).

**P8:** on tune seeds 0–2, then held-out seeds 3–7, AUC of the best RB soft rule at K = 64 (χ² β = 0.05 or
χ²-TR δ = 0.5, chosen on seeds 0–2) ≥ AUC of the better of {argmax K = 2, argmax K = 2 + rb} + 0.03 in the
hard regime, and ≥ the same baseline − 0.01 in the easy regime.

## Verdicts so far (tune seeds 0–2; no held-out run was launched for a hypothesis that failed here)
| prediction | result | verdict |
|---|---|---|
| P1 argmax K-inversion | K = 2/4/16/64 → final 0.90/0.76/0.65/0.45 | holds (tune seeds) |
| P3 χ² beats tuned argmax | χ² 0.910 vs argmax K = 2 0.895 | fails (difference within noise) |
| P6 χ² wins when coverage is scarce | AUC 0.574 vs 0.630 | fails |
| P7 lcbopt beats argmax (same critic) | AUC 0.763 vs 0.808 | fails |
| P8 RB soft K = 64 beats argmax K = 2 (+rb) | hard: AUC 0.655 vs 0.663; easy: 0.809 vs 0.829 | fails |

RB re-selection helps every rule a little (argmax K = 2: +0.017 easy, +0.033 hard). That is an
engineering gain, not a new algorithm.

## Amendment D (written before any take-off method run): take-off beyond the demonstrations
Testbed: demos with table goals only (`demo_goals="table"`, demo noise 0.45). Target = goals 0.10–0.30 m in
the air, where the base success is 0.00 on all 3 seeds. Diagnosis (`rsi/results/takeoff_diag`): argmax K = 2
for 10 rounds stays at 0.00 target success, whether it trains on target goals or on all heights 0–0.3 m.

Infrastructure (standard methods, used as factors):
- frontier curriculum over goal height: p ∝ m(1 − m) + 0.02 per 2.5 cm bin (SEC / PLR);
- hindsight final-state relabelling for the critic and the distillation data (HER / GCSL).

Hypothesis linking the earlier findings: beyond the demonstrations, coverage is the bottleneck. Best-of-K
gain (1 − (1 − m)^K) − m peaks at m* = 1 − K^{−1/(K−1)} (0.50 for K = 2, 0.06 for K = 64). Large K is
therefore needed, and it is usable only with a small χ² step (χ²-TR), because argmax-of-64 over-optimises.

Runs (seeds 0–2, 10 rounds, train heights 0–0.3 m, evaluation on 0.10–0.30 m):
- E1 argmax K = 2 + curriculum
- E2 argmax K = 2 + HER
- E3 argmax K = 2 + curriculum + HER
- E4 χ²-TR δ = 0.5, K = 64 + curriculum + HER
- E5 argmax K = 64 + curriculum + HER

**P9a:** E3 takes off (mean final target success ≥ 0.10).
**P9b:** E4 target AUC ≥ E3 + 0.05, and E4 > E5.

## Amendment E (written after the take-off runs E1–E5 on seeds 0–2, before any `balanced` run)
Take-off facts (10 rounds):
- argmax K = 64 + curriculum + HER takes off on seed 0 (target success 0.21, object lift 19 cm) and starts
  on seed 2 (lift 6 cm at round 10). Seed 1 does not take off.
- Every small-step variant stays at 0 and sharpens back to table behaviour (the lift decreases): argmax K = 2
  with or without curriculum or HER, and χ²-TR δ = 0.5 at K = 64.
- In-distribution the opposite holds: argmax K = 2 is the best, and K = 64 over-optimises.

**Unified rule (`balanced`).** For an episode whose task has current success estimate m (per goal-height
bin, from past training rounds only), select by argmax over a random subset of K_eff(m) of the K = 64
candidates, where 1 − (1 − m)^{K_eff} = 1/2, clipped to [2, 64]. This is the p(1 − p)-maximising point for
the selected system. It reduces to argmax-of-2 where m ≥ 0.5 and to argmax-of-64 where m → 0. Without goal
bins, m is the previous round's training success. Nothing is tuned.

**P10:**
- (a) take-off regime: final target success ≥ argmax K = 64 (same curriculum + HER);
- (b) easy regime: final J ≥ argmax K = 2 − 0.02;
- (c) hard regime: AUC ≥ argmax K = 2 − 0.02.

All on seeds 0–2 first; held-out seeds 3–7 if (a)–(c) hold.

## Verdicts for P9–P10 (tune seeds 0–2)
| prediction | result | verdict |
|---|---|---|
| P9a curriculum + HER (argmax K = 2) takes off | 0.00 target success on 3/3 seeds; the lift shrinks | fails |
| P9b χ²-TR K = 64 beats argmax K = 2 in take-off | both 0.00 | fails |
| (exploratory) argmax K = 64 + curriculum + HER | seed 0: 0.21, seed 2: lift 6 cm, seed 1: none | 1/3 takes off |
| P10a balanced ≥ argmax K = 64 in take-off | 0.010 vs 0.072 | fails |
| P10b balanced ≈ argmax K = 2 in the easy regime | 0.912 vs 0.895 | holds |
| P10c balanced ≈ argmax K = 2 in the hard regime (AUC) | see final line below | fails |

P10c final (3 seeds): balanced AUC 0.595, final 0.697 vs argmax K = 2 AUC 0.630, final 0.753 -> fails.

## Amendment F (written before any CFG-RSI run): self-improving CFG in the take-off regime
Context: see `rsi/results/A1_REPORT.md`. The in-distribution regimes are explained by demo labels
(filtered BC ≈ 0.93), so the CFG-RSI test runs in the take-off regime (table demos, target goals
0.10–0.30 m, curriculum + HER, 10 rounds, seeds 0–2).

Methods:
- CFG-RSI (`rsi/cfg.py`), (w, v) ∈ {(0, 0), (1, 0), (2, 0), (3, 0), (2, 1)}.
- Baselines from filtered-BC init: argmax K = 2 and argmax K = 64 (curriculum + HER).

Predictions:
- **P11a:** some CFG config with w ≥ 2 reaches target success ≥ 0.05 at round 10 on ≥ 2 of 3 seeds.
- **P11b:** the best CFG config's mean final target success ≥ argmax K = 64 (filtered init).
- Mechanism, reported whatever the outcome: lift per round increases with w (guidance extrapolates);
  v > 0 keeps action diversity higher than v = 0 at the same w; the fraction of success labels and the
  conditional-vs-unconditional gap show whether the model learns the outcome structure.

## Amendment G (written before any expo run): extrapolation across rounds (idea C)
P11 failed (see `rsi/results/P11_REPORT.md`). Only critic-argmax with K = 64 moves the frontier, and its
frontier stalls near 10 cm. Method: after each distillation, θ ← θ + α(θ − θ_prev) (ExPO,
weak-to-strong extrapolation), applied on top of argmax K = 64, filtered init, curriculum + HER, 10 rounds,
seeds 0–2. α ∈ {0.5, 1.0}; the control α = 0 already exists (`takeoff_filt`).

**P12:** with the better α, mean train lift p95 at round 5 ≥ control + 0.02 m, and at round 10 ≥ control
+ 0.03 m, without lower table-goal success (frontier bin 0) than the control at round 10.
Mechanism to report: per-height success bins over rounds (frontier speed), table-goal retention, J_gen.

## Verdicts for P11–P12 (take-off, seeds 0–2)
| prediction | result | verdict |
|---|---|---|
| P11a CFG (w ≥ 2) takes off | 0 target success; lift ≤ 2.5 cm at every w | fails (M1: the CFG density ratio points down) |
| P11b best CFG ≥ argmax K = 64 (filtered init) | 0.000 vs 0.018 | fails |
| P12 ExPO speeds up the frontier | α = 0.5: r5 −0.017, r10 +0.018; α = 1.0 unstable | fails (amplifies a low-SNR update) |

## Amendment H (written before any stability run): replay and acceptance gate (P13)
Base: argmax K = 64, filtered-BC init, curriculum + HER, take-off regime, 10 rounds, seeds 0–2. Problems
measured on this base (P12 control): the frontier oscillates (3 collapses > 3 cm between rounds), table goals
are forgotten (cumulative 0.87 → 0.4–0.47), and goals above the frontier are out of distribution.

Factors (2 × 2, every run uses the same new evaluation sets):
- **replay** (`distill_data="replay"`): distil all rounds' executed chunks, not only the current round's.
  Rationale: accumulate-vs-replace (Gerstgrasser et al. 2024) and continual-learning replay against forgetting.
- **gate**: after distillation, run the new and the old generator (same refit critic) on the same 100 fresh
  initial states with goals uniform on 0–0.3 m. Keep the new one unless mean(Y_new − Y_old) <
  −1 · s.e. (paired). The episode budget is matched: gated runs use 200 training + 2 × 100 gate episodes per
  round (= 400).

New metrics: J_easy (goals 0–5 cm, 100 episodes), J_full (0–30 cm, 100 episodes), evaluation lift p95,
per-round per-height training success, and the number of lift collapses.

Predictions (3 seeds, means; control = no replay, no gate, rerun with the same code):
- **P13a** replay: J_easy at round 10 ≥ control + 0.15, and lift collapses ≤ control.
- **P13b** gate: lift collapses ≤ 1 in total, with mean final training lift ≥ control − 0.01.
- **P13c** replay + gate: final J_full ≥ control + 0.05.

## Verdicts for P13 (tune seeds 0–2) — see rsi/results/P13_REPORT.md
| prediction | result | verdict |
|---|---|---|
| P13a replay: J_easy ≥ control + 0.15, collapses ≤ control | 0.63 vs 0.44; 1 vs 2 | **holds** |
| P13b gate: ≤ 1 collapse and lift ≥ control − 0.01 | 0 collapses; lift 0.083 vs 0.125 | fails |
| P13c replay + gate: J_full ≥ control + 0.05 | 0.193 vs 0.180 | fails |

**Held-out test (written before running it):** replay vs control on seeds 3–7, same configuration.
Pass if the mean paired difference in J_easy at round 10 is ≥ +0.10 AND the mean paired difference in
J_full at round 10 is ≥ +0.05.

**Held-out result (seeds 3–7):** J_easy +0.330 [0.086, 0.574], J_full +0.200 [0.045, 0.355] → **passes**.
J_target +0.105 [−0.020, 0.230] (not a criterion).

## Amendment I (written before any P14 run): generator plasticity and sampling temperature
Context: `rsi/results/M3_M5_FRONTIER.md`. Base: the P13 replay configuration (argmax K = 64, filtered init,
curriculum + HER, replay, 10 rounds); control = `rsi/results/stability` replay runs, seeds 0–2.
Factors:
- **plasticity:** distil lr 3e-5 → 3e-4. Mechanism: the generator learns goal-height-dependent lifting.
- **temperature:** sampling temperature 1.0 → 1.5 (initial and per-step DDPM noise scaled), which widens the
  candidate spread. Mechanism: a larger reach per decision.

Grid: 2 × 2 with the control reused (3 new configs × seeds 0–2).
Predictions (means over seeds 0–2, round 10):
- **P14a** (plasticity): the generator's goal slope (M5: mean up at h = 0.30 minus at h = 0, held states) is
  ≥ 3 × the control's, and J_full ≥ control + 0.05.
- **P14b** (temperature): training lift p95 ≥ control + 0.03, and J_target ≥ control + 0.05.
- **P14c** (both): J_target ≥ control + 0.08.

Mechanism metrics are reported whatever the outcome: M4 lift per decision and M5 goal slope from the saved
final models.

## Verdicts for P14 (seeds 0–2) — see rsi/results/P14_REPORT.md
| prediction | result | verdict |
|---|---|---|
| P14a lr 3e-4: goal slope ≥ 3 × control AND J_full ≥ control + 0.05 | slope +0.010 vs +0.011; J_full 0.370 vs 0.287 | fails (outcome met, mechanism refuted) |
| P14b τ = 1.5: lift ≥ +0.03 and J_target ≥ +0.05 | collapses at round 0 (J_easy 0.07–0.09) | fails |
| P14c both: J_target ≥ +0.08 | collapse | fails |

**New outcome-only hypothesis (written before running it):** distil lr 3e-4 vs 3e-5 on the replay take-off
base, held-out seeds 3–7. The control reuses `rsi/results/stability_heldout` (replay, lr 3e-5, same
configuration and seeds). Pass if the mean paired difference in J_full ≥ +0.05 AND in J_target ≥ +0.05 at
round 10.

## Amendment J (written before any qgrad run): directed exploration along the critic's action gradient (P15)
Motivation (P14): isotropic widening breaks the grasp, and the generator stays goal-blind. Method
(`qgrad_eta`):
- Every candidate is moved one step of L2 length η along ∇_a Q̂(s, a). This is the DPG action-improvement
  step (Silver et al. 2014), i.e. classifier/value guidance applied to the samples.
- The moved version is executed where the critic scores it higher, then argmax over K = 64 as before.
- Exploration then widens only along the direction the goal-aware critic prefers. Distilling the moved
  chunks gives goal-dependent targets.

Base: the P14 best configuration (replay, distil lr 3e-4, argmax K = 64, filtered init, curriculum + HER,
10 rounds, seeds 0–2). Tuning budget: η ∈ {0.1, 0.3}.
Predictions (round 10, means over seeds 0–2, versus the base `rsi/results/frontier_p14` lr 3e-4 τ = 1):
- **P15a** (mechanism): the generator goal slope (M5) is ≥ 2 × the base (base +0.010).
- **P15b** (outcome): J_target ≥ base + 0.05 and J_easy ≥ base − 0.05, for the better η.

Report the risk regardless of outcome: the gradient step is a larger selection step along a learned critic
(the step-size law). Track opt0-style optimism, collapses and J_easy.

## Verdicts: P14 held-out and P15
| prediction | result | verdict |
|---|---|---|
| lr 3e-4 held-out: J_full ≥ +0.05 and J_target ≥ +0.05 | J_full −0.024, J_target +0.049 (lift +0.033, CI excludes 0) | fails (frontier up, easy goals forgotten) |
| P15a qgrad: goal slope ≥ 2 × base | η = 0.1: +0.022 vs +0.010 | **holds** |
| P15b qgrad: J_target ≥ +0.05 and J_easy ≥ −0.05 | η = 0.1: +0.032 / −0.15 | fails |

## Amendment K (written before any goal-relative run): break the goal coupling (P16)
Diagnosis (P13–P15): every intervention that raises the frontier also costs easy-goal success. A single
generator with weak goal dependence shifts its vertical behaviour at every goal height together.

Method:
- Add the goal relative to the object (desired − achieved goal, 3 dims) to the inputs of the generator and
  the critic (`RSI_GOALREL=1`).
- "Move the object toward the goal" then has the same form at every height (translation invariance in z),
  so goal-dependent behaviour should be easier to represent and less likely to interfere across heights.
- Combined with the critic-gradient step η = 0.1, the only intervention shown to induce goal dependence (P15a).

Base: the held-out-confirmed configuration (replay, distil lr 3e-5, argmax K = 64, filtered init,
curriculum + HER, 10 rounds), seeds 0–2. Control = `rsi/results/stability` replay runs:
J_easy 0.63, J_full 0.287, J_target 0.085.
Runs:
- G1 goal-relative;
- G2 goal-relative + qgrad η = 0.1;
- G3 qgrad η = 0.1 without goal-relative (attribution).

**P16:** G2 J_target ≥ control + 0.08 AND J_easy ≥ control − 0.05 (round 10, mean of 3 seeds).
Mechanism to report: the M5 goal slope and the generator's mean up at h = 0 (the leakage to table goals),
per config.
If P16 holds on tune seeds → held-out on seeds 3–7 with the same criterion, paired against control.

## P16 verdict (seeds 0–2, partial: the goal-relative arm was stopped at round 3)
**Fails.**
- With the goal-relative feature, the loop degrades from round 1:
  - training success 0.04–0.12 vs 0.26–0.45 without it;
  - J_easy 0.73 → 0.05–0.32;
  - critic optimism up to +0.14.
- Unverified mechanism hypothesis: the explicit goal − object feature makes the trivial hindsight rule
  ("object did not move ⇒ success for the relabelled goal") easy for the critic, and argmax over 64 exploits it.
- The plain qgrad η = 0.1 run on the lr 3e-5 base (G3) is kept for attribution.

## Amendment L (written before any balanced run): goal-height-balanced distillation (P17)
Diagnosis: the curriculum concentrates new data at the frontier, so the distillation set under-represents
easy goals, and each frontier-pushing update leaks to the table goals (P14 held-out, P15).

Method (`balanced=True`):
- Importance-weight the distillation loss so that every 2.5 cm goal-height bin carries equal total weight,
  i.e. the training-goal distribution matches the uniform evaluation range.
- The weight per row is 1 / max(bin count, 2% of rows).

Configurations on the replay base (argmax K = 64, filtered init, curriculum + HER, 10 rounds, seeds 0–2):
- B1 balanced (attribution);
- B2 balanced + qgrad η = 0.1;
- B3 balanced + distil lr 3e-4.

The B2 / B3 choice is made on seeds 0–2 (tuning budget 2).
Control: the stability replay runs (J_target 0.085, J_easy 0.63, J_full 0.287).

**P17:** the better of B2/B3 has J_target ≥ control + 0.08 AND J_easy ≥ control − 0.05 AND J_full ≥ control + 0.05.
If it holds → held-out seeds 3–7 with the same three criteria, paired against `stability_heldout` replay.

## Amendment M (written before any FAS run): frontier-adaptive selection pressure (P18)
Diagnosis (M6, `rsi/analysis_kgoal.py`, deployment-only evaluation of saved replay-control models):
- Easy goals (0–5 cm) show the K-inversion at deployment:
  seed 3 K = 1/2/4/64 → 0.79/0.92/0.93/0.84; seed 4 → 0.65/0.81/0.78/0.50.
- High goals (10–30 cm) reach non-zero success only at K = 64 (seed 3: 0.15).
- Disclosure: these two models are held-out-seed models (seeds 3, 4). They were used for a mechanism
  diagnosis, not for choosing a config. To keep the held-out test clean, the P18 held-out set is **seeds 5–9**
  (controls for seeds 8, 9 are run fresh). The same diagnosis is rerun on tune seeds 0–2
  (`rsi/results/m6_kgoal_tune.log`) for the record.

Reading: the right amount of selection pressure depends on the goal.
- Lemma 2: χ²(argmax-of-K ‖ generator) = (K−1)²/(2K−1). Lemma 1: the exploitation of critic error grows with
  √χ² · sd(e). So K sets the step size.
- Where the generator already succeeds (easy goals), the possible improvement is small and large K mostly buys
  critic exploitation.
- Where it never succeeds (high goals), only a large step reaches the rare upward candidates.
- This is the robotics analogue of compute-optimal test-time scaling (Snell et al. 2024): allocate
  best-of-N pressure per difficulty bin, with difficulty estimated from the model's own outcomes rather than
  fixed.

Method (`fas=True`):
- For each 2.5 cm goal-height bin, a Thompson-sampling bandit over arms K ∈ {2, 64}.
  - Arm K selects the critic argmax over a random subset of size K of the 64 sampled candidates.
- Bandit reward: max(success, fraction of the initial object–goal distance removed) ∈ [0, 1], made Bernoulli by
  the Agrawal–Goyal trick, so arms stay distinguishable where success is 0.
- Discount γ = 0.7 per round (the system changes every round). No updates while the critic is untrained.
- Training episodes draw arms by Thompson sampling. Evaluation uses the arm with the higher posterior mean.
- Everything else is the held-out-confirmed replay base: argmax over 64 samples, distil lr 3e-5, filtered init,
  curriculum + HER, 10 rounds.
- Tuning budget: 1 config (arms, γ and the reward fixed here, before any run).

Control: stability replay (seeds 0–2): J_target 0.085, J_easy 0.63, J_full 0.287.

**P18:** FAS at round 10, mean of seeds 0–2: J_easy ≥ control + 0.10 AND J_target ≥ control − 0.02 AND
J_full ≥ control + 0.05.
If it holds → held-out seeds 5–9, paired against replay control on the same seeds; the same three criteria,
plus a paired 95% CI on J_full excluding 0.
Mechanism to report:
- the chosen arm per height bin over rounds (`fas_arm`): prediction 2 on low bins and 64 on high bins;
- the training success per bin.

M6 on tune seeds 0–2 (replay-control models, deployment only; recorded after P18 launched, before any P18 result):

| seed | easy K = 1/2/4/64 | high K = 1/2/4/64 |
|---|---|---|
| 0 | 0.51 / 0.78 / 0.76 / 0.52 | 0.00 / 0.00 / 0.02 / 0.03 |
| 1 | 0.77 / 0.88 / 0.89 / 0.58 | 0.00 / 0.01 / 0.02 / 0.16 |
| 2 | 0.67 / 0.91 / 0.84 / 0.75 | 0.00 / 0.01 / 0.08 / 0.05 |
| mean | 0.65 / **0.86** / 0.83 / 0.62 | 0.00 / 0.01 / 0.04 / **0.08** |

The K-inversion on easy goals reproduces on all three tune seeds (+0.24 from K = 64 → 2, deployment only), and high
goals need the large K.

**P18b (fairness, added before any P18 result was read):** a fixed-K argmax baseline gets the tuning budget
FAS does not use: K ∈ {4, 16} on the same replay base (with K = 64 = control, 3 configs). FAS counts as positive
only if its J_full ≥ the best fixed K's J_full (tune seeds 0–2, round 10). The best fixed K, chosen on J_full,
is also the second held-out comparator.

## P18 verdict (seeds 0–2, round 10)
| | J_target | J_easy | J_full |
|---|---|---|---|
| control (replay, K = 64) | 0.085 | 0.63 | 0.287 |
| FAS (progress reward) | 0.000 | 0.96 | 0.257 |
| criterion | ≥ 0.065 ✗ | ≥ 0.73 ✓ | ≥ 0.337 ✗ |

**Fails.**
- Easy goals are fully protected (+0.33), but the frontier is lost: J_target = 0 on every seed and every round.
- The bandit chose K = 2 in most high bins (`fas_arm`).

Mechanism (M7, `rsi/analysis_proxy.py`, control models, in-air goals, deployment only):

| seed | K | success | progress reward | final xy err | max lift |
|---|---|---|---|---|---|
| 0 | 2 / 64 | 0.00 / 0.01 | 0.276 / 0.200 | 0.055 / 0.177 | 0.026 / 0.040 |
| 1 | 2 / 64 | 0.00 / 0.14 | 0.278 / 0.343 | 0.051 / 0.112 | 0.021 / 0.053 |
| 2 | 2 / 64 | 0.01 / 0.13 | 0.342 / 0.357 | 0.054 / 0.120 | 0.043 / 0.074 |

- The distance-reduction proxy is dominated by xy placement.
- K = 2 (close to the table-only demos) places accurately on the table and earns as much progress or more.
- K = 64 is the arm that lifts and succeeds.
- The bandit was gamed by its own proxy, the same failure as a gamed verifier.

Lesson: the step-size controller must be rewarded by the target event itself. Where the target has never
been observed, there is no evidence for a smaller step, and the coverage argument (1 − (1 − m)^K) says the
default must be the large K.

## P17 verdict (partial: B3 still running)
| config | J_target | J_easy | J_full |
|---|---|---|---|
| B1 balanced | 0.110 | 0.53 | 0.260 |
| B2 balanced + qgrad 0.1 | **0.190** | 0.437 | 0.300 |
| G3 qgrad 0.1 (P16 attribution) | 0.147 | 0.367 | 0.240 |

- B2 fails on J_easy (−0.19) and J_full (+0.013).
- Balanced weighting does not protect easy goals; it slightly helps with qgrad (0.44 vs 0.37 J_easy,
  0.19 vs 0.15 J_target).

## Amendment N (written before any FAS-v2 run): FAS v2, rewarded by the target event (P19)
Diagnosis from P18, M6 and M7:
- (i) The progress proxy was gamed (M7).
- (ii) The FAS-v1 generator alone (K = 1) reaches 0.92 on easy goals vs 0.65 for the control generator.
  Small K *during training* stops the corruption of easy-goal behaviour, so this is a training effect, not
  only a deployment one.
- (iii) The FAS-v1 frontier is gone even at K = 64. Large K *during training* at the frontier is what
  creates take-off.
- (iv) B2 (balanced + qgrad) models: K choice at deployment recovers only part of the easy loss
  (K = 2: 0.66 vs K = 64: 0.52; generator 0.47). The rest is forgetting caused by training.

FAS v2 (`fas_reward="success"`, `fas_gate=True`), two arms K ∈ {2, 64}:
- **Reward:** success only (the target event, no proxy). Discount γ = 0.7, as in v1.
- **Default:** K = 64 (coverage: 1 − (1 − m)^K) in every bin with fewer than 1 discounted success.
- **Training:** Thompson sampling in eligible bins.
- **Deployment:** K = 2 only if P(θ₂ > θ₆₄ | data) ≥ 0.8 AND arm 2 has ≥ 5 effective trials; otherwise 64.
- The smoke test (`seed 99`, tiny budget) found that the uniform prior on an untried arm wins against a
  low-success incumbent. This is why the deployment rule uses superiority probability with a minimum of
  evidence, fixed before any real run.

Configurations (tune seeds 0–2, same replay base as P18):
- F2: FAS v2.
- F3: FAS v2 + balanced + qgrad η = 0.1, i.e. combined with the highest-frontier config B2.

Tuning budget of the FAS family: v1, F2, F3 = 3 configs, the same as fixed K ∈ {4, 16, 64}.

Predictions:
- F2: J_easy ≥ 0.85, J_target within ±0.03 of the control (0.085), arm 2 on bins below 5 cm and 64 above
  10 cm.
- F3: J_target ≥ 0.15 and J_easy ≥ 0.70.
- **P19 (pass/fail):** the better of F2/F3 on J_full (tune seeds, round 10) satisfies the P18 criteria
  (J_easy ≥ control + 0.10, J_target ≥ control − 0.02, J_full ≥ control + 0.05) AND P18b
  (J_full ≥ best fixed K).
  - If it passes → held-out seeds 5–9, paired against the replay control and the best fixed K on the same
    seeds.

## P17 verdict (final, tune seeds 0–2)
| config | J_target | J_easy | J_full |
|---|---|---|---|
| control | 0.085 | 0.63 | 0.287 |
| B1 balanced | 0.110 | 0.53 | 0.260 |
| B2 balanced + qgrad 0.1 | 0.190 | 0.437 | 0.300 |
| **B3 balanced + distil lr 3e-4** | **0.188** | **0.593** | **0.357** |
| criterion | ≥ 0.165 | ≥ 0.58 | ≥ 0.337 |

**B3 holds on tune seeds** (selected over B2 on J_full).
- Per seed (J_target / J_easy / J_full): 0.19/0.54/0.35, 0.03/0.47/0.24, 0.345/0.77/0.48. High variance.
- The same lr without balancing (P14 held-out) failed through easy-goal forgetting; balancing is the change.
- As pre-registered: held-out seeds 3–7 with the same three criteria, paired against `stability_heldout` replay.

## Amendment O (written before any outcome-consistent run): outcome-consistent distillation (P20)
Diagnosis (M8 goal slope ≈ 0 in every config, plus reading the distillation code):
- The distillation set contains the critic-selected actions of *every* training episode under the commanded
  goal, plus (HER) the same actions under the achieved goal.
- In-air bins have 0–55% (mostly ≤ 20%) training success at round 10 (`round_bins`). So most in-air
  commanded rows are action sequences that ended with the object on the table, and they also appear as HER
  rows labelled with the table goal.
- Identical actions labelled with two different goal heights teach the generator to ignore the goal height.
- With a goal-blind generator, lifting learned at the frontier leaks to table goals (B3: generator up at h = 0 is
  +0.03, the same as at h = 0.2). This is the frontier ↔ easy-goal coupling seen in P14, P15, P17 and P18/19.

Method (`distill_filter="success"`):
- Commanded-goal rows only from successful episodes, as in outcome-filtered self-training (STaR, ReST, RAFT).
- HER rows from every episode (GCSL).
- Every distilled row then pairs an action sequence with a goal it actually achieved. The critic still uses all
  data, and selection is unchanged.

Configurations (tune seeds 0–2, tuning budget 2):
- D1: control replay base + filter.
- D2: B3 base (balanced, distil lr 3e-4) + filter.

Predictions:
- Mechanism (M8 probe): D1 generator goal slope ≥ 0.022 (2× control +0.011), and generator mean up at h = 0
  ≤ control's.
- **P20 (pass/fail):** the better of D1/D2 on J_full (round 10, mean of seeds 0–2), against the control
  (0.085 / 0.63 / 0.287):
  - J_full ≥ control + 0.05 AND J_target ≥ control + 0.03 AND J_easy ≥ control − 0.05.
  - In words: both ends move up, with no trade-off.
- If it passes → held-out seeds 5–9 against the control and B3 on the same seeds.

## P18b result (fixed-K baselines, tune seeds 0–2, round 10)
| fixed K | J_target | J_easy | J_full |
|---|---|---|---|
| 4 | 0.000 | 0.967 | 0.220 |
| 16 | 0.005 | 0.903 | 0.253 |
| 64 (control) | 0.085 | 0.63 | 0.287 |

- Best fixed K on J_full = 64 (the control).
- Smaller K protects easy goals but never takes off: no seed reaches J_target > 0.015.
- The trade-off is monotone in K, so no single K serves both ends.

## P19 interim
- F2 (FAS v2 on the control base), final: J_target 0.058, J_easy 0.883, J_full 0.257 → fails J_full and
  J_target.
  - Arms behave as predicted: K = 2 below 5–7.5 cm, 64 above.
  - Take-off is slower than the control's.
- F3 (FAS v2 + balanced + qgrad 0.1), round 10, seeds 0 and 2: 0.20/0.77/0.32 and 0.385/0.83/0.43.
  - Seed 1 was OOM-killed at round 4 (memory cgroup, 12 jobs at once) and is being rerun from scratch,
    alone.
  - No verdict until seed 1 finishes.

## P17 held-out verdict (B3 = balanced + distil lr 3e-4, seeds 3–7, paired against `stability_heldout` replay)
| metric | B3 | control | paired diff | 95% CI | per-seed diffs |
|---|---|---|---|---|---|
| J_target | 0.232 | 0.174 | +0.058 | [−0.130, +0.246] | −0.16 +0.12 +0.20 +0.16 −0.03 |
| J_easy | 0.714 | 0.702 | +0.012 | [−0.069, +0.093] | +0.03 +0.05 −0.06 +0.09 −0.05 |
| J_full | 0.390 | 0.368 | +0.022 | [−0.112, +0.156] | −0.08 +0.09 +0.16 +0.03 −0.09 |

**Fails** (J_target +0.058 < +0.08; J_full +0.022 < +0.05).
- Balancing does remove the easy-goal loss that sank P14 held-out (+0.012 vs P14's forgetting).
- The frontier gain is not reliable: seed 3 never takes off.
- The control is much stronger on held-out seeds (J_target 0.174) than on tune seeds (0.085), so tune-seed
  margins over the control are optimistic.

## P19 verdict (tune seeds 0–2, round 10)
| config | J_target | J_easy | J_full |
|---|---|---|---|
| control (= best fixed K) | 0.085 | 0.63 | 0.287 |
| B2 balanced + qgrad (no FAS, P17) | 0.190 | 0.437 | 0.300 |
| F2 FAS v2 | 0.058 | 0.883 | 0.257 |
| **F3 FAS v2 + balanced + qgrad** | **0.253** | **0.807** | **0.367** |
| criterion (P18 + P18b) | ≥ 0.065 | ≥ 0.73 | ≥ 0.337 and ≥ 0.287 |

**F3 holds on tune seeds** (selected over F2 on J_full).
- Paired per-seed diffs vs control:
  - J_target +0.18 / +0.03 / +0.30;
  - J_easy +0.28 / +0.20 / +0.05;
  - J_full +0.15 / −0.03 / +0.12.
- Attribution vs B2 (the same loop without FAS): J_easy +0.37, J_target +0.06, J_full +0.07.
  The frontier push (qgrad + balancing) and the per-goal step-size controller are complementary.
- Mechanism: arms K = 2 on bins below 7.5–10 cm and K = 64 above (`fas_arm`), as predicted.

Held-out as pre-registered: seeds 5–9, paired against the replay control (= best fixed K) on the same seeds;
criteria J_easy ≥ +0.10, J_target ≥ −0.02, J_full ≥ +0.05, and the paired 95% CI on J_full excluding 0.
Extra (not part of the verdict, for attribution): B2 on seeds 5–9.

## P20 interim
- D1 (outcome-consistent distillation, control base), final: 0.097 / 0.68 / 0.263.
- Mechanism prediction holds: goal slope +0.034 / +0.025 / +0.020 (mean 0.026 vs control 0.011, ≥ 0.022 ✓).
  Generator up at h = 0 is about equal to control (−0.007 vs −0.005).
- Outcome not yet improved. D2 still running.

## P20 verdict (tune seeds 0–2, round 10)
| config | J_target | J_easy | J_full |
|---|---|---|---|
| control | 0.085 | 0.630 | 0.287 |
| B3 balanced + lr 3e-4 (P17) | 0.188 | 0.593 | 0.357 |
| D1 filter (control base) | 0.097 | 0.680 | 0.263 |
| **D2 filter + balanced + lr 3e-4** | **0.497** | **0.857** | **0.630** |
| criterion | ≥ 0.115 | ≥ 0.58 | ≥ 0.337 |

**D2 holds on tune seeds** (selected over D1 on J_full).
- Paired diffs vs control:
  - J_target +0.412, CI [+0.141, +0.683];
  - J_easy +0.227, CI [−0.061, +0.515];
  - J_full +0.343, CI [+0.305, +0.381].
- Attribution vs B3 (the same loop without the filter), positive on every seed:
  J_target +0.31, J_easy +0.26, J_full +0.27.

Reading:
- With goal-consistent distillation data, the larger distillation step (lr 3e-4) no longer leaks lifting to
  table goals. Both ends rise together; the coupling of P14–P19 is gone.
- D1 at lr 3e-5 shows the mechanism (goal slope 2.4× control) but moves too slowly for 10 rounds.

Held-out as pre-registered: seeds 5–9, paired against the control and against B3 on the same seeds. B3 seeds 8
and 9 are run fresh.

Mechanism for D2 (tune-seed models, deployment-only probes, recorded before any held-out result):

M8 goal slope:

| models | goal slope | gen up h = 0 | gen up h = 0.2 | critic pick up h = 0.2 |
|---|---|---|---|---|
| control | +0.011 | −0.005 | +0.002 | +0.10 |
| B3 (same loop, no filter) | −0.009 | +0.031 | +0.024 | +0.10 |
| D1 (filter, lr 3e-5) | +0.026 | −0.007 | +0.010 | +0.10 |
| **D2** | **+0.141** | +0.038 | **+0.136** | +0.20 |

M6 success by K:

| D2 | K = 1 | K = 2 | K = 4 | K = 64 |
|---|---|---|---|---|
| easy (0–5 cm) | 0.69 | 0.90 | 0.94 | 0.86 |
| high (10–30 cm) | **0.47** | 0.55 | 0.60 | 0.44 |

Findings:
- The generator becomes goal-conditioned: slope 13× the control. The coupling is broken at its source.
- The generator alone succeeds on in-air goals (0.47; 0.00 for every earlier config). Improvement is now
  carried by the generator, not only by critic selection.
- The K-inversion is still present, and now on high goals too (seed 0: K = 64 0.22 vs K = 2 0.66). This
  leaves room for per-goal selection pressure on top. That is a separate test (P21), to be pre-registered
  before it is run.

## P19 held-out verdict (F3, seeds 5–9, paired against the replay control = best fixed K)
| metric | F3 | control | paired diff | 95% CI | per-seed diffs |
|---|---|---|---|---|---|
| J_target | 0.309 | 0.224 | +0.085 | [−0.005, +0.175] | +0.00 +0.08 +0.20 +0.07 +0.07 |
| J_easy | 0.826 | 0.688 | +0.138 | [−0.034, +0.310] | +0.20 +0.21 +0.20 +0.19 −0.11 |
| J_full | 0.450 | 0.378 | +0.072 | [−0.022, +0.166] | −0.03 +0.07 +0.15 +0.14 +0.03 |

**Not confirmed under the strict pre-registered rule.**
- The three point criteria hold: J_easy +0.138 ≥ 0.10, J_target +0.085 ≥ −0.02, J_full +0.072 ≥ 0.05.
- The paired 95% CI on J_full includes 0.
- Directionally consistent: J_full is positive on 4 of 5 seeds and J_target is ≥ 0 on all 5.

## P20 held-out verdict (D2, seeds 5–9) — **CONFIRMED**
Against the replay control (= best fixed K) on the same seeds:

| metric | D2 | control | paired diff | 95% CI | per-seed diffs |
|---|---|---|---|---|---|
| J_target | 0.450 | 0.224 | **+0.226** | [+0.021, +0.431] | +0.20 −0.05 +0.34 +0.35 +0.29 |
| J_easy | 0.840 | 0.688 | **+0.152** | [+0.066, +0.238] | +0.17 +0.19 +0.05 +0.23 +0.12 |
| J_full | 0.594 | 0.378 | **+0.216** | [+0.090, +0.342] | +0.22 +0.07 +0.19 +0.35 +0.25 |

All three pre-registered criteria hold (J_full ≥ +0.05, J_target ≥ +0.03, J_easy ≥ −0.05), and every paired
95% CI excludes 0.

Against B3 (the identical loop without the outcome filter), on the same seeds:

| metric | D2 | B3 | paired diff | 95% CI | per-seed diffs |
|---|---|---|---|---|---|
| J_target | 0.450 | 0.347 | +0.103 | [−0.172, +0.378] | +0.00 −0.21 +0.38 +0.19 +0.16 |
| J_easy | 0.840 | 0.736 | +0.104 | [+0.002, +0.206] | +0.23 +0.10 +0.10 +0.00 +0.09 |
| J_full | 0.594 | 0.492 | +0.102 | [−0.022, +0.226] | +0.06 +0.04 +0.28 +0.07 +0.06 |

- The filter's own contribution is positive on J_full for 5/5 seeds; its CI is only significant for J_easy.
- The held-out gain is smaller than on tune seeds (J_full +0.22 vs +0.34), as expected from selection on
  tune seeds.

Attribution for FAS (extra, not a verdict): F3 vs B2 on seeds 5–9 gives J_easy +0.228 [+0.121, +0.335],
J_target +0.059 [−0.164, +0.282], J_full +0.070 [−0.098, +0.238]. The easy-goal protection of the per-goal
step size is reliable; its frontier effect is not.

## Amendment P (written 3 Oct 2026, before any scale-up run): scale-up of P20 on the RTX 5090 host
Setup:
- Code: `rsi/loop_fetch.py` with `RSI_DEVICE=cuda`. The CPU path is verified bit-identical to the pre-change
  code on a reference run.
- Runs are resumable per round; an interrupted-then-resumed run is verified bit-identical to an uninterrupted
  one.
- Runner `scale/rsi_run.py`, specs `scale/rsi_specs.py`, verdicts `scale/rsi_summarize.py` (which reproduces the
  P20 held-out tables exactly).
- Configs are frozen from P20; nothing is re-tuned.
- Seeds 10–29 are fresh. Seeds 0–9 are never used for any verdict below.

**S1 (replication with power, longer horizon).**
- Methods: control, B3, D2 (P20 configs). Seeds 10–29 (20 seeds), 20 rounds × 400 training episodes.
- S1a (primary, round 10, the P20 horizon), D2 vs control: J_full ≥ +0.05 with paired 95% CI excluding 0,
  J_target ≥ +0.03, J_easy ≥ −0.05.
  - Prediction: J_full diff in [+0.10, +0.30].
- S1b (persistence, round 20): D2 vs control J_full paired CI excludes 0.
  - Prediction: the diff is still ≥ +0.10. The control's goal-blind generator keeps the frontier ↔ easy-goal
    coupling, so more rounds do not close the gap.
- S1c (attribution, rounds 10 and 20): D2 vs B3 J_full paired CI excludes 0. At 5 seeds the diff was +0.102,
  CI [−0.022, +0.226].
  - Prediction: holds at both rounds.
- Mechanism (reported, no criterion): goal slope (M8) and K-by-goal (M6) on the final models of seeds 10–14
  for each method.
  - Prediction: D2 slope ≥ 5× control.

**S2 (data and model scale).**
- Methods: control, B3, D2. Seeds 10–19, 10 rounds.
- 4× training episodes per round (1600), 2× network width (512), 2× critic and distillation steps
  (4000 / 3000). The same scaling applies to every method; lr and all else are unchanged.
- S2a: D2 vs control at round 10, the same three criteria as S1a.
  - Prediction: holds. More data does not remove the label contradiction, because failures at the frontier
    still dominate the commanded in-air rows.
- Reported without a criterion: control S2 vs control S1 at round 10 (does 4× data close the gap?), and
  D2 vs B3.

Verdicts are only computed after every run of a block is done. They are reported in `scale/results/<block>/REPORT.md`
together with the registry rows and wall-clock.

**S3 (second goal-conditioned task):** to be pre-registered separately after a CPU pilot on tune seeds 0–2.

## Amendment Q (written before any push loop run): second task, push take-off by direction (P22, CPU pilot)
Task (`task="push"`, `rsi/fetch.py`):
- FetchPush. The goal lies at distance U(0.10, 0.20) m from the object, at angle ±h from the +x axis.
- Demos come from a scripted pusher (100% success with no noise, 58% at noise 0.45), with front goals only
  (h ≤ 90°). 600 demos, noise 0.45, filtered BC: the same protocol as pick-and-place.
- Pushing towards the back requires going around the object, which is never demonstrated.

BC base, generator alone (seed 0, 100 episodes per bin):

| angle | 0–45° | 45–90° | 90–120° | 120–150° | 150–180° |
|---|---|---|---|---|---|
| success | 0.19 | 0.15 | 0.06 | 0.02 | 0.03 |

There is a take-off gradient, and the front-goal success is low, so both ends have headroom.

Loop:
- Identical to P20 except the difficulty axis: train_h [0, π], target eval_h [2π/3, π], easy_h [0, π/4],
  full_h [0, π].
- Curriculum, FAS and balanced bins: 12 bins over the angle.
- Balanced weighting uses each row's remaining push direction |angle(goal − object)|. This is fixed here.
- Methods: control, B3, D2 with the P20 hyper-parameters unchanged (no tuning for push). Tune seeds 0–2,
  10 rounds × 400 episodes.

**P22 (pilot gate for S3):** D2 vs control on seeds 0–2, round 10:
- J_full ≥ +0.05 AND J_target ≥ +0.03 AND J_easy ≥ −0.05 (the P20 criteria).

Outcomes:
- If it holds → S3 = held-out seeds 10–29 on the GPU host, with the S1 protocol.
- If it fails → report it, and analyse whether the label contradiction is present at all. Mechanism to
  report: training success per angle bin (`round_bins`). Prediction: back bins < 20% for most rounds, so the
  contradiction is present.

## P22 verdict (push pilot, tune seeds 0–2, round 10) — **fails; S3 is not added to the GPU grid**
| method | J_target (120–180°) | J_easy (0–45°) | J_full (0–180°) |
|---|---|---|---|
| round 0 (BC, all methods) | 0.005–0.025 | 0.23–0.24 | 0.10–0.15 |
| control | 0.025 | 0.120 | 0.067 |
| B3 | 0.028 | 0.117 | 0.070 |
| D2 | 0.037 | 0.120 | 0.077 |

D2 vs control: J_full +0.010 [−0.015, +0.035], J_target +0.012, J_easy +0.000.

The base loop itself does not self-improve on push. Every method ends *below* its round 0. The P20 fix has nothing
to act on.

Mechanism:
- Training success falls from ≈ 0.10 (round 1, untrained critic = generator samples) to 0.01–0.02 at round 2,
  the first round selected by the critic. It stays at 0.04–0.09 afterwards.
- The critic is strongly over-optimistic about its own picks: opt0 = +0.17 to +0.54 (predicted success
  0.3–0.6 vs ≈ 0.05 realised).
- The generator alone stays flat (J_gen 0.01–0.04 on targets). The loss is in selection: the K-inversion of the
  first phase of this project, in a regime with sparse successes, where the critic's ranking signal is weak and
  argmax over 64 exploits its errors.
- Success vs K at deployment: `rsi/results/m6_kgoal_push_p22_s*.log`, to be added below when done.

Reading: outcome-consistent distillation fixes the distillation data. It does not make a poor selector good.
The P20 claim stays scoped to loops whose critic selection already improves the frontier, as in pick-and-place.
