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
