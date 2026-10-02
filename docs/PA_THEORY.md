# Pessimistic Amplification: theory notes (draft, 1 Oct 2026)

Status: working notes. Lemmas 1–3 are proved below (elementary). Theorem 4 is a sketch. Everything else is a
hypothesis to be tested on Fetch (CPU) and then Robomimic Square (RTX 5090).

## 0. Setting: the robot RSI loop
Round r has a generator (diffusion policy) μ_r and a critic Q̂_r learned from the robot's own episodes.
At every decision the robot samples K action chunks a_1..a_K ~ μ_r(·|s), a selection rule turns the critic
scores into a distribution w over the K candidates, and it executes a_J with J ~ w. Then:
1. the critic is refit on the new episodes (Monte-Carlo targets = task success);
2. the executed chunks are distilled into the generator: μ_{r+1} ≈ π_r, the law of the executed action,
   mixed with a fraction ρ of demonstrations.

This is the loop of PA-RL, of V-GPS with distillation, and of `ppa/robo/loop.py`. Write f = π/μ for the
density ratio that the selection rule induces at a state, and e = Q̂ − Q for the critic error.

## 1. The failure: the winner's curse is distilled
Argmax over K noisy scores selects candidates whose error is large and positive. Executing them costs this
round; distilling them also moves the next proposal μ_{r+1} towards the critic's mistakes. Fetch evidence
(`rsi/results/d1b_fetch`): with K = 64 the system with selection does worse than the generator alone
(0.26–0.41 vs 0.55–0.66 at round 2), while the critic over-rates its own choices by +0.14 to +0.35.

## 2. Lemmas
**Lemma 1 (what a selection rule can certify).** For any state, proposal μ and selection-induced π ≪ μ,

  E_π[Q] − E_μ[Q] ≥ (E_π[Q̂] − E_μ[Q̂]) − sqrt(χ²(π‖μ)) · sd_μ(e).

*Proof.* E_π[e] − E_μ[e] = E_μ[(f − 1)(e − c)] for any constant c, because E_μ[f − 1] = 0. Take c = E_μ[e] and
apply Cauchy–Schwarz: |·| ≤ sqrt(E_μ[(f−1)²]) · sd_μ(e) = sqrt(χ²(π‖μ)) · sd_μ(e). ∎

Two consequences:
- A constant offset of the critic does not matter. Only the part of the error that varies across
  candidates (sd_μ(e)) can be exploited. This is also why the calibration of PPA labels did not matter for
  critic-argmax.
- The exploitable error grows with the selection's χ²-distance from the proposal, not with K itself.

**Lemma 2 (argmax-of-K).** If Q̂ has a continuous distribution under μ, argmax-of-K has density ratio
f = K·U^{K−1}, where U = F_μ(Q̂) is uniform, so

  χ²(π_K‖μ) = K²/(2K − 1) − 1 = (K − 1)²/(2K − 1) ≈ K/2.

(Checked numerically for K = 2, 4, 16, 64.) The certified loss term therefore grows as sqrt(K/2)·sd(e),
while the estimated gain grows only as sd_μ(Q̂)·sqrt(2 log K) for Gaussian-like scores. **The lower bound for
argmax peaks at a finite K* and becomes negative for large K.** More sampling compute makes certified
self-improvement worse.

**Lemma 3 (χ²-regularised selection).** π_β = argmax_π E_π[Q̂] − β χ²(π‖μ) has
f_β = max(0, 1 + (Q̂ − λ)/(2β)), with λ chosen so that E_μ f_β = 1 (Huang et al., 2025, "Is Best-of-N the Best
of Them?"). Properties:
- f_β ≤ 1 + range(Q̂)/(2β), independent of K. With K empirical candidates the weights are
  w_k ∝ max(0, 1 + (q_k − λ)/(2β)) (`rsi/loop_nav.py:chi2_weights`), which converge to π_β as K → ∞.
  More K only reduces Monte-Carlo error and never increases exploitation (scaling-monotone).
- Without truncation: gain = Var_μ(Q̂)/(2β) and χ² = Var_μ(Q̂)/(4β²). By Lemma 1 the certified improvement
  is σ̂(σ̂ − ε)/(2β), where σ̂ = sd_μ(Q̂) and ε = sd_μ(e). **Improvement is certified exactly where the
  critic's spread over the candidates exceeds its error spread**, which is a per-state signal-to-noise
  condition. Where σ̂ ≤ ε the bound says do not select, i.e. keep the generator.

**Corollary (argmax couples coverage and step size).** In the loop, K plays two roles:
- **Coverage.** P(some candidate is good) = 1 − (1 − m)^K.
- **Step size.** χ²(π_K‖μ) ≈ K/2 is how far one round moves the proposal, and by Lemma 1 it is also how much
  critic error the round can exploit.

Argmax ties the two together, so buying coverage with more K also takes a larger, less certified step.
A χ² *trust region* separates them: maximise E_w[Q̂] subject to χ²(w‖uniform_K) ≤ δ. The solution is the χ²
tilt with a per-state β chosen so the constraint binds (`ppa/select.py:chi2_trust_weights`).
- K then sets coverage only.
- δ sets the step size. It is scale-free (invariant to affine rescaling of Q̂), and δ = (K₀−1)²/(2K₀−1)
  reproduces the step of argmax-of-K₀.

Prediction P5: with δ ≈ 1.3, the step of argmax-of-4, the trust-region rule at K = 64 beats argmax at
K = 4, because the step is the same and coverage is better.

**Corollary 5 (the right K depends on the context; added 2 Oct 2026, motivates FAS / P18).**
Combine Lemma 1 with Lemma 2 for Gaussian-like scores at one context s (one goal):
- certified gain ≈ σ̂(s)·sqrt(2 ln K) − ε(s)·sqrt(K/2), where σ̂ = sd_μ(Q̂) and ε = sd_μ(e);
- setting the derivative in K to zero gives **K* ln K* ≈ 4·SNR(s)²**, where SNR(s) = σ̂(s)/ε(s).

So the bound-optimal selection pressure grows roughly with the squared critic signal-to-noise ratio of the
context.

In the take-off task, SNR differs sharply across goal heights (M6):
- On table goals almost every candidate succeeds. σ̂ is small relative to ε, so K* is small (K = 2–4 is best
  at deployment; K = 64 loses 0.24).
- On in-air goals only rare upward candidates make progress, and only K = 64 finds them.

One global K, or a global δ, cannot serve both.

ε(s) is not observable from the critic itself, and earlier attempts that proxied it by ensemble disagreement
(LCB, χ² with a global β) did not win. Instead, FAS learns the per-context step directly from realised outcomes:
- a bandit over K per goal-height bin;
- rewarded by real success or progress;
- discounted because the system changes each round.

This is the robotics analogue of compute-optimal test-time scaling (Snell et al., 2024), where the best-of-N
budget is allocated by estimated difficulty.

## 3. RSI-specific statements (to be proved or refuted)
**Theorem 4 (sketch: monotone self-improvement).** Assume the critic of round r is fit on data from π_{r−1}
and distillation is exact (μ_r = π_{r−1}). Then the error that Lemma 1 needs, sd_{μ_r}(e_r), is an
*on-policy* regression error. With χ² selection, per-round improvement is at least
E_s[σ̂(σ̂ − ε)/(2β)] − (state-distribution shift term) − δ_distill. With argmax, the χ² term grows
like K/2 and is also compounded, because μ_{r+1} carries the over-rated actions of round r.
Open: bound the state-distribution shift with the per-state ratio bound f ≤ 1 + 1/(2β) over the horizon.

**Self-calibrated pessimism (predictable).** On fresh episodes, before the critic is refit on them,
E[Q̂(s, a_exec)] − E[Y] is an unbiased estimate of the realised exploitation E_π[e]. Lemma 1 gives
ε ≥ E_π[e]/sqrt(χ²), so the robot can measure its own winner's curse each round and set the next β from it.
Only past data is used, so the rule is predictable in the sense of `ppa/labels.py`.

## 4. Pre-registered predictions (Fetch, then Square)
- P1. Argmax: final true success is non-monotone in K, and the realised optimism grows with K.
- P2. χ² selection: final success is non-decreasing in K (within 1 s.e.), and its best K beats argmax's best K.
- P3. χ² beats KL-regularised (softmax) selection and ensemble-LCB pessimism, each given the same tuning budget.

## 5. Related work to position against
- Huang et al. 2025 (ICML): χ²-pessimism for inference-time alignment with a fixed reward model. No
  generator updates, no on-policy critic, no compounding.
- Gao et al. 2023: reward-model over-optimisation scaling laws (LLM).
- Thrun & Schwartz 1993; van Hasselt 2010: overestimation bias of max over noisy Q-values.
- PA-RL, V-GPS, RoboMonkey, UF-OPS (2603.10282), OGPO (2605.03065, uses heuristic "conservative advantages"
  against critic over-exploitation): sample-and-select robot methods with argmax or heuristic pessimism.
