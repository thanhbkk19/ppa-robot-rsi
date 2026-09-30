# Handoff spec: Prediction-Powered Amplification (PPA) for robot self-improvement

**Status (30 Sep 2026):** toy-validated direction, not robot-validated, not a certified novelty claim.
**Supersedes as main direction:** "frozen generator + model-control-variate selector" (THEORY_FIRST_MODEL_ASSISTED_RSI.md). That spec's factual-gradient discipline is kept. Its frozen generator and α-calibration are demoted (see §6).

## 0. One-paragraph pitch
A robot improves itself by (i) sampling K candidate action chunks from its generative policy, (ii) selecting among them with a learned selector or critic, and (iii) distilling the selected behaviour back into the generator (amplification). Self-improvement at scale needs a **self-verifier** V (a learned success detector, VLM judge or progress model) to label outcomes, and V can be gamed. PPA labels every episode with the prediction-powered (doubly robust) label

  Ỹ = V + r̂ + (A/p)(Y − V − r̂),  A ~ Bernoulli(p), p decided before Y is seen,

where Y is a scarce ground-truth anchor (human check or privileged sim success) and r̂ is an optional predictable residual model. Because E[Ỹ | episode] = Y, **every learner that is linear in its labels optimizes the true success rate in expectation, however wrong V is**. That rules out reward hacking in expectation, while the loop still uses the cheap verifier for >95% of the labelling.

## 1. Evidence from the toy loop (`toy/`, CPU, 5 held-out seeds, tuned baselines)
Multi-step 2-D navigation. T = 6, sparse terminal success, diffusion generator, K = 4, 800 episodes per round, 6 rounds.

**True reward (R1, held-out seeds 3–7)**

| Method | final J | AUC |
|---|---|---|
| PG selector + distill | 0.96 | 0.758 |
| Critic-argmax + distill (PA-RL-like) | 0.90 | 0.751 |
| Filtered BC | 0.53 | 0.433 |
| Frozen generator + selector (old spec) | 0.28 | 0.276 |

**Hackable self-verifier (R2).** V also fires on a trap pocket that 90% of demos reach. Anchor budget p = 5% unless stated. Critic learner (PG in brackets).

| Labels | final true J | self-estimated J |
|---|---|---|
| Oracle Y everywhere | 0.91 (0.96) | |
| Self V only | 0.10 (0.00) | 0.81 (0.90) → **hacked** |
| Naive mix (Y if anchored else V) | 0.06 (0.00) | → **hacked** |
| Anchors only | 0.79 (0.42) | |
| **PPA-DR** | **0.90** (0.39 at lr 100 → 0.91 with active anchoring at lr 30) | |
| PPA-DR at p = 2% / 1% | 0.93 / 0.86 | anchors only: 0.41 / 0.20 |
| PPA-DR with r̂ ≡ 0 | 0.89 | |
| "Fine-tune reward model on anchors" (plug-in V + r̂), r̂ well-specified, p = 2% | 0.92 | |
| Same plug-in, r̂ misspecified (cannot see where V is fooled), p = 2% | **0.02 (hacked)** | |
| PPA-DR with the same misspecified r̂, p = 2% | 0.73 | |

**Negative or neutral findings, so claims stay honest:**
- Active anchoring ("verify claimed successes") gave no consistent gain over uniform anchoring.
- Online power tuning of the r̂ coefficient did not help (0.64 ± 0.15 vs 0.73).
- A PG selector with high-variance DR labels needs a smaller learning rate.

## 2. Mathematical core (proofs are short; write them fully in the paper)
- **P1 (unbiased labels).** If p(τ) ≥ p_min > 0 depends only on pre-label information and r̂ is predictable (fit on past batches), then E[Ỹ | τ] = Y. PPI/DR identity, inherited.
- **C1 (no hacking in expectation).** For any learner whose update is linear in labels (likelihood-ratio PG; least-squares critic on Monte-Carlo targets), the expected update equals the update under true labels. The population objective is the true J, not V. TD critics with bootstrapping are only approximately covered; this is a known gap, so test it.
- **P2 (variance / label budget).** Var(Ỹ | τ) = (1 − p)/p · (Y − V − r̂)². The optimal allocation under E[p] = budget is p ∝ |Y − V − r̂| (Neyman). If V has no false negatives, anchors on V = 0 episodes are wasted. The effective sample size is n_eff ≈ n / (1 + E[(1−p)/p · e²]/Var Y).
- **Lemma A (K-candidate trust region).** Any selector over K i.i.d. candidates satisfies π_sel/μ ≤ K, so D_∞(π_sel‖μ) ≤ log K. Each distillation round is a bounded mirror-descent-like step.
- **Amplification map with learnability.** m_{r+1} ≥ q(1 − (1 − m_r)^K) − δ, verified in the toy: corr 0.98, median relative error 3.6%. q is not constant: q ≈ 1/K (chance) until the selector has seen about 10 successes per context. Conjecture to prove or test: the take-off condition is n_eff · K · m ≳ c, which ties anchor budget and verifier quality to whether RSI starts at all. **This is the main new theory target.**
- **Gate (phase 2).** Accept a distilled generator only if an anytime-valid PPI lower bound (e-process / confidence sequence; Ville) on ΔJ, computed on *fresh* episodes, exceeds −τ. Total false-acceptance probability ≤ δ over the whole run. This addresses the "calibration that optimization invalidates" gap in the robot-verifier survey: anchors are drawn on-policy every round, so calibration follows the policy.

## 3. Algorithm (implementation contract)
```
inputs: pretrained generator μ0 (diffusion/flow), K, self-verifier V, anchor oracle Y (budget p),
        learner ∈ {critic-argmax (default), softmax-PG selector (ablation)}, distill mix ρ
for round k:
  for each batch of episodes (on-policy, current μ_k + learner):
     execute; record V for every episode
     decide A ~ Bern(p(τ)) from pre-label info only; query Y where A = 1
     Ỹ = V + r̂(τ) + A/p (Y − V − r̂(τ))     # r̂ from previous batches only
     learner update with Ỹ (critic: MC targets; PG: return-centred score)
     update r̂ on anchored episodes (for future batches)
  distill: fine-tune μ on executed (unfiltered) selected actions + ρ demos → μ_{k+1}
  [phase 2] gate μ_{k+1} with PPI lower bound on fresh episodes; reject → keep μ_k
log: true J (eval only), self-estimated J, anchors used, |J_self − J| (hacking gap), q, m
```
Do not: filter distillation data by V (that is filtered BC and it gets hacked), refit r̂ on the current batch before labelling it, or clip Ỹ (clipping breaks P1; use a smaller lr or a larger batch instead).

## 4. Scale-up plan (for the coding agent, on the user's RTX 5090)
1. **Backbone:** PA-RL codebase (sample K → critic select → distill) on Robomimic Square / Transport / ToolHang with DPPO pretrained diffusion checkpoints (non-saturated success). Check the framework (JAX or PyTorch) first.
2. **Self-verifiers, three regimes:**
   - (a) a success classifier trained on demos plus a few failures (naturally hackable)
   - (b) a VLM judge on final frames
   - (c) controlled injected false positives on a specified state region (a reproducible hacking testbed)
   - The anchor is sim ground-truth success, with budget p ∈ {1, 2, 5}%.
3. **Baselines (tune each on seeds 0–2, report held-out seeds ≥ 5):**
   - PA-RL with true reward (oracle upper bound)
   - PA-RL with V only
   - anchors-only
   - naive mix
   - plug-in reward-model fine-tuning on anchors (the strongest practical baseline)
   - filtered BC
   - DSRL (true reward)
   - frozen-generator selector (old spec)
4. **Primary metrics:**
   - true success vs episodes (AUC and final)
   - hacking gap J_self − J
   - anchors used
   - wall-clock
5. **Pre-registered go/no-go:**
   - GO if, in the hackable regimes, PPA-DR at ≤ 2% anchors is within 0.05 of oracle final success and beats plug-in fine-tuning by ≥ 0.10 in at least one regime where the verifier's failure is not visible to the reward-model features.
   - NO-GO if plug-in fine-tuning matches PPA everywhere. The unbiasedness then has no practical value on these benchmarks; pivot to the theory paper (learnability threshold) or a different benchmark.
   - Separately test whether a TD critic breaks C1 (compare MC vs TD targets with DR labels).

## 5. Novelty positioning (checked 30 Sep 2026; re-check before submission)
- **Closest work:**
  - SGM (certified self-modification; frozen evaluator; no proxy; no robots)
  - SureSim (PPI for robot *evaluation*, not training)
  - CARE, 2605.25864 (active label acquisition for LLM RLVR; biased, no unbiasedness guarantee)
  - RLVR with noisy verifiers, 2510.00915 (assumes a known noise model)
  - PA-RL and V-GPS (critic selection; no verifier-error handling)
  - "No Free Checker" survey, 2609.09250 (names the cheap-vs-credible verifier gap and the "calibration that optimization invalidates" as open)
- **Honest weak point:** the DR/PPI identity itself is standard statistics. The contribution must be the amplification loop, the learnability/threshold theory with n_eff, the gate, and robot evidence that plug-in reward-model fine-tuning fails where PPA does not.

## 6. What happened to the previous spec's ideas
- The factual-gradient and predictability rules are kept; they are exactly what makes P1 hold.
- The frozen generator is dropped as the main method: it plateaus at 0.28 vs 0.90+ with distillation.
- The model control variate with OGD-α and the label power tuning gave no measurable benefit in two separate tests. They can stay as optional variance tools, not as claims.
