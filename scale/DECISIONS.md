# Decisions log

(append: date, decision, reason, affected milestones)

## 2026-10-01 — D1: backbone = PA-RL mechanism re-implemented in PyTorch on DPPO (not the PA-RL codebase)
- **Decision.** PA-RL (`third_party/PolicyAgnosticRL`, commit a26a192) is cloned for reference but not run. The
  PA-RL mechanism (sample K chunks from a diffusion generator → critic-argmax → distill the executed chunks back into
  the generator) is re-implemented in `ppa/robo/` (PyTorch) on top of DPPO's (`third_party/dppo`, commit cc7234a)
  pretrained Robomimic diffusion checkpoints and environment wrappers.
- **Reason.** (1) PA-RL's public code is JAX 0.4.24 / jaxlib-cuda12 of early 2024 + TF 2.15; that stack has no
  sm_120 (RTX 5090 Blackwell) support. (2) The public PA-RL code has **no Robomimic support at all** (only D4RL
  antmaze/kitchen, CALVIN, real robot; `grep robomimic` is empty), so the spec's "PA-RL on Robomimic with DPPO
  checkpoints" does not exist as runnable code in either case. (3) DPPO is PyTorch, ships the exact checkpoints the
  spec names, and its env wrappers run on current torch 2.11+cu128.
- **Deviations inside the re-implementation (vs. PA-RL paper):** global candidate selection only (argmax over K,
  no local gradient refinement of candidates); critic = 2-member MLP ensemble on (state, chunk, t/T) with MC or
  SARSA-TD targets instead of Cal-QL; distillation = supervised DDPM loss on the round's executed chunks + ρ = 10%
  demos (spec §3). These make the learner linear in labels for the MC variant, which is what C1 needs.
- **Affected:** M1–M4.

## 2026-10-01 — D2: tasks = Square (Transport only if budget allows); ToolHang dropped
- DPPO releases no ToolHang checkpoint (Robomimic tasks in DPPO: lift/can/square/transport). Square pretrained
  diffusion-MLP success ≈ 0.38–0.41, i.e. non-saturated. Transport (≈ 2× episode length, bimanual) is deferred to
  the budget in `BUDGET.md`.

## 2026-10-01 — D3: episodes are fixed length; Y = success at the final step
- Episodes run the full 400 env steps (100 chunk decisions) with no termination on success: terminating on
  success would leak Y to every learner through the episode length. Y = sim success over the whole final chunk.
  Verifiers judge the final state. DPPO's any-time success is logged as `J_any` for reference only.
- Train / evaluation / verifier-calibration initial states come from disjoint integer seed ranges
  (train 1e8 + seed·1e6 + round·1e4 + i; eval 9e8 + seed·1e4 + i, the same 192 states every round; calibration
  5e6..5e6+959). Evaluation episodes are never given to any learner.

## 2026-10-01 — D4: M1 reproduction tolerance (pre-registered before the oracle runs)
- PA-RL has no published Robomimic numbers, so "published numbers" can only refer to the DPPO checkpoint.
  Tolerance: our harness's pretrained success must agree with DPPO's own eval script
  (`script/run.py --config-name=eval_diffusion_mlp`, 250 episodes) within **±0.07** (≈ 2 s.e.).
  Measured: DPPO harness 0.412 (n = 250) vs ours 0.380 any-time / 0.379 final (n = 960) → |Δ| = 0.032, PASS.
- Backbone criterion for M1 (true reward = oracle labels): mean final true success on tune seeds 0–2 must exceed
  the pretrained success by **≥ +0.10** within the M4 per-run episode budget. Otherwise the loop has no headroom
  to be hacked or recovered and M1 stops.

## 2026-10-01 — D5: verifier regimes (M2) and reward-model features
- (c) `topdown`: V = 1 iff the nut is within the peg's ±0.03 m xy window, ignoring height (a top-down judge
  cannot see insertion depth). False positives = nut hovering/jammed above the peg: 14.9% of pretrained episodes
  (24% of failures) on the 960-episode calibration set → reachable. Reward-model / r̂ features: nut and
  end-effector xy relative to the peg (top-down-visible only) → the failure is **invisible** to r̂.
- (a) `classifier`: sklearn MLP(64,64) on final states of the 300 MH demos (label 1) + the first 20 failures of
  the calibration set (label 0); frozen once (`scale/results/raw/calib/square_classifier_v1.pkl`). On the other
  940 calibration episodes: P(V=1) 0.55 vs J 0.39, FPR 0.31, TPR 0.92. r̂ features: the full final state
  (failure in principle **visible**).
- (b) VLM judge: not implemented (optional in the spec).
- r̂ (PPA) and the plug-in reward model use the same ridge model on [1, V, x, V·x], λ = 1. The plug-in RM uses Y
  on anchored episodes, clip(V + r̂) elsewhere, refits on all anchors including the current batch and relabels
  the whole replay each round (strongest practical variant; no predictability constraint).

## 2026-10-01 — D6: DSRL omitted
- DSRL (optional in CLAUDE.md) is not run; it needs a separate latent-noise SAC stack and the budget is spent on
  the pre-registered comparison instead.
