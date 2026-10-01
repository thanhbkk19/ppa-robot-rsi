# Literature review: self-improvement ideas from LLM / CV that target our measured failures (1 Oct 2026)

Our loop: sample K chunks from a diffusion policy → select with a learned critic → distil. On Fetch we
measured two failures (`docs/RSI_STATUS.md`):

- **F1 (inside the demonstration support).**
  - The critic is noisy and overstates the spread between candidates.
  - Larger selection steps over-optimise it (K-inversion), so small steps win.
  - Small steps plateau: 0.90 easy, 0.75 hard. The generator does not fully absorb the system's gains.
- **F2 (beyond the demonstrations, "take-off").** Base success is 0.
  - Positive-only distillation *sharpens back*: object lift shrinks 2.8 → 0.5 cm.
  - Only aggressive argmax-of-64 moved the frontier, and only on 1 of 3 seeds.

Both failures have close analogues in LLM RL and in diffusion fine-tuning, where fixes have been
published, often with theory.

## 1. What the other fields found (same phenomena)
| our observation | analogue elsewhere | source |
|---|---|---|
| Selection over many samples exploits a noisy critic (K-inversion) | Best-of-N reward hacking at large N; verifier hacking when searching diffusion noise | Huang et al. 2025 (ICML); Ma et al. 2025 (CVPR) |
| Positive-only self-distillation narrows behaviour, so the lift shrinks | RL on positives raises pass@1 but lowers pass@k at large k (narrowing); entropy collapse | Yue et al. 2025 (limit-of-RLVR); DAPO 2025; entropy-perspective 2506.14758 |
| Reward fine-tuning of diffusion collapses diversity | DDPO / DRaFT over-optimisation, mode collapse | 2410.08315; GARDO 2512.24138 |
| No progress where success is 0 | "Zero-baseline" problems need expansion, not sharpening; boundary-aware curricula | MATH-Beyond (ICLR 2026); 2606.22317 |

The negative results we got are therefore expected, and the cures below already work in those fields.

## 2. Candidate ideas, ranked by fit to F1/F2, theory, and novelty for robot RSI

### A. Self-improving classifier-free guidance (CFG) instead of select-then-distill — **top pick**
- **Source.** "Diffusion Guidance Is a Controllable Policy Improvement Operator" (CFGRL, Frans, Park, Abbeel,
  Levine 2025).
  - Train one diffusion policy conditioned on an optimality flag o, with the flag dropped some of the time.
  - Sample from μ(a|s)·[μ(a|s,o=1)/μ(a|s)]^w.
  - They prove guidance is a policy-improvement operator, and the weight w sets the step size.
- **Why it fits.**
  - It removes the argmax over a noisy critic. The "critic" is the density ratio learned by the same network
    on the action manifold, so there is no winner's curse over K samples (F1).
  - w is exactly the step-size knob our step-size law says matters, but continuous and per sample.
  - w > 1 extrapolates beyond the data, which is the mechanism autoguidance uses in CV (Karras et al.
    NeurIPS 2024). That is a candidate cure for F2.
- **RSI version (new as far as the searches show).**
  - Each round: label executed chunks with the episode outcome (plus hindsight relabelling for goals),
    retrain the conditional generator, and deploy guided sampling.
  - Theory to inherit: the CFGRL improvement result.
  - New piece: iterating it with on-policy data, and how w trades exploitation against exploration.
- **Cheap test.** Fetch easy, hard and take-off regimes, w ∈ {1, 1.5, 2, 3}, against argmax K = 2 and
  argmax K = 64.

### B. Negative-sample reinforcement for diffusion policies
- **Source.** "The Surprising Effectiveness of Negative Reinforcement in LLM Reasoning" (2506.01347).
  Training only on penalising failures improves pass@k across k ≤ 256 and preserves diversity, while
  positive-only training narrows.
- **Why it fits.** Our distillation is positive-only, and its sharpening is exactly the F2 failure.
- **Diffusion version.** Guide *away* from failures, i.e. CFG with a "failure" condition
  (μ·[μ/μ(·|o=0)]^w). Or use a Diffusion-DPO-style contrast with failed chunks as losers. This is the
  negative half of idea A and can be ablated inside it.

### C. Extrapolating across RSI rounds (critic-free acceleration)
- **Sources.**
  - ExPO, "Weak-to-Strong Extrapolation Expedites Alignment" (ICML 2024): θ = θ_new + α(θ_new − θ_old) is a
    first-order step along the alignment objective.
  - Autoguidance: guide a model with a worse version of itself.
- **Why it fits.**
  - In take-off the lift grows only a few mm per round (balanced rule: 2.7 → 5.5 cm).
  - Extrapolating μ_r against μ_{r−1}, either in weights or by guidance μ_r^{1+w}/μ_{r−1}^{w}, amplifies
    the direction that already improved, with no extra data and no critic.
- **Cheap test.** It is an add-on to any loop. The risk is extrapolating noise, which is why the step is
  kept small.

### D. Train the generator for the deployed best-of-K (pass@k-aware objectives)
- **Sources.**
  - Inference-aware BoN fine-tuning (Chow et al., ICLR 2025).
  - max@k / pass@k training (2510.23393).
  - DAPO's dynamic sampling, which drops zero-advantage groups.
- **Why it fits.** We deploy selection over K but distil as if deploying the single sample. Optimising the
  best-of-K objective keeps the generator diverse where selection helps (F1 plateau and F2 narrowing).
- **Cost.** A leave-one-out (max-of-K) weighted distillation; we already store the K candidates (rb code).

### E. Search or steering in the diffusion noise space instead of i.i.d. samples
- **Sources.**
  - DSRL (CoRL 2025): RL over the latent noise of a frozen diffusion policy.
  - Inference-time scaling of diffusion by zero-order / path search over noise (Ma et al. 2025).
  - Noise-trajectory search (2506.03164).
- **Why it fits.** For F2, local search around the best noise can climb further than i.i.d. best-of-K, and
  it stays on the generator's manifold.
- **Caveat.** The same papers report verifier hacking, so it needs a step control (Section 1).

### F. Others, lower priority
- **SPIN-Diffusion.** Self-play against the previous iterate. It is reward-free with a stationary-point
  guarantee, and may help the distillation gap.
- **Go-Explore.** Archive and return-to-frontier for sparse Fetch pick-and-place. It needs state resets,
  so it is simulation-only.
- **Boundary-aware curriculum (2606.22317).** Close to our frontier curriculum.
- **TTRL (majority-vote pseudo-rewards).** Relevant only if ground-truth success is unavailable.

## 3. Proposed next loop (same discipline: pre-register, tune seeds 0–2, held-out 3–7)
1. **A + B (CFG self-improvement with positive and negative guidance).**
   - Predictions:
     - (i) easy / hard regimes: final success ≥ argmax K = 2;
     - (ii) take-off: target success > 0 on ≥ 2/3 seeds within 10 rounds, beating argmax K = 64;
     - (iii) w orders the outcome like the χ² step (small w best in-distribution, larger w needed for
       take-off).
2. **C as an add-on** to the best of step 1. Prediction: faster take-off (rounds to lift ≥ 10 cm).
3. **D or E** only if 1–2 fail on F1 or F2 respectively.

## Sources
- Huang et al. 2025, Is Best-of-N the Best of Them? https://arxiv.org/pdf/2503.21878
- Ma et al. 2025, Scaling inference-time compute for diffusion models (CVPR) https://cvpr.thecvf.com/virtual/2025/poster/32892
- Limit of RLVR (Yue et al. 2025) https://arxiv.org/html/2504.13837v1
- DAPO https://arxiv.org/pdf/2503.14476
- Reasoning with exploration: an entropy perspective https://arxiv.org/pdf/2506.14758
- Negative sample reinforcement https://arxiv.org/abs/2506.01347
- MATH-Beyond (ICLR 2026) https://en.papernotes.org/ICLR2026/reinforcement_learning/math-beyond_a_benchmark_for_rl_to_expand_beyond_the_base_model/
- Boundary-aware curriculum RL https://arxiv.org/html/2606.22317
- CFGRL https://arxiv.org/abs/2505.23458
- Autoguidance https://arxiv.org/pdf/2406.02507
- ExPO https://huggingface.co/papers/2404.16792
- Inference-aware BoN fine-tuning https://arxiv.org/abs/2412.15287
- Best of N worlds (max@k) https://arxiv.org/pdf/2510.23393
- Avoiding mode collapse in diffusion RL fine-tuning https://arxiv.org/pdf/2410.08315
- GARDO https://arxiv.org/html/2512.24138
- DSRL https://diffusion-steering.github.io/
- Noise trajectory search https://arxiv.org/html/2506.03164v1
- SPIN-Diffusion https://arxiv.org/pdf/2402.10210
- Go-Explore https://arxiv.org/pdf/2004.12919
- TTRL https://arxiv.org/pdf/2504.16084
