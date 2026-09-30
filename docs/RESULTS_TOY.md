# Toy verification: generator–selector amplification loop (30 Sep 2026)

CPU-only toy experiments (2 cores). They test internal consistency and the regimes where the idea works. They are **not** robot evidence.

## Setups
- **Exp A/B, contextual bandit** (`bandit.py`, `baselines.py`). 6 contexts, 2-D actions. The good region is a disk with P(success) = 0.95 inside and 0.02 outside. Demos put a fraction eps on the good mode. Generator = per-context GMM (6 components). Selector = softmax over K candidates on 144 RBF features, trained by factual on-policy PG with a predictable baseline and plain SGD. Distillation = GMM refit on selector-induced samples plus 5% demos.
- **Exp C, multi-step** (`multistep.py`). 2-D navigation, T = 6 macro-steps, sparse terminal success, walls with a trap route and a narrow good route (eps = 0.1 of demos). Generator = tiny conditional DDPM (20 steps). Selector features = RBF of the nominal next position. Distillation = fine-tuning the diffusion model on executed (on-policy, *unfiltered*) selector actions plus 10% demos. Each round is 800 episodes.

## Results
1. **One-step map is accurate.** Predicted π = q(1−(1−m)^K) vs measured: corr 0.983, median relative error 3.6% (296 rounds).
2. **The threshold exists, but q drives it, not δ.** q stays at chance (qK ≈ 1.0) when the selector sees fewer than about 1 success per context per round. qK rises to 1.5 at 10–30 successes and to about 5 at more than 100. So q = q(n_sel·K·m). With eps ≤ 0.003 the loop never took off.
3. **The rounds formula under-predicts** by about 1.5–2.5 rounds, because q(m) is small early on.
4. **The outer loop is the key ingredient.** In the multi-step task at lr = 30 (3 seeds for amplify_warm; 1 seed for the frozen-selector row, which is the only frozen run at that lr):
   - frozen generator + selector (current spec): plateaus at J ≈ 0.30
   - amplify, selector reset each round: 0.72–0.83
   - amplify, warm selector: 0.89–0.96
   - critic-argmax + distill (PA-RL-like): 0.93
   - filtered BC: 0.56
   - The generator alone went from 0.03 to 0.65.
5. **The factual-PG selector is not better than critic selection** in these toys. It is very sensitive to lr: at lr = 2 in the multi-step task it failed completely (J ≈ 0.04), while at lr = 10–30 it matched the critic. In the one-step bandit, filtered BC was the most sample-efficient (eps = 0.01, J = 0.84 at 16k episodes vs 0.79 for tuned amplify and 0.73 for critic).
6. **The spec's model control variate (G_α with OGD on α) gave no measurable benefit** (0.47 vs 0.51, within noise).

## Implications
- Keep: the outer distillation loop, Lemma A, and the amplification map (empirically predictive).
- Revise the theory so that q depends on m (a learnability threshold). This is the true bistability mechanism here.
- Novelty cannot rest on "PG selector beats critic selection". The unbiasedness advantage needs a test with a misspecified or hackable critic (aliased features, noisy success detector). The critic here had well-specified features.

## Round 2 (30 Sep 2026): tuned baselines and a hackable self-verifier
See PPA_SPEC.md §1 for the full tables. Scripts: ms2.py, r1_tune.py, r2_run.py, r2b_run.py, r2c_run.py, r2d_run.py. Raw data: r1_tune.json, r2*.json.
- With the true reward and tuned hyperparameters (held-out seeds 3–7), the PG selector and the PA-RL-like critic tie on AUC (0.758 vs 0.751). Frozen generator: 0.28.
- With a verifier that is fooled on a trap region, every self-reward method is hacked (true J 0–0.10 while self-estimated J is 0.8–0.9). DR/PPI labels with 1–5% anchors recover 0.86–0.93 (oracle 0.91). Anchors-only gets 0.20–0.79.
- Plug-in reward-model fine-tuning works when its features can see the failure (0.92). It is fully hacked when they cannot (0.02), whereas DR still reaches 0.73.
- Active anchoring and online power tuning showed no consistent gain.
