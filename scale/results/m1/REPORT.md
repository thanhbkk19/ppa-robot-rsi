# M1 — backbone reproduction (Square, true reward)

Backbone: PA-RL mechanism (K = 4 candidates from the DPPO diffusion-MLP checkpoint → 2-member MC critic argmax →
distil executed chunks + 10% demos), re-implemented in PyTorch (`ppa/robo/`, see DECISIONS D1). Per run: 6 rounds ×
480 training episodes (2 880 episodes, 1.15 M env steps); 192 fixed evaluation episodes per round (disjoint seeds,
never given to any learner). True success J = sim success at the final step.

## 1. Harness vs published checkpoint (tolerance pre-registered in DECISIONS D4: ±0.07)
| Harness | Episodes | Pretrained success |
|---|---|---|
| DPPO `eval_diffusion_mlp` (their script) | 250 | 0.412 |
| ours (`ppa.robo.env.Collector`), final-step / any-time | 960 | 0.379 / 0.380 |

|Δ| = 0.032 → **PASS**.

## 2. Oracle loop on tune seeds 0–2 (criterion D4: mean final J ≥ pretrained + 0.10)
| Config | seed 0 | seed 1 | seed 2 | mean final J | mean AUC (rounds 0–6) |
|---|---|---|---|---|---|
| oracle, distill lr 3e-5 | 0.71 | 0.73 | 0.77 | **0.737** | 0.595 |
| oracle, distill lr 1e-4 | 0.71 | 0.76 | 0.67 | 0.713 | 0.605 |
| frozen generator + selector (no distillation) | 0.48 | 0.41 | 0.52 | 0.470 | 0.432 |

Round-0 J (pretrained generator, untrained critic) is 0.36 / 0.38 / 0.41 on the three seeds' evaluation sets.
Oracle learning curve (lr 3e-5): 0.38 → 0.46 → 0.59 → 0.62 → 0.67 → 0.70 → 0.74 (seed mean).

**Verdict: PASS** (+0.36 over pretrained with lr 3e-5; the criterion needs +0.10). Distill lr 3e-5 is selected for
all methods (higher mean final J on tune seeds). As in the toy, the outer distillation loop is what matters: the
frozen-generator selector gains only +0.09.

## 3. Throughput (1× RTX 5090, 24 CPU threads, 48 env workers)
- ≈ 5.8 episodes/s (≈ 21 k episodes/h); simulation-bound (GPU < 1 GB, K-candidate sampling ≈ 15% of wall-clock).
- One 6-round run (≈ 4 000 episodes incl. evaluation + critic/distil updates) ≈ 740 s.
- See `scale/BUDGET.md` for the M4 grid.
