# Budget: RSI scale-up (Amendment P), 1× RTX 5090 + 24 CPU threads

## Measured on the 4-core CPU container (no GPU)
- One 10-round run (n_train 400, K = 64, 1 000 episodes per round including evaluations): 1 450–1 900 s on one
  core (stability / P20 held-out runs, 3–4 runs in parallel on 4 cores).
  → **≈ 150–190 core-seconds per run-round**.
- About half of that is CPU diffusion sampling (64 candidates × 50 envs × 20 denoising steps per decision). The
  rest is env stepping (in-process, 1 core), critic and distillation fits, and evaluations.
- Host RAM ≈ 1.5 GB per run at round 10. The replay grows linearly with rounds, so ≈ 2 GB at round 20 and
  ≈ 4–5 GB for S2 (4× data).

## Grid
| block | runs | rounds | relative cost per round | run-rounds (S1 units) |
|---|---|---|---|---|
| S0 profile | 1 + 4 + 8 + 12 + 16 (+ warm-up) | 2 | 1 | ≈ 100 |
| S1: 3 methods × 20 seeds | 60 | 20 | 1 | 1 200 |
| S2: 3 methods × 10 seeds | 30 | 10 | ≈ 4–5 (4× episodes, 2× width / steps) | ≈ 1 350 |
| **total** | | | | **≈ 2 650** |

## Wall-clock estimate (S0 replaces this with measured numbers)
| assumption | throughput | total |
|---|---|---|
| pessimistic: no GPU gain, 8 effective parallel runs | ≈ 170 run-rounds/h | ≈ 16 h |
| expected: sampling and fits on GPU, 12 parallel runs | ≈ 500 run-rounds/h | ≈ 5–6 h |

- Far below the ≈ 10 GPU-day cap in `CLAUDE.md`, so nothing is shrunk.
- S3 (second task) and P21 (per-goal K on top of D2) fit in the remaining budget once pre-registered.
- Memory guard: S2 runs with half the S1 workers (`scale/rsi_pipeline.sh`). If S0 shows host memory > 80% at
  the recommended worker count, lower `WORKERS`.
