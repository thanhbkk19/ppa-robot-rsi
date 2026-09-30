# Budget (measured in M1, 1× RTX 5090 + 24 CPU threads)

## Measured
- Simulation-bound: ≈ 5.8 episodes/s with 48 async env workers (400 env steps each), ≈ 21 k episodes/h.
- K = 4 DDPM(20-step) candidate sampling ≈ 15% of wall-clock; GPU memory < 1 GB; critic (3 000 steps) + distillation
  (1 500 steps) ≈ 20 s per round.
- One run = round-0 eval + 6 × (480 train + 192 eval) episodes = 4 224 episodes ≈ **740 s**.
- Two concurrent runs: no gain (CPU-bound) → one run at a time.

## Per run (matched across methods)
| item | value |
|---|---|
| rounds | 6 |
| training episodes / round | 480 (10 labelling batches of 48) |
| training episodes / run | 2 880 (1.15 M env steps) |
| evaluation episodes / round | 192, fixed per seed, never seen by learners |
| anchors at 1 / 2 / 5 % | ≈ 29 / 58 / 144 per run |

## Grid (task = Square; regimes topdown (c) + classifier (a); DECISIONS D2, D8)
| block | configs | seeds | runs |
|---|---|---|---|
| M2 V-only hackability | 2 | 3 | 6 |
| M4 tuning (2 configs per method, budget 2%) | 21 method-units × 2 | 3 | 126 (15 reused from M1/M2) |
| M4 held-out: 7 anchor methods × 3 budgets × 2 regimes + 2 V-methods × 2 regimes + 3 shared | 49 | 5 (3–7) | 245 |
| M3 TD-critic check (oracle, self, plug-in, PPA-DR uniform; topdown 2%) | 4 | 5 | 20 |
| **total** | | | **≈ 380 runs ≈ 78 h ≈ 3.3 GPU-days** |

Within the ≈ 10 GPU-day cap, so no further shrinking. Transport (≈ 2× longer episodes, 2× larger action chunks) is
not in the grid; it would roughly double the cost and is left for after the Square verdict.
