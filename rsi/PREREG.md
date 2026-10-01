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
