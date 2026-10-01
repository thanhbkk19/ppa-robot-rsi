# P11: self-improving CFG in the take-off regime (3 seeds, 10 rounds) — verdict: fails
| method | train lift p95 at rounds 2 / 5 / 10 (m) | final target success |
|---|---|---|
| argmax K2 (BC init) | 0.023 / 0.014 / 0.016 | 0.000 |
| argmax K64 (BC init) | 0.017 / 0.026 / 0.093 | 0.072 |
| chi2tr K64 (BC init) | 0.022 / 0.015 / 0.020 | 0.000 |
| argmax K2 (filtered init) | 0.023 / 0.022 / 0.023 | 0.000 |
| argmax K64 (filtered init) | 0.039 / 0.096 / 0.087 | 0.018 |
| CFG w0 v0 | 0.017 / 0.012 / 0.010 | 0.000 |
| CFG w1 v0 | 0.020 / 0.013 / 0.013 | 0.000 |
| CFG w2 v0 | 0.022 / 0.017 / 0.016 | 0.000 |
| CFG w3 v0 | 0.025 / 0.020 / 0.021 | 0.000 |
| CFG w2 v1 | 0.021 / 0.015 / 0.016 | 0.000 |

- **P11a** (some CFG config with w ≥ 2 reaches target success ≥ 0.05 on ≥ 2/3 seeds): fails (0 on all).
- **P11b** (best CFG ≥ argmax K64, filtered init): fails.

Mechanism:
- **M1** (`M1_DIRECTION.md`): the CFG density ratio is anti-aligned with lifting for in-air goals; the
  critic is aligned.
- **Within CFG, larger w gives slightly more lift** (1.0 → 2.1 cm). The extrapolation is real but tiny,
  because guidance tilts the local density and cannot reach behaviour the success-conditioned model never
  saw.
- **v > 0 (negative guidance) does not help lift**, and its diversity does not differ meaningfully from v = 0.

**M2 (frontier analysis of argmax K64, filtered init, 5 rounds):**
- Per-height training success shows a moving frontier: 5–7.5 cm 0.40, 7.5–10 cm 0.18, 10–12.5 cm 0.15
  (all ≈ 0 at the start).
- **Easy goals are forgotten** (table goals 0.87 → 0.48).
- For goals beyond the frontier (≥ 12.5 cm) the policy does not lift, nor even move the object to the goal
  xy (xy error 0.17 m): an out-of-distribution goal input.
- The uniform target metric (0.10–0.30 m) hides this progress. From now on the frontier height is reported
  as well.
