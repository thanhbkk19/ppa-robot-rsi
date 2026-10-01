# RSI for robots: where the search stands (1 Oct 2026)

Loop studied: sample K chunks from a diffusion policy → select with a learned critic → distil the executed
chunks back into the policy. This is PA-RL / V-GPS + distillation, run here on a CPU MuJoCo testbed
(Fetch pick-and-place). All numbers below are from tune seeds 0–2. Held-out seeds 3–7 are running.

## Established
1. **K-inversion.** With critic argmax, more candidates make self-improvement worse.
   - Final success for K = 2 / 3 / 4 / 16 / 64: 0.90 / 0.85 / 0.76 / 0.65 / 0.45 (easy regime, base 0.42).
   - Hard regime (base 0.18): AUC 0.63 at K = 2 vs 0.23 at K = 64.
   - At K = 64, selection is worse than no selection at all.
2. **Step-size law.** The outcome is set by the χ²-distance of the selection from the proposal, not by K or by
   the rule.
   - argmax-of-K has χ² = (K−1)²/(2K−1) (Lemma 2, proved and tested).
   - A χ² trust region at the same χ² gives the same result: δ = 1.3 → 0.78, argmax-of-4 → 0.76.
   - Lemma 1 explains why: the critic error a selection can exploit is at most sqrt(χ²) · sd(error).
3. **Diagnostics behave as the theory says.** Over rounds, the critic's spread across candidates shrinks
   and its optimism about its own choices flips from negative to positive. The large-step rules start
   declining exactly when the optimism turns positive.

## Refuted (pre-registered, see `rsi/PREREG.md`)
- **P3:** χ² selection at large K does not beat tuned small-K argmax. Easy regime: 0.91 vs 0.90.
- **P6:** this does not change when coverage is scarce. Hard regime AUC: 0.59 vs 0.63.

The χ² rule fixes the K-inversion, but so does simply using K = 2. As an algorithm it is not a contribution
on this testbed.

## Second loop (after the user chose "find a new algorithm")
- **Probe of true candidate values.** At most states the K candidates are nearly equivalent in true value.
  The critic overstates their spread, and its Spearman correlation with the true values is 0.34. Selection
  pays off only at a few states.
- **H7 / P7, per-state certified-bound selection (lcbopt, bootstrap ensemble).** Fails: AUC 0.763 vs 0.808.
- **H8 / P8, Rao–Blackwellised distillation.** It shrinks the distillation gap for soft rules, but argmax
  K = 2 with the same re-selection is as good or better.

Conclusion on Fetch: once the χ² step is small, the selection/distillation design space is saturated.
Argmax K = 2 (optionally re-selected with the refit critic) matches every principled alternative we tried.

## Artifacts
- `ppa/select.py` + tests: argmax, LCB, softmax, χ², χ²-trust-region selection
- `docs/PA_THEORY.md`: Lemmas 1–3
- `rsi/`: testbeds, runners, results, REPORTs
- Square harness (`ppa/robo/loop.py`) wired for every rule. Ready for the RTX 5090.
