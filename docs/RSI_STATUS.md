# RSI for robots: where the search stands (2 Oct 2026)

## Update 2 Oct 2026: first held-out-confirmed algorithmic result (P20, `rsi/results/P20_REPORT.md`)
**Outcome-consistent distillation** confirms on held-out seeds 5–9.
- Method: commanded-goal rows are distilled only from successful episodes, and hindsight rows from all
  episodes. Combined with goal-balanced weights and lr 3e-4.
- Against the best fixed-K control: J_target +0.226 [+0.021, +0.431], J_easy +0.152 [+0.066, +0.238],
  J_full +0.216 [+0.090, +0.342].
- Mechanism: mixing failed commanded rows with their hindsight copies labels the same actions with two goals,
  which makes the diffusion generator goal-blind (slope 0.011).
- With the filter, the slope is 0.141, and the generator alone reaches 0.47 on in-air goals (from 0.00).
- This breaks the frontier ↔ easy-goal coupling that sank P14–P19.

Text below this line is the state before this update.


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

## Third loop: take-off beyond the demonstrations (the user chose this problem)
Testbed: demos with table goals only; target goals 0.10–0.30 m in the air, base success 0.00.
- **The standard loop never takes off.** Argmax K = 2, with or without the frontier curriculum and
  hindsight relabelling, stays at 0 for 10 rounds and sharpens back to table behaviour (lift 2.8 → 0.5 cm).
- **Aggressive selection is what moves the frontier.** Argmax K = 64 + curriculum + HER takes off on 1 of 3
  seeds (0.21 target success, 19 cm lift) and starts on a second. χ²-TR (small step) does not take off.
- **Regime dependence.** The optimal χ² step is small inside the demonstration support (exploitation:
  pessimism wins) and large outside it (exploration: optimism wins). This matches the LLM theory that
  sharpening cannot exceed the base policy's coverage without exploration (Huang et al. 2024; XPO).
- **The success-balanced rule fails to unify both regimes.** It picks K_eff with 1 − (1 − m)^K_eff = 1/2.
  It matches argmax K = 2 in the easy regime, is slightly worse in the hard regime, and is too slow to take
  off within 10 rounds (P10).

## Overall verdict of the search
- No algorithm found here beats the simplest tuned baseline in a pre-registered test.
- What is robust and theory-backed:
  - Lemmas 1–2 and the χ² step-size law;
  - the K-inversion;
  - the critic overstates the spread between candidates;
  - the regime-dependent optimal step (small in-distribution, large for take-off).
- These support an analysis paper, not yet an algorithm paper. The most promising algorithmic lead is
  reliable take-off. It needs longer runs (20–30 rounds) and more seeds than 4 CPU cores allow, which the
  RTX 5090 host (24 threads; Fetch is CPU-bound) can provide.

## Artifacts
- `ppa/select.py` + tests: argmax, LCB, softmax, χ², χ²-trust-region selection
- `docs/PA_THEORY.md`: Lemmas 1–3
- `rsi/`: testbeds, runners, results, REPORTs
- Square harness (`ppa/robo/loop.py`) wired for every rule. Ready for the RTX 5090.
