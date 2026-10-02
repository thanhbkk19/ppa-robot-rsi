# Why the take-off frontier stalls at 10–15 cm (replay models, held-out seeds 3 and 5)
Scripts: `rsi/analysis_frontier.py` (M3), `rsi/analysis_timing.py` (M4), `rsi/analysis_goalcond.py` (M5).

**M3, coverage and choice are fine.** At held states with high goals, binned by the current lift up to
13–20 cm:
- the best upward candidate among K = 64 has up +0.13 to +0.25;
- the critic's pick has up +0.06 to +0.17;
- Spearman(Q, up) is +0.53 to +0.74.

The critic proposes and chooses lifting at every height. The object is grasped in 54–100% of episodes, but
the max lift saturates near 9.5 cm (seed 3) whether the goal is 15 or 20 cm.

**M4, time budget (seed 3, 50 episodes per goal).**

| goal | first held decision | lift per decision after the grasp | lift at the end (13 decisions) |
|---|---|---|---|
| 15 cm, system | ≈ 4 | ≈ 0.9 cm | 7.9 cm |
| 15 cm, scripted controller | ≈ 3–4 | ≈ 7–8 cm (14 cm in 2 decisions) | 14.2 cm |
| 25 cm, system | ≈ 4 | ≈ 0 | ≈ 0 |

Frontier ≈ (13 − 4) decisions × ≈ 1 cm + the 5 cm success tolerance ≈ 14 cm, which matches the observed
10–15 cm. The frontier is a lifting-speed × time-budget limit.

**M5, the generator ignores the goal height.** At 200 fixed held states, only the goal input changed:

| goal height | 0 | 0.10 | 0.20 | 0.30 |
|---|---|---|---|---|
| generator mean up (seed 3) | −0.005 | 0.000 | +0.005 | +0.009 |
| generator p90 up (seed 3) | 0.066 | 0.069 | 0.071 | 0.074 |
| critic pick up (seed 3) | −0.097 | +0.079 | +0.093 | +0.002 |

Seed 5 is similar (generator −0.006 → +0.022; picks up to +0.17).

Reading:
1. **All goal dependence lives in the critic. The generator learned the goal-marginal behaviour.** Its
   weights on the goal-height input were trained on table-only demos, where that input never varies, and a
   small-lr fine-tune (3e-5 × 1 500 steps per round) barely moves them. Each round therefore moves the
   system only as far as one tail pick of a goal-blind generator: about 1 cm per decision.
2. **The critic's goal extrapolation fades beyond about 25 cm** (pick up drops to +0.002 at 30 cm on seed 3).
3. The candidate spread (p90 ≈ 0.07–0.11) caps the per-decision reach. Best-of-K gain scales like
   σ·sqrt(2 ln K), so widening σ (sampling temperature) is cheaper than raising K.
