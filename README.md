# PPA: Prediction-Powered Amplification for robot self-improvement

Research code for a robot **recursive self-improvement** loop that stays anchored to reality:

1. **Sample.** Draw K candidate action chunks from a generative policy (diffusion/flow).
2. **Select.** Choose among them with a learned critic or selector.
3. **Distill.** Distill the selected behaviour back into the generator (amplification).
4. **Label.** Label episodes with a cheap **self-verifier** V, corrected by a small random budget of ground-truth checks Y through the prediction-powered (doubly robust) label `Ỹ = V + r̂ + (A/p)(Y − V − r̂)`. This makes the learning signal unbiased for true success even when V is gamed.

**Status (30 Sep 2026):** validated only on CPU toy problems. The scale-up to simulated manipulation (Robomimic, via a PA-RL backbone) is the next step, with pre-registered go/no-go criteria. See `docs/PPA_SPEC.md`.

## Toy headline (multi-step navigation, diffusion generator, 5 held-out seeds)

| Setting | True success (final) |
|---|---|
| Oracle labels (critic learner) | 0.91 |
| Self-verifier only (gamed) | 0.10, with a self-estimate of 0.81, i.e. reward hacking |
| Anchors only, 2% budget | 0.41 |
| **PPA-DR, 2% / 1% anchors** | **0.93 / 0.86** |
| Plug-in reward-model fine-tuning, failure invisible to its features | 0.02 (hacked) |
| PPA-DR, same misspecified residual | 0.73 |

## Repo map
```
CLAUDE.md                 instructions + milestones for the coding agent (start here)
docs/PPA_SPEC.md          research spec: algorithm, math, baselines, go/no-go
docs/RESULTS_TOY.md       toy results log (rounds 1–2)
docs/prior/               earlier spec (frozen-generator + control-variate); kept for provenance
ppa/labels.py             framework-agnostic PPA label contract (use this in the scale-up)
tests/                    unit tests for the label contract (pytest -q)
toy/                      CPU toy code + raw results; `python summarize.py` reprints all tables
scale/                    scale-up plan, grid config, decisions, results (to be filled)
```

## Quickstart
```bash
pip install -r requirements-toy.txt
pytest -q
cd toy && python summarize.py      # reprint toy tables from results/*.json
python r2_run.py                   # rerun the main toy grid (about 30 min on 2 CPU cores)
```
