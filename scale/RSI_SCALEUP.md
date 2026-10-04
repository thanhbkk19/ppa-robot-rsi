# Runbook: RSI scale-up on the RTX 5090 host

What it tests: whether outcome-consistent distillation (P20, `rsi/results/P20_REPORT.md`) holds with:
- 20 fresh seeds and 20 rounds (S1);
- 4× data and 2× network width (S2).

The pre-registered criteria are in `rsi/PREREG.md`, Amendment P.

S3 (a second task) was gated on a CPU pilot (P22, push by direction). The pilot failed, so S3 is not part of this
grid; see `rsi/PREREG.md`, P22 verdict.

## One-time setup
```bash
git fetch origin claude/session-hardware-info-tasocb && git checkout claude/session-hardware-info-tasocb
source .venv/bin/activate          # the env from M0/M1 (PyTorch >= 2.7, CUDA >= 12.8 for sm_120)
pip install -r requirements-toy.txt "gymnasium-robotics>=1.2" mujoco scipy
export MUJOCO_GL=egl
```

## Run everything (resumable: rerun the same command after any interruption)
```bash
mkdir -p scale/results/logs && nohup bash scale/rsi_pipeline.sh > scale/results/logs/pipeline_rsi.out 2>&1 &
```
Steps:
1. Checks CUDA, then runs the unit tests.
2. **S0 profile** (≈ 30–60 min) → `scale/results/s0/profile.json`, recommended number of parallel runs.
3. **S1**: 60 runs → `scale/results/s1/*.json`, verdict tables `scale/results/s1/verdict_tables.md`.
4. **S2**: 30 runs → `scale/results/s2/…`.
5. Mechanism probe (goal slope) on S1 seeds 10–14.

Progress: `scale/results/runs.csv` (one row per run with status and wall-clock) and
`scale/results/logs/s1.log`. Every result JSON is updated after each round.

Override the number of parallel runs with `WORKERS=12 bash scale/rsi_pipeline.sh`.

## What to send back
Commit `scale/results/s0`, `s1`, `s2` (JSON, `verdict_tables.md`, `summary.json`, `mechanism_slope.log`) and
`scale/results/runs.csv`. Checkpoints and models (`*.pt`) are git-ignored. I then write the REPORT.md files with
the verdicts.
