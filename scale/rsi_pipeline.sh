#!/usr/bin/env bash
# RSI scale-up on the GPU host: S0 profile -> S1 -> S2 -> verdict tables. Every step is resumable / idempotent:
# rerun this script after any interruption and it continues where it stopped.
#   usage: bash scale/rsi_pipeline.sh            (optional: WORKERS=12 to skip the profile's recommendation)
set -e
cd "$(dirname "$0")/.."
[ -f .venv/bin/activate ] && source .venv/bin/activate
export RSI_DEVICE=${RSI_DEVICE:-cuda} MUJOCO_GL=${MUJOCO_GL:-egl}
mkdir -p scale/results/logs scale/results/s1 scale/results/s2

python -c "import torch; assert torch.cuda.is_available() or '$RSI_DEVICE' == 'cpu', 'no CUDA device'; print(torch.__version__, torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"
pytest -q tests >/dev/null && echo "unit tests ok"

if [ -z "$WORKERS" ]; then
  [ -f scale/results/s0/profile.json ] || python scale/rsi_profile.py 2>&1 | tee scale/results/logs/s0.log
  WORKERS=$(python -c "import json; print(json.load(open('scale/results/s0/profile.json'))['recommended_workers'])")
fi
# S2 holds 4x the replay per run: use half the workers (at least 1)
W2=$(( WORKERS / 2 > 0 ? WORKERS / 2 : 1 ))
echo "workers: S1 $WORKERS, S2 $W2"

python scale/rsi_specs.py
python scale/rsi_run.py scale/specs/rsi_s1.jsonl --workers "$WORKERS" 2>&1 | tee -a scale/results/logs/s1_runner.log
python scale/rsi_summarize.py scale/results/s1 --rounds 10 20 --json scale/results/s1/summary.json | tee scale/results/s1/verdict_tables.md
python scale/rsi_run.py scale/specs/rsi_s2.jsonl --workers "$W2" 2>&1 | tee -a scale/results/logs/s2_runner.log
python scale/rsi_summarize.py scale/results/s2 --rounds 10 --json scale/results/s2/summary.json | tee scale/results/s2/verdict_tables.md

# mechanism probes (M8 goal slope, M6 K-by-goal) on seeds 10-14 of S1
for s in 10 11 12 13 14; do
  python -m rsi.analysis_slope "scale/results/s1/*seed${s}_*.pt" >> scale/results/s1/mechanism_slope.log 2>&1 || true
done
echo RSI_PIPELINE_DONE | tee -a scale/results/logs/pipeline_rsi.log
