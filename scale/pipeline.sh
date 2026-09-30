#!/usr/bin/env bash
# Chain: finish tuning -> select configs -> held-out grid -> TD check. Each step is resumable/idempotent.
set -e
cd "$(dirname "$0")/.."
source .venv/bin/activate
while pgrep -f "scale/run.py m4tune" >/dev/null; do sleep 60; done
python scale/run.py m4tune scale/specs/m4tune.jsonl >> scale/results/logs/m4tune.log 2>&1   # retries any failed run
python scale/make_specs.py heldout
python scale/run.py m4 scale/specs/m4.jsonl >> scale/results/logs/m4.log 2>&1
python scale/make_td_specs.py
python scale/run.py m3td scale/specs/m3td.jsonl >> scale/results/logs/m3td.log 2>&1
echo PIPELINE_DONE >> scale/results/logs/pipeline.log
