#!/bin/sh
# M-tune (Fetch): every pessimism family gets 3 configs at K = 64 on tune seeds 0-2 (distill lr 3e-5 fixed,
# chosen on the argmax baseline in d1b). Argmax's own tuning budget is its K in {4, 16, 64} (d1b).
cd "$(dirname "$0")/.." || exit 1
python3 -m rsi.loop_fetch tune_fetch '{"rule":["chi2"],"K":[64],"beta":[0.02,0.05,0.1],"seed":[0,1,2]}' 4 >> rsi/results/tune_fetch.log 2>&1
python3 -m rsi.loop_fetch tune_fetch '{"rule":["softmax"],"K":[64],"temp":[0.01,0.03,0.1],"seed":[0,1,2]}' 4 >> rsi/results/tune_fetch.log 2>&1
python3 -m rsi.loop_fetch tune_fetch '{"rule":["lcb"],"K":[64],"kappa":[1.0,2.0,4.0],"seed":[0,1,2]}' 4 >> rsi/results/tune_fetch.log 2>&1
