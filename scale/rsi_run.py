"""Run the RSI scale-up specs (rsi/loop_fetch.py configs) with N parallel worker processes.

usage: python scale/rsi_run.py <specs.jsonl> [--workers N] [--retries R]
  each line: {"milestone": "s1", "cfg": {...loop_fetch config...}}

- One subprocess per run, so each has its own CUDA context and an OOM kill loses only that run.
- Results: scale/results/<milestone>/<config key>.json, written every round. Final models (.pt) are git-ignored.
- Resumable:
  - finished runs (done = true) are skipped;
  - interrupted runs restart from their last per-round checkpoint (<key>.ckpt.pt), with identical results;
  - a failed run is retried up to R times.
- Registry: scale/results/runs.csv (run id, milestone, git SHA, config hash, seed, status, wall-clock).
- Device: set RSI_DEVICE=cuda in the environment (inherited by the runs).
"""
import os
import sys
import csv
import json
import time
import fcntl
import hashlib
import argparse
import subprocess
import datetime
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "scale", "results")
REG = os.path.join(RES, "runs.csv")
FIELDS = ["run_id", "milestone", "git_sha", "config_hash", "seed", "status", "wall_clock_s", "updated"]


def git_sha():
    try:
        return subprocess.check_output(["git", "-C", ROOT, "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


sys.path.insert(0, ROOT)
from rsi.runkey import run_key as cfg_key  # noqa: E402  (same naming as rsi/loop_fetch.py:job)


def cfg_hash(cfg):
    return hashlib.sha1(json.dumps({k: v for k, v in cfg.items() if k != "seed"}, sort_keys=True).encode()).hexdigest()[:10]


def register(row):
    os.makedirs(RES, exist_ok=True)
    with open(REG + ".lock", "w") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        rows = []
        if os.path.exists(REG):
            with open(REG) as fh:
                rows = [r for r in csv.DictReader(fh) if r["run_id"] != row["run_id"]]
        rows.append(row)
        with open(REG + ".tmp", "w", newline="") as fh:
            w = csv.DictWriter(fh, FIELDS); w.writeheader(); w.writerows(rows)
        os.replace(REG + ".tmp", REG)


def is_done(ms, cfg):
    f = os.path.join(RES, ms, cfg_key(cfg) + ".json")
    try:
        return json.load(open(f)).get("done", False)
    except Exception:
        return False


def run_one(spec, retries, sha):
    ms, cfg = spec["milestone"], spec["cfg"]
    rid = f"rsi_{ms}_{cfg_hash(cfg)}_s{cfg['seed']}"
    row = dict(run_id=rid, milestone=ms, git_sha=sha, config_hash=cfg_hash(cfg), seed=cfg["seed"])
    if is_done(ms, cfg):
        return rid, "skipped"
    os.makedirs(os.path.join(RES, "logs"), exist_ok=True)
    log = os.path.join(RES, "logs", f"{ms}.log")
    env = dict(os.environ, RSI_RESULTS=RES, PYTHONPATH=ROOT)
    code = "import sys, json; from rsi.loop_fetch import job; job((sys.argv[1], json.loads(sys.argv[2])))"
    status = "failed"
    for attempt in range(retries + 1):
        t = time.time()
        register(dict(row, status=f"running(try {attempt + 1})", wall_clock_s="",
                      updated=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
        with open(log, "a") as fh:
            p = subprocess.run([sys.executable, "-c", code, ms, json.dumps(cfg)], cwd=ROOT, env=env, stdout=fh, stderr=subprocess.STDOUT)
        if p.returncode == 0 and is_done(ms, cfg):
            status = "done"
        register(dict(row, status=status if status == "done" else f"failed(rc {p.returncode})",
                      wall_clock_s=int(time.time() - t), updated=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
        if status == "done":
            break
    return rid, status


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("specs"); ap.add_argument("--workers", type=int, default=8); ap.add_argument("--retries", type=int, default=1)
    a = ap.parse_args()
    specs = [json.loads(l) for l in open(a.specs) if l.strip()]
    sha = git_sha()
    print(f"{len(specs)} runs, {sum(is_done(s['milestone'], s['cfg']) for s in specs)} already done, "
          f"{a.workers} workers, device {os.environ.get('RSI_DEVICE', 'cpu')}", flush=True)
    with ThreadPoolExecutor(a.workers) as ex:
        for rid, st in ex.map(lambda s: run_one(s, a.retries, sha), specs):
            print(f"{datetime.datetime.now():%H:%M:%S} {rid}: {st}", flush=True)


if __name__ == "__main__":
    main()
