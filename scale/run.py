"""Run a list of configs sequentially (resumable) and keep the run registry scale/results/runs.csv.

usage: python scale/run.py <milestone> <specs.jsonl>   # one JSON dict of make_cfg kwargs per line
Completed runs (status "done" in their JSON) are skipped, interrupted runs resume from their checkpoint.
"""
import os
import sys
import csv
import json
import time
import fcntl
import subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from ppa.robo.loop import Run, make_cfg, run_id, cfg_hash  # noqa: E402
from ppa.robo.env import Collector  # noqa: E402

REG = os.path.join(ROOT, "scale", "results", "runs.csv")
FIELDS = ["run_id", "milestone", "git_sha", "config_hash", "seed", "status", "wall_clock_s", "updated"]


def git_sha():
    try:
        return subprocess.check_output(["git", "-C", ROOT, "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def register(row):
    os.makedirs(os.path.dirname(REG), exist_ok=True)
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


def main(milestone, spec_path):
    out = os.path.join(ROOT, "scale", "results", milestone)
    raw = os.path.join(ROOT, "scale", "results", "raw", milestone)
    specs = [json.loads(l) for l in open(spec_path) if l.strip()]
    sha = git_sha()
    cols = None
    for sp in specs:
        cfg = make_cfg(**sp)
        rid = run_id(cfg)
        jp = os.path.join(out, rid + ".json")
        if os.path.exists(jp) and json.load(open(jp)).get("status") == "done":
            continue
        if cols is None:
            cols = {"train": Collector(cfg["task"], cfg["n_envs"]), "eval": Collector(cfg["task"], cfg["n_envs"])}
        base = dict(run_id=rid, milestone=milestone, git_sha=sha, config_hash=cfg_hash(cfg), seed=cfg["seed"])
        register(dict(base, status="running", wall_clock_s="", updated=time.strftime("%F %T")))
        print(f"[{time.strftime('%T')}] start {rid}", flush=True)
        try:
            run = Run(cfg, out, raw, cols)
            hist = run.run()
            run.cleanup()
            register(dict(base, status="done", wall_clock_s=f"{run.wall:.0f}", updated=time.strftime("%F %T")))
            print(f"[{time.strftime('%T')}] done {rid} J: " + " ".join(f"{h['J']:.2f}" for h in hist), flush=True)
        except Exception as e:  # keep going; the run resumes next time
            register(dict(base, status=f"failed:{type(e).__name__}", wall_clock_s="", updated=time.strftime("%F %T")))
            print(f"FAILED {rid}: {e!r}", flush=True)
            import traceback; traceback.print_exc()
    if cols:
        for c in cols.values():
            c.close()


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
