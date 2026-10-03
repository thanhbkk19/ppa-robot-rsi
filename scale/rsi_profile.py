"""S0: profile the RSI loop on this host and pick the number of parallel runs.

usage: RSI_DEVICE=cuda python scale/rsi_profile.py [--workers 1 4 8 12 16]

For each worker count N it runs N independent 2-round D2 runs (seeds 1000+, never used elsewhere, n_train 400) at
the same time and measures:
- throughput in run-rounds per hour;
- peak GPU memory (nvidia-smi);
- host RSS.
Writes scale/results/s0/profile.json and prints the recommended --workers: the highest throughput, provided
host memory stays below 80%.
"""
import os
import sys
import json
import time
import argparse
import subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "scale", "results", "s0")
CFG = dict(rule="argmax", K=64, distill_data="replay", gate=False, n_train=400, extra_evals=True, curriculum=True,
           her=True, demo_goals="table", demo_filter=True, train_h=[0.0, 0.3], eval_h=[0.1, 0.3], balanced=True,
           distill_lr=3e-4, distill_filter="success", rounds=2, ckpt=False)


def gpu_mem():
    try:
        q = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
                                    text=True).split("\n")[0].split(",")
        return int(q[0]), int(q[1])
    except Exception:
        return None, None


def host_mem():
    info = dict(l.split(":")[0:2] for l in open("/proc/meminfo"))
    tot = int(info["MemTotal"].split()[0]); avail = int(info["MemAvailable"].split()[0])
    return (tot - avail) / tot


def trial(n, tag):
    env = dict(os.environ, RSI_RESULTS=os.path.join(OUT, "runs"), PYTHONPATH=ROOT)
    code = "import sys, json; from rsi.loop_fetch import job; job((sys.argv[1], json.loads(sys.argv[2])))"
    t = time.time(); procs = []
    for i in range(n):
        cfg = dict(CFG, seed=1000 + i)
        procs.append(subprocess.Popen([sys.executable, "-c", code, f"{tag}_n{n}", json.dumps(cfg)], cwd=ROOT, env=env,
                                      stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))
    peak_gpu, peak_host = 0, 0.0
    while any(p.poll() is None for p in procs):
        g, _ = gpu_mem(); peak_gpu = max(peak_gpu, g or 0); peak_host = max(peak_host, host_mem()); time.sleep(5)
    wall = time.time() - t
    ok = sum(p.returncode == 0 for p in procs)
    # the first run of each seed also pre-trains its BC policy (cached afterwards); report it separately
    return dict(workers=n, ok=ok, wall_s=round(wall), run_rounds_per_h=round(2 * ok / wall * 3600, 1),
                peak_gpu_mb=peak_gpu, peak_host_mem_frac=round(peak_host, 3))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--workers", type=int, nargs="+", default=[1, 4, 8, 12, 16])
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    _, gpu_total = gpu_mem()
    res = dict(device=os.environ.get("RSI_DEVICE", "cpu"), cpu_threads=os.cpu_count(), gpu_total_mb=gpu_total,
               warmup=trial(max(a.workers), "warmup"))   # builds the BC caches for seeds 1000+ so trials time the loop only
    res["trials"] = []
    for n in a.workers:
        r = trial(n, "prof"); res["trials"].append(r); print(r, flush=True)
        json.dump(res, open(os.path.join(OUT, "profile.json"), "w"), indent=1)
    good = [r for r in res["trials"] if r["ok"] == r["workers"] and r["peak_host_mem_frac"] < 0.8]
    best = max(good, key=lambda r: r["run_rounds_per_h"]) if good else None
    res["recommended_workers"] = best["workers"] if best else 1
    json.dump(res, open(os.path.join(OUT, "profile.json"), "w"), indent=1)
    print("recommended --workers", res["recommended_workers"])


if __name__ == "__main__":
    main()
