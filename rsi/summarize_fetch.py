"""Tables for Fetch results: python -m rsi.summarize_fetch <dir> [<dir> ...] [--seeds 3,4,5,6,7]
Unit of replication = seed. final = J_sys after the last round; AUC = mean J_sys over rounds 0..R."""
from __future__ import annotations
import sys
import json
import glob
import argparse
from collections import defaultdict
import numpy as np
from scipy import stats

PARAM = {"chi2": "beta", "softmax": "temp", "lcb": "kappa", "argmax": None}


def load(dirs, seeds=None, lr=3e-5):
    runs = []
    for d in dirs:
        for f in glob.glob(f"rsi/results/{d}/*.json"):
            r = json.load(open(f))
            if not r.get("done") or (seeds and r["cfg"]["seed"] not in seeds) or r["cfg"]["distill_lr"] != lr:
                continue
            runs.append(r)
    return runs


def key(c):
    p = PARAM[c["rule"]] if c["K"] > 1 else None
    return (c["rule"] if c["K"] > 1 else "none", c["K"], None if p is None else c[p])


def table(runs):
    g = defaultdict(dict)
    for r in runs:
        g[key(r["cfg"])][r["cfg"]["seed"]] = r["hist"]
    rows = []
    for k in sorted(g, key=lambda k: (k[0], k[1], k[2] or 0)):
        H = g[k]
        fin = np.array([H[s][-1]["J_sys"] for s in sorted(H)])
        auc = np.array([np.mean([h["J_sys"] for h in H[s]]) for s in sorted(H)])
        opt = np.array([np.nanmean([h["opt0"] for h in H[s][1:]]) for s in sorted(H)])
        se = fin.std(ddof=1) / np.sqrt(len(fin)) if len(fin) > 1 else float("nan")
        rows.append(dict(rule=k[0], K=k[1], param=k[2], n=len(fin), final=fin.mean(), se=se, auc=auc.mean(),
                         opt=opt.mean(), finals=dict(zip(sorted(H), fin.round(3).tolist()))))
    return rows, g


def paired(g, a, b):
    """Paired difference of final J (a - b) over common seeds, with a 95% t-interval."""
    s = sorted(set(g[a]) & set(g[b]))
    d = np.array([g[a][i][-1]["J_sys"] - g[b][i][-1]["J_sys"] for i in s])
    if len(d) < 2:
        return dict(n=len(d), mean=float(d.mean()) if len(d) else float("nan"))
    h = stats.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))
    return dict(n=len(d), mean=float(d.mean()), lo=float(d.mean() - h), hi=float(d.mean() + h))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--seeds", default=None)
    a = ap.parse_args()
    seeds = None if a.seeds is None else [int(x) for x in a.seeds.split(",")]
    rows, _ = table(load(a.dirs, seeds))
    print("| rule | K | param | n | final J ± s.e. | AUC | mean opt0 | per seed |")
    print("|---|---|---|---|---|---|---|---|")
    for r in rows:
        print(f"| {r['rule']} | {r['K']} | {r['param']} | {r['n']} | {r['final']:.3f} ± {r['se']:.3f} | "
              f"{r['auc']:.3f} | {r['opt']:+.3f} | {r['finals']} |")
