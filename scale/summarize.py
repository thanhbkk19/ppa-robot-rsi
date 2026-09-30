"""Aggregate run JSONs of a milestone into per-config tables (unit of replication = seed).

usage: python scale/summarize.py <milestone> [--seeds 3,4,5,6,7] [--md out.md]
"""
import os
import sys
import json
import glob
import argparse
from collections import defaultdict
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load(milestone, seeds=None):
    runs = []
    for f in glob.glob(os.path.join(ROOT, "scale", "results", milestone, "*.json")):
        d = json.load(open(f))
        if d.get("status") != "done":
            continue
        if seeds is not None and d["cfg"]["seed"] not in seeds:
            continue
        runs.append(d)
    return runs


def key(cfg):
    name = cfg.get("name", cfg["method"])
    return (cfg["verifier"], name, cfg["budget"], cfg["critic_target"], cfg["critic_lr"], cfg["distill_lr"])


def table(runs):
    g = defaultdict(list)
    for d in runs:
        g[key(d["cfg"])].append(d)
    rows = []
    for k, ds in sorted(g.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][2])):
        fin = np.array([d["history"][-1]["J"] for d in ds])
        auc = np.array([np.mean([h["J"] for h in d["history"]]) for d in ds])
        gap = np.array([d["history"][-1]["gap"] for d in ds])
        jself = np.array([d["history"][-1]["J_self"] for d in ds])
        anch = np.array([d["history"][-1]["anchors"] for d in ds])
        wall = np.array([d["wall_clock_s"] for d in ds])
        se = lambda x: x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else float("nan")
        rows.append(dict(verifier=k[0], method=k[1], budget=k[2], target=k[3], critic_lr=k[4], distill_lr=k[5],
                         n=len(ds), J=fin.mean(), J_se=se(fin), auc=auc.mean(), J_self=jself.mean(), gap=gap.mean(),
                         gap_se=se(gap), anchors=anch.mean(), wall=wall.mean(), seeds=sorted(d["cfg"]["seed"] for d in ds),
                         finals=fin.tolist()))
    return rows


def md(rows):
    out = ["| verifier | method | budget | target | critic lr | n | final J (± s.e.) | AUC | J_self | gap J_self−J | anchors | wall s |",
           "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        out.append(f"| {r['verifier']} | {r['method']} | {r['budget']} | {r['target']} | {r['critic_lr']:g} | {r['n']} | "
                   f"{r['J']:.3f} ± {r['J_se']:.3f} | {r['auc']:.3f} | {r['J_self']:.3f} | {r['gap']:+.3f} | "
                   f"{r['anchors']:.0f} | {r['wall']:.0f} |")
    return "\n".join(out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("milestone")
    ap.add_argument("--seeds", default=None)
    ap.add_argument("--md", default=None)
    a = ap.parse_args()
    seeds = None if a.seeds is None else [int(s) for s in a.seeds.split(",")]
    rows = table(load(a.milestone, seeds))
    s = md(rows)
    print(s)
    if a.md:
        open(a.md, "w").write(s + "\n")
