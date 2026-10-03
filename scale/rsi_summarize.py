"""Paired verdict tables for the RSI scale-up blocks (pre-registered in rsi/PREREG.md, Amendment P).

usage: python scale/rsi_summarize.py <results dir> [--rounds 10 20] [--json out.json]
  <results dir>: e.g. scale/results/s1 (any directory of loop_fetch result JSONs)

Methods are identified from the config:
- D2 = distill_filter "success";
- B3 = balanced without the filter;
- control = neither.
Only finished runs (done = true) count, and only seeds present for both methods of a pair.
CIs are paired 95% t-intervals over seeds (the seed is the unit of replication).
"""
import sys
import json
import glob
import argparse
import numpy as np
from scipy.stats import t as tdist

METRICS = ("J_target", "J_easy", "J_full")


def method_of(c):
    if c.get("distill_filter", "none") == "success":
        return "D2" if c.get("balanced") else "D1"
    return "B3" if c.get("balanced") else "control"


def load(d):
    runs = {}
    for f in glob.glob(f"{d}/*.json"):
        try:
            j = json.load(open(f))
        except Exception:
            continue
        if not j.get("done"):
            continue
        c = j["cfg"]
        runs.setdefault(method_of(c), {})[c["seed"]] = {h["round"]: dict(J_target=h["J_sys"], J_easy=h.get("J_easy"),
                                                                           J_full=h.get("J_full")) for h in j["hist"]}
    return runs


def paired(A, B, rnd):
    seeds = sorted(s for s in set(A) & set(B) if rnd in A[s] and rnd in B[s])
    out = dict(n=len(seeds), seeds=seeds)
    for m in METRICS:
        a = np.array([A[s][rnd][m] for s in seeds], float); b = np.array([B[s][rnd][m] for s in seeds], float)
        d = a - b
        hw = tdist.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else float("nan")
        out[m] = dict(a=a.mean() if len(d) else np.nan, b=b.mean() if len(d) else np.nan, diff=d.mean() if len(d) else np.nan,
                      lo=d.mean() - hw if len(d) else np.nan, hi=d.mean() + hw if len(d) else np.nan,
                      pos=int((d > 0).sum()))
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("dir"); ap.add_argument("--rounds", type=int, nargs="+", default=[10])
    ap.add_argument("--json"); a = ap.parse_args()
    runs = load(a.dir)
    print("finished runs per method:", {m: len(v) for m, v in runs.items()})
    summary = {}
    for rnd in a.rounds:
        for x, y in (("D2", "control"), ("D2", "B3"), ("B3", "control")):
            if x not in runs or y not in runs:
                continue
            p = paired(runs[x], runs[y], rnd); summary[f"{x}-{y}@r{rnd}"] = p
            print(f"\n### {x} vs {y}, round {rnd} ({p['n']} paired seeds)")
            print("| metric | " + x + " | " + y + " | paired diff | 95% CI | seeds with diff > 0 |")
            print("|---|---|---|---|---|---|")
            for m in METRICS:
                r = p[m]
                print(f"| {m} | {r['a']:.3f} | {r['b']:.3f} | {r['diff']:+.3f} | [{r['lo']:+.3f}, {r['hi']:+.3f}] | "
                      f"{r['pos']}/{p['n']} |")
            if (x, y) == ("D2", "control"):
                ok = (p["J_full"]["diff"] >= 0.05 and p["J_full"]["lo"] > 0 and p["J_target"]["diff"] >= 0.03
                      and p["J_easy"]["diff"] >= -0.05)
                print(f"\nP20 criteria (J_full ≥ +0.05 with CI > 0, J_target ≥ +0.03, J_easy ≥ −0.05): "
                      f"{'HOLD' if ok else 'FAIL'}")
            else:
                print(f"\nJ_full paired CI excludes 0: {'YES' if p['J_full']['lo'] > 0 or p['J_full']['hi'] < 0 else 'NO'}")
    if a.json:
        json.dump(summary, open(a.json, "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
