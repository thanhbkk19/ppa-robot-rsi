"""Paired per-seed comparison of two result sets at the final round (t-based 95% CI over seeds).
usage: python -m rsi.paired <dir_a> <glob_a> <dir_b> <glob_b>      (a = method, b = comparator)"""
import sys, glob, json
import numpy as np
from scipy.stats import t as tdist


def load(d, pat):
    out = {}
    for f in glob.glob(f"rsi/results/{d}/{pat}"):
        if not f.endswith(".json"):
            continue
        j = json.load(open(f))
        if j.get("done"):
            h = j["hist"][-1]
            out[j["cfg"]["seed"]] = dict(J_target=h["J_sys"], J_easy=h["J_easy"], J_full=h["J_full"])
    return out


def main(da, pa, db, pb):
    A, B = load(da, pa), load(db, pb)
    seeds = sorted(set(A) & set(B))
    print(f"seeds {seeds}")
    print("| metric | method | comparator | paired diff | 95% CI | per-seed diffs |")
    print("|---|---|---|---|---|---|")
    for m in ("J_target", "J_easy", "J_full"):
        a = np.array([A[s][m] for s in seeds]); b = np.array([B[s][m] for s in seeds]); d = a - b
        hw = tdist.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else float("nan")
        print(f"| {m} | {a.mean():.3f} | {b.mean():.3f} | {d.mean():+.3f} | [{d.mean() - hw:+.3f}, {d.mean() + hw:+.3f}] | "
              + " ".join(f"{x:+.2f}" for x in d) + " |")


if __name__ == "__main__":
    main(*sys.argv[1:5])
