"""P13 tables: python -m rsi.summarize_stability [results_dir]"""
import sys, json, glob
import numpy as np

d0 = sys.argv[1] if len(sys.argv) > 1 else "stability"
G = {}
for f in glob.glob(f"rsi/results/{d0}/*.json"):
    try:
        d = json.load(open(f))
    except Exception:
        continue
    if not d.get("done"):
        continue
    c = d["cfg"]
    G.setdefault((c["distill_data"], c["gate"]), {})[c["seed"]] = d["hist"]
print("| replay | gate | n | train lift r5 / r10 | eval lift p95 r10 | collapses | J_easy r0 → r10 | J_full r10 | J_target r10 | best J_full | gate accept rate |")
print("|---|---|---|---|---|---|---|---|---|---|---|")
for (dd, g), runs in sorted(G.items()):
    S = sorted(runs)
    L = np.array([[h["max_lift"] for h in runs[s][1:]] for s in S])
    col = int((np.diff(L, axis=1) < -0.03).sum())
    je0 = np.mean([runs[s][0].get("J_easy", np.nan) for s in S]); je = np.mean([runs[s][-1]["J_easy"] for s in S])
    jf = np.mean([runs[s][-1]["J_full"] for s in S]); jt = np.mean([runs[s][-1]["J_sys"] for s in S])
    bf = np.mean([max(h["J_full"] for h in runs[s]) for s in S])
    el = np.mean([runs[s][-1]["eval_lift_p95"] for s in S])
    acc = [h["gate_accept"] for s in S for h in runs[s][1:] if "gate_accept" in h]
    print(f"| {dd == 'replay'} | {g} | {len(S)} | {L[:,4].mean():.3f} / {L[:,9].mean():.3f} | {el:.3f} | {col} | {je0:.2f} → {je:.2f} | "
          f"{jf:.3f} | {jt:.3f} | {bf:.3f} | {np.mean(acc) if acc else float('nan'):.2f} |")
