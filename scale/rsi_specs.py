"""Write the RSI scale-up specs (pre-registered in rsi/PREREG.md, Amendment P; budget in scale/BUDGET_RSI.md).

usage: python scale/rsi_specs.py      -> scale/specs/rsi_s1.jsonl, rsi_s2.jsonl

Configs are frozen from P20 (rsi/results/P20_REPORT.md); nothing is re-tuned. Seeds 10-29 were never used before.
"""
import os
import json

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = dict(rule="argmax", K=64, distill_data="replay", gate=False, n_train=400, extra_evals=True, curriculum=True,
            her=True, demo_goals="table", demo_filter=True, train_h=[0.0, 0.3], eval_h=[0.1, 0.3])
METHODS = {
    "control": {},                                                               # replay loop, = best fixed K
    "B3": dict(balanced=True, distill_lr=3e-4),                                  # same loop without the filter
    "D2": dict(balanced=True, distill_lr=3e-4, distill_filter="success"),        # P20 method
}
# S2: 4x training episodes per round, 2x network width, 2x critic / distillation steps (same for every method)
S2_SCALE = dict(n_train=1600, width=512, critic_steps=4000, distill_steps=3000)


def specs():
    out = {"rsi_s1": [], "rsi_s2": []}
    for m, kw in METHODS.items():
        for seed in range(10, 30):
            out["rsi_s1"].append(dict(milestone="s1", method=m, cfg=dict(BASE, **kw, rounds=20, seed=seed)))
        for seed in range(10, 20):
            out["rsi_s2"].append(dict(milestone="s2", method=m, cfg=dict(BASE, **kw, **S2_SCALE, rounds=10, seed=seed)))
    return out


if __name__ == "__main__":
    os.makedirs(os.path.join(ROOT, "scale", "specs"), exist_ok=True)
    for name, rows in specs().items():
        # interleave methods so a partial run of the block is still a paired comparison
        rows.sort(key=lambda r: (r["cfg"]["seed"], r["method"]))
        with open(os.path.join(ROOT, "scale", "specs", name + ".jsonl"), "w") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")
        print(name, len(rows), "runs")
