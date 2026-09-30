"""Write the M4 spec files (tuning on seeds 0-2, held-out seeds 3-7) and select tuned configs.

usage:
  python scale/make_specs.py tune            -> scale/specs/m4tune.jsonl
  python scale/make_specs.py heldout         -> scale/results/m4tune/SELECTED.json + scale/specs/m4.jsonl
Protocol (pre-registered in DECISIONS D7): every method gets exactly 2 configs on tune seeds 0-2 at budget 0.02;
the one with the higher mean final J is used for every budget on held-out seeds 3-7.
"""
import os
import sys
import json
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from ppa.robo.loop import make_cfg, run_id  # noqa: E402

REGIMES = ["topdown", "classifier"]
ANCHOR_METHODS = ["anchors_only", "naive_mix", "plugin_rm", "ppa_dr_uniform", "ppa_dr_uniform_rhat",
                  "ppa_dr_claimed", "ppa_dr_claimed_rhat"]
PER_REGIME = ANCHOR_METHODS + ["self_only", "filtered_bc_self"]
SHARED = ["oracle", "frozen_selector", "filtered_bc_true"]  # learning does not depend on V -> run once (topdown)
KNOB = {"filtered_bc_self": ("distill_lr", [3e-5, 1e-4]), "filtered_bc_true": ("distill_lr", [3e-5, 1e-4])}
DEFAULT_KNOB = ("critic_lr", [3e-4, 1e-4])
BUDGETS = [0.01, 0.02, 0.05]
TUNE_SEEDS, HELDOUT_SEEDS = [0, 1, 2], [3, 4, 5, 6, 7]


def units():
    for v in REGIMES:
        for m in PER_REGIME:
            yield v, m
    for m in SHARED:
        yield "topdown", m


def tune_specs():
    out = []
    for v, m in units():
        knob, vals = KNOB.get(m, DEFAULT_KNOB)
        for val in vals:
            for s in TUNE_SEEDS:
                out.append(dict(name=m, verifier=v, budget=0.02, seed=s, **{knob: val}))
    return out


def select():
    sel = {}
    for v, m in units():
        knob, vals = KNOB.get(m, DEFAULT_KNOB)
        means = []
        for val in vals:
            fin = []
            for s in TUNE_SEEDS:
                rid = run_id(make_cfg(name=m, verifier=v, budget=0.02, seed=s, **{knob: val}))
                d = json.load(open(os.path.join(ROOT, "scale", "results", "m4tune", rid + ".json")))
                assert d["status"] == "done", rid
                fin.append(d["history"][-1]["J"])
            means.append(float(np.mean(fin)))
        best = int(np.argmax(means))  # ties -> first (default) config
        sel[f"{v}/{m}"] = dict(knob=knob, value=vals[best], tune_means=dict(zip(map(str, vals), means)))
    return sel


def heldout_specs(sel):
    out = []
    for v, m in units():
        s = sel[f"{v}/{m}"]
        budgets = BUDGETS if m in ANCHOR_METHODS else [0.02]
        for b in budgets:
            for seed in HELDOUT_SEEDS:
                out.append(dict(name=m, verifier=v, budget=b, seed=seed, **{s["knob"]: s["value"]}))
    return out


def write(path, specs):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        fh.write("\n".join(json.dumps(x) for x in specs) + "\n")
    print(path, len(specs))


if __name__ == "__main__":
    if sys.argv[1] == "tune":
        write(os.path.join(ROOT, "scale", "specs", "m4tune.jsonl"), tune_specs())
    else:
        sel = select()
        json.dump(sel, open(os.path.join(ROOT, "scale", "results", "m4tune", "SELECTED.json"), "w"), indent=1)
        write(os.path.join(ROOT, "scale", "specs", "m4.jsonl"), heldout_specs(sel))
