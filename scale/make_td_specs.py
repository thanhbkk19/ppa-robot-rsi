"""M3 TD-critic check: selected (MC-tuned) configs of 4 methods, critic_target=td, topdown 2%, held-out seeds 3-7."""
import os, json
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sel = json.load(open(os.path.join(ROOT, "scale", "results", "m4tune", "SELECTED.json")))
out = []
for m in ["oracle", "self_only", "plugin_rm", "ppa_dr_uniform"]:
    s = sel[f"topdown/{m}"]
    for seed in [3, 4, 5, 6, 7]:
        out.append(dict(name=m, verifier="topdown", budget=0.02, seed=seed, critic_target="td", **{s["knob"]: s["value"]}))
open(os.path.join(ROOT, "scale", "specs", "m3td.jsonl"), "w").write("\n".join(json.dumps(x) for x in out) + "\n")
print(len(out))
