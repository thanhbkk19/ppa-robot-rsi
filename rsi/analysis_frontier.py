"""M3: why does the take-off frontier stall? (saved critic-argmax model, take-off regime)

For goal heights h below, at and above the frontier, run the system (generator + argmax over K candidates) and
decompose each episode into: grasped (object lifted > 1 cm at some point), max lift, final lift, xy error.
At the first decision where the object is held (lift > 1 cm), sample K candidates and measure their vertical
command 'up' (mean z-displacement of the chunk):
  - coverage: the best 'up' among the K candidates (is an upward chunk proposed at all?)
  - choice:   'up' of the critic's pick, and Spearman(critic score, up) (does the critic prefer it?)
usage: python -m rsi.analysis_frontier <model.pt> [seed]
"""
import sys
import numpy as np
import torch
from scipy.stats import spearmanr

from rsi.models import Diffusion, Critic
from rsi.fetch import Envs, NDEC, OBS_DIM, ACT_DIM, H, STEPS

UP = [2, 6, 10, 14]
TABLE = 0.4247


def main(path, seed=0, K=64, n=50):
    st = torch.load(path, weights_only=False)
    gen = Diffusion(OBS_DIM, ACT_DIM); gen.load_state_dict(st["gen"])
    cr = Critic(OBS_DIM, ACT_DIM, n_ens=len(st["critic"]))
    for net, sd in zip(cr.nets, st["critic"]):
        net.load_state_dict(sd)
    rng = np.random.default_rng(seed)
    print("h    | succ | grasp | max lift | final lift | xy err | held-state: best up among K | up of pick | spearman(Q, up)")
    bins = [0.01, 0.04, 0.07, 0.10, 0.13, 0.20]
    by_lift = {b: dict(best=[], pick=[], rho=[], gen=[]) for b in range(len(bins) - 1)}   # all held decisions, high goals
    for h in [0.05, 0.10, 0.15, 0.20, 0.25]:
        envs = Envs(n); seeds = [9_700_000 + int(h * 1000) * 100 + i for i in range(n)]
        o = envs.reset(seeds, np.full(n, h))
        maxz = np.zeros(n); probed = np.zeros(n, bool); best_up, pick_up, rhos = [], [], []
        for t in range(NDEC):
            S = np.c_[o, np.full(n, t / NDEC)].astype(np.float32)
            C = gen.sample(np.repeat(S, K, 0), rng.integers(1 << 30)).reshape(n, K, ACT_DIM)
            q = cr(np.concatenate([np.repeat(S[:, None], K, 1), C], -1)).mean(0)
            j = q.argmax(1)
            held = (o[:, 5] - TABLE > 0.01) & ~probed
            for i in np.flatnonzero(held):
                up = C[i][:, UP].mean(1)
                best_up.append(up.max()); pick_up.append(up[j[i]])
                if up.std() > 0:
                    rhos.append(spearmanr(q[i], up)[0])
            probed |= held
            if h >= 0.15:   # every held decision, binned by the object's current lift
                for i in np.flatnonzero(o[:, 5] - TABLE > 0.01):
                    b = np.searchsorted(bins, o[i, 5] - TABLE, side="right") - 1
                    if 0 <= b < len(bins) - 1:
                        up = C[i][:, UP].mean(1)
                        by_lift[b]["best"].append(up.max()); by_lift[b]["pick"].append(up[j[i]])
                        by_lift[b]["gen"].append(up.mean())
                        if up.std() > 0:
                            by_lift[b]["rho"].append(spearmanr(q[i], up)[0])
            o, succ = envs.step_chunk(C[np.arange(n), j], min(H, STEPS - t * H))
            maxz = np.maximum(maxz, o[:, 5] - TABLE)
        fin = o[:, 5] - TABLE; xy = np.linalg.norm(o[:, 3:5] - o[:, 25:27], axis=1)
        g = maxz > 0.01
        print(f"{h:.2f} | {succ.mean():.2f} | {g.mean():.2f}  | {np.mean(maxz[g]) if g.any() else 0:.3f}    | "
              f"{np.mean(fin[g]) if g.any() else 0:.3f}      | {xy.mean():.3f}  | "
              f"{np.mean(best_up) if best_up else float('nan'):+.3f}                       | "
              f"{np.mean(pick_up) if pick_up else float('nan'):+.3f}      | {np.mean(rhos) if rhos else float('nan'):+.2f}")
    print("goals >= 0.15, all held decisions by current lift: lift bin | n | mean up of candidates | best up | up of pick | spearman")
    for b, v in by_lift.items():
        if v["best"]:
            print(f"  {bins[b]:.2f}-{bins[b+1]:.2f} | {len(v['best']):4d} | {np.mean(v['gen']):+.3f} | {np.mean(v['best']):+.3f} | "
                  f"{np.mean(v['pick']):+.3f} | {np.mean(v['rho']) if v['rho'] else float('nan'):+.2f}")


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 0)
