"""M9 (push task): is the generator goal-sensitive? The push analogue of the M8 goal slope.

At the same initial states (50 seeds), set the goal in front of the object (angle 0) or behind it (angle pi) at the
same distance. Sample 64 first-decision chunks from the generator for each and report, per model:
- sensitivity: |E[a | back] - E[a | front]| over the xyz displacement entries of the chunk (0 = goal-blind);
- dx_front / dx_back: mean x displacement of the chunk. Pushing towards the back requires approaching from +x,
  pushing towards the front from -x.
usage: python -m rsi.analysis_push <glob of .pt>
"""
import sys
import glob
import numpy as np
from rsi.models import load_system
from rsi.fetch import Envs

XYZ = [i for j in range(4) for i in (4 * j, 4 * j + 1, 4 * j + 2)]
DX = [0, 4, 8, 12]


def probe(gen, n=50, K=64, seed=0):
    envs = Envs(n, "push"); seeds = [9_990_000 + i for i in range(n)]
    out = {}
    for name, ang in (("front", 0.0), ("back", np.pi)):
        o = envs.reset(seeds)
        for i, e in enumerate(envs.envs):   # same distance for both goals: 0.15 m at the exact angle (no sign flip)
            u = e.unwrapped; obj = o[i, 3:6]
            u.goal = obj + 0.15 * np.array([np.cos(ang), np.sin(ang), 0.0])
            o[i, 25:28] = u.goal
        S = np.c_[o, np.zeros(n)].astype(np.float32)
        C = gen.sample(np.repeat(S, K, 0), seed).reshape(n, K, -1)
        out[name] = C.mean(1)                       # (n, 16) mean chunk per state
    sens = np.linalg.norm(out["back"][:, XYZ] - out["front"][:, XYZ], axis=1).mean()
    return sens, out["front"][:, DX].mean(), out["back"][:, DX].mean()


if __name__ == "__main__":
    for path in sorted(glob.glob(sys.argv[1])):
        gen, _ = load_system(path)
        s, f, b = probe(gen)
        seed = path.split("seed")[1].split("_")[0] if "seed" in path else path.rsplit("_s", 1)[-1][:-3]
        print(f"seed {seed} | sensitivity {s:.3f} | dx front {f:+.3f} | dx back {b:+.3f} | {path.split('/')[-1][:60]}", flush=True)
