"""M6: is the easy-goal loss a K-inversion (selection over-optimisation) rather than forgetting?
Evaluate a saved take-off model (generator + critic) on easy (0-5 cm) and high (10-30 cm) goals with K in {1, 2, 4, 64}.
usage: python -m rsi.analysis_kgoal <glob of .pt> [push]   (push: easy 0-45 deg, high 120-180 deg)"""
import sys, glob, numpy as np, torch
from rsi.models import Diffusion, Critic, load_system
from rsi.fetch import Envs, NDEC, OBS_DIM, ACT_DIM, H, STEPS
from rsi.loop_fetch import heights_for


TASK = sys.argv[2] if len(sys.argv) > 2 else "pnp"
EASY, HIGH = ([0.0, np.pi / 4], [2 * np.pi / 3, np.pi]) if TASK == "push" else ([0.0, 0.05], [0.1, 0.3])


def method_of(path):   # from the result JSON next to the model (keys may be hashed)
    import json
    c = json.load(open(path[:-3] + ".json"))["cfg"]
    return "D2" if c.get("distill_filter") == "success" else "B3" if c.get("balanced") else "control"


def evaluate(gen, cr, K, h_range, n=100, seed=0):
    rng = np.random.default_rng(seed); envs = Envs(50, TASK); Ys = []
    seeds = [9_950_000 + i for i in range(n)]
    hs = heights_for(seeds, h_range)
    for b in range(0, n, 50):
        o = envs.reset(seeds[b:b + 50], hs[b:b + 50])
        for t in range(NDEC):
            S = np.c_[o, np.full(50, t / NDEC)].astype(np.float32)
            C = gen.sample(np.repeat(S, K, 0), rng.integers(1 << 30)).reshape(50, K, ACT_DIM)
            j = np.zeros(50, int) if K == 1 else cr(np.concatenate([np.repeat(S[:, None], K, 1), C], -1)).mean(0).argmax(1)
            o, succ = envs.step_chunk(C[np.arange(50), j], min(H, STEPS - t * H))
        Ys.append(succ)
    return float(np.concatenate(Ys).mean())


for path in sorted(glob.glob(sys.argv[1])):
    gen, cr = load_system(path)
    seed = path.split('seed')[1].split('_')[0]
    row = {K: (evaluate(gen, cr, K, EASY), evaluate(gen, cr, K, HIGH)) for K in [1, 2, 4, 64]}
    print(f"seed {seed} | easy  K=1/2/4/64: " + " / ".join(f"{row[K][0]:.2f}" for K in row)
          + " | high K=1/2/4/64: " + " / ".join(f"{row[K][1]:.2f}" for K in row)
          + (" | " + method_of(path) if TASK == "push" else ""),
          flush=True)
