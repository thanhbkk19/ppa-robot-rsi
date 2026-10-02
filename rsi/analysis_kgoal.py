"""M6: is the easy-goal loss a K-inversion (selection over-optimisation) rather than forgetting?
Evaluate a saved take-off model (generator + critic) on easy (0-5 cm) and high (10-30 cm) goals with K in {1, 2, 4, 64}.
usage: python -m rsi.analysis_kgoal <glob of .pt>"""
import sys, glob, numpy as np, torch
from rsi.models import Diffusion, Critic
from rsi.fetch import Envs, NDEC, OBS_DIM, ACT_DIM, H, STEPS
from rsi.loop_fetch import heights_for


def evaluate(gen, cr, K, h_range, n=100, seed=0):
    rng = np.random.default_rng(seed); envs = Envs(50); Ys = []
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
    st = torch.load(path, weights_only=False)
    gen = Diffusion(OBS_DIM, ACT_DIM); gen.load_state_dict(st["gen"])
    cr = Critic(OBS_DIM, ACT_DIM, n_ens=len(st["critic"]))
    for net, sd in zip(cr.nets, st["critic"]):
        net.load_state_dict(sd)
    seed = path.split('seed')[1].split('_')[0]
    row = {K: (evaluate(gen, cr, K, [0.0, 0.05]), evaluate(gen, cr, K, [0.1, 0.3])) for K in [1, 2, 4, 64]}
    print(f"seed {seed} | easy  K=1/2/4/64: " + " / ".join(f"{row[K][0]:.2f}" for K in row)
          + " | high K=1/2/4/64: " + " / ".join(f"{row[K][1]:.2f}" for K in row), flush=True)
