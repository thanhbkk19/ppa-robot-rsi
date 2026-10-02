"""M7: is the FAS progress reward aligned with take-off on in-air goals?
For saved models, on in-air goals (10-30 cm) and K in {2, 64}: success, mean progress reward
(max(success, 1 - final distance / initial distance)), and its decomposition: final xy error, final lift,
max lift during the episode.
usage: python -m rsi.analysis_proxy <glob of .pt>"""
import sys, glob, numpy as np, torch
from rsi.models import Diffusion, Critic
from rsi.fetch import Envs, NDEC, OBS_DIM, ACT_DIM, H, STEPS
from rsi.loop_fetch import heights_for

TABLE = 0.4247


def rollout(gen, cr, K, h_range, n=100, seed=0):
    rng = np.random.default_rng(seed); envs = Envs(50); out = []
    seeds = [9_960_000 + i for i in range(n)]
    hs = heights_for(seeds, h_range)
    for b in range(0, n, 50):
        o = envs.reset(seeds[b:b + 50], hs[b:b + 50])
        d0 = np.linalg.norm(o[:, 25:28] - o[:, 3:6], axis=1); maxz = np.zeros(50)
        for t in range(NDEC):
            S = np.c_[o, np.full(50, t / NDEC)].astype(np.float32)
            C = gen.sample(np.repeat(S, K, 0), rng.integers(1 << 30)).reshape(50, K, ACT_DIM)
            j = cr(np.concatenate([np.repeat(S[:, None], K, 1), C], -1)).mean(0).argmax(1)
            o, succ = envs.step_chunk(C[np.arange(50), j], min(H, STEPS - t * H))
            maxz = np.maximum(maxz, o[:, 5] - TABLE)
        d1 = np.linalg.norm(o[:, 25:28] - o[:, 3:6], axis=1)
        prog = np.maximum(succ, np.clip(1 - d1 / np.maximum(d0, 1e-6), 0, 1))
        out.append(np.c_[succ, prog, np.linalg.norm(o[:, 3:5] - o[:, 25:27], axis=1), o[:, 5] - TABLE, maxz])
    return np.concatenate(out).mean(0)


if __name__ == "__main__":
    print("seed | K  | succ | progress reward | final xy err | final lift | max lift")
    for path in sorted(glob.glob(sys.argv[1])):
        st = torch.load(path, weights_only=False)
        gen = Diffusion(OBS_DIM, ACT_DIM); gen.load_state_dict(st["gen"])
        cr = Critic(OBS_DIM, ACT_DIM, n_ens=len(st["critic"]))
        for net, sd in zip(cr.nets, st["critic"]):
            net.load_state_dict(sd)
        seed = path.split('seed')[1].split('_')[0]
        for K in (2, 64):
            s, p, xy, z, mz = rollout(gen, cr, K, [0.1, 0.3])
            print(f"{seed}    | {K:2d} | {s:.2f} | {p:.3f}           | {xy:.3f}        | {z:+.3f}     | {mz:.3f}",
                  flush=True)
