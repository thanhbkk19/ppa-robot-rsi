"""M5 summary over many saved models: generator goal slope (mean up at h=0.30 minus h=0 at fixed held states),
generator p90 up at h=0.20, and critic-pick up at h=0.20. usage: python -m rsi.analysis_slope <glob of .pt> [temp]"""
import sys, glob, numpy as np, torch
from rsi.models import Diffusion, Critic
from rsi.fetch import Envs, NDEC, OBS_DIM, ACT_DIM, H, STEPS

UP = [2, 6, 10, 14]; TABLE = 0.4247


def held_states(gen, cr, K=64, n=40, seed=0):
    rng = np.random.default_rng(seed); envs = Envs(n)
    o = envs.reset([9_900_000 + i for i in range(n)], np.full(n, 0.10)); out = []
    for t in range(NDEC):
        S = np.c_[o, np.full(n, t / NDEC)].astype(np.float32)
        C = gen.sample(np.repeat(S, K, 0), rng.integers(1 << 30)).reshape(n, K, ACT_DIM)
        q = cr(np.concatenate([np.repeat(S[:, None], K, 1), C], -1)).mean(0)
        out += [S[i] for i in np.flatnonzero(o[:, 5] - TABLE > 0.01)]
        o, _ = envs.step_chunk(C[np.arange(n), q.argmax(1)], min(H, STEPS - t * H))
    return np.array(out)[:150]


def main(pattern, temp=1.0, K=64):
    for path in sorted(glob.glob(pattern)):
        st = torch.load(path, weights_only=False)
        gen = Diffusion(OBS_DIM, ACT_DIM); gen.load_state_dict(st["gen"]); gen.temp = temp
        cr = Critic(OBS_DIM, ACT_DIM, n_ens=len(st["critic"]))
        for net, sd in zip(cr.nets, st["critic"]):
            net.load_state_dict(sd)
        Sh = held_states(gen, cr)
        res = {}
        for h in [0.0, 0.20, 0.30]:
            S2 = Sh.copy(); S2[:, 27] = TABLE + h
            C = gen.sample(np.repeat(S2, K, 0), 1).reshape(len(S2), K, ACT_DIM); up = C[:, :, UP].mean(-1)
            q = cr(np.concatenate([np.repeat(S2[:, None], K, 1), C], -1)).mean(0)
            res[h] = (up.mean(), np.quantile(up, 0.9), up[np.arange(len(up)), q.argmax(1)].mean())
        tag = path.split('/')[-1]
        seed = tag.split('seed')[1].split('_')[0]
        print(f"seed {seed} | slope {res[0.30][0] - res[0.0][0]:+.3f} | gen mean up h=0.2 {res[0.2][0]:+.3f} "
              f"| p90 h=0.2 {res[0.2][1]:+.3f} | pick up h=0.2 {res[0.2][2]:+.3f} | {len(Sh)} held states", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], float(sys.argv[2]) if len(sys.argv) > 2 else 1.0)
