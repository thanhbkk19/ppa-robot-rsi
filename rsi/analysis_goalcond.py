"""M5: does the generator condition its vertical command on the goal height? At states where the object is held,
mean / 90th-percentile 'up' of generator samples vs goal height (the state is fixed, only the goal input varies),
plus the critic pick's 'up'. usage: python -m rsi.analysis_goalcond <model.pt>"""
import sys, numpy as np, torch
from rsi.models import Diffusion, Critic
from rsi.fetch import Envs, NDEC, OBS_DIM, ACT_DIM, H, STEPS

UP = [2, 6, 10, 14]; TABLE = 0.4247
st = torch.load(sys.argv[1], weights_only=False)
gen = Diffusion(OBS_DIM, ACT_DIM); gen.load_state_dict(st["gen"])
cr = Critic(OBS_DIM, ACT_DIM, n_ens=len(st["critic"]))
for net, sd in zip(cr.nets, st["critic"]):
    net.load_state_dict(sd)
rng = np.random.default_rng(0); n, K = 60, 64
envs = Envs(n); o = envs.reset([9_900_000 + i for i in range(n)], np.full(n, 0.10))
held_S = []
for t in range(NDEC):     # collect held states under a 10 cm goal
    S = np.c_[o, np.full(n, t / NDEC)].astype(np.float32)
    C = gen.sample(np.repeat(S, K, 0), rng.integers(1 << 30)).reshape(n, K, ACT_DIM)
    q = cr(np.concatenate([np.repeat(S[:, None], K, 1), C], -1)).mean(0)
    held_S += [S[i] for i in np.flatnonzero(o[:, 5] - TABLE > 0.01)]
    o, _ = envs.step_chunk(C[np.arange(n), q.argmax(1)], min(H, STEPS - t * H))
Sh = np.array(held_S)[:200]
print(f"{len(Sh)} held states | goal height -> generator mean up / p90 up | critic-pick up")
for h in [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]:
    S2 = Sh.copy(); S2[:, 27] = TABLE + h            # only the goal height changes
    C = gen.sample(np.repeat(S2, K, 0), 1).reshape(len(S2), K, ACT_DIM)
    up = C[:, :, UP].mean(-1)
    q = cr(np.concatenate([np.repeat(S2[:, None], K, 1), C], -1)).mean(0)
    print(f"  h={h:.2f}: {up.mean():+.3f} / {np.quantile(up, 0.9):+.3f} | {up[np.arange(len(up)), q.argmax(1)].mean():+.3f}")
