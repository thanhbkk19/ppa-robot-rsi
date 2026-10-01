"""M2: where do lifted episodes fail? Evaluate a saved critic-argmax model on in-air goals and decompose the
final error into height shortfall, xy error and drops."""
import sys, numpy as np, torch
from rsi.models import Diffusion, Critic
from rsi.fetch import Envs, NDEC, OBS_DIM, ACT_DIM, H, STEPS
from rsi.loop_fetch import heights_for

path, seed, K = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 0, 64
st = torch.load(path, weights_only=False)
gen = Diffusion(OBS_DIM, ACT_DIM); gen.load_state_dict(st["gen"])
cr = Critic(OBS_DIM, ACT_DIM, n_ens=len(st["critic"]))
for n, sd in zip(cr.nets, st["critic"]):
    n.load_state_dict(sd)
rng = np.random.default_rng(seed); n = 50
rows = []
for b in range(4):
    envs = Envs(n); seeds = [9_500_000 + b * 100 + i for i in range(n)]
    hs = heights_for(seeds, [0.1, 0.3]); o = envs.reset(seeds, hs)
    maxz = np.zeros(n)
    for t in range(NDEC):
        S = np.c_[o, np.full(n, t / NDEC)].astype(np.float32)
        C = gen.sample(np.repeat(S, K, 0), rng.integers(1 << 30)).reshape(n, K, ACT_DIM)
        q = cr(np.concatenate([np.repeat(S[:, None], K, 1), C], -1)).mean(0)
        o, succ = envs.step_chunk(C[np.arange(n), q.argmax(1)], min(H, STEPS - t * H))
        maxz = np.maximum(maxz, o[:, 5] - 0.4247)
    obj, goal = o[:, 3:6], o[:, 25:28]
    rows.append(np.c_[hs, obj[:, 2] - 0.4247, maxz, np.linalg.norm(obj[:, :2] - goal[:, :2], axis=1),
                      np.linalg.norm(obj - goal, axis=1), succ])
R = np.concatenate(rows)
h, z, mz, xy, d, s = R.T
print(f"episodes {len(R)} | success {s.mean():.3f}")
print(f"final lift: mean {z.mean():.3f} m, p95 {np.quantile(z, .95):.3f} | max lift during episode p95 {np.quantile(mz, .95):.3f}")
for lo, hi in [(0.1, 0.15), (0.15, 0.2), (0.2, 0.3)]:
    m = (h >= lo) & (h < hi)
    print(f"goal {lo:.2f}-{hi:.2f}: n {m.sum():3d} success {s[m].mean():.2f} | final lift {z[m].mean():.3f} "
          f"(shortfall {np.mean(h[m] - z[m]):.3f}) | xy err {xy[m].mean():.3f} | lifted>2cm {np.mean(z[m] > .02):.2f} "
          f"| dropped (max-final>3cm) {np.mean(mz[m] - z[m] > .03):.2f}")
