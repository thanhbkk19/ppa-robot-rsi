"""M4: is the frontier limited by the time budget? Lift trajectory per decision for high goals (saved model),
and the decision at which the object is first held. Compared with the scripted controller (physical reference)."""
import sys, numpy as np, torch
from rsi.models import Diffusion, Critic
from rsi.fetch import Envs, NDEC, OBS_DIM, ACT_DIM, H, STEPS, scripted

TABLE = 0.4247


def run_system(path, h, n=50, K=64, seed=0):
    st = torch.load(path, weights_only=False)
    gen = Diffusion(OBS_DIM, ACT_DIM); gen.load_state_dict(st["gen"])
    cr = Critic(OBS_DIM, ACT_DIM, n_ens=len(st["critic"]))
    for net, sd in zip(cr.nets, st["critic"]):
        net.load_state_dict(sd)
    rng = np.random.default_rng(seed); envs = Envs(n)
    o = envs.reset([9_800_000 + i for i in range(n)], np.full(n, h))
    lifts = []
    for t in range(NDEC):
        S = np.c_[o, np.full(n, t / NDEC)].astype(np.float32)
        C = gen.sample(np.repeat(S, K, 0), rng.integers(1 << 30)).reshape(n, K, ACT_DIM)
        q = cr(np.concatenate([np.repeat(S[:, None], K, 1), C], -1)).mean(0)
        o, succ = envs.step_chunk(C[np.arange(n), q.argmax(1)], min(H, STEPS - t * H))
        lifts.append(o[:, 5] - TABLE)
    return np.array(lifts).T, succ


def run_scripted(h, n=50):
    envs = Envs(n); o = envs.reset([9_800_000 + i for i in range(n)], np.full(n, h))
    phase = np.zeros(n, int); rng = np.random.default_rng(0); lifts = []
    for t in range(STEPS):
        a = scripted(o, phase, rng, 0.0)
        out = []
        for i, e in enumerate(envs.envs):
            oo, _, _, _, info = e.step(a[i]); out.append(np.r_[oo["observation"], oo["desired_goal"]])
        o = np.stack(out).astype(np.float32)
        if (t + 1) % H == 0 or t == STEPS - 1:
            lifts.append(o[:, 5] - TABLE)
    return np.array(lifts).T


if __name__ == "__main__":
    path = sys.argv[1]
    for h in [0.15, 0.25]:
        L, s = run_system(path, h)
        held = L > 0.01
        first = np.where(held.any(1), held.argmax(1), -1)
        print(f"system   h={h}: success {s.mean():.2f} | first held decision (mean, of held eps) "
              f"{first[first >= 0].mean():.1f} | mean lift per decision: {L.mean(0).round(3)}")
        Ls = run_scripted(h)
        print(f"scripted h={h}: mean lift per decision: {Ls.mean(0).round(3)}")
