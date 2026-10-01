"""M1: does the improvement signal point 'up' for in-air goals?

For a trained CFG model and a trained critic-argmax model (take-off regime), at states reached after a few
decisions with goals 0.10-0.30 m in the air:
  - sample Kc candidate chunks from the (unconditional / deployed) generator,
  - measure each candidate's vertical command  up = mean of the z-displacement actions in the chunk,
  - score candidates with the method's own improvement signal:
      CFG    : implicit ratio log mu(a|s,succ) - log mu(a|s)  (ELBO estimate, CondDiffusion.implicit_q)
      critic : ensemble-mean Q(s, a)
  - report Spearman(score, up) per state (averaged), the 'up' of the top-scored candidate minus the median,
    and, for CFG, how far guidance w shifts the mean vertical command relative to w = 0.
usage: python -m rsi.analysis_direction <cfg_model.pt> <argmax_model.pt> [seed]
"""
import sys
import numpy as np
import torch
from scipy.stats import spearmanr

from rsi.cfg import CondDiffusion
from rsi.models import Diffusion, Critic
from rsi.fetch import Envs, NDEC, OBS_DIM, ACT_DIM
from rsi.loop_fetch import heights_for

UP = [2, 6, 10, 14]          # z-displacement entries of the 4 low-level actions in a chunk


def states(gen_sample, seed, n=48, t_stop=5):
    """Roll the given sampler for t_stop decisions on in-air goals and return the states reached."""
    envs = Envs(n); rng = np.random.default_rng(seed)
    seeds = [7_100_000 + seed * 1000 + i for i in range(n)]
    o = envs.reset(seeds, heights_for(seeds, [0.1, 0.3]))
    for t in range(t_stop):
        S = np.c_[o, np.full(n, t / NDEC)].astype(np.float32)
        o, _ = envs.step_chunk(gen_sample(S, rng.integers(1 << 30)), 4)
    return np.c_[o, np.full(n, t_stop / NDEC)].astype(np.float32)


def report(name, S, cands, scores):
    up = cands[:, :, UP].mean(-1)                                  # (n, Kc)
    rho = [spearmanr(s, u)[0] for s, u in zip(scores, up) if np.std(s) > 0 and np.std(u) > 0]
    top = up[np.arange(len(up)), scores.argmax(1)] - np.median(up, 1)
    print(f"{name}: spearman(score, up) = {np.mean(rho):+.3f} | top-scored up minus median = {top.mean():+.4f} "
          f"| mean up of candidates = {up.mean():+.4f}")


def main(cfg_path, am_path, seed=0, Kc=32):
    torch.manual_seed(seed)
    cg = CondDiffusion(OBS_DIM, ACT_DIM); cg.load_state_dict(torch.load(cfg_path, weights_only=False))
    st = torch.load(am_path, weights_only=False)
    ag = Diffusion(OBS_DIM, ACT_DIM); ag.load_state_dict(st["gen"])
    cr = Critic(OBS_DIM, ACT_DIM, n_ens=len(st["critic"]))
    for n, sd in zip(cr.nets, st["critic"]):
        n.load_state_dict(sd)
    cr.trained = True

    # CFG: states reached by its own deployed sampler (w = 2), candidates from the unconditional model
    S = states(lambda s, sd: cg.sample(s, sd, w=2.0, v=0.0), seed)
    C = cg.sample(np.repeat(S, Kc, 0), seed + 1, w=0.0, v=0.0).reshape(len(S), Kc, ACT_DIM)
    q = cg.implicit_q(np.repeat(S, Kc, 0), C.reshape(-1, ACT_DIM), n_mc=32, seed=seed).reshape(len(S), Kc)
    report("CFG implicit ratio", S, C, q)
    ups = {w: cg.sample(np.repeat(S, Kc, 0), seed + 2, w=w, v=0.0)[:, UP].mean() for w in [0.0, 1.0, 2.0, 3.0]}
    print("CFG mean vertical command by guidance w:", {w: round(float(u), 4) for w, u in ups.items()})

    # critic-argmax: states reached by its own system (generator + argmax over 64), candidates from its generator
    def am_sample(s, sd):
        c = ag.sample(np.repeat(s, 64, 0), sd).reshape(len(s), 64, ACT_DIM)
        qq = cr(np.concatenate([np.repeat(s[:, None], 64, 1), c], -1)).mean(0)
        return c[np.arange(len(s)), qq.argmax(1)]
    S2 = states(am_sample, seed)
    C2 = ag.sample(np.repeat(S2, Kc, 0), seed + 1).reshape(len(S2), Kc, ACT_DIM)
    q2 = cr(np.concatenate([np.repeat(S2[:, None], Kc, 1), C2], -1)).mean(0)
    report("critic Q          ", S2, C2, q2)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 0)
