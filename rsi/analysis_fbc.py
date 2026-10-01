"""A1: is the round-0 CFG gain just 'use the successful demos'? Filtered BC (unconditional diffusion on the
successful demos only) vs conditional BC (w=1) and guided sampling, same demos, same steps."""
import numpy as np, torch
from multiprocessing import Pool
from rsi.cfg import pretrained_cond, DEF, REGIMES, SUCC
from rsi.models import Diffusion
from rsi.fetch import Envs, OBS_DIM, ACT_DIM
from rsi.loop_fetch import episodes


def one(args):
    regime, seed = args
    c = dict(DEF); c.update(regime=regime, seed=seed); R = REGIMES[regime]
    _, S, A, C = pretrained_cond(c, R)
    keep = C[:, 0] > 0.5
    torch.manual_seed(seed)
    fbc = Diffusion(OBS_DIM, ACT_DIM); fbc.fit(S[keep], A[keep], c["bc_steps"], seed)
    envs = Envs(50); rng = np.random.default_rng(seed)
    ev = [900_000 + seed * 10_000 + i for i in range(200)]
    _, _, Y, _ = episodes(fbc, envs, ev, 1, None, rng, h_range=R["eval_h"])
    return f"{regime} seed {seed} filtered-BC:{Y.mean():.2f} (success demos {keep.mean():.2f} of rows)"


if __name__ == "__main__":
    with Pool(4) as p:
        for line in p.imap_unordered(one, [(r, s) for r in ["easy", "hard"] for s in [0, 1, 2]]):
            print(line, flush=True)
