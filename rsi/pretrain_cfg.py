"""Build cached conditional BC models and report round-0 success for several guidance weights.
python -m rsi.pretrain_cfg  -> prints, per regime and seed, success of w in {0, 1, 2, 3} (v = 0) and (2, 1)."""
import sys
import numpy as np
from multiprocessing import Pool
from rsi.cfg import pretrained_cond, DEF, REGIMES
from rsi.fetch import Envs
from rsi.loop_fetch import episodes


def one(args):
    regime, seed = args
    c = dict(DEF); c.update(regime=regime, seed=seed); R = REGIMES[regime]
    gen, *_ = pretrained_cond(c, R)
    envs = Envs(50); rng = np.random.default_rng(seed)
    ev = [900_000 + seed * 10_000 + i for i in range(200)]
    out = []
    for w, v in [(0, 0), (1, 0), (2, 0), (3, 0), (2, 1)]:
        gen.w, gen.v = w, v
        _, _, Y, d = episodes(gen, envs, ev, 1, None, rng, h_range=R["eval_h"])
        out.append(f"w{w}v{v}:{Y.mean():.2f}")
    return f"{regime} seed {seed} " + " ".join(out)


if __name__ == "__main__":
    jobs = [(r, s) for r in ["easy", "hard", "takeoff"] for s in [0, 1, 2]]
    with Pool(4) as p:
        for line in p.imap_unordered(one, jobs):
            print(line, flush=True)
