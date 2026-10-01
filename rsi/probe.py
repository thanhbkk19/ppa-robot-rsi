"""Ground-truth probe of the selector: true values of the K candidates at saved states.

At decision t of fresh episodes (run with the current system), save the MuJoCo state, sample Kc candidate chunks
from the generator, and estimate each candidate's true value by R rollouts from the saved state (execute the
candidate, then continue with the current system to the end; Y = final success). Compares the critic's scores
with these Monte-Carlo values: rank correlation, regret of the critic's pick, and the winner's curse
(score of the pick minus its true value).
"""
from __future__ import annotations
import numpy as np
from scipy.stats import spearmanr

from rsi.fetch import Envs, NDEC, STEPS, H, ACT_DIM


def probe(gen, critic, sel, K, seed, n_states=16, t_probe=4, Kc=8, R=16, rng=None):
    rng = rng or np.random.default_rng(seed)
    base = Envs(n_states)
    o = base.reset([7_000_000 + seed * 1000 + i for i in range(n_states)])
    for t in range(t_probe):        # drive to decision t with the system
        S = np.c_[o, np.full(n_states, t / NDEC)].astype(np.float32)
        C = gen.sample(np.repeat(S, K, 0), rng.integers(1 << 30)).reshape(n_states, K, ACT_DIM)
        j = np.zeros(n_states, int) if K == 1 else sel(S, C)[0]
        o, _ = base.step_chunk(C[np.arange(n_states), j], H)
    sims = Envs(Kc * R)
    sims.reset(list(range(sims.n)))     # gymnasium requires a reset before the first step; states are overwritten
    out = []
    for i in range(n_states):
        st = base.get_state(i)
        S0 = np.c_[o[i:i + 1], [[t_probe / NDEC]]].astype(np.float32)
        cand = gen.sample(np.repeat(S0, Kc, 0), rng.integers(1 << 30))
        qhat = critic(np.c_[np.repeat(S0, Kc, 0), cand]).mean(0)
        for e in range(sims.n):
            sims.set_state(e, st)
        oo, succ = sims.step_chunk(np.repeat(cand, R, 0), H)
        for t in range(t_probe + 1, NDEC):
            S = np.c_[oo, np.full(sims.n, t / NDEC)].astype(np.float32)
            C = gen.sample(np.repeat(S, K, 0), rng.integers(1 << 30)).reshape(sims.n, K, ACT_DIM)
            j = np.zeros(sims.n, int) if K == 1 else sel(S, C)[0]
            oo, succ = sims.step_chunk(C[np.arange(sims.n), j], min(H, STEPS - t * H))
        qtrue = succ.reshape(Kc, R).mean(1)
        out.append((qhat, qtrue))
    qh = np.array([x[0] for x in out]); qt = np.array([x[1] for x in out])
    k = qh.argmax(1); ar = np.arange(len(qh))
    rho = [spearmanr(a, b)[0] for a, b in zip(qh, qt) if np.std(b) > 0 and np.std(a) > 0]
    return dict(spearman=float(np.mean(rho)) if rho else float("nan"),
                regret=float((qt.max(1) - qt[ar, k]).mean()),
                curse=float((qh[ar, k] - qt[ar, k]).mean()),
                gain_vs_random=float((qt[ar, k] - qt.mean(1)).mean()),
                spread_hat=float(qh.std(1).mean()), spread_true=float(qt.std(1).mean()),
                err_sd=float((qh - qt).std(1).mean()), qt_mean=float(qt.mean()))
