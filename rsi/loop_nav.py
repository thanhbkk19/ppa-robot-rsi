"""Sample-K -> select -> distill loop on the nav testbed, with pluggable selection rules and diagnostics.

Per round (same order as ppa/robo/loop.py): collect n_sel episodes with the current system, refit the critic on
the whole replay (MC targets), distil the executed actions (+ rho demos) into the generator, then evaluate
  J_sys  : deployed system (generator + K candidates + selection rule)  -- the RSI objective
  J_gen  : generator alone (K = 1)
  opt0   : mean Q_hat of the executed first action minus realised success (critic optimism on its own choices)
  pB     : fraction of generator first actions heading to the narrow good route (x > 0): a diversity probe
"""
from __future__ import annotations
import json
import sys
import time
import itertools
import numpy as np
import torch

from rsi.nav import Critic, Diff, demos, feats, rollout, T

DEF = dict(rule="argmax", K=4, seed=0, rounds=6, n_sel=800, n_eval=600, eps=0.1, rho=0.1,
           critic_steps=2000, n_ens=2, beta=0.1, temp=0.05, kappa=1.0)


def make_select(cfg, critic, rng):
    rule, K = cfg["rule"], cfg["K"]

    def sel(S, C):
        n = len(S)
        if K == 1 or not critic.trained:
            return np.zeros(n, int)
        q = critic(feats(S, C))                      # (E, n, K)
        qm = q.mean(0)
        if rule == "argmax":
            return qm.argmax(1)
        if rule == "lcb":                            # ensemble pessimism
            return (qm - cfg["kappa"] * q.std(0)).argmax(1)
        if rule == "softmax":                        # KL-regularised (Boltzmann) tilt of the empirical proposal
            z = qm / cfg["temp"]; p = np.exp(z - z.max(1, keepdims=True)); p /= p.sum(1, keepdims=True)
            return (p.cumsum(1) < rng.random((n, 1))).sum(1).clip(0, K - 1)
        if rule == "chi2":                           # chi^2-regularised tilt of the empirical proposal
            p = chi2_weights(qm, cfg["beta"])
            return (p.cumsum(1) < rng.random((n, 1))).sum(1).clip(0, K - 1)
        raise ValueError(rule)
    return sel


def chi2_weights(q, beta):
    """Selection probabilities over K candidates that maximise  sum_k w_k q_k - beta * chi2(w || uniform_K),
    chi2(w || u) = K * sum_k w_k^2 - 1.  KKT: K w_k = max(0, 1 + (q_k - lam) / (2 beta)), lam set by sum w = 1.
    Equivalent to the chi^2-regularised tilt  pi(a) = mu(a) * max(0, 1 + (Q(a) - lam) / (2 beta))  of the
    proposal mu, evaluated on its K empirical samples (Huang et al., 2025, InferenceTimePessimism)."""
    n, K = q.shape
    qs = -np.sort(-q, 1)                                   # descending
    m = np.arange(1, K + 1)
    # support = top-m candidates: sum_{k<=m} (1 + (q_k - lam)/(2 beta)) = K  ->  lam_m
    lam = (qs.cumsum(1) + 2 * beta * (m - K)) / m          # (n, K)
    ok = 1 + (qs - lam) / (2 * beta) > 0                   # top-m support is feasible
    mstar = K - np.argmax(ok[:, ::-1], 1) - 1              # largest feasible m (0-based index)
    l = lam[np.arange(n), mstar][:, None]
    w = np.maximum(0, 1 + (q - l) / (2 * beta))
    return w / w.sum(1, keepdims=True)


def run(cfg):
    c = dict(DEF); c.update(cfg)
    rng = np.random.default_rng(c["seed"]); torch.manual_seed(c["seed"])
    Sd, Ad = demos(1000, c["eps"], rng)
    gen = Diff(); gen.fit(Sd, Ad, 3000, c["seed"])
    critic = Critic(c["n_ens"], seed=c["seed"])
    RX, RY = [], []
    hist = []

    def evaluate(rnd):
        sel = make_select(c, critic, rng)
        log, Y, _ = rollout(gen, c["n_eval"], c["K"], rng, sel)
        opt0 = float("nan")
        if critic.trained:
            X0 = feats(log[0]["S"], log[0]["a"][:, None]); opt0 = float(critic(X0).mean() - Y.mean())
        _, Yg, _ = rollout(gen, c["n_eval"], 1, rng, lambda S, C: np.zeros(len(S), int))
        S0 = np.tile(np.array([[0., 0., 0.]], np.float32), (2000, 1))
        a0 = gen.sample(S0, rng.integers(1 << 30))
        hist.append(dict(round=rnd, J_sys=float(Y.mean()), J_gen=float(Yg.mean()), opt0=opt0,
                         pB=float((a0[:, 0] > 0).mean())))

    evaluate(0)
    for r in range(1, c["rounds"] + 1):
        sel = make_select(c, critic, rng)
        Sx, Ax = [], []
        for _ in range(c["n_sel"] // 100):
            log, Y, _ = rollout(gen, 100, c["K"], rng, sel)
            for L in log:
                m = L["active"]
                RX.append(feats(L["S"][m], L["a"][m][:, None])[:, 0]); RY.append(Y[m])
                Sx.append(L["S"][m]); Ax.append(L["a"][m])
        critic.fit(np.concatenate(RX), np.concatenate(RY), c["critic_steps"])
        S = np.concatenate(Sx); A = np.concatenate(Ax)
        nm = max(int(c["rho"] * len(S)), 200); i = rng.integers(0, len(Sd), nm)
        gen.fit(np.concatenate([S, Sd[i]]), np.concatenate([A, Ad[i]]), 1500, c["seed"] * 100 + r)
        evaluate(r)
    return dict(cfg=c, hist=hist)


def job(cfg):
    t = time.time(); out = run(cfg); out["sec"] = time.time() - t
    h = out["hist"][-1]
    print(f"{cfg} J_sys={h['J_sys']:.2f} J_gen={h['J_gen']:.2f} opt0={h['opt0']:+.2f} pB={h['pB']:.2f} "
          f"({out['sec']:.0f}s)", flush=True)
    return out


def grid(spec):
    keys = list(spec)
    return [dict(zip(keys, v)) for v in itertools.product(*[spec[k] for k in keys])]


if __name__ == "__main__":
    from multiprocessing import Pool
    name, spec = sys.argv[1], json.loads(sys.argv[2])
    cfgs = grid(spec)
    with Pool(int(sys.argv[3]) if len(sys.argv) > 3 else 4) as p:
        res = p.map(job, cfgs, chunksize=1)
    json.dump(res, open(f"rsi/results/{name}.json", "w"))
