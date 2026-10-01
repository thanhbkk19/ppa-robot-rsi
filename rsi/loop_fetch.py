"""Sample-K -> select -> distill loop on Fetch pick-and-place (CPU). Same protocol as rsi/loop_nav.py.

Pretrained generator = BC diffusion on noisy scripted demos (cached per seed). Per round: collect n_train
episodes with the current system, refit the critic on the replay (MC targets = final success), distil the
executed chunks (+ rho demos), evaluate the system (K candidates) and the generator alone (K = 1) on fixed
evaluation initial states that no learner sees.
"""
from __future__ import annotations
import os
import json
import sys
import time
import itertools
import numpy as np
import torch

from rsi.fetch import Envs, NDEC, STEPS, H, OBS_DIM, ACT_DIM, collect_demos
from rsi.models import Diffusion, Critic
from rsi.loop_nav import chi2_weights

torch.set_num_threads(1)
ROOT = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(ROOT, "cache")

DEF = dict(rule="argmax", K=4, seed=0, rounds=6, n_train=400, n_eval=200, n_envs=50, rho=0.1,
           n_demo=600, demo_noise=0.45, bc_steps=15000, distill_steps=1500, distill_lr=3e-5,
           critic_steps=2000, critic_lr=3e-4, n_ens=2, distill_data="round", critic_target="mc", boot=False, z=1.0, beta=0.1, temp=0.05, kappa=1.0, delta=1.0)


def pretrained(c):
    os.makedirs(CACHE, exist_ok=True)
    f = os.path.join(CACHE, f"bc_n{c['n_demo']}_z{c['demo_noise']}_s{c['seed']}.pt")
    if os.path.exists(f):
        st = torch.load(f, weights_only=False)
        gen = Diffusion(OBS_DIM, ACT_DIM); gen.load_state_dict(st["gen"])
        return gen, st["S"], st["A"]
    S, A, Y = collect_demos(c["n_demo"], c["demo_noise"], seed=1000 + c["seed"])
    S = S.reshape(-1, OBS_DIM); A = A.reshape(-1, ACT_DIM)
    torch.manual_seed(c["seed"])
    gen = Diffusion(OBS_DIM, ACT_DIM); gen.fit(S, A, c["bc_steps"], c["seed"])
    torch.save(dict(gen=gen.state_dict(), S=S, A=A, demo_success=float(Y.mean())), f)
    gen.opt = None
    return gen, S, A


def select_fn(c, critic, rng):
    rule, K = c["rule"], c["K"]

    def sel(S, C):  # S (n, s), C (n, K, a)
        n = len(S)
        if K == 1 or not critic.trained:
            return np.zeros(n, int), None
        X = np.concatenate([np.repeat(S[:, None], K, 1), C], -1)
        q = critic(X)
        qm = q.mean(0)
        if rule == "argmax":
            j = qm.argmax(1)
        elif rule == "lcb":
            j = (qm - c["kappa"] * q.std(0)).argmax(1)
        elif rule == "softmax":
            z = qm / c["temp"]; p = np.exp(z - z.max(1, keepdims=True)); p /= p.sum(1, keepdims=True)
            j = (p.cumsum(1) < rng.random((n, 1))).sum(1).clip(0, K - 1)
        elif rule == "chi2":
            p = chi2_weights(qm, c["beta"])
            j = (p.cumsum(1) < rng.random((n, 1))).sum(1).clip(0, K - 1)
        elif rule == "lcbopt":     # maximise the certified lower bound w.q - z eps(s) ||w||, eps(s) = ensemble sd
            from ppa.select import lcb_opt_weights
            p = lcb_opt_weights(qm, c["z"] * q.std(0).mean(1))
            sel.chi2.append(float((K * (p ** 2).sum(1) - 1).mean()))
            j = (p.cumsum(1) < rng.random((n, 1))).sum(1).clip(0, K - 1)
        elif rule == "chi2tr":
            from ppa.select import chi2_trust_weights
            p = chi2_trust_weights(qm, c["delta"])
            j = (p.cumsum(1) < rng.random((n, 1))).sum(1).clip(0, K - 1)
        else:
            raise ValueError(rule)
        return j, q
    sel.chi2 = []
    return sel


def episodes(gen, envs, seeds, K, sel, rng):
    """Run len(seeds) episodes (batches of envs.n). Returns S (n, NDEC, s), A (n, NDEC, a), Y (n,), diag."""
    S_all, A_all, Y_all, spread, ens_sd = [], [], [], [], []
    for b in range(0, len(seeds), envs.n):
        o = envs.reset(seeds[b:b + envs.n])
        S_ep, A_ep = [], []
        for t in range(NDEC):
            S = np.c_[o, np.full(envs.n, t / NDEC)].astype(np.float32)
            C = gen.sample(np.repeat(S, K, 0), rng.integers(1 << 30)).reshape(envs.n, K, ACT_DIM)
            j, q = (np.zeros(envs.n, int), None) if K == 1 else sel(S, C)
            if q is not None:
                spread.append(q.mean(0).std(1).mean()); ens_sd.append(q.std(0).mean())
            a = C[np.arange(envs.n), j]
            o, succ = envs.step_chunk(a, min(H, STEPS - t * H))
            S_ep.append(S); A_ep.append(a)
        S_all.append(np.stack(S_ep, 1)); A_all.append(np.stack(A_ep, 1)); Y_all.append(succ)
    d = dict(q_spread=float(np.mean(spread)) if spread else float("nan"),
             q_ens_sd=float(np.mean(ens_sd)) if ens_sd else float("nan"))
    return np.concatenate(S_all), np.concatenate(A_all), np.concatenate(Y_all), d


def run(cfg, out=None):
    c = dict(DEF); c.update(cfg)
    rng = np.random.default_rng(c["seed"]); torch.manual_seed(c["seed"])
    gen, Sd, Ad = pretrained(c)
    critic = Critic(OBS_DIM, ACT_DIM, c["n_ens"], c["critic_lr"], seed=c["seed"])
    envs = Envs(c["n_envs"])
    eval_seeds = [900_000 + c["seed"] * 10_000 + i for i in range(c["n_eval"])]
    RX, RY, hist, DS, DA, RXn, RL, RW = [], [], [], [], [], [], [], []
    t0 = time.time()

    def evaluate(rnd, extra):
        S, A, Y, d = episodes(gen, envs, eval_seeds, c["K"], select_fn(c, critic, rng), rng)
        opt0 = float("nan")
        if critic.trained:
            opt0 = float(critic(np.c_[S[:, 0], A[:, 0]]).mean() - Y.mean())
        _, _, Yg, _ = episodes(gen, envs, eval_seeds, 1, select_fn(c, critic, rng), rng)
        hist.append(dict(round=rnd, J_sys=float(Y.mean()), J_gen=float(Yg.mean()), opt0=opt0,
                         wall=time.time() - t0, **d, **extra))
        if out:
            json.dump(dict(cfg=c, hist=hist), open(out, "w"))

    evaluate(0, {})
    for r in range(1, c["rounds"] + 1):
        seeds = [100_000_000 + c["seed"] * 1_000_000 + r * 10_000 + i for i in range(c["n_train"])]
        sel = select_fn(c, critic, rng)
        S, A, Y, d = episodes(gen, envs, seeds, c["K"], sel, rng)
        d["sel_chi2"] = float(np.mean(sel.chi2)) if sel.chi2 else float("nan")
        X3 = np.concatenate([S, A], -1)                                   # (n, NDEC, s + a)
        RX.append(X3.reshape(-1, OBS_DIM + ACT_DIM)); RY.append(np.repeat(Y, NDEC))
        if c["boot"]:   # per-episode bootstrap weights (drawn only here, so other runs keep their random stream)
            RW.append(np.repeat(rng.poisson(1.0, (len(Y), c["n_ens"])), NDEC, 0))
        RXn.append(np.concatenate([X3[:, 1:], X3[:, -1:]], 1).reshape(-1, OBS_DIM + ACT_DIM))
        RL.append(np.tile(np.arange(NDEC) == NDEC - 1, len(Y)))
        if c["critic_target"] == "mc":
            closs = critic.fit(np.concatenate(RX), np.concatenate(RY), c["critic_steps"],
                               wboot=np.concatenate(RW) if c["boot"] else None)
        else:
            closs = critic.fit_sarsa(np.concatenate(RX), np.concatenate(RXn), np.concatenate(RL),
                                     np.concatenate(RY), c["critic_steps"])
        DS.append(S.reshape(-1, OBS_DIM)); DA.append(A.reshape(-1, ACT_DIM))
        if c["distill_data"] == "round":
            Sx, Ax = DS[-1], DA[-1]
        else:  # accumulate every round's executed chunks
            Sx, Ax = np.concatenate(DS), np.concatenate(DA)
        nm = int(c["rho"] * len(Sx)); i = rng.integers(0, len(Sd), nm)
        gen.fit(np.concatenate([Sx, Sd[i]]), np.concatenate([Ax, Ad[i]]), c["distill_steps"],
                c["seed"] * 100 + r, lr=c["distill_lr"])
        extra = dict(train_J=float(Y.mean()), critic_loss=closs, train_spread=d["q_spread"],
                     train_ens_sd=d["q_ens_sd"], sel_chi2=d["sel_chi2"])
        if r in c.get("probe", []):
            from rsi.probe import probe
            extra["probe"] = probe(gen, critic, select_fn(c, critic, rng), c["K"], c["seed"] * 100 + r, rng=rng)
        evaluate(r, extra)
    return dict(cfg=c, hist=hist)


def job(args):
    name, cfg = args
    key = "_".join(f"{k}{v}" for k, v in sorted(cfg.items()))
    out = os.path.join(ROOT, "results", name, key + ".json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    if os.path.exists(out) and json.load(open(out)).get("done"):
        return
    t = time.time(); res = run(cfg, out); res["done"] = True; res["sec"] = time.time() - t
    json.dump(res, open(out, "w"))
    h = res["hist"][-1]
    print(f"{cfg} J_sys={h['J_sys']:.2f} J_gen={h['J_gen']:.2f} opt0={h['opt0']:+.2f} "
          f"curve={[round(x['J_sys'], 2) for x in res['hist']]} ({res['sec']:.0f}s)", flush=True)


if __name__ == "__main__":
    from multiprocessing import Pool
    name, spec = sys.argv[1], json.loads(sys.argv[2] if not sys.argv[2].endswith(".json") else open(sys.argv[2]).read())
    specs = spec if isinstance(spec, list) else [spec]   # a list of grids is run in one pool
    cfgs = []
    for sp in specs:
        keys = list(sp)
        cfgs += [dict(zip(keys, v)) for v in itertools.product(*[sp[k] for k in keys])]
    cfgs.sort(key=lambda c: -c.get("K", 4))             # longest runs first
    with Pool(int(sys.argv[3]) if len(sys.argv) > 3 else 4) as p:
        list(p.imap_unordered(job, [(name, c) for c in cfgs], chunksize=1))
