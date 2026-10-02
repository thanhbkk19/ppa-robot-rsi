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
import copy
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
           critic_steps=2000, critic_lr=3e-4, n_ens=2, distill_data="round", critic_target="mc", rb=False, rb_draws=4, demo_goals="env", demo_filter=False, expo=0.0, extra_evals=False, samp_temp=1.0, gate=False, n_gate=100, gate_z=1.0, train_h=None, eval_h=None, curriculum=False, her=False, boot=False, z=1.0, beta=0.1, temp=0.05, kappa=1.0, delta=1.0)


def pretrained(c):
    os.makedirs(CACHE, exist_ok=True)
    tag = ("" if c["demo_goals"] == "env" else f"_{c['demo_goals']}") + ("_filt" if c["demo_filter"] else "")
    f = os.path.join(CACHE, f"bc_n{c['n_demo']}_z{c['demo_noise']}{tag}_s{c['seed']}.pt")
    if os.path.exists(f):
        st = torch.load(f, weights_only=False)
        gen = Diffusion(OBS_DIM, ACT_DIM); gen.load_state_dict(st["gen"])
        return gen, st["S"], st["A"]
    S, A, Y = collect_demos(c["n_demo"], c["demo_noise"], seed=1000 + c["seed"], table_only=c["demo_goals"] == "table")
    if c["demo_filter"]:   # filtered BC: keep only the successful demonstrations
        S, A = S[Y > 0.5], A[Y > 0.5]
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
        elif rule == "balanced":   # success-balanced step: argmax over a random subset of K_eff(m) candidates
            from ppa.select import balanced_delta
            m = sel.m_cur if sel.m_cur is not None else np.full(n, sel.m_global)
            _, keff = balanced_delta(m, K)
            ke = np.maximum(2, np.round(keff)).astype(int)
            score = np.where(rng.random((n, K)).argsort(1) < ke[:, None], qm, -np.inf)   # keep ke random candidates
            j = score.argmax(1)
            sel.chi2.append(float(((ke - 1) ** 2 / (2 * ke - 1)).mean()))
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
    sel.m_cur, sel.m_global, sel.m_of_h = None, 0.5, None
    return sel


def select_weights(c, critic, S, C):
    """Selection probabilities [n, K] of rule c["rule"] for candidates C [n, K, a] at states S [n, s]."""
    from ppa.select import chi2_trust_weights, lcb_opt_weights
    K = C.shape[1]
    q = critic(np.concatenate([np.repeat(S[:, None], K, 1), C], -1))
    qm = q.mean(0)
    rule = c["rule"]
    if rule == "argmax":
        return np.eye(K)[qm.argmax(1)]
    if rule == "chi2":
        return chi2_weights(qm, c["beta"])
    if rule == "chi2tr":
        return chi2_trust_weights(qm, c["delta"])
    if rule == "softmax":
        z = qm / c["temp"]; p = np.exp(z - z.max(1, keepdims=True)); return p / p.sum(1, keepdims=True)
    if rule == "lcbopt":
        return lcb_opt_weights(qm, c["z"] * q.std(0).mean(1))
    raise ValueError(rule)


def heights_for(seeds, rng_h):
    """Goal heights for a list of episode seeds: None (env default) or [lo, hi] -> uniform, deterministic per seed."""
    if rng_h is None:
        return None
    lo, hi = rng_h
    return np.array([lo + (hi - lo) * ((s * 0.6180339887498949) % 1.0) for s in seeds])


def episodes(gen, envs, seeds, K, sel, rng, keep_cands=False, h_range=None, heights=None):
    """Run len(seeds) episodes (batches of envs.n). Returns S (n, NDEC, s), A (n, NDEC, a), Y (n,), diag.
    keep_cands: diag["C"] holds every candidate set (n, NDEC, K, a) for Rao-Blackwellised distillation."""
    S_all, A_all, Y_all, spread, ens_sd, C_all, F_all = [], [], [], [], [], [], []
    hs = heights_for(seeds, h_range) if heights is None else np.asarray(heights)
    for b in range(0, len(seeds), envs.n):
        o = envs.reset(seeds[b:b + envs.n], None if hs is None else hs[b:b + envs.n])
        if getattr(sel, "m_of_h", None) is not None and hs is not None:
            sel.m_cur = sel.m_of_h(hs[b:b + envs.n])
        S_ep, A_ep, C_ep = [], [], []
        for t in range(NDEC):
            S = np.c_[o, np.full(envs.n, t / NDEC)].astype(np.float32)
            C = gen.sample(np.repeat(S, K, 0), rng.integers(1 << 30)).reshape(envs.n, K, ACT_DIM)
            j, q = (np.zeros(envs.n, int), None) if K == 1 else sel(S, C)
            if q is not None:
                spread.append(q.mean(0).std(1).mean()); ens_sd.append(q.std(0).mean())
            a = C[np.arange(envs.n), j]
            o, succ = envs.step_chunk(a, min(H, STEPS - t * H))
            S_ep.append(S); A_ep.append(a)
            if keep_cands:
                C_ep.append(C.astype(np.float16))
        S_all.append(np.stack(S_ep, 1)); A_all.append(np.stack(A_ep, 1)); Y_all.append(succ)
        F_all.append(o[:, 3:6].copy())           # final object position (achieved goal)
        if keep_cands:
            C_all.append(np.stack(C_ep, 1))
    d = dict(q_spread=float(np.mean(spread)) if spread else float("nan"),
             q_ens_sd=float(np.mean(ens_sd)) if ens_sd else float("nan"))
    if keep_cands:
        d["C"] = np.concatenate(C_all)
    d["final_obj"] = np.concatenate(F_all)
    if hs is not None:
        d["heights"] = np.asarray(hs, float)
    return np.concatenate(S_all), np.concatenate(A_all), np.concatenate(Y_all), d


def run(cfg, out=None):
    c = dict(DEF); c.update(cfg)
    rng = np.random.default_rng(c["seed"]); torch.manual_seed(c["seed"])
    gen, Sd, Ad = pretrained(c)
    gen.temp = c["samp_temp"]
    critic = Critic(OBS_DIM, ACT_DIM, c["n_ens"], c["critic_lr"], seed=c["seed"])
    envs = Envs(c["n_envs"])
    eval_seeds = [900_000 + c["seed"] * 10_000 + i for i in range(c["n_eval"])]
    RX, RY, hist, DS, DA, RXn, RL, RW = [], [], [], [], [], [], [], []
    t0 = time.time()

    def evaluate(rnd, extra):
        S, A, Y, d = episodes(gen, envs, eval_seeds, c["K"], prime(select_fn(c, critic, rng)), rng, h_range=c["eval_h"])
        opt0 = float("nan")
        if critic.trained:
            opt0 = float(critic(np.c_[S[:, 0], A[:, 0]]).mean() - Y.mean())
        _, _, Yg, _ = episodes(gen, envs, eval_seeds, 1, select_fn(c, critic, rng), rng, h_range=c["eval_h"])
        d = {k: v for k, v in d.items() if not isinstance(v, np.ndarray)}
        if c["extra_evals"]:
            for tag, hr in (("J_easy", [0.0, 0.05]), ("J_full", [0.0, 0.3])):
                _, _, Ye, de = episodes(gen, envs, eval_seeds[:100], c["K"], prime(select_fn(c, critic, rng)), rng,
                                        h_range=hr)
                extra = dict(extra, **{tag: float(Ye.mean())})
                if tag == "J_full":
                    extra["eval_lift_p95"] = float(np.quantile(de["final_obj"][:, 2] - 0.4247, 0.95))
        if c["eval_h"] is not None:   # where the evaluation objects end up (lift above the table, 95th pct)
            extra = dict(extra)
        hist.append(dict(round=rnd, J_sys=float(Y.mean()), J_gen=float(Yg.mean()), opt0=opt0,
                         wall=time.time() - t0, **d, **extra))
        if out:
            json.dump(dict(cfg=c, hist=hist), open(out, "w"))

    # frontier curriculum over goal heights: bins of 2.5 cm in train_h, success counts with a Beta(1, 1) prior,
    # training heights drawn per bin with probability proportional to m (1 - m) + floor (SEC / PLR-style)
    nb = 12
    edges = None if c["train_h"] is None else np.linspace(c["train_h"][0], c["train_h"][1], nb + 1)
    succ_n, tot_n = np.ones(nb), np.full(nb, 2.0)

    m_last = [0.5]

    def m_of_h(h):
        b = np.clip(np.searchsorted(edges, h, side="right") - 1, 0, nb - 1)
        return succ_n[b] / tot_n[b]

    def prime(sel):   # give the selection rule predictable success estimates (past training rounds only)
        sel.m_global = m_last[0]
        sel.m_of_h = m_of_h if edges is not None else None
        return sel

    def curriculum_heights(n):
        m = succ_n / tot_n
        p = m * (1 - m) + 0.02
        b = rng.choice(nb, size=n, p=p / p.sum())
        return edges[b] + rng.random(n) * (edges[b + 1] - edges[b])

    evaluate(0, {})
    for r in range(1, c["rounds"] + 1):
        seeds = [100_000_000 + c["seed"] * 1_000_000 + r * 10_000 + i for i in range(c["n_train"])]
        sel = prime(select_fn(c, critic, rng))
        hts = curriculum_heights(len(seeds)) if c["curriculum"] else None
        S, A, Y, d = episodes(gen, envs, seeds, c["K"], sel, rng, keep_cands=c["rb"], h_range=c["train_h"],
                              heights=hts)
        round_bins = None
        if edges is not None and "heights" in d:
            bi = np.clip(np.searchsorted(edges, d["heights"], side="right") - 1, 0, nb - 1)
            np.add.at(succ_n, bi, Y); np.add.at(tot_n, bi, 1.0)
            cnt = np.bincount(bi, minlength=nb); sc = np.bincount(bi, weights=Y, minlength=nb)
            round_bins = [round(float(a / b), 3) if b > 0 else None for a, b in zip(sc, cnt)]   # this round only
        m_last[0] = float(Y.mean())
        if c["her"]:
            # hindsight (final-state) relabelling: the achieved object position becomes the goal, and the episode
            # is a success for it. Appended to the critic replay and to the distillation data (GCSL-style).
            S_h = S.copy(); S_h[:, :, 25:28] = d["final_obj"][:, None, :]
            X3h = np.concatenate([S_h, A], -1)
            RX.append(X3h.reshape(-1, OBS_DIM + ACT_DIM)); RY.append(np.ones(len(Y) * NDEC))
            RXn.append(np.concatenate([X3h[:, 1:], X3h[:, -1:]], 1).reshape(-1, OBS_DIM + ACT_DIM))
            RL.append(np.tile(np.arange(NDEC) == NDEC - 1, len(Y)))
            if c["boot"]:
                RW.append(np.repeat(rng.poisson(1.0, (len(Y), c["n_ens"])), NDEC, 0))
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
        if c["rb"]:
            # Rao-Blackwellised distillation: re-weight ALL K candidates of every visited state with the selection
            # rule under the freshly refit critic, then draw rb_draws candidates per state from those weights
            # (stratified). Same target distribution as distilling the selected action, lower variance.
            Cs = d.pop("C").reshape(-1, c["K"], ACT_DIM).astype(np.float32)
            Ss = S.reshape(-1, OBS_DIM)
            Wt = []
            for b in range(0, len(Ss), 512):
                Wt.append(select_weights(c, critic, Ss[b:b + 512], Cs[b:b + 512]))
            Wt = np.concatenate(Wt)
            u = rng.random((len(Ss), c["rb_draws"], 1))
            idx = (Wt.cumsum(1)[:, None, :] < u).sum(-1).clip(0, c["K"] - 1)        # (n, draws)
            A_rb = Cs[np.arange(len(Ss))[:, None], idx].reshape(-1, ACT_DIM)
            DS.append(np.repeat(Ss, c["rb_draws"], 0)); DA.append(A_rb)
        else:
            DS.append(S.reshape(-1, OBS_DIM)); DA.append(A.reshape(-1, ACT_DIM))
        if c["her"]:
            DS[-1] = np.concatenate([DS[-1], S_h.reshape(-1, OBS_DIM)]); DA[-1] = np.concatenate([DA[-1], A.reshape(-1, ACT_DIM)])
        if c["distill_data"] == "round":
            Sx, Ax = DS[-1], DA[-1]
        else:  # accumulate every round's executed chunks
            Sx, Ax = np.concatenate(DS), np.concatenate(DA)
        nm = int(c["rho"] * len(Sx)); i = rng.integers(0, len(Sd), nm)
        prev = [p.detach().clone() for p in gen.parameters()] if c["expo"] > 0 else None
        old_state = copy.deepcopy(gen.state_dict()) if c["gate"] else None
        gen.fit(np.concatenate([Sx, Sd[i]]), np.concatenate([Ax, Ad[i]]), c["distill_steps"],
                c["seed"] * 100 + r, lr=c["distill_lr"])
        if prev is not None:   # ExPO-style extrapolation along this round's update: theta += alpha (theta - theta_prev)
            with torch.no_grad():
                for p, q0 in zip(gen.parameters(), prev):
                    p.add_(c["expo"] * (p - q0))
        gate_info = {}
        if c["gate"]:
            # paired acceptance test on fresh gate episodes (same initial states and goal heights 0-0.3 m for both
            # generators, same refit critic): keep the new generator unless it is worse by more than gate_z s.e.
            gs = [200_000_000 + c["seed"] * 1_000_000 + r * 10_000 + i for i in range(c["n_gate"])]
            new_state = copy.deepcopy(gen.state_dict())
            _, _, Yn, _ = episodes(gen, envs, gs, c["K"], prime(select_fn(c, critic, rng)), rng, h_range=[0.0, 0.3])
            gen.load_state_dict(old_state)
            _, _, Yo, _ = episodes(gen, envs, gs, c["K"], prime(select_fn(c, critic, rng)), rng, h_range=[0.0, 0.3])
            dlt = Yn - Yo
            se = dlt.std(ddof=1) / np.sqrt(len(dlt)) if dlt.std() > 0 else 0.0
            accept = bool(dlt.mean() >= -c["gate_z"] * se)
            if accept:
                gen.load_state_dict(new_state)
            gate_info = dict(gate_accept=accept, gate_new=float(Yn.mean()), gate_old=float(Yo.mean()))
        extra = dict(train_J=float(Y.mean()), critic_loss=closs, train_spread=d["q_spread"], **gate_info,
                     max_lift=float(np.quantile(d["final_obj"][:, 2] - 0.4247, 0.95)),
                     curr_m=(succ_n / tot_n).round(3).tolist() if c["curriculum"] else None, round_bins=round_bins,
                     train_ens_sd=d["q_ens_sd"], sel_chi2=d["sel_chi2"])
        if r in c.get("probe", []):
            from rsi.probe import probe
            extra["probe"] = probe(gen, critic, select_fn(c, critic, rng), c["K"], c["seed"] * 100 + r, rng=rng)
        evaluate(r, extra)
    if out:   # final generator + critic, for mechanism analysis (git-ignored)
        torch.save(dict(gen=gen.state_dict(), critic=[n.state_dict() for n in critic.nets]), out[:-5] + ".pt")
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
