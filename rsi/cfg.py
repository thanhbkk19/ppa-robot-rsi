"""Self-improving classifier-free guidance (CFG) for the Fetch testbed (docs/LIT_REVIEW.md, ideas A + B).

One diffusion policy mu(a | s, c) with outcome condition c in {success, failure, null}. The null condition is
obtained by dropping c with probability p_drop during training (classifier-free guidance). The robot samples

    eps = eps_null + w (eps_succ - eps_null) + v (eps_null - eps_fail)

i.e. mu(a|s) * [mu(a|s, succ) / mu(a|s)]^w * [mu(a|s) / mu(a|s, fail)]^v (CFGRL, Frans et al. 2025; the v-term is
negative-sample guidance). w = v = 0 is the unconditional generator; w = 1, v = 0 is success-conditioned BC.
No critic and no selection over K candidates: the improvement direction is the density ratio learned by the
generator itself.

RSI loop per round: collect episodes with guided sampling, label every executed chunk with its episode
outcome (and, with HER, add copies relabelled to the achieved goal as successes), refit the conditional model on
all own data so far plus the labelled demos, evaluate.
"""
from __future__ import annotations
import os
import sys
import json
import time
import itertools
import numpy as np
import torch
import torch.nn as nn

from rsi.fetch import Envs, NDEC, OBS_DIM, ACT_DIM, collect_demos
from rsi.models import mlp
from rsi.loop_fetch import episodes

torch.set_num_threads(1)
ROOT = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(ROOT, "cache")
SUCC, FAIL, NULL = np.array([1., 0.]), np.array([0., 1.]), np.array([0., 0.])


class CondDiffusion(nn.Module):
    def __init__(self, s_dim, a_dim, N=20, h=256):
        super().__init__()
        self.N, self.a_dim = N, a_dim
        b = torch.linspace(1e-4, 0.2, N); self.register_buffer("b", b)
        al = torch.cumprod(1 - b, 0); self.register_buffer("al", al)
        al_prev = torch.cat([torch.ones(1), al[:-1]])
        self.register_buffer("sig2", b * (1 - al_prev) / (1 - al), persistent=False)
        self.net = mlp(a_dim + s_dim + 2 + 1, a_dim, h)
        self.opt = None
        self.w, self.v = 0.0, 0.0          # guidance used by sample() (set by the loop)

    def eps(self, x, S, C, k):
        return self.net(torch.cat([x, S, C, torch.full((len(S), 1), k / self.N) if np.isscalar(k) else k], 1))

    def fit(self, S, A, C, steps, seed, lr=1e-4, batch=256, p_drop=0.2):
        g = torch.Generator().manual_seed(seed)
        S = torch.as_tensor(S, dtype=torch.float32); A = torch.as_tensor(A, dtype=torch.float32)
        C = torch.as_tensor(C, dtype=torch.float32)
        if self.opt is None:
            self.opt = torch.optim.Adam(self.parameters(), lr)
        for gr in self.opt.param_groups:
            gr["lr"] = lr
        for _ in range(steps):
            i = torch.randint(0, len(S), (batch,), generator=g)
            c = C[i] * (torch.rand(batch, 1, generator=g) > p_drop)
            k = torch.randint(0, self.N, (batch,), generator=g); e = torch.randn(batch, self.a_dim, generator=g)
            ab = self.al[k][:, None]
            x = ab.sqrt() * A[i] + (1 - ab).sqrt() * e
            loss = ((self.eps(x, S[i], c, k[:, None].float() / self.N) - e) ** 2).mean()
            self.opt.zero_grad(); loss.backward(); self.opt.step()
        return loss.item()

    @torch.no_grad()
    def sample(self, S, seed, w=None, v=None):
        w = self.w if w is None else w; v = self.v if v is None else v
        g = torch.Generator().manual_seed(int(seed))
        S = torch.as_tensor(S, dtype=torch.float32); n = len(S)
        x = torch.randn(n, self.a_dim, generator=g)
        conds = [torch.zeros(n, 2)]
        if w != 0:
            conds.append(torch.tensor(SUCC, dtype=torch.float32).expand(n, 2))
        if v != 0:
            conds.append(torch.tensor(FAIL, dtype=torch.float32).expand(n, 2))
        m = len(conds)
        Sm, Cm = S.repeat(m, 1), torch.cat(conds)
        for k in reversed(range(self.N)):
            out = self.eps(x.repeat(m, 1), Sm, Cm, k).view(m, n, -1)
            e = out[0]
            j = 1
            if w != 0:
                e = e + w * (out[j] - out[0]); j += 1
            if v != 0:
                e = e + v * (out[0] - out[j])
            ab, b = self.al[k], self.b[k]
            x = (x - b / (1 - ab).sqrt() * e) / (1 - b).sqrt()
            if k > 0:
                x += self.sig2[k].sqrt() * torch.randn(x.shape, generator=g)
        return x.clamp(-1, 1).numpy()

    @torch.no_grad()
    def implicit_q(self, S, A, n_mc=16, seed=0):
        """ELBO estimate of log mu(a|s,succ) - log mu(a|s,null) (up to the DDPM weighting): the 'critic' that the
        guidance implicitly follows. Used only for analysis."""
        g = torch.Generator().manual_seed(seed)
        S = torch.as_tensor(S, dtype=torch.float32); A = torch.as_tensor(A, dtype=torch.float32)
        n = len(S); out = torch.zeros(n)
        cs, cn = torch.tensor(SUCC, dtype=torch.float32).expand(n, 2), torch.zeros(n, 2)
        for _ in range(n_mc):
            k = torch.randint(0, self.N, (n,), generator=g); e = torch.randn(n, self.a_dim, generator=g)
            ab = self.al[k][:, None]; x = ab.sqrt() * A + (1 - ab).sqrt() * e
            kk = k[:, None].float() / self.N
            out += ((self.eps(x, S, cn, kk) - e) ** 2).sum(1) - ((self.eps(x, S, cs, kk) - e) ** 2).sum(1)
        return (out / n_mc).numpy()


DEF = dict(regime="easy", w=1.0, v=0.0, seed=0, rounds=6, n_train=400, n_eval=200, n_envs=50,
           n_demo=600, bc_steps=15000, fit_steps=2000, fit_lr=1e-4, p_drop=0.2)
REGIMES = {
    "easy": dict(demo_noise=0.45, table=False, train_h=None, eval_h=None, curriculum=False, her=False),
    "hard": dict(demo_noise=0.9, table=False, train_h=None, eval_h=None, curriculum=False, her=False),
    "takeoff": dict(demo_noise=0.45, table=True, train_h=[0.0, 0.3], eval_h=[0.1, 0.3], curriculum=True, her=True),
}


def pretrained_cond(c, R):
    """Conditional BC on the labelled demos (cached per seed and regime)."""
    os.makedirs(CACHE, exist_ok=True)
    f = os.path.join(CACHE, f"cbc_n{c['n_demo']}_z{R['demo_noise']}{'_table' if R['table'] else ''}_s{c['seed']}.pt")
    gen = CondDiffusion(OBS_DIM, ACT_DIM)
    if os.path.exists(f):
        st = torch.load(f, weights_only=False); gen.load_state_dict(st["gen"])
        return gen, st["S"], st["A"], st["C"]
    S, A, Y = collect_demos(c["n_demo"], R["demo_noise"], seed=1000 + c["seed"], table_only=R["table"])
    C = np.where(np.repeat(Y, NDEC)[:, None] > 0.5, SUCC, FAIL)
    S = S.reshape(-1, OBS_DIM); A = A.reshape(-1, ACT_DIM)
    torch.manual_seed(c["seed"])
    gen.fit(S, A, C, c["bc_steps"], c["seed"], lr=1e-3, p_drop=c["p_drop"])
    torch.save(dict(gen=gen.state_dict(), S=S, A=A, C=C, demo_success=float(Y.mean())), f)
    gen.opt = None
    return gen, S, A, C


def run(cfg, out=None):
    c = dict(DEF); c.update(cfg); R = REGIMES[c["regime"]]
    rng = np.random.default_rng(c["seed"]); torch.manual_seed(c["seed"])
    gen, Sd, Ad, Cd = pretrained_cond(c, R)
    envs = Envs(c["n_envs"])
    eval_seeds = [900_000 + c["seed"] * 10_000 + i for i in range(c["n_eval"])]
    RS, RA, RC, hist = [Sd], [Ad], [Cd], []
    t0 = time.time()
    nb = 12
    edges = None if R["train_h"] is None else np.linspace(R["train_h"][0], R["train_h"][1], nb + 1)
    succ_n, tot_n = np.ones(nb), np.full(nb, 2.0)
    probe_S = None

    def sample_with(w, v):
        gen.w, gen.v = w, v
        return gen

    def evaluate(rnd, extra):
        nonlocal probe_S
        res = {}
        for tag, (w, v) in dict(J_sys=(c["w"], c["v"]), J_uncond=(0.0, 0.0), J_cond=(1.0, 0.0)).items():
            S, A, Y, d = episodes(sample_with(w, v), envs, eval_seeds, 1, None, rng, h_range=R["eval_h"])
            res[tag] = float(Y.mean())
            if tag == "J_sys":
                if probe_S is None:   # fixed probe states (round-0 evaluation visits), for diversity tracking
                    probe_S = S[:64, 3].copy()
                res["lift_p95"] = float(np.quantile(d["final_obj"][:, 2] - 0.4247, 0.95))
        # action diversity of the deployed sampler at fixed states: mean per-state std over 16 samples
        gen.w, gen.v = c["w"], c["v"]
        smp = gen.sample(np.repeat(probe_S, 16, 0), rng.integers(1 << 30)).reshape(len(probe_S), 16, ACT_DIM)
        res["diversity"] = float(smp.std(1).mean())
        hist.append(dict(round=rnd, wall=time.time() - t0, **res, **extra))
        if out:
            json.dump(dict(cfg=c, hist=hist), open(out + ".tmp", "w")); os.replace(out + ".tmp", out)

    evaluate(0, {})
    for r in range(1, c["rounds"] + 1):
        seeds = [100_000_000 + c["seed"] * 1_000_000 + r * 10_000 + i for i in range(c["n_train"])]
        hts = None
        if R["curriculum"]:
            m = succ_n / tot_n; p = m * (1 - m) + 0.02
            b = rng.choice(nb, size=len(seeds), p=p / p.sum())
            hts = edges[b] + rng.random(len(seeds)) * (edges[b + 1] - edges[b])
        S, A, Y, d = episodes(sample_with(c["w"], c["v"]), envs, seeds, 1, None, rng, h_range=R["train_h"], heights=hts)
        if edges is not None and "heights" in d:
            bi = np.clip(np.searchsorted(edges, d["heights"], side="right") - 1, 0, nb - 1)
            np.add.at(succ_n, bi, Y); np.add.at(tot_n, bi, 1.0)
        lab = np.where(np.repeat(Y, NDEC)[:, None] > 0.5, SUCC, FAIL)
        RS.append(S.reshape(-1, OBS_DIM)); RA.append(A.reshape(-1, ACT_DIM)); RC.append(lab)
        if R["her"]:
            Sh = S.copy(); Sh[:, :, 25:28] = d["final_obj"][:, None, :]
            RS.append(Sh.reshape(-1, OBS_DIM)); RA.append(A.reshape(-1, ACT_DIM)); RC.append(np.tile(SUCC, (len(lab), 1)))
        loss = gen.fit(np.concatenate(RS), np.concatenate(RA), np.concatenate(RC), c["fit_steps"],
                       c["seed"] * 100 + r, lr=c["fit_lr"], p_drop=c["p_drop"])
        Cc = np.concatenate(RC)
        evaluate(r, dict(train_J=float(Y.mean()), fit_loss=loss, frac_succ_labels=float(Cc[:, 0].mean()),
                         train_lift_p95=float(np.quantile(d["final_obj"][:, 2] - 0.4247, 0.95))))
    if out:   # final model, for mechanism analysis (not committed: *.pt is git-ignored)
        torch.save(gen.state_dict(), out[:-5] + ".pt")
    return dict(cfg=c, hist=hist)


def job(args):
    name, cfg = args
    key = "_".join(f"{k}{v}" for k, v in sorted(cfg.items()))
    out = os.path.join(ROOT, "results", name, key + ".json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    if os.path.exists(out) and json.load(open(out)).get("done"):
        return
    t = time.time(); res = run(cfg, out); res["done"] = True; res["sec"] = time.time() - t
    json.dump(res, open(out + ".tmp", "w")); os.replace(out + ".tmp", out)
    h = res["hist"][-1]
    print(f"{cfg} J_sys={h['J_sys']:.2f} curve={[round(x['J_sys'], 2) for x in res['hist']]} ({res['sec']:.0f}s)",
          flush=True)


if __name__ == "__main__":
    from multiprocessing import Pool
    name, spec = sys.argv[1], json.loads(sys.argv[2])
    specs = spec if isinstance(spec, list) else [spec]
    cfgs = []
    for sp in specs:
        keys = list(sp)
        cfgs += [dict(zip(keys, v)) for v in itertools.product(*[sp[k] for k in keys])]
    cfgs.sort(key=lambda c: -c.get("rounds", 6))
    with Pool(int(sys.argv[3]) if len(sys.argv) > 3 else 4) as p:
        list(p.imap_unordered(job, [(name, c) for c in cfgs], chunksize=1))
