"""Generic conditional DDPM generator and MC critic ensemble (PyTorch, CPU-friendly).

Device: RSI_DEVICE=cuda puts every network, batch and random generator on the GPU (default cpu). Inputs and
outputs stay numpy on the host. On cpu the code path, and so every result, is unchanged.
"""
from __future__ import annotations
import os
import numpy as np
import torch
import torch.nn as nn

DEV = torch.device(os.environ.get("RSI_DEVICE", "cpu"))


def gen_on(seed):
    """A seeded random generator on the compute device."""
    return torch.Generator(device=DEV).manual_seed(int(seed))


def cpu_state(module):
    """state_dict with every tensor on the cpu (checkpoints load anywhere)."""
    return {k: v.detach().cpu() for k, v in module.state_dict().items()}


def mlp(i, o, h=256, n=3):
    layers, d = [], i
    for _ in range(n):
        layers += [nn.Linear(d, h), nn.SiLU()]; d = h
    return nn.Sequential(*layers, nn.Linear(d, o))


class Diffusion(nn.Module):
    """x_0 = action chunk in [-1, 1]^A conditioned on s. 20-step DDPM with epsilon prediction."""

    def __init__(self, s_dim, a_dim, N=20, h=256, var="posterior"):
        super().__init__()
        self.N, self.a_dim = N, a_dim
        b = torch.linspace(1e-4, 0.2, N); self.register_buffer("b", b)
        al = torch.cumprod(1 - b, 0); self.register_buffer("al", al)
        # reverse-process variance: "beta" (sigma_k^2 = beta_k, DDPM's upper choice) or "posterior"
        # (sigma_k^2 = beta_k (1 - al_{k-1}) / (1 - al_k), the true posterior variance, DDPM's lower choice)
        al_prev = torch.cat([torch.ones(1), al[:-1]])
        self.register_buffer("sig2", b if var == "beta" else b * (1 - al_prev) / (1 - al), persistent=False)
        self.net = mlp(a_dim + s_dim + 1, a_dim, h)
        self.opt = None
        self.temp = 1.0       # sampling temperature: scales the initial and the per-step noise
        # goal classifier-free guidance: during training the desired goal (obs dims 25:28) is replaced by a null
        # value (0, far outside the workspace) with probability goal_drop; sampling uses
        # eps = eps_null + goal_w (eps_goal - eps_null). goal_w = 1 is the plain conditional model.
        self.goal_drop, self.goal_w = 0.0, 1.0
        self.to(DEV)

    def loss(self, S, A, g, w=None):
        n = len(S)
        if self.goal_drop > 0:
            S = S.clone(); S[torch.rand(n, generator=g, device=DEV) < self.goal_drop, 25:28] = 0.0
        k = torch.randint(0, self.N, (n,), generator=g, device=DEV)
        e = torch.randn(n, self.a_dim, generator=g, device=DEV)
        ab = self.al[k][:, None]
        x = ab.sqrt() * A + (1 - ab).sqrt() * e
        l = ((self.net(torch.cat([x, S, k[:, None] / self.N], 1)) - e) ** 2).mean(1)
        return l.mean() if w is None else (w * l).sum() / w.sum().clamp_min(1e-8)

    def fit(self, S, A, steps, seed, lr=1e-3, batch=256, w=None):
        g = gen_on(seed)
        S = torch.as_tensor(S, dtype=torch.float32, device=DEV); A = torch.as_tensor(A, dtype=torch.float32, device=DEV)
        W = None if w is None else torch.as_tensor(w, dtype=torch.float32, device=DEV)
        if self.opt is None:
            self.opt = torch.optim.Adam(self.parameters(), lr)
        for gr in self.opt.param_groups:
            gr["lr"] = lr
        for _ in range(steps):
            i = torch.randint(0, len(S), (batch,), generator=g, device=DEV)
            loss = self.loss(S[i], A[i], g, None if W is None else W[i])
            self.opt.zero_grad(); loss.backward(); self.opt.step()
        return loss.item()

    @torch.no_grad()
    def sample(self, S, seed):
        g = gen_on(seed)
        S = torch.as_tensor(S, dtype=torch.float32, device=DEV)
        x = self.temp * torch.randn(len(S), self.a_dim, generator=g, device=DEV)
        for k in reversed(range(self.N)):
            kk = torch.full((len(S), 1), k / self.N, device=DEV)
            eps = self.net(torch.cat([x, S, kk], 1))
            if self.goal_w != 1.0:
                S0 = S.clone(); S0[:, 25:28] = 0.0
                e0 = self.net(torch.cat([x, S0, kk], 1))
                eps = e0 + self.goal_w * (eps - e0)
            ab, b = self.al[k], self.b[k]
            x = (x - b / (1 - ab).sqrt() * eps) / (1 - b).sqrt()
            if k > 0:
                x += self.temp * self.sig2[k].sqrt() * torch.randn(x.shape, generator=g, device=DEV)
        return x.clamp(-1, 1).cpu().numpy()


class Critic:
    """Ensemble Q(s, a) -> P(success) by least squares on MC targets (linear in labels)."""

    def __init__(self, s_dim, a_dim, n_ens=2, lr=3e-4, seed=0, h=256):
        torch.manual_seed(seed)
        self.nets = [mlp(s_dim + a_dim, 1, h).to(DEV) for _ in range(n_ens)]
        self.opts = [torch.optim.Adam(n.parameters(), lr) for n in self.nets]
        self.g = gen_on(seed + 1)
        self.trained = False

    def fit(self, X, y, steps=2000, batch=512, wboot=None):
        """wboot [N, n_ens]: per-row bootstrap weights (Poisson(1) per episode) -> each member is fit on its own
        bootstrap resample, so ensemble disagreement estimates the critic's statistical error."""
        X = torch.as_tensor(X, dtype=torch.float32, device=DEV); y = torch.as_tensor(y, dtype=torch.float32, device=DEV)
        W = None if wboot is None else torch.as_tensor(wboot, dtype=torch.float32, device=DEV)
        for m, (net, opt) in enumerate(zip(self.nets, self.opts)):
            for _ in range(steps):
                if W is None:
                    i = torch.randint(0, len(X), (batch,), generator=self.g, device=DEV)
                else:
                    i = torch.multinomial(W[:, m], batch, replacement=True, generator=self.g)
                loss = ((net(X[i]).squeeze(-1) - y[i]) ** 2).mean()
                opt.zero_grad(); loss.backward(); opt.step()
        self.trained = True
        return loss.item()

    def fit_sarsa(self, X, Xn, last, y, steps=2000, batch=512, tau=0.005):
        """SARSA targets on executed chunks: y_t = Q_targ(s_{t+1}, a_{t+1}) for t < T-1, y_{T-1} = success.
        Lower-variance than MC (future randomness is replaced by its estimate), biased by bootstrapping."""
        import copy
        X = torch.as_tensor(X, dtype=torch.float32, device=DEV); Xn = torch.as_tensor(Xn, dtype=torch.float32, device=DEV)
        last = torch.as_tensor(last, dtype=torch.bool, device=DEV); y = torch.as_tensor(y, dtype=torch.float32, device=DEV)
        if not hasattr(self, "targs"):
            self.targs = [copy.deepcopy(n) for n in self.nets]
        for _ in range(steps):
            i = torch.randint(0, len(X), (batch,), generator=self.g, device=DEV)
            with torch.no_grad():
                qn = torch.stack([t(Xn[i]).squeeze(-1) for t in self.targs]).mean(0)
                tgt = torch.where(last[i], y[i], qn)
            for net, opt in zip(self.nets, self.opts):
                loss = ((net(X[i]).squeeze(-1) - tgt) ** 2).mean()
                opt.zero_grad(); loss.backward(); opt.step()
            with torch.no_grad():
                for net, t in zip(self.nets, self.targs):
                    for p, pt in zip(net.parameters(), t.parameters()):
                        pt.lerp_(p, tau)
        self.trained = True
        return loss.item()

    @torch.no_grad()
    def __call__(self, X):
        X = torch.as_tensor(X, dtype=torch.float32, device=DEV)
        return torch.stack([n(X).squeeze(-1) for n in self.nets]).cpu().numpy()


def load_system(path):
    """Load a saved (generator, critic) pair from rsi/loop_fetch.py, inferring the network width."""
    from rsi.fetch import OBS_DIM, ACT_DIM
    st = torch.load(path, weights_only=False, map_location="cpu")
    h = st["gen"]["net.0.weight"].shape[0]
    gen = Diffusion(OBS_DIM, ACT_DIM, h=h); gen.load_state_dict(st["gen"])
    if "critic" not in st:   # a pretrained (BC) cache file: generator only
        return gen, None
    cr = Critic(OBS_DIM, ACT_DIM, n_ens=len(st["critic"]), h=h)
    for net, sd in zip(cr.nets, st["critic"]):
        net.load_state_dict(sd)
    cr.trained = True
    return gen, cr
