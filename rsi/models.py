"""Generic conditional DDPM generator and MC critic ensemble (PyTorch, CPU-friendly)."""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn


def mlp(i, o, h=256, n=3):
    layers, d = [], i
    for _ in range(n):
        layers += [nn.Linear(d, h), nn.SiLU()]; d = h
    return nn.Sequential(*layers, nn.Linear(d, o))


class Diffusion(nn.Module):
    """x_0 = action chunk in [-1, 1]^A conditioned on s. 20-step DDPM with epsilon prediction."""

    def __init__(self, s_dim, a_dim, N=20, h=256):
        super().__init__()
        self.N, self.a_dim = N, a_dim
        b = torch.linspace(1e-4, 0.2, N); self.register_buffer("b", b)
        self.register_buffer("al", torch.cumprod(1 - b, 0))
        self.net = mlp(a_dim + s_dim + 1, a_dim, h)
        self.opt = None

    def loss(self, S, A, g, w=None):
        n = len(S)
        k = torch.randint(0, self.N, (n,), generator=g); e = torch.randn(n, self.a_dim, generator=g)
        ab = self.al[k][:, None]
        x = ab.sqrt() * A + (1 - ab).sqrt() * e
        l = ((self.net(torch.cat([x, S, k[:, None] / self.N], 1)) - e) ** 2).mean(1)
        return l.mean() if w is None else (w * l).sum() / w.sum().clamp_min(1e-8)

    def fit(self, S, A, steps, seed, lr=1e-3, batch=256, w=None):
        g = torch.Generator().manual_seed(seed)
        S = torch.as_tensor(S, dtype=torch.float32); A = torch.as_tensor(A, dtype=torch.float32)
        W = None if w is None else torch.as_tensor(w, dtype=torch.float32)
        if self.opt is None:
            self.opt = torch.optim.Adam(self.parameters(), lr)
        for gr in self.opt.param_groups:
            gr["lr"] = lr
        for _ in range(steps):
            i = torch.randint(0, len(S), (batch,), generator=g)
            loss = self.loss(S[i], A[i], g, None if W is None else W[i])
            self.opt.zero_grad(); loss.backward(); self.opt.step()
        return loss.item()

    @torch.no_grad()
    def sample(self, S, seed):
        g = torch.Generator().manual_seed(int(seed))
        S = torch.as_tensor(S, dtype=torch.float32)
        x = torch.randn(len(S), self.a_dim, generator=g)
        for k in reversed(range(self.N)):
            eps = self.net(torch.cat([x, S, torch.full((len(S), 1), k / self.N)], 1))
            ab, b = self.al[k], self.b[k]
            x = (x - b / (1 - ab).sqrt() * eps) / (1 - b).sqrt()
            if k > 0:
                x += b.sqrt() * torch.randn(x.shape, generator=g)
        return x.clamp(-1, 1).numpy()


class Critic:
    """Ensemble Q(s, a) -> P(success) by least squares on MC targets (linear in labels)."""

    def __init__(self, s_dim, a_dim, n_ens=2, lr=3e-4, seed=0, h=256):
        torch.manual_seed(seed)
        self.nets = [mlp(s_dim + a_dim, 1, h) for _ in range(n_ens)]
        self.opts = [torch.optim.Adam(n.parameters(), lr) for n in self.nets]
        self.g = torch.Generator().manual_seed(seed + 1)
        self.trained = False

    def fit(self, X, y, steps=2000, batch=512):
        X = torch.as_tensor(X, dtype=torch.float32); y = torch.as_tensor(y, dtype=torch.float32)
        for net, opt in zip(self.nets, self.opts):
            for _ in range(steps):
                i = torch.randint(0, len(X), (batch,), generator=self.g)
                loss = ((net(X[i]).squeeze(-1) - y[i]) ** 2).mean()
                opt.zero_grad(); loss.backward(); opt.step()
        self.trained = True
        return loss.item()

    @torch.no_grad()
    def __call__(self, X):
        X = torch.as_tensor(X, dtype=torch.float32)
        return np.stack([n(X).squeeze(-1).numpy() for n in self.nets])
