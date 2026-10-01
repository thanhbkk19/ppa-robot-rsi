"""Diagnostic testbed for the sample-K -> select -> distill loop with a *learned* NN critic.

Same 2-D multi-step navigation task, demos and diffusion generator as toy/multistep.py, but the selector is a
neural-network critic ensemble trained on executed (state, selected action) pairs with Monte-Carlo targets,
as in ppa/robo/critic.py. The critic never sees unselected candidates, so its error on rarely-executed
actions is real (this is what the ridge/RBF critic of the original toy could not show).
"""
from __future__ import annotations
import os
import sys
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "toy"))
from multistep import Diff, step, T, demos, success  # noqa: E402

torch.set_num_threads(1)


class Critic:
    """Ensemble of MLPs Q(s, a, t) -> P(success), trained by least squares on MC targets (label linear)."""

    def __init__(self, n_ens=2, hidden=256, lr=1e-3, seed=0):
        g = torch.Generator().manual_seed(seed)
        self.nets = []
        for _ in range(n_ens):
            net = nn.Sequential(nn.Linear(5, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU(),
                                nn.Linear(hidden, 1))
            for p in net.parameters():
                with torch.no_grad():
                    p.copy_(torch.randn(p.shape, generator=g) * (1.0 / np.sqrt(p.shape[-1])) if p.dim() > 1
                            else torch.zeros_like(p))
            self.nets.append(net)
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

    @torch.no_grad()
    def __call__(self, X):
        """Returns (n_ens, ...) predictions."""
        X = torch.as_tensor(X, dtype=torch.float32)
        return np.stack([n(X).squeeze(-1).numpy() for n in self.nets])


def feats(S, C):
    """S: (n, 3) [x, y, t/T]; C: (n, K, 2) -> (n, K, 5)."""
    n, K, _ = C.shape
    return np.concatenate([np.repeat(S[:, None, :2], K, 1), C, np.repeat(S[:, None, 2:], K, 1)], -1).astype(np.float32)


def rollout(gen, n, K, rng, select):
    """select(S, C) -> index (n,) of the executed candidate. Returns per-step logs, Y, final state."""
    s = np.zeros((n, 2)) + 0.1 * rng.standard_normal((n, 2)); done = np.zeros(n, bool); Y = np.zeros(n)
    log = []
    for t in range(T):
        S = np.c_[s, np.full(n, t / T)].astype(np.float32)
        C = gen.sample(np.repeat(S, K, 0), rng.integers(1 << 30)).reshape(n, K, 2)
        j = select(S, C)
        a = C[np.arange(n), j]
        log.append(dict(S=S, C=C, j=j, a=a, active=~done))
        s2 = step(s, a); s = np.where(done[:, None], s, s2)
        newly = success(s) & ~done; Y[newly] = 1; done |= newly
    return log, Y, s
