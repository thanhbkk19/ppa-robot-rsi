"""Chunk critic Q(s, a_chunk, t) used for critic-argmax selection over K generator candidates.

Targets (episode label L is the only reward, at the final decision T-1):
  mc : y_t = gamma^(T-1-t) * L       -- linear in L, so C1 holds exactly (least squares)
  td : y_t = gamma * Q_targ(s_{t+1}, a_{t+1}, t+1) for t < T-1, y_{T-1} = L   (SARSA on executed chunks;
       bootstrapping -> C1 only approximate)
"""
from __future__ import annotations
import copy
import numpy as np
import torch
import torch.nn as nn


class QNet(nn.Module):
    def __init__(self, in_dim, hidden=256, depth=3, n_ens=2):
        super().__init__()
        self.members = nn.ModuleList()
        for _ in range(n_ens):
            layers, d = [], in_dim
            for _ in range(depth):
                layers += [nn.Linear(d, hidden), nn.LayerNorm(hidden), nn.Mish()]
                d = hidden
            layers.append(nn.Linear(d, 1))
            self.members.append(nn.Sequential(*layers))

    def forward(self, x):  # -> [n_ens, B]
        return torch.stack([m(x).squeeze(-1) for m in self.members])


class Critic:
    def __init__(self, obs_dim, act_flat, T, gamma=0.99, lr=3e-4, n_ens=2, target="mc", device="cuda", seed=0):
        torch.manual_seed(seed)
        self.T, self.gamma, self.target, self.dev = T, gamma, target, device
        self.q = QNet(obs_dim + act_flat + 1, n_ens=n_ens).to(device)
        self.q_targ = copy.deepcopy(self.q)
        self.opt = torch.optim.Adam(self.q.parameters(), lr=lr)
        self.n_updates = 0
        self.g = torch.Generator(device="cpu").manual_seed(seed)

    def _x(self, obs, act, t):
        tt = t.float()[:, None] / self.T
        return torch.cat([obs, act.flatten(1), tt], 1)

    @torch.no_grad()
    def score(self, obs, cands, t):
        """obs [B, D] torch, cands [B, K, H, A] torch -> Q [B, K] (ensemble mean)."""
        B, K = cands.shape[:2]
        o = obs.repeat_interleave(K, 0)
        tt = torch.full((B * K,), t, device=obs.device)
        return self.q(self._x(o, cands.reshape(B * K, *cands.shape[2:]), tt)).mean(0).view(B, K)

    @torch.no_grad()
    def score_ens(self, obs, cands, t):
        """Like score, but returns every ensemble member: [n_ens, B, K] (for pessimistic selection rules)."""
        B, K = cands.shape[:2]
        o = obs.repeat_interleave(K, 0)
        tt = torch.full((B * K,), t, device=obs.device)
        return self.q(self._x(o, cands.reshape(B * K, *cands.shape[2:]), tt)).view(-1, B, K)

    def fit(self, obs, act, labels, steps, batch=1024):
        """obs [N, T+1, D], act [N, T, H, A], labels [N] (one per episode; any real value)."""
        if len(obs) == 0 or steps == 0:
            return float("nan")
        N, T = act.shape[:2]
        dev = self.dev
        O = torch.as_tensor(obs, dtype=torch.float32, device=dev)
        A = torch.as_tensor(act, dtype=torch.float32, device=dev)
        L = torch.as_tensor(labels, dtype=torch.float32, device=dev)
        disc = self.gamma ** torch.arange(T - 1, -1, -1, device=dev, dtype=torch.float32)  # gamma^(T-1-t)
        losses = []
        for s in range(steps):
            ei = torch.randint(0, N, (batch,), generator=self.g).to(dev)
            ti = torch.randint(0, T, (batch,), generator=self.g).to(dev)
            x = self._x(O[ei, ti], A[ei, ti], ti)
            if self.target == "mc":
                y = disc[ti] * L[ei]
            else:
                with torch.no_grad():
                    last = ti == T - 1
                    tn = torch.clamp(ti + 1, max=T - 1)
                    qn = self.q_targ(self._x(O[ei, tn], A[ei, tn], tn)).mean(0)
                    y = torch.where(last, L[ei], self.gamma * qn)
            q = self.q(x)
            loss = ((q - y[None]) ** 2).mean()
            self.opt.zero_grad(set_to_none=True); loss.backward(); self.opt.step()
            if self.target == "td":
                with torch.no_grad():
                    for p, pt in zip(self.q.parameters(), self.q_targ.parameters()):
                        pt.lerp_(p, 0.005)
            losses.append(loss.item())
            self.n_updates += 1
        return float(np.mean(losses[-max(1, steps // 10):]))
