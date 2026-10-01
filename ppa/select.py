"""Selection rules over K generator candidates scored by a critic ensemble (framework-agnostic: NumPy or torch).

  argmax  : pick the best ensemble-mean score (PA-RL / V-GPS)
  lcb     : argmax of mean - kappa * ensemble std (ensemble pessimism)
  softmax : sample from exp(q / temp)  (KL-regularised tilt of the empirical proposal)
  chi2    : sample from the chi^2-regularised tilt  K w_k = max(0, 1 + (q_k - lam) / (2 beta))
            (Huang et al. 2025, InferenceTimePessimism; docs/PA_THEORY.md Lemma 3). The density ratio to the
            proposal is at most 1 + range(q) / (2 beta) for every K, so more candidates never increase the
            critic error the selection can exploit.
"""
from __future__ import annotations
import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


def _is_torch(x):
    return torch is not None and isinstance(x, torch.Tensor)


def chi2_weights(q, beta):
    """q [B, K] -> w [B, K] maximising sum_k w_k q_k - beta * (K sum_k w_k^2 - 1) over the simplex."""
    if _is_torch(q):
        K = q.shape[1]
        qs = torch.sort(q, 1, descending=True).values
        m = torch.arange(1, K + 1, device=q.device, dtype=q.dtype)
        lam = (qs.cumsum(1) + 2 * beta * (m - K)) / m
        ok = (1 + (qs - lam) / (2 * beta)) > 0
        mstar = K - 1 - torch.argmax(ok.flip(1).to(torch.int8), 1)   # largest feasible support size - 1
        l = lam.gather(1, mstar[:, None])
        w = torch.clamp(1 + (q - l) / (2 * beta), min=0)
        return w / w.sum(1, keepdim=True)
    q = np.asarray(q, float)
    n, K = q.shape
    qs = -np.sort(-q, 1)
    m = np.arange(1, K + 1)
    lam = (qs.cumsum(1) + 2 * beta * (m - K)) / m
    ok = 1 + (qs - lam) / (2 * beta) > 0
    mstar = K - np.argmax(ok[:, ::-1], 1) - 1
    l = lam[np.arange(n), mstar][:, None]
    w = np.maximum(0, 1 + (q - l) / (2 * beta))
    return w / w.sum(1, keepdims=True)


def selection_probs(q_ens, rule="argmax", beta=0.05, temp=0.03, kappa=1.0):
    """q_ens [E, B, K] ensemble scores -> probabilities [B, K] over candidates (one-hot for argmax / lcb)."""
    T = _is_torch(q_ens)
    qm = q_ens.mean(0)
    if rule in ("argmax", "lcb"):
        z = qm if rule == "argmax" else qm - kappa * (q_ens.std(0) if T else q_ens.std(0, ddof=0))
        idx = z.argmax(1)
        if T:
            return torch.nn.functional.one_hot(idx, qm.shape[1]).to(qm.dtype)
        return np.eye(qm.shape[1])[idx]
    if rule == "softmax":
        z = qm / temp
        return torch.softmax(z, 1) if T else np.exp(z - z.max(1, keepdims=True)) / np.exp(
            z - z.max(1, keepdims=True)).sum(1, keepdims=True)
    if rule == "chi2":
        return chi2_weights(qm, beta)
    raise ValueError(rule)


def sample_index(p, rng=None, generator=None):
    """Draw one candidate index per row from probabilities p [B, K]."""
    if _is_torch(p):
        return torch.multinomial(p, 1, generator=generator).squeeze(1)
    rng = rng or np.random.default_rng()
    return (p.cumsum(1) < rng.random((len(p), 1))).sum(1).clip(0, p.shape[1] - 1)


def chi2_divergence(p):
    """chi^2(p || uniform_K) per row: K * sum p^2 - 1 (diagnostic: how far the selection moved from the proposal)."""
    K = p.shape[1]
    return K * (p ** 2).sum(1) - 1
