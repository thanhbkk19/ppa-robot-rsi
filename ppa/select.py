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


def chi2_trust_weights(q, delta, iters=40):
    """chi^2 trust region: maximise sum_k w_k q_k  s.t.  chi2(w || uniform_K) <= delta, per row.
    The solution is the chi^2 tilt chi2_weights(q, beta) with the row's beta set (bisection on log beta) so that
    the constraint binds; if even argmax stays inside the region (delta >= K - 1) the row is argmax.
    delta is scale-free (it does not depend on the units of q) and decouples the step size from K:
    argmax-of-K has chi2 = (K-1)^2/(2K-1) to the proposal, so it couples both (docs/PA_THEORY.md Lemma 2)."""
    T = _is_torch(q)
    xp_log = (lambda x: torch.log(torch.as_tensor(x, dtype=q.dtype, device=q.device))) if T else np.log
    n, K = q.shape
    lo = xp_log(1e-6) * (torch.ones(n, 1, dtype=q.dtype, device=q.device) if T else np.ones((n, 1)))
    hi = xp_log(1e3) * (torch.ones(n, 1, dtype=q.dtype, device=q.device) if T else np.ones((n, 1)))
    exp = torch.exp if T else np.exp
    for _ in range(iters):
        mid = (lo + hi) / 2
        w = _chi2_rowbeta(q, exp(mid))
        too_far = chi2_divergence(w)[:, None] > delta          # beta too small -> raise it
        lo = (mid * too_far + lo * ~too_far) if T else np.where(too_far, mid, lo)
        hi = (hi * too_far + mid * ~too_far) if T else np.where(too_far, hi, mid)
    return _chi2_rowbeta(q, exp(hi))


def _chi2_rowbeta(q, beta):
    """chi2_weights with a per-row beta [B, 1]."""
    T = _is_torch(q)
    K = q.shape[1]
    if T:
        qs = torch.sort(q, 1, descending=True).values
        m = torch.arange(1, K + 1, device=q.device, dtype=q.dtype)
        lam = (qs.cumsum(1) + 2 * beta * (m - K)) / m
        ok = (1 + (qs - lam) / (2 * beta)) > 0
        mstar = K - 1 - torch.argmax(ok.flip(1).to(torch.int8), 1)
        l = lam.gather(1, mstar[:, None])
        w = torch.clamp(1 + (q - l) / (2 * beta), min=0)
        return w / w.sum(1, keepdim=True)
    qs = -np.sort(-q, 1)
    m = np.arange(1, K + 1)
    lam = (qs.cumsum(1) + 2 * beta * (m - K)) / m
    ok = 1 + (qs - lam) / (2 * beta) > 0
    mstar = K - np.argmax(ok[:, ::-1], 1) - 1
    l = lam[np.arange(len(q)), mstar][:, None]
    w = np.maximum(0, 1 + (q - l) / (2 * beta))
    return w / w.sum(1, keepdims=True)


def lcb_opt_weights(q, c, iters=50):
    """Per-row maximiser of Lemma 1's certified lower bound for the empirical proposal (uniform over K):
        w.q - c * sqrt(chi2(w || u)),   chi2(w || u) = K ||w||^2 - 1,   c = z * eps(s) >= 0.
    KKT: w is proportional to u = (q - lam)_+ with  sd_k(u) = c  (population sd over the K entries).
    If sd_k(q) <= c the critic's spread does not exceed its error and the solution is uniform (no selection,
    docs/PA_THEORY.md Lemma 3); otherwise lam rises, truncating low candidates, until the truncated spread is c.
    q [B, K]; c [B] or scalar. NumPy only (Fetch testbed)."""
    q = np.asarray(q, float)
    n, K = q.shape
    c = np.broadcast_to(np.asarray(c, float), (n,))[:, None]
    sd = lambda lam: np.maximum(q - lam, 0).std(1, keepdims=True)
    lo = q.min(1, keepdims=True); hi = q.max(1, keepdims=True)      # sd(lo) = sd(q), sd(hi) = 0
    for _ in range(iters):
        mid = (lo + hi) / 2
        big = sd(mid) > c
        lo = np.where(big, mid, lo); hi = np.where(big, hi, mid)
    u = np.maximum(q - lo, 0)
    flat = q.std(1, keepdims=True) <= c                               # no certified improvement: uniform
    w = np.where(flat, 1.0, u)
    s_ = w.sum(1, keepdims=True)
    w = np.where(s_ > 0, w / np.where(s_ > 0, s_, 1), np.eye(K)[q.argmax(1)])
    return w


def selection_probs(q_ens, rule="argmax", beta=0.05, temp=0.03, kappa=1.0, delta=1.0):
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
    if rule == "chi2tr":
        return chi2_trust_weights(qm, delta)
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
