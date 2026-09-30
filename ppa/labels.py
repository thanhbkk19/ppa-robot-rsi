"""Prediction-powered (doubly robust) outcome labels for self-improving robot policies.

Framework-agnostic (NumPy). Import these into the scale-up codebase (PA-RL / DPPO / DSRL fork);
do not re-derive them inline.

    Y_tilde = V + r_hat + (A / p) * (Y - V - r_hat),   A ~ Bernoulli(p)

V      self-verifier output for the episode, in [0, 1] (learned success detector, VLM judge, progress model)
Y      ground-truth outcome, queried ONLY where A == 1 (human check or privileged sim success)
p      anchor probability, fixed BEFORE Y is observed, using pre-label information only
r_hat  prediction of (Y - V) from a residual model fitted on PREVIOUS batches only

Guarantee (P1): E[Y_tilde | episode] = Y for any V and any predictable r_hat, provided p > 0.
Hence every learner whose update is linear in its labels (likelihood-ratio PG, least-squares critic
on Monte-Carlo targets) has the same expected update as with ground-truth labels (C1).

Contract, where breaking any line voids the guarantee:
  * never clip or rescale Y_tilde (use a smaller lr or a larger batch instead)
  * never refit r_hat on a batch before labelling that same batch
  * the anchor decision must not depend on Y or on anything computed from it
  * p must be > 0 for every episode that can occur
"""
from __future__ import annotations
import numpy as np


def ppa_labels(v, y, anchored, p, r_hat=None):
    """Return unbiased labels. y may be NaN wherever anchored is False (it is never read there)."""
    v = np.asarray(v, float); anchored = np.asarray(anchored, bool); p = np.asarray(p, float)
    r = np.zeros_like(v) if r_hat is None else np.asarray(r_hat, float)
    if np.any(p <= 0) or np.any(p > 1):
        raise ValueError("anchor probabilities must lie in (0, 1]")
    y = np.asarray(y, float)
    if np.any(np.isnan(y[anchored])):
        raise ValueError("ground truth missing for an anchored episode")
    resid = np.where(anchored, np.nan_to_num(y) - v - r, 0.0)
    return v + r + resid / p


class AnchorSampler:
    """Chooses which episodes get a ground-truth check.

    mode='uniform': p = budget for every episode.
    mode='claimed': Neyman-style allocation for verifiers with (almost) no false negatives: spend the
                    budget on episodes the verifier calls successes; keep a floor p_lo = floor_frac*budget
                    elsewhere so P1 still holds if false negatives exist. The expected budget uses the
                    PREVIOUS running estimate of P(V > thr) (predictable).
    In the toy, 'claimed' gave no consistent gain over 'uniform'; keep both and report both.
    """

    def __init__(self, budget: float, mode: str = "uniform", floor_frac: float = 0.1,
                 thr: float = 0.5, ema: float = 0.1, seed: int | None = None):
        assert 0 < budget <= 1
        self.budget, self.mode, self.floor_frac, self.thr, self.ema = budget, mode, floor_frac, thr, ema
        self.f_claimed = 0.5
        self.rng = np.random.default_rng(seed)
        self.n_queried = 0

    def probs(self, v):
        v = np.asarray(v, float)
        if self.mode == "uniform":
            return np.full(v.shape, self.budget)
        p_lo = self.floor_frac * self.budget
        p_hi = min(1.0, (self.budget - p_lo * (1 - self.f_claimed)) / max(self.f_claimed, 1e-3))
        return np.where(v > self.thr, p_hi, p_lo)

    def sample(self, v):
        """Returns (anchored mask, p). Call update(v) AFTER labels for this batch are computed."""
        p = self.probs(v)
        a = self.rng.random(p.shape) < p
        self.n_queried += int(a.sum())
        return a, p

    def update(self, v):
        self.f_claimed = (1 - self.ema) * self.f_claimed + self.ema * float(np.mean(np.asarray(v) > self.thr))


class RidgeResidual:
    """Minimal predictable residual model r_hat(x) = E[Y - V | x], fitted on anchored episodes only.
    Replace by any regressor with the same predict/add/refit protocol. r_hat = 0 is always valid."""

    def __init__(self, dim: int, lam: float = 1.0):
        self.A = lam * np.eye(dim); self.b = np.zeros(dim); self.w = np.zeros(dim)

    def predict(self, x):
        return np.asarray(x, float) @ self.w

    def add(self, x, target, weight=None):
        x = np.asarray(x, float); t = np.asarray(target, float)
        w = np.ones(len(t)) if weight is None else np.asarray(weight, float)
        self.A += (x * w[:, None]).T @ x; self.b += (x * w[:, None]).T @ t

    def refit(self):
        self.w = np.linalg.solve(self.A, self.b)


def label_batch(v, y_oracle, feats, sampler: AnchorSampler, residual: RidgeResidual | None):
    """One predictable labelling step for a batch of finished episodes.
    y_oracle: callable(indices) -> ground truth for those episodes (the ONLY place Y is read)."""
    v = np.asarray(v, float)
    a, p = sampler.sample(v)
    r = residual.predict(feats) if residual is not None else np.zeros_like(v)
    y = np.full(v.shape, np.nan)
    idx = np.flatnonzero(a)
    if len(idx):
        y[idx] = y_oracle(idx)
    lab = ppa_labels(v, y, a, p, r)
    if residual is not None and len(idx):  # update AFTER labelling -> used for future batches only
        residual.add(np.asarray(feats)[idx], y[idx] - v[idx]); residual.refit()
    sampler.update(v)
    return lab, dict(anchored=a, p=p, n_queried=int(a.sum()))
