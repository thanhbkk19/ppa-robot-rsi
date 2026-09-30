"""Unit tests for the PPA label contract. Run: pytest -q"""
import numpy as np, pytest
from ppa.labels import ppa_labels, AnchorSampler, RidgeResidual, label_batch


def _mc_mean_labels(v, y, p, r_hat, n=200_000, seed=0):
    rng = np.random.default_rng(seed)
    a = rng.random((n, len(v))) < p
    lab = ppa_labels(np.broadcast_to(v, a.shape), np.broadcast_to(y, a.shape), a,
                     np.broadcast_to(p, a.shape), np.broadcast_to(r_hat, a.shape))
    return lab.mean(0)


@pytest.mark.parametrize("r_hat", [0.0, 0.7, -3.0])
def test_unbiased_for_any_verifier_and_residual(r_hat):
    y = np.array([0., 1., 0., 1.])
    v = np.array([1., 1., 0., 0.3])   # fooled, correct, correct, wrong
    p = np.array([0.05, 0.05, 0.2, 0.5])
    m = _mc_mean_labels(v, y, p, np.full(4, r_hat))
    assert np.allclose(m, y, atol=0.05), m


def test_missing_ground_truth_only_allowed_when_not_anchored():
    v = np.array([1., 0.]); p = np.array([.5, .5])
    ppa_labels(v, np.array([np.nan, np.nan]), np.array([False, False]), p)
    with pytest.raises(ValueError):
        ppa_labels(v, np.array([np.nan, 0.]), np.array([True, False]), p)


def test_zero_probability_rejected():
    with pytest.raises(ValueError):
        ppa_labels(np.array([1.]), np.array([0.]), np.array([False]), np.array([0.]))


def test_claimed_sampler_respects_budget_in_expectation():
    s = AnchorSampler(0.05, mode="claimed", seed=1)
    rng = np.random.default_rng(2); used = 0; n = 0
    for _ in range(2000):
        v = (rng.random(100) < 0.7).astype(float)
        a, p = s.sample(v); used += a.sum(); n += 100; s.update(v)
    assert abs(used / n - 0.05) < 0.005


def test_label_batch_is_predictable():
    """The residual used to label a batch must not have seen that batch's ground truth."""
    res = RidgeResidual(dim=2, lam=1e-6); s = AnchorSampler(1.0, seed=0)
    feats = np.array([[1., 0.], [0., 1.]]); v = np.array([1., 1.]); y = np.array([0., 1.])
    lab, info = label_batch(v, lambda i: y[i], feats, s, res)
    assert np.allclose(lab, y)                        # p = 1: labels equal ground truth exactly
    assert np.allclose(res.predict(feats), y - v, atol=1e-3)   # residual learned only afterwards
