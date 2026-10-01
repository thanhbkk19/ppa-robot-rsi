import numpy as np
import pytest
import torch

from ppa.select import chi2_weights, selection_probs, sample_index, chi2_divergence


def test_chi2_weights_simplex_and_known_case():
    q = np.array([[0.1, 0.5, 0.9, 0.2], [0.5, 0.5, 0.5, 0.5]])
    w = chi2_weights(q, 0.1)
    assert np.allclose(w.sum(1), 1)
    assert np.allclose(w[0], [0, 0.25, 0.75, 0])     # support {0.9, 0.5}, lam = 0.5
    assert np.allclose(w[1], 0.25)                    # ties -> uniform


def test_chi2_limits_and_ratio_bound():
    rng = np.random.default_rng(0)
    q = rng.random((50, 64))
    assert np.allclose(chi2_weights(q, 1e6), 1 / 64, atol=1e-6)             # beta -> inf: no selection
    assert (chi2_weights(q, 1e-6).argmax(1) == q.argmax(1)).all()            # beta -> 0: argmax
    for beta in [0.02, 0.05, 0.2]:
        w = chi2_weights(q, beta)
        assert (64 * w <= 1 + (q.max(1) - q.min(1))[:, None] / (2 * beta) + 1e-9).all()


def test_chi2_is_optimal_against_random_simplex_points():
    rng = np.random.default_rng(1)
    q = rng.random((1, 8)); beta = 0.05
    obj = lambda w: (w * q).sum() - beta * (8 * (w ** 2).sum() - 1)
    best = obj(chi2_weights(q, beta)[0])
    for _ in range(2000):
        w = rng.dirichlet(np.ones(8) * 0.3)
        assert obj(w) <= best + 1e-9


def test_torch_matches_numpy():
    q = np.random.default_rng(2).random((20, 16))
    for beta in [0.01, 0.05, 0.3]:
        assert np.allclose(chi2_weights(torch.tensor(q), beta).numpy(), chi2_weights(q, beta), atol=1e-6)


@pytest.mark.parametrize("rule", ["argmax", "lcb", "softmax", "chi2"])
def test_selection_probs_shapes(rule):
    q = np.random.default_rng(3).random((2, 5, 7))
    p = selection_probs(q, rule)
    assert p.shape == (5, 7) and np.allclose(p.sum(1), 1)
    pt = selection_probs(torch.tensor(q), rule)
    assert np.allclose(pt.numpy(), p, atol=1e-6)
    assert sample_index(p, np.random.default_rng(0)).shape == (5,)
    assert (chi2_divergence(p) >= -1e-9).all()


def test_argmax_of_K_chi2_formula():
    # docs/PA_THEORY.md Lemma 2: chi^2(argmax-of-K || mu) = (K-1)^2 / (2K-1)
    rng = np.random.default_rng(0)
    for K in [2, 8, 32]:
        u = rng.random((400_000, K)).max(1)
        assert abs((K * u ** (K - 1)).mean() - 1 - (K - 1) ** 2 / (2 * K - 1)) < 0.05 * K
