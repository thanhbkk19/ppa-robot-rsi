"""M3 contract tests for the Robomimic loop's labelling step (no simulator needed)."""
import numpy as np
import pytest

loop = pytest.importorskip("ppa.robo.loop")
from ppa.labels import AnchorSampler  # noqa: E402


class _V:
    def __call__(self, final):
        return (final[:, 0] > 0).astype(float)

    def features(self, final):
        return final[:, :3].astype(float)


def _run(method, budget, residual="none", anchoring="uniform", seed=0):
    cfg = loop.make_cfg(method=method, budget=budget, residual=residual, anchoring=anchoring, seed=seed)
    r = loop.Run.__new__(loop.Run)
    r.cfg, r.V, r.resid, r.n_queried = cfg, _V(), None, 0
    r.sampler = AnchorSampler(budget, mode=anchoring, seed=seed) if method in loop.ANCHORED else None
    return r


def _episodes(n=200, seed=0):
    rng = np.random.default_rng(seed)
    obs = rng.normal(size=(n, 5, 4)).astype(np.float32)
    y = (rng.random(n) < 0.4).astype(np.float32)  # V is wrong on ~half the episodes
    return dict(obs=obs, y=y)


@pytest.mark.parametrize("residual", ["none", "learned"])
def test_ppa_with_budget_one_is_exactly_the_true_reward_backbone(residual):
    ep = _episodes()
    lab_dr, anch, _, _ = _run("dr", 1.0, residual)._label(ep)
    lab_or, _, _, _ = _run("oracle", 1.0)._label(ep)
    assert anch.all()
    np.testing.assert_allclose(lab_dr, lab_or, atol=1e-12)


def test_dr_labels_are_unbiased_and_unclipped():
    ep = _episodes(n=4000, seed=1)
    labs = [_run("dr", 0.05, "learned", seed=s)._label(ep)[0] for s in range(300)]
    mean = np.mean(labs, 0)
    assert np.max(np.abs(labs)) > 1.5  # never clipped
    assert abs(mean.mean() - ep["y"].mean()) < 0.02
    assert np.corrcoef(mean, ep["y"])[0, 1] > 0.9


def test_oracle_is_read_only_for_anchored_episodes():
    ep = _episodes(n=480)
    r = _run("naive_mix", 0.02)
    lab, anch, v, _ = r._label(ep)
    assert r.n_queried == anch.sum()
    np.testing.assert_array_equal(lab[~anch], v[~anch])
