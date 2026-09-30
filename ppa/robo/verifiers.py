"""Self-verifiers V (cheap, gameable) and the reward-model feature maps used by r_hat / plug-in RM.

All verifiers judge the FINAL state of an episode (like a judge on the final frame).

Regimes (docs/PPA_SPEC.md §4.2):
  topdown     (c) controlled false positives: V fires when the nut is within the peg's xy tolerance,
                  ignoring height (a top-down camera cannot see insertion depth). False positives are
                  the "hovering / jammed above the peg" states, which the pretrained policy reaches in
                  ~15% of episodes. Reward-model features are the top-down-visible quantities only, so
                  the failure is INVISIBLE to r_hat / the plug-in reward model.
  classifier  (a) an MLP success classifier trained on the final states of the demos (label 1) plus
                  20 failed final states of the pretrained policy (label 0). Reward-model features are
                  the full final state, so its failures are (in principle) VISIBLE to r_hat.
"""
from __future__ import annotations
import os
import pickle
import numpy as np

from ppa.robo.env import ROOT, task_cfg

PEG1_XY = np.array([0.23, 0.10])   # NutAssemblySquare peg1 (fixed); success needs |dxy| < 0.03 and z < 0.87
XY_TOL = 0.03
CALIB = os.path.join(ROOT, "scale", "results", "raw", "calib", "square_pretrained_960.npz")
CLF_PATH = os.path.join(ROOT, "scale", "results", "raw", "calib", "square_classifier_v1.pkl")


class _Unnorm:
    def __init__(self, task):
        n = np.load(task_cfg(task)["normalization_path"])
        self.lo, self.hi = n["obs_min"], n["obs_max"]

    def __call__(self, o):
        return (o / 2 + 0.5) * (self.hi - self.lo + 1e-6) + self.lo


class TopDownVerifier:
    name = "topdown"

    def __init__(self, task="square"):
        assert task == "square"
        self.un = _Unnorm(task)

    def __call__(self, final_obs):
        f = self.un(final_obs)
        d = np.abs(f[:, 9:11] - PEG1_XY)
        return (d.max(1) < XY_TOL).astype(np.float64)

    def features(self, final_obs):
        """Top-down-visible reward-model features: nut and end-effector xy relative to the peg."""
        f = self.un(final_obs)
        return 10.0 * np.concatenate([f[:, 9:11] - PEG1_XY, f[:, 0:2] - PEG1_XY], 1)


class ClassifierVerifier:
    name = "classifier"

    def __init__(self, task="square"):
        assert task == "square"
        if not os.path.exists(CLF_PATH):
            fit_classifier()
        with open(CLF_PATH, "rb") as fh:
            self.clf = pickle.load(fh)

    def __call__(self, final_obs):
        return self.clf.predict(np.asarray(final_obs, np.float64)).astype(np.float64)

    def features(self, final_obs):
        return np.asarray(final_obs, np.float64)


def fit_classifier(n_fail=20):
    """Frozen once: demos' final states (1) + the first n_fail failures of the calibration rollouts (0).
    Calibration seeds (5e6 range) are disjoint from every train/eval seed."""
    from sklearn.neural_network import MLPClassifier
    from ppa.robo.generator import load_demos
    _, _, fin = load_demos("square", 4)
    d = np.load(CALIB)
    fail = np.flatnonzero(d["y"] == 0)[:n_fail]
    X = np.concatenate([fin, d["obs"][fail, -1]]).astype(np.float64)
    L = np.r_[np.ones(len(fin)), np.zeros(n_fail)]
    clf = MLPClassifier((64, 64), max_iter=2000, random_state=0).fit(X, L)
    with open(CLF_PATH, "wb") as fh:
        pickle.dump(clf, fh)


VERIFIERS = {"topdown": TopDownVerifier, "classifier": ClassifierVerifier}


def residual_design(v, feats):
    """Design matrix for the ridge residual r_hat(x) ~ E[Y - V | x]: [1, V, x, V*x]."""
    v = np.asarray(v, float)[:, None]
    return np.concatenate([np.ones_like(v), v, feats, v * feats], 1)
