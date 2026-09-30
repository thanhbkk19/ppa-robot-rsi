"""Vectorised Robomimic (low-dim) environments built on DPPO's wrappers, plus a fixed-length
episode collector.

Episodes always run for the full horizon (no early termination on success): terminating on true
success would leak the ground truth Y through the episode length. Y is the sim success at the final
step and is only handed to the learner through the anchor oracle.
"""
from __future__ import annotations
import os
import sys
import json
import numpy as np
from omegaconf import OmegaConf

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DPPO = os.path.join(ROOT, "third_party", "dppo")
DATA = os.path.join(ROOT, "data")
if DPPO not in sys.path:
    sys.path.insert(0, DPPO)
os.environ.setdefault("MUJOCO_GL", "egl")

TASKS = {
    # obs_dim, act_dim, horizon (chunk), env steps per episode, checkpoint
    "square": dict(obs_dim=23, act_dim=7, horizon=4, max_steps=400,
                   ckpt="ckpt/square_pre_diffusion_mlp_ta4_td20_state_8000.pt",
                   low_dim_keys=["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos", "object"]),
    "transport": dict(obs_dim=59, act_dim=14, horizon=8, max_steps=800,
                      ckpt="ckpt/transport_pre_diffusion_mlp_ta8_td20_state_8000.pt",
                      low_dim_keys=["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos",
                                    "robot1_eef_pos", "robot1_eef_quat", "robot1_gripper_qpos", "object"]),
}


def task_cfg(task):
    c = dict(TASKS[task])
    c["n_decisions"] = c["max_steps"] // c["horizon"]
    c["normalization_path"] = os.path.join(DATA, "robomimic", task, "normalization.npz")
    c["env_meta_path"] = os.path.join(DPPO, "cfg", "robomimic", "env_meta", f"{task}.json")
    return c


def make_venv(task, n_envs):
    from env.gym_utils import make_async
    c = task_cfg(task)
    wrappers = OmegaConf.create({
        "robomimic_lowdim": {"normalization_path": c["normalization_path"], "low_dim_keys": c["low_dim_keys"]},
        "multi_step": {"n_obs_steps": 1, "n_action_steps": c["horizon"], "max_episode_steps": c["max_steps"],
                       "reset_within_step": False},
    })
    venv = make_async(task, env_type=None, num_envs=n_envs, asynchronous=True, max_episode_steps=c["max_steps"],
                      wrappers=wrappers, robomimic_env_cfg_path=c["env_meta_path"], shape_meta=None,
                      use_image_obs=False, render=False, render_offscreen=False,
                      obs_dim=c["obs_dim"], action_dim=c["act_dim"])
    return venv


class Collector:
    """Runs batches of fixed-length episodes. Initial states are set by per-episode integer seeds, so
    train and evaluation episodes can be drawn from disjoint seed ranges."""

    def __init__(self, task, n_envs):
        self.task, self.n_envs = task, n_envs
        self.c = task_cfg(task)
        self.venv = make_venv(task, n_envs)
        norm = np.load(self.c["normalization_path"])
        self.obs_min, self.obs_max = norm["obs_min"], norm["obs_max"]

    def unnormalize_obs(self, obs):
        return (obs / 2 + 0.5) * (self.obs_max - self.obs_min + 1e-6) + self.obs_min

    def run(self, act_fn, seeds):
        """act_fn(obs[B, obs_dim], t) -> actions [B, horizon, act_dim] (normalised).
        Returns dict of arrays over len(seeds) episodes:
          obs   [N, T+1, obs_dim]  normalised observations at each decision (last = final state)
          act   [N, T, horizon, act_dim]
          rew   [N, T]  env reward summed within each chunk (sparse success indicator)
          y     [N]     ground-truth success at the final step
          y_any [N]     success at any step (DPPO's metric; reporting only)
        """
        seeds = list(seeds)
        T = self.c["n_decisions"]
        out = {k: [] for k in ("obs", "act", "rew")}
        for b0 in range(0, len(seeds), self.n_envs):
            bs = seeds[b0:b0 + self.n_envs]
            n = len(bs)
            opts = [{"seed": int(s)} for s in bs] + [{"seed": int(bs[0])}] * (self.n_envs - n)
            obs = self.venv.reset_arg(options_list=opts)
            if isinstance(obs, list):
                obs = {"state": np.stack([o["state"] for o in obs])}
            O = np.zeros((self.n_envs, T + 1, self.c["obs_dim"]), np.float32)
            A = np.zeros((self.n_envs, T, self.c["horizon"], self.c["act_dim"]), np.float32)
            R = np.zeros((self.n_envs, T), np.float32)
            for t in range(T):
                o = obs["state"][:, -1]
                O[:, t] = o
                a = act_fn(o, t)
                A[:, t] = a
                obs, r, term, trunc, info = self.venv.step(a)
                R[:, t] = r
            O[:, T] = obs["state"][:, -1]
            out["obs"].append(O[:n]); out["act"].append(A[:n]); out["rew"].append(R[:n])
        res = {k: np.concatenate(v) for k, v in out.items()}
        res["y"] = (res["rew"][:, -1] >= self.c["horizon"] - 0.5).astype(np.float32)
        res["y_any"] = (res["rew"].max(1) > 0).astype(np.float32)
        res["seeds"] = np.asarray(seeds)
        return res

    def close(self):
        self.venv.close()
