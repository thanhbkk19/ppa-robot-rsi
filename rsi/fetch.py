"""Fetch pick-and-place testbed (MuJoCo, CPU) for the sample-K -> select -> distill loop.

- Actions are chunks of H = 4 low-level 4-D actions (xyz displacement + gripper), 13 decisions per episode
  (50 env steps, the last chunk truncated). Episodes are fixed length; Y = env success at the final step
  (no early termination, so episode length never leaks Y; same convention as scale/DECISIONS.md D3).
- Demos come from a noisy scripted controller; the pretrained diffusion policy is fit on them by BC.
"""
from __future__ import annotations
import os
import numpy as np
import gymnasium as gym
import gymnasium_robotics

gym.register_envs(gymnasium_robotics)
H, STEPS = 4, 50
NDEC = (STEPS + H - 1) // H
# RSI_GOALREL=1 appends the goal relative to the object (desired - achieved goal), a translation-invariant goal
# feature: "move the object toward the goal" is then the same function at every goal height.
GOALREL = os.environ.get("RSI_GOALREL") == "1"
OBS_DIM = 25 + 3 + (3 if GOALREL else 0) + 1     # observation, desired goal, [goal - object], t / NDEC


def obs_vec(o):
    v = [o["observation"], o["desired_goal"]]
    if GOALREL:
        v.append(o["desired_goal"] - o["achieved_goal"])
    return np.concatenate(v)
ACT_DIM = 4 * H


class Envs:
    """A list of Fetch envs stepped in lockstep in this process."""

    def __init__(self, n, task="FetchPickAndPlace-v4"):
        self.envs = [gym.make(task, max_episode_steps=10_000) for _ in range(n)]
        self.n = n

    def reset(self, seeds, heights=None):
        """heights: optional goal height above the object's resting height, per env (None = env default
        goal distribution: on the table or, half of the time, up to 0.45 m in the air)."""
        obs = [e.reset(seed=int(s))[0] for e, s in zip(self.envs, seeds)]
        if heights is not None:
            for i, (e, h) in enumerate(zip(self.envs, heights)):
                u = e.unwrapped
                u.goal = u.goal.copy(); u.goal[2] = obs[i]["achieved_goal"][2] + float(h)
                obs[i] = u._get_obs()
        return np.stack([obs_vec(o) for o in obs]).astype(np.float32)

    def get_state(self, i):
        import mujoco
        u = self.envs[i].unwrapped
        st = np.empty(mujoco.mj_stateSize(u.model, mujoco.mjtState.mjSTATE_INTEGRATION))
        mujoco.mj_getState(u.model, u.data, st, mujoco.mjtState.mjSTATE_INTEGRATION)
        return st, u.goal.copy()

    def set_state(self, i, state):
        import mujoco
        u = self.envs[i].unwrapped
        mujoco.mj_setState(u.model, u.data, state[0], mujoco.mjtState.mjSTATE_INTEGRATION)
        u.goal = state[1].copy()
        mujoco.mj_forward(u.model, u.data)
        o = u._get_obs()
        return obs_vec(o).astype(np.float32)

    def step_chunk(self, chunk, n_steps):
        """chunk (n, H*4) -> executes the first n_steps low-level actions. Returns obs, success flags."""
        a = chunk.reshape(self.n, H, 4)
        out, succ = [], []
        for i, e in enumerate(self.envs):
            for h in range(n_steps):
                o, _, _, _, info = e.step(np.clip(a[i, h], -1, 1))
            out.append(obs_vec(o)); succ.append(info["is_success"])
        return np.stack(out).astype(np.float32), np.array(succ, float)


def scripted(o, phase, rng, noise):
    """Noisy pick-and-place controller. o: (n, 28). phase: (n,) int state, updated in place."""
    grip, obj, goal = o[:, 0:3], o[:, 3:6], o[:, 25:28]
    n = len(o); a = np.zeros((n, 4))
    above = obj + np.array([0, 0, 0.05])
    d_above = np.linalg.norm(grip - above, axis=1); d_obj = np.linalg.norm(grip - obj, axis=1)
    phase[(phase == 0) & (d_above < 0.015)] = 1
    phase[(phase == 1) & (d_obj < 0.01)] = 2
    tgt = np.where(phase[:, None] == 0, above, np.where(phase[:, None] == 1, obj, goal))
    a[:, :3] = 10.0 * (tgt - grip)
    a[:, 3] = np.where(phase >= 2, -1.0, 1.0)
    # after closing, wait a few steps before moving (phase 2 -> 3 counter kept in phase >= 2)
    closing = (phase >= 2) & (phase < 5)
    a[closing, :3] = 0.0
    phase[closing] += 1
    a[:, :3] += noise * rng.standard_normal((n, 3))
    return np.clip(a, -1, 1)


def collect_demos(n_eps, noise, seed, n_envs=50, table_only=False):
    """Scripted episodes -> (S, A chunks, success). S rows: [obs(28), t/NDEC].
    table_only: every demo goal lies on the table (height 0), so lifting is never demonstrated."""
    rng = np.random.default_rng(seed)
    E = Envs(n_envs)
    S, A, Ys = [], [], []
    for b in range(0, n_eps, n_envs):
        o = E.reset([seed * 100_000 + b + i for i in range(n_envs)], heights=np.zeros(n_envs) if table_only else None)
        phase = np.zeros(n_envs, int)
        # per-episode noise level: a mixture of good and sloppy operators
        lvl = noise * rng.exponential(1.0, n_envs)[:, None]
        ep_S, ep_A = [], []
        for t in range(NDEC):
            n_steps = min(H, STEPS - t * H)
            ch = np.zeros((n_envs, H, 4))
            cur = o
            for h in range(H):
                ch[:, h] = np.clip(scripted(cur, phase, rng, 0.0) + lvl * rng.standard_normal((n_envs, 4)), -1, 1)
                if h < n_steps:
                    # step each env one action so the controller is closed-loop inside the chunk
                    nxt = []
                    for i, e in enumerate(E.envs):
                        oo, _, _, _, info = e.step(ch[i, h])
                        nxt.append(obs_vec(oo))
                    cur = np.stack(nxt).astype(np.float32)
            ep_S.append(np.c_[o, np.full(n_envs, t / NDEC)]); ep_A.append(ch.reshape(n_envs, -1))
            o = cur
        y = np.array([float(e.unwrapped._is_success(e.unwrapped._get_obs()["achieved_goal"],
                                                     e.unwrapped.goal)) for e in E.envs])
        S.append(np.stack(ep_S, 1)); A.append(np.stack(ep_A, 1)); Ys.append(y)
    return (np.concatenate(S).astype(np.float32), np.concatenate(A).astype(np.float32), np.concatenate(Ys))
