"""Diffusion generator mu (DPPO's pretrained DiffusionMLP) with K-candidate sampling and distillation."""
from __future__ import annotations
import os
import copy
import numpy as np
import torch

from ppa.robo.env import DATA, task_cfg  # noqa: F401  (sets sys.path for DPPO)
from model.diffusion.diffusion import DiffusionModel
from model.diffusion.mlp_diffusion import DiffusionMLP


def load_generator(task, device="cuda"):
    c = task_cfg(task)
    net = DiffusionMLP(action_dim=c["act_dim"], horizon_steps=c["horizon"], cond_dim=c["obs_dim"], time_dim=32,
                       mlp_dims=[1024, 1024, 1024], cond_mlp_dims=[512, 64], residual_style=True)
    model = DiffusionModel(network=net, horizon_steps=c["horizon"], obs_dim=c["obs_dim"], action_dim=c["act_dim"],
                           network_path=os.path.join(DATA, c["ckpt"]), device=device, denoised_clip_value=1.0,
                           randn_clip_value=3, final_action_clip_value=1.0, denoising_steps=20,
                           predict_epsilon=True)
    model.eval()
    return model


@torch.no_grad()
def sample(model, obs, k):
    """obs [B, obs_dim] (np or torch) -> candidates [B, k, horizon, act_dim] (torch, on device)."""
    dev = model.betas.device
    o = torch.as_tensor(obs, dtype=torch.float32, device=dev)
    B = o.shape[0]
    cond = {"state": o.repeat_interleave(k, 0)[:, None]}
    # DDPM ancestral sampling, as DiffusionModel.forward (whose upstream version passes an unsupported kwarg)
    n = B * k
    x = torch.randn((n, model.horizon_steps, model.action_dim), device=dev)
    for t in reversed(range(model.denoising_steps)):
        t_b = torch.full((n,), t, device=dev, dtype=torch.long)
        mean, logvar = model.p_mean_var(x=x, t=t_b, cond=cond)
        std = torch.zeros_like(mean) if t == 0 else torch.exp(0.5 * logvar).clip(min=1e-3)
        x = mean + std * torch.randn_like(x).clamp_(-model.randn_clip_value, model.randn_clip_value)
    x = x.clamp(-model.final_action_clip_value, model.final_action_clip_value)
    return x.view(B, k, *x.shape[1:])


def distill(model, obs, act, demo_obs, demo_act, rho, steps, lr, batch=512, seed=0, wd=0.0):
    """Fine-tune the generator on executed (unfiltered) chunks plus a rho fraction of demos.
    obs [N, obs_dim], act [N, horizon, act_dim]. Returns mean loss of the last 10% of steps."""
    dev = model.betas.device
    g = torch.Generator(device="cpu").manual_seed(seed)
    O = torch.as_tensor(obs, dtype=torch.float32); A = torch.as_tensor(act, dtype=torch.float32)
    DO = torch.as_tensor(demo_obs, dtype=torch.float32); DA = torch.as_tensor(demo_act, dtype=torch.float32)
    n_demo = int(round(rho * batch)) if len(DO) else 0
    n_on = batch - n_demo
    opt = torch.optim.AdamW(model.network.parameters(), lr=lr, weight_decay=wd)
    model.train()
    losses = []
    for s in range(steps):
        i = torch.randint(0, len(O), (n_on,), generator=g)
        xo, xa = O[i], A[i]
        if n_demo:
            j = torch.randint(0, len(DO), (n_demo,), generator=g)
            xo, xa = torch.cat([xo, DO[j]]), torch.cat([xa, DA[j]])
        loss = model.loss(xa.to(dev), {"state": xo.to(dev)[:, None]})
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        losses.append(loss.item())
    model.eval()
    return float(np.mean(losses[-max(1, steps // 10):]))


def load_demos(task, horizon):
    """Demo (state, action-chunk) pairs from DPPO's preprocessed (normalised) train.npz."""
    d = np.load(os.path.join(DATA, "robomimic", task, task, "train.npz"))
    S, A, L = d["states"], d["actions"], d["traj_lengths"]
    obs, act, finals = [], [], []
    start = 0
    for l in L:
        s, a = S[start:start + l], A[start:start + l]
        for t in range(0, l - horizon + 1):
            obs.append(s[t]); act.append(a[t:t + horizon])
        finals.append(s[l - 1])
        start += l
    return np.asarray(obs, np.float32), np.asarray(act, np.float32), np.asarray(finals, np.float32)
