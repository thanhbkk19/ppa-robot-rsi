"""PA-RL-style amplification loop (sample K -> critic-argmax -> distill) with pluggable episode labels.

One run = one (task, verifier, method, anchor budget, critic target, seed). Results are written after
every round to <out>/<run_id>.json and the run resumes from <raw>/<run_id>/ckpt.pt.

Label methods (docs/PPA_SPEC.md §3/§4; scale/configs/grid.yaml):
  oracle        Y everywhere (upper bound)
  self          V everywhere
  anchor_only   critic trains only on anchored episodes, label Y
  naive_mix     Y if anchored else V
  plugin        plug-in reward model: Y if anchored else clip(V + r_hat(x), 0, 1), with r_hat refit on ALL
                anchors so far (including the current batch) and the whole replay relabelled every round.
                No unbiasedness constraint -> strongest practical variant.
  dr            PPA labels from ppa.labels.label_batch (predictable r_hat, never clipped)
Ground truth Y is read only through `oracle(idx)` inside the labelling step; evaluation success is
computed on separate evaluation episodes (disjoint seeds) that no learner sees.
"""
from __future__ import annotations
import os
import json
import time
import hashlib
import numpy as np
import torch

from ppa.labels import AnchorSampler, RidgeResidual, label_batch
from ppa.robo.env import Collector, ROOT, task_cfg
from ppa.robo.generator import load_generator, sample, distill, load_demos
from ppa.robo.critic import Critic
from ppa.robo.verifiers import VERIFIERS, residual_design

DEFAULTS = dict(
    task="square", verifier="topdown", method="oracle", anchoring="uniform", residual="none", budget=0.02,
    critic_target="mc", seed=0,
    K=4, rounds=6, n_train=480, n_eval=192, n_envs=48,
    gamma=0.99, critic_lr=3e-4, critic_steps=3000, critic_batch=1024, n_ens=2,
    distill_steps=1500, distill_lr=3e-5, distill_batch=512, rho=0.1,
    select=True, distill=True, filter_labels=None,   # filtered BC: select=False, K=1, filter_labels in {self,true}
    ridge_lam=1.0,
)

METHODS = {
    "oracle": dict(method="oracle"),
    "self_only": dict(method="self"),
    "anchors_only": dict(method="anchor_only"),
    "naive_mix": dict(method="naive_mix"),
    "plugin_rm": dict(method="plugin", residual="learned"),
    "ppa_dr_uniform": dict(method="dr", anchoring="uniform", residual="none"),
    "ppa_dr_uniform_rhat": dict(method="dr", anchoring="uniform", residual="learned"),
    "ppa_dr_claimed": dict(method="dr", anchoring="claimed", residual="none"),
    "ppa_dr_claimed_rhat": dict(method="dr", anchoring="claimed", residual="learned"),
    "filtered_bc_self": dict(method="self", select=False, K=1, filter_labels="self"),
    "filtered_bc_true": dict(method="oracle", select=False, K=1, filter_labels="true"),
    "frozen_selector": dict(method="oracle", distill=False),
}
ANCHORED = {"anchor_only", "naive_mix", "plugin", "dr"}


def make_cfg(**kw):
    cfg = dict(DEFAULTS)
    if "name" in kw:
        cfg.update(METHODS[kw["name"]])
    cfg.update(kw)
    if cfg["method"] not in ANCHORED:
        cfg["budget"] = 1.0 if cfg["method"] == "oracle" else 0.0
        cfg["anchoring"], cfg["residual"] = "none", "none"
    return cfg


def cfg_hash(cfg):
    keys = sorted(k for k in cfg if k not in ("seed", "name"))
    return hashlib.sha1(json.dumps({k: cfg[k] for k in keys}, sort_keys=True).encode()).hexdigest()[:10]


def run_id(cfg):
    return f"{cfg.get('name', cfg['method'])}_{cfg['verifier']}_b{cfg['budget']}_{cfg['critic_target']}_{cfg_hash(cfg)}_s{cfg['seed']}"


def train_seeds(seed, rnd, n):
    return [100_000_000 + seed * 1_000_000 + rnd * 10_000 + i for i in range(n)]


def eval_seeds(seed, n):
    return [900_000_000 + seed * 10_000 + i for i in range(n)]


class Run:
    def __init__(self, cfg, out_dir, raw_dir, collectors=None):
        self.cfg = cfg
        self.id = run_id(cfg)
        self.out = os.path.join(out_dir, f"{self.id}.json")
        self.ck = os.path.join(raw_dir, self.id, "ckpt.pt")
        os.makedirs(out_dir, exist_ok=True); os.makedirs(os.path.dirname(self.ck), exist_ok=True)
        self.collectors = collectors

    # ---------------------------------------------------------------- state
    def _init(self):
        c, s = self.cfg, self.cfg["seed"]
        np.random.seed(s); torch.manual_seed(s)
        tc = task_cfg(c["task"])
        self.T = tc["n_decisions"]
        self.gen = load_generator(c["task"])
        self.critic = Critic(tc["obs_dim"], tc["horizon"] * tc["act_dim"], self.T, gamma=c["gamma"], lr=c["critic_lr"],
                             n_ens=c["n_ens"], target=c["critic_target"], seed=s)
        self.V = VERIFIERS[c["verifier"]](c["task"])
        self.sampler = AnchorSampler(c["budget"], mode=c["anchoring"] if c["anchoring"] != "none" else "uniform",
                                     seed=10_000 + s) if c["method"] in ANCHORED else None
        self.resid = None
        self.rng = np.random.default_rng(20_000 + s)
        self.rep = dict(obs=[], act=[], label=[], v=[], feats=[], anch=[], y_anch=[])  # per-episode replay
        self.hist, self.round, self.wall, self.n_queried = [], 0, 0.0, 0
        self.demo_obs, self.demo_act, _ = load_demos(c["task"], tc["horizon"])

    def _save(self):
        st = dict(gen=self.gen.state_dict(), q=self.critic.q.state_dict(), q_targ=self.critic.q_targ.state_dict(),
                  opt=self.critic.opt.state_dict(), n_upd=self.critic.n_updates, cg=self.critic.g.get_state(),
                  sampler=None if self.sampler is None else dict(f=self.sampler.f_claimed, n=self.sampler.n_queried,
                                                                    rng=self.sampler.rng.bit_generator.state),
                  resid=None if self.resid is None else (self.resid.A, self.resid.b, self.resid.w),
                  rng=self.rng.bit_generator.state, rep=self.rep, hist=self.hist, round=self.round, wall=self.wall,
                  n_queried=self.n_queried, torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state())
        torch.save(st, self.ck + ".tmp"); os.replace(self.ck + ".tmp", self.ck)

    def _load(self):
        st = torch.load(self.ck, weights_only=False)
        self.gen.load_state_dict(st["gen"]); self.critic.q.load_state_dict(st["q"])
        self.critic.q_targ.load_state_dict(st["q_targ"]); self.critic.opt.load_state_dict(st["opt"])
        self.critic.n_updates = st["n_upd"]; self.critic.g.set_state(st["cg"])
        if st["sampler"] is not None:
            self.sampler.f_claimed, self.sampler.n_queried = st["sampler"]["f"], st["sampler"]["n"]
            self.sampler.rng.bit_generator.state = st["sampler"]["rng"]
        if st["resid"] is not None:
            self.resid = RidgeResidual(st["resid"][0].shape[0]); self.resid.A, self.resid.b, self.resid.w = st["resid"]
        self.rng.bit_generator.state = st["rng"]
        self.rep, self.hist, self.round, self.wall = st["rep"], st["hist"], st["round"], st["wall"]
        self.n_queried = st["n_queried"]
        torch.set_rng_state(st["torch_rng"]); torch.cuda.set_rng_state(st["cuda_rng"])

    def _write(self, status):
        with open(self.out + ".tmp", "w") as fh:
            json.dump(dict(run_id=self.id, cfg=self.cfg, status=status, wall_clock_s=self.wall, history=self.hist), fh,
                      indent=1)
        os.replace(self.out + ".tmp", self.out)

    # ---------------------------------------------------------------- acting
    def _act_fn(self, select):
        c = self.cfg
        has_q = select and self.critic.n_updates > 0

        def act(o, t):
            cands = sample(self.gen, o, c["K"])
            if has_q:
                q = self.critic.score(torch.as_tensor(o, dtype=torch.float32, device=cands.device), cands, t)
                idx = q.argmax(1)
            else:
                idx = torch.zeros(len(o), dtype=torch.long, device=cands.device)
            return cands[torch.arange(len(o), device=cands.device), idx].cpu().numpy()
        return act

    def _collect(self, kind, seeds):
        col = self.collectors[kind] if self.collectors else None
        if col is None:
            col = Collector(self.cfg["task"], self.cfg["n_envs"])
            self.collectors = self.collectors or {}
            self.collectors[kind] = col
        return col.run(self._act_fn(self.cfg["select"]), seeds)

    # ---------------------------------------------------------------- labelling
    def _label(self, ep):
        """Labels for a freshly collected round, batch by batch (r_hat predictable between batches).
        Returns labels (np) and the anchored mask. Y is read only via oracle(idx)."""
        c = self.cfg
        y_true = ep["y"]  # held by the environment; read only through oracle() below
        final = ep["obs"][:, -1]
        v = self.V(final)
        feats = self.V.features(final)
        n = len(v)
        m = c["method"]
        if m == "oracle":
            self.n_queried += n
            return y_true.astype(float).copy(), np.ones(n, bool), v, feats
        if m == "self":
            return v.copy(), np.zeros(n, bool), v, feats
        labels = np.zeros(n); anch = np.zeros(n, bool)
        bs = c["n_envs"]
        for b0 in range(0, n, bs):
            sl = slice(b0, min(n, b0 + bs))
            idx0 = np.arange(sl.start, sl.stop)

            def oracle(idx, idx0=idx0):
                self.n_queried += len(idx)
                return y_true[idx0[idx]]

            if m == "dr":
                phi = residual_design(v[sl], feats[sl])
                if c["residual"] == "learned" and self.resid is None:
                    self.resid = RidgeResidual(phi.shape[1], lam=c["ridge_lam"])
                lab, info = label_batch(v[sl], oracle, phi, self.sampler,
                                        self.resid if c["residual"] == "learned" else None)
                labels[sl], anch[sl] = lab, info["anchored"]
            else:
                a, p = self.sampler.sample(v[sl])
                y = np.full(len(a), np.nan)
                if a.any():
                    y[a] = oracle(np.flatnonzero(a))
                self.sampler.update(v[sl])
                anch[sl] = a
                if m in ("anchor_only", "naive_mix"):
                    labels[sl] = np.where(a, np.nan_to_num(y), v[sl])
                else:  # plugin: store anchors; labels are (re)computed for the whole replay below
                    labels[sl] = np.where(a, np.nan_to_num(y), np.nan)
        return labels, anch, v, feats

    def _plugin_relabel(self):
        v = np.concatenate(self.rep["v"]); f = np.concatenate(self.rep["feats"])
        anch = np.concatenate(self.rep["anch"]); ya = np.concatenate(self.rep["y_anch"])
        phi = residual_design(v, f)
        r = RidgeResidual(phi.shape[1], lam=self.cfg["ridge_lam"])
        if anch.any():
            r.add(phi[anch], ya[anch] - v[anch]); r.refit()
        lab = np.where(anch, ya, np.clip(v + r.predict(phi), 0.0, 1.0))
        out, i = [], 0
        for arr in self.rep["v"]:
            out.append(lab[i:i + len(arr)]); i += len(arr)
        self.rep["label"] = out

    # ---------------------------------------------------------------- main
    def evaluate(self):
        ev = self._collect("eval", eval_seeds(self.cfg["seed"], self.cfg["n_eval"]))
        v = self.V(ev["obs"][:, -1])
        return dict(J=float(ev["y"].mean()), J_any=float(ev["y_any"].mean()), J_self=float(v.mean()),
                    gap=float(v.mean() - ev["y"].mean()))

    def run(self):
        c = self.cfg
        self._init()
        if os.path.exists(self.ck):
            self._load()
        if self.round == 0 and not self.hist:
            t0 = time.time()
            e = self.evaluate()
            self.wall += time.time() - t0
            self.hist.append(dict(round=0, episodes=0, anchors=0, **e))
            self._write("running"); self._save()
        while self.round < c["rounds"]:
            t0 = time.time()
            r = self.round + 1
            ep = self._collect("train", train_seeds(c["seed"], r, c["n_train"]))
            labels, anch, v, feats = self._label(ep)
            # replay (episode-level). Anchored ground truth kept only where it was queried.
            y_anch = np.where(anch, ep["y"], np.nan)
            keep = anch if c["method"] == "anchor_only" else np.ones(len(v), bool)
            for k, arr in (("obs", ep["obs"]), ("act", ep["act"]), ("label", labels), ("v", v), ("feats", feats),
                           ("anch", anch), ("y_anch", y_anch)):
                self.rep[k].append(arr[keep])
            if c["method"] == "plugin":
                self._plugin_relabel()
            L = np.concatenate(self.rep["label"])
            closs = float("nan")
            if c["select"] and len(L):
                closs = self.critic.fit(np.concatenate(self.rep["obs"]), np.concatenate(self.rep["act"]), L,
                                        c["critic_steps"], c["critic_batch"])
            dloss = float("nan")
            if c["distill"]:
                # filtered BC keeps only episodes whose label (V for "self", Y for "true") says success
                sel = np.ones(len(v), bool) if c["filter_labels"] is None else labels >= 0.5
                if sel.any():
                    o = ep["obs"][sel, :-1].reshape(-1, ep["obs"].shape[-1])
                    a = ep["act"][sel].reshape(-1, *ep["act"].shape[2:])
                    dloss = distill(self.gen, o, a, self.demo_obs, self.demo_act, c["rho"], c["distill_steps"],
                                    c["distill_lr"], c["distill_batch"], seed=c["seed"] * 1000 + r)
            e = self.evaluate()
            self.wall += time.time() - t0
            self.round = r
            self.hist.append(dict(round=r, episodes=r * c["n_train"], anchors=int(self.n_queried),
                                  train_J=float(ep["y"].mean()), train_V=float(v.mean()),
                                  label_mean=float(np.mean(labels[~np.isnan(labels)])) if len(labels) else float("nan"),
                                  label_min=float(np.nanmin(labels)), label_max=float(np.nanmax(labels)),
                                  critic_loss=closs, distill_loss=dloss, wall_s=self.wall, **e))
            self._write("running"); self._save()
        self._write("done")
        return self.hist

    def cleanup(self):
        if os.path.exists(self.ck):
            os.remove(self.ck)
