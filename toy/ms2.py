"""Round 1-2: configurable multi-step testbed.
- learner: 'pg' (warm softmax selector, factual PG), 'critic' (argmax over K of MC-regression critic, PA-RL-like),
           'fbc' (filtered BC: execute generator, distill episodes whose label>0.5), 'frozen' (pg selector, no distillation)
- label:   'true' (oracle Y), 'self' (robot's own detector V), 'anchor_only' (only anchored episodes are used),
           'naive_mix' (Y if anchored else V), 'dr' (V + r_hat + A/p (Y - V - r_hat)); anchor prob uniform or 'active'
- hack:    if True, the detector V also fires (w.p. 0.9) when the trap pocket F is reached (false positive).
True J (goal reached) is always what is reported. Episode budget counts every executed episode."""
import numpy as np, torch
from multistep import Diff, step, T, GOAL, GR, demos, CEN, phi, D
torch.set_num_threads(1)
FPOS, FR = np.array([-1.5, 3.2]), 0.5
def in_goal(s): return np.linalg.norm(s-GOAL, axis=-1) < GR
def in_F(s): return np.linalg.norm(s-FPOS, axis=-1) < FR

class Ridge:
    def __init__(s, d, lam=1.0): s.A = lam*np.eye(d); s.b = np.zeros(d); s.w = np.zeros(d)
    def add(s, F, Y, w=None):
        w = np.ones(len(Y)) if w is None else w
        s.A += (F*w[:, None]).T@F; s.b += (F*w[:, None]).T@Y
    def refit(s): s.w = np.linalg.solve(s.A, s.b)
    def __call__(s, F): return F@s.w

def softmax(z): z = z-z.max(-1, keepdims=True); e = np.exp(z); return e/e.sum(-1, keepdims=True)

def rollout(gen, n, K, r, choose, hack):
    s = np.zeros((n, 2))+0.1*r.standard_normal((n, 2)); done = np.zeros(n, bool)
    Y = np.zeros(n); V = np.zeros(n); fooled = r.random(n) < 0.9; sfin = s.copy(); log = []
    for t in range(T):
        S = np.c_[s, np.full(n, t/T)].astype(np.float32)
        C = gen.sample(np.repeat(S, K, 0), r.integers(1 << 30)).reshape(n, K, 2)
        P = choose(s, C); j = (P.cumsum(-1) < r.random((n, 1))).sum(-1).clip(0, K-1)
        a = C[np.arange(n), j]; log.append(dict(S=S, C=C, P=P, j=j, active=~done))
        s2 = step(s, a); s = np.where(done[:, None], s, s2)
        g = in_goal(s) & ~done; f = (in_F(s) & fooled & ~done) if hack else np.zeros(n, bool)
        Y[g] = 1; V[g | f] = 1; done |= g | f
    return log, Y, V, s

def run(learner, label='true', hack=False, eps=0.1, K=4, seed=0, rounds=6, n_sel=800, B=100,
        lr=30.0, lam=1.0, rho=0.1, p_anchor=0.05, active=False, rlam=1.0, rfeat='pos', tune_lam=False):
    r = np.random.default_rng(seed); torch.manual_seed(seed)
    Sd, Ad = demos(1000, eps, r); gen = Diff(); gen.fit(Sd, Ad, 3000, seed)
    theta = np.zeros(D); base = Ridge(D); critic = Ridge(D, lam); corr = Ridge(2*D, rlam); out = []
    n_anchor = 0; fV = 0.5; lamc = 1.0; sw_er = 0.0; sw_rr = 1e-6   # running P(V=1), predictable
    uniform = lambda s, C: np.full(C.shape[:2], 1/C.shape[1])
    soft = lambda s, C: softmax(phi(s[:, None]+C)@theta)
    def argq(s, C):
        z = critic(phi(s[:, None]+C))+1e-9*r.random(C.shape[:2]); return (z == z.max(-1, keepdims=True)).astype(float)
    behave = {'pg': soft, 'frozen': soft, 'critic': argq, 'fbc': uniform}[learner]
    Kb = 1 if learner == 'fbc' else K
    for k in range(rounds):
        Sx, Ax = [], []
        for _ in range(n_sel//B):
            log, Y, V, sf = rollout(gen, B, Kb, r, behave, hack)
            # ----- labels -----
            if active:   # verify claimed successes; expected budget = p_anchor using the *previous* estimate of P(V=1)
                plo = 0.1*p_anchor; phi_ = min(1.0, (p_anchor - plo*(1-fV))/max(fV, 1e-3))
                pa = np.where(V > 0, phi_, plo)
            else: pa = np.full(B, p_anchor)
            fV = 0.9*fV + 0.1*V.mean()
            A = r.random(B) < pa; n_anchor += A.sum()
            Ffin = np.c_[phi(sf)*V[:, None], phi(sf)*(1-V)[:, None]]   # residual model conditions on (final pos, V)
            if rfeat == 'V_only': Ffin = np.zeros((B, 2*D)); Ffin[:, 0] = V; Ffin[:, 1] = 1-V  # misspecified: cannot see where
            if label == 'true': lab, w = Y, np.ones(B)
            elif label == 'self': lab, w = V, np.ones(B)
            elif label == 'naive_mix': lab, w = np.where(A, Y, V), np.ones(B)
            elif label == 'anchor_only': lab, w = Y, A.astype(float)
            elif label == 'plugin':   # 'fine-tune the reward model on anchors' baseline: V + r_hat, no IPW residual term
                rh = corr(Ffin); lab, w = V + rh, np.ones(B)
                corr.add(Ffin[A], (Y - V)[A]); corr.refit()
            elif label == 'dr':
                rh = lamc*corr(Ffin)   # lamc: predictable power-tuning coefficient (PPI++ / spec's alpha)
                lab, w = V + rh + A/pa*(Y - V - rh), np.ones(B)
                if tune_lam and A.any():   # minimise IPW estimate of E[(1-p)/p (Y-V-lam r)^2] on past anchors
                    rr = corr(Ffin)[A]; e0 = (Y - V)[A]; ww = (1-pa[A])/pa[A]**2
                    sw_er += (ww*e0*rr).sum(); sw_rr += (ww*rr*rr).sum(); lamc = float(np.clip(sw_er/sw_rr, 0, 1.5))
                corr.add(Ffin[A], (Y - V)[A], (1/pa[A]) if active else None); corr.refit()  # predictable: used next batch
            # ----- learner update -----
            if learner in ('pg', 'frozen'):
                g = np.zeros(D)
                for L in log:
                    m = L['active'] & (w > 0); F = phi(L['S'][:, None, :2]+L['C']); p = L['P']; j = L['j']
                    sc = F[np.arange(B), j]-(p[..., None]*F).sum(1)
                    bl = np.clip(base(phi(L['S'][:, :2])), 0, 1)
                    g += ((sc*(lab-bl)[:, None])[m]).sum(0)
                theta += lr*g/B
                for L in log:
                    m = L['active'] & (w > 0); base.add(phi(L['S'][m, :2]), lab[m])
                base.refit()
            elif learner == 'critic':
                for L in log:
                    m = L['active'] & (w > 0); a = L['C'][np.arange(B), L['j']]
                    critic.add(phi(L['S'][m, :2]+a[m]), lab[m])
                critic.refit()
            for L in log:   # distillation data
                m = L['active'] & ((lab > 0.5) if learner == 'fbc' else np.ones(B, bool))
                Sx.append(L['S'][m]); Ax.append(L['C'][np.arange(B), L['j']][m])
        _, Yd, Vd, _ = rollout(gen, 600, Kb, r, behave, hack)
        if learner != 'frozen':
            S = np.concatenate(Sx); Aa = np.concatenate(Ax)
            nm = max(int(rho*len(S)), 200); i = r.integers(0, len(Sd), nm)
            if len(S) >= 50: gen.fit(np.concatenate([S, Sd[i]]), np.concatenate([Aa, Ad[i]]), 1500, seed+k+1)
        if learner == 'fbc': _, Yd, Vd, _ = rollout(gen, 600, 1, r, uniform, hack)
        out.append(dict(episodes=(k+1)*n_sel, J=float(Yd.mean()), Jself=float(Vd.mean()), anchors=int(n_anchor)))
    return out

