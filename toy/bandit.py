"""Exp A/B: contextual-bandit test of the amplification loop (generator + learned selector + distillation).
Ground truth is known, so m (good mass), q (selector recognition), delta (distillation loss) and J are exact MC estimates."""
import numpy as np, warnings
from sklearn.mixture import GaussianMixture
warnings.filterwarnings("ignore")

P_IN, P_OUT, RAD = 0.95, 0.02, 0.3
GRID = np.stack(np.meshgrid(np.linspace(-3,3,12), np.linspace(-3,3,12)), -1).reshape(-1,2)
def feat(a):  # RBF features, a: (...,2) -> (...,144)
    d2 = ((a[...,None,:]-GRID)**2).sum(-1); return np.exp(-d2/(2*0.45**2))

class Task:
    def __init__(s, C, eps, seed, n_demo=1000):
        r = np.random.default_rng(seed); s.C = C; s.good = []; s.demos = []
        for c in range(C):
            dist = r.uniform(-2.3,2.3,(4,2))
            while True:
                g = r.uniform(-2.3,2.3,2)
                if np.min(np.linalg.norm(dist-g,axis=1))>1.2: break
            s.good.append(g)
            ng = r.binomial(n_demo, eps)
            A = np.concatenate([dist[r.integers(0,4,n_demo-ng)]+0.25*r.standard_normal((n_demo-ng,2)),
                                g+0.1*r.standard_normal((ng,2))])
            s.demos.append(A)
        s.good = np.array(s.good)
    def is_good(s, c, a): return np.linalg.norm(a-s.good[c],axis=-1) < RAD
    def prob(s, c, a): return np.where(s.is_good(c,a), P_IN, P_OUT)

class Gen:  # per-context GMM generator; pools for fast sampling
    def __init__(s, data, M, seed, pool=40000):
        r = np.random.default_rng(seed); s.pools=[]
        for c,X in enumerate(data):
            g = GaussianMixture(M, covariance_type='full', reg_covar=1e-3, random_state=seed+c).fit(X)
            P,_ = g.sample(pool); s.pools.append(P[r.permutation(pool)])  # shuffle: sklearn returns grouped by component
    def cand(s, c, n, K, r): return s.pools[c][r.integers(0,len(s.pools[c]),(n,K))]

def logits(theta, c, A): return feat(A) @ theta[c]           # A (n,K,2) -> (n,K)
def softmax(z): z = z-z.max(-1,keepdims=True); e=np.exp(z); return e/e.sum(-1,keepdims=True)

def measure(task, gen, theta, K, r, n=4000, selector='soft'):
    out = dict(m=[],q=[],pi=[],J=[])
    for c in range(task.C):
        A = gen.cand(c,n,K,r); G = task.is_good(c,A).astype(float)
        if selector=='soft': P = softmax(logits(theta,c,A))
        elif selector=='uniform': P = np.full(G.shape,1/K)
        else: z=selector(c,A); P=(z==z.max(-1,keepdims=True)).astype(float); P/=P.sum(-1,keepdims=True)
        pig = (P*G).sum(-1); anyg = G.max(-1)>0
        out['m'].append(G.mean()); out['pi'].append(pig.mean())
        out['q'].append(pig[anyg].mean() if anyg.any() else np.nan)
        out['J'].append(P_IN*pig.mean()+P_OUT*(1-pig.mean()))
    return {k:np.nanmean(v) for k,v in out.items()}

def train_selector(task, gen, theta, K, n_ep, r, lr=3.0, B=50):
    """Factual on-policy PG, predictable per-context baseline, plain SGD (spec A3-A6)."""
    b = np.full(task.C, 0.05); used=0
    while used < n_ep:
        cs = r.integers(0,task.C,B); g = np.zeros_like(theta); Ys=np.zeros(B)
        for i,c in enumerate(cs):
            A = gen.cand(c,1,K,r)[0]; F = feat(A); p = softmax(F@theta[c])
            j = r.choice(K,p=p); Y = r.random() < task.prob(c,A[j])
            g[c] += (F[j]-p@F)*(Y-b[c]); Ys[i]=Y
        theta += lr*g/B
        for c in np.unique(cs): b[c] = 0.9*b[c]+0.1*Ys[cs==c].mean()   # baseline updated after batch -> predictable
        used += B
    return theta

def distill(task, gen, choose, K, n_d, rho, M, seed, r):
    data=[]
    for c in range(task.C):
        A = gen.cand(c,n_d,K,r); j = choose(c,A)
        X = A[np.arange(n_d),j]
        nmix = int(rho*n_d)
        if nmix: X = np.concatenate([X, task.demos[c][r.integers(0,len(task.demos[c]),nmix)]])
        data.append(X)
    return Gen(data, M, seed)

def soft_choose(theta, r):
    def f(c,A):
        P = softmax(logits(theta,c,A)); u=r.random((len(A),1)); return (P.cumsum(-1)<u).sum(-1).clip(0,A.shape[1]-1)
    return f

def run_loop(eps, K, seed, rounds=8, n_sel=2000, n_d=2000, rho=0.05, M=6, C=6, reset=True, lr=3.0, verbose=False):
    r = np.random.default_rng(seed); task = Task(C,eps,seed)
    gen = Gen(task.demos, M, seed); theta = np.zeros((C,GRID.shape[0])); hist=[]
    for k in range(rounds):
        if reset: theta = np.zeros_like(theta)
        theta = train_selector(task, gen, theta, K, n_sel, r, lr)
        st = measure(task, gen, theta, K, r); st['episodes']=(k+1)*n_sel
        new = distill(task, gen, soft_choose(theta,r), K, n_d, rho, M, seed+100*(k+1), r)
        st['m_next'] = measure(task,new,theta,K,r,selector='uniform')['m']
        st['delta'] = st['pi']-st['m_next']; hist.append(st); gen = new
        if verbose: print(k, {a:round(v,4) for a,v in st.items()})
    return hist
if __name__=="__main__":
    run_loop(0.02, 4, 0, verbose=True)
