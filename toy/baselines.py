"""Exp B: equal env-episode budget comparison on the contextual bandit."""
import numpy as np, json, itertools, sys
from multiprocessing import Pool
from bandit import Task, Gen, feat, logits, softmax, measure, distill, soft_choose, GRID, P_IN, P_OUT

D = GRID.shape[0]
class Critic:  # per-context ridge regression on RBF features, fitted only on past data (predictable)
    def __init__(s,C,lam=1.0): s.A=np.stack([lam*np.eye(D)]*C); s.b=np.zeros((C,D)); s.w=np.zeros((C,D))
    def add(s,c,F,Y): s.A[c]+=np.outer(F,F); s.b[c]+=F*Y
    def refit(s): s.w=np.linalg.solve(s.A,s.b[...,None])[...,0]
    def pred(s,c,A): return np.clip(feat(A)@s.w[c],0,1)

def train_selector(task, gen, theta, K, n_ep, r, lr=3.0, B=50, cv=False, critic=None, st=None):
    """Factual PG; optional predictable control variate G_alpha = G0 + alpha*C with OGD on alpha (spec Sec. 7-9)."""
    b = st.setdefault('b', np.full(task.C,0.05)); used=0
    while used < n_ep:
        cs = r.integers(0,task.C,B); g0=np.zeros_like(theta); Cv=np.zeros_like(theta); Ys=np.zeros(B)
        for i,c in enumerate(cs):
            A = gen.cand(c,1,K,r)[0]; F = feat(A); p = softmax(F@theta[c])
            j = r.choice(K,p=p); Y = float(r.random() < task.prob(c,A[j]))
            s_all = F - p@F                       # score for every candidate index
            g0[c] += s_all[j]*(Y-b[c]); Ys[i]=Y
            if cv:
                z = critic.pred(c,A) - b[c]
                Cv[c] += (p[:,None]*s_all*z[:,None]).sum(0) - s_all[j]*z[j]
                critic.add(c,F[j],Y)
            elif critic is not None: critic.add(c,F[j],Y)
        a = st.get('alpha',0.0)
        G = (g0 + a*Cv)/B
        theta += lr*G
        if cv:   # OGD on per-batch second moment, alpha used above was chosen before this batch
            gv, cvv = g0.ravel()/B, Cv.ravel()/B
            st['alpha'] = float(np.clip(a - st.get('beta',0.5)*2*cvv@(gv+a*cvv), -1.5, 1.5))
            critic.refit()
        elif critic is not None: critic.refit()
        for c in np.unique(cs): b[c] = 0.9*b[c]+0.1*Ys[cs==c].mean()
        used += B
    return theta

def run(method, eps, K, seed, rounds=8, n_sel=2000, n_d=2000, rho=0.05, M=6, C=6):
    r = np.random.default_rng(seed); task = Task(C,eps,seed); gen = Gen(task.demos,M,seed)
    theta = np.zeros((C,D)); st={}; curve=[]; critic = Critic(C)
    for k in range(rounds):
        if method=='frozen_selector':            # spec's backbone: generator frozen, selector keeps learning
            theta = train_selector(task,gen,theta,K,n_sel,r,st=st)
            J = measure(task,gen,theta,K,r)['J']
        elif method in ('amplify','amplify_cv','amplify_warm'):
            if method!='amplify_warm': theta=np.zeros_like(theta); st.pop('b',None)
            theta = train_selector(task,gen,theta,K,n_sel,r,cv=(method=='amplify_cv'),critic=critic if method=='amplify_cv' else None,st=st)
            J = measure(task,gen,theta,K,r)['J']
            gen = distill(task,gen,soft_choose(theta,r),K,n_d,rho,M,seed+100*(k+1),r)
        elif method=='critic_distill':           # PA-RL-like: act = argmax critic over K samples, distill argmax
            for _ in range(n_sel):
                c = r.integers(task.C); A = gen.cand(c,1,K,r)[0]
                j = int(np.argmax(critic.pred(c,A)+1e-9*r.random(K)))
                critic.add(c,feat(A[j]),float(r.random()<task.prob(c,A[j])))
            critic.refit()
            argsel = lambda c,A: np.argmax(critic.pred(c,A)+1e-9*r.random(A.shape[:2]),-1)
            J = measure(task,gen,None,K,r,selector=lambda c,A: critic.pred(c,A)+1e-9*r.random(A.shape[:2]))['J']
            gen = distill(task,gen,argsel,K,n_d,rho,M,seed+100*(k+1),r)
        elif method=='filtered_bc':              # execute generator, keep successes, refit (+rho demos)
            data=[]
            for c in range(task.C):
                A = gen.cand(c,n_sel//task.C,1,r)[:,0]; Y = r.random(len(A)) < task.prob(c,A)
                X = A[Y]; X = np.concatenate([X, task.demos[c][r.integers(0,len(task.demos[c]),max(int(rho*len(X)),20))]]) if len(X)>=M else task.demos[c]
                data.append(X)
            gen = Gen(data,M,seed+100*(k+1))
            J = measure(task,gen,None,1,r,selector='uniform')['J']   # deployable policy after consuming this round's data
        curve.append(float(J))
    return curve

def job(a): m,eps,K,seed=a; return dict(method=m,eps=eps,K=K,seed=seed,curve=run(m,eps,K,seed))
if __name__=="__main__":
    methods=['frozen_selector','amplify','amplify_warm','amplify_cv','critic_distill','filtered_bc']
    grid=list(itertools.product(methods,[0.003,0.01,0.03],[4],range(5)))
    with Pool(2) as p: res=p.map(job,grid)
    json.dump(res,open('results/baselines.json','w')); print('done',len(res))
