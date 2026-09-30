"""Exp C: multi-step, sparse terminal success, diffusion-policy generator, K-candidate selector, distillation loop."""
import numpy as np, torch, torch.nn as nn, json, itertools, sys, time
torch.set_num_threads(1)
T, MAXSTEP = 6, 1.3
GOAL, GR = np.array([0.,3.2]), 0.5
def in_wall(p):
    x,y = p[...,0], p[...,1]
    horiz = (y>1.4)&(y<1.6)&~((x>-2.6)&(x<-1.4))&~((x>1.7)&(x<2.3))
    vert = (x>-1.1)&(x<-0.9)&(y>1.6)&(y<4.5)
    box = (np.abs(x)>4.5)|(y<-1.5)|(y>5)
    return horiz|vert|box
def step(s,a):
    n=np.linalg.norm(a,axis=-1,keepdims=True); a=a*np.minimum(1,MAXSTEP/np.maximum(n,1e-9))
    p=s.copy(); alive=np.ones(len(s),bool)
    for i in range(1,11):
        q=s+a*i/10; hit=in_wall(q); alive&=~hit; p[alive]=q[alive]
    return p
def success(s): return np.linalg.norm(s-GOAL,axis=-1)<GR
WP_B=[(1.0,0.5),(2.0,1.1),(2.0,2.1),(1.0,2.9),(0.0,3.2),(0.0,3.2)]
WP_A=[(-1.0,0.5),(-2.0,1.1),(-2.0,2.1),(-1.5,3.0),(-1.5,3.2),(-1.5,3.2)]
def demos(n, eps, r):
    S,A=[],[]
    for i in range(n):
        wp = WP_B if r.random()<eps else WP_A
        s=np.array([0.,0.])+0.1*r.standard_normal(2)
        for t in range(T):
            a=np.array(wp[t])-s+0.12*r.standard_normal(2); S.append(np.r_[s,t/T]); A.append(a); s=step(s[None],a[None])[0]
            if success(s): break
    return np.array(S,np.float32),np.array(A,np.float32)

class Diff(nn.Module):  # tiny conditional DDPM, 20 steps
    def __init__(s,N=20):
        super().__init__(); s.N=N; b=torch.linspace(1e-4,0.2,N); s.al=torch.cumprod(1-b,0); s.b=b
        s.net=nn.Sequential(nn.Linear(2+3+1,128),nn.SiLU(),nn.Linear(128,128),nn.SiLU(),nn.Linear(128,128),nn.SiLU(),nn.Linear(128,2))
    def fit(s,S,A,steps,seed):
        g=torch.Generator().manual_seed(seed); S=torch.tensor(S); A=torch.tensor(A)/2
        opt=torch.optim.Adam(s.parameters(),1e-3)
        for _ in range(steps):
            i=torch.randint(0,len(S),(256,),generator=g); k=torch.randint(0,s.N,(256,),generator=g)
            e=torch.randn(256,2,generator=g); ab=s.al[k][:,None]
            x=ab.sqrt()*A[i]+(1-ab).sqrt()*e
            loss=((s.net(torch.cat([x,S[i],k[:,None]/s.N],1))-e)**2).mean(); opt.zero_grad(); loss.backward(); opt.step()
    @torch.no_grad()
    def sample(s,S,seed):
        g=torch.Generator().manual_seed(int(seed)); S=torch.tensor(S,dtype=torch.float32); x=torch.randn(len(S),2,generator=g)
        for k in reversed(range(s.N)):
            eps=s.net(torch.cat([x,S,torch.full((len(S),1),k/s.N)],1)); ab=s.al[k]; b=s.b[k]
            x=(x-b/(1-ab).sqrt()*eps)/(1-b).sqrt()
            if k>0: x+=b.sqrt()*torch.randn(x.shape,generator=g)
        return (x*2).numpy()

GX,GY=np.meshgrid(np.linspace(-4,4,14),np.linspace(-1,4.5,10)); CEN=np.stack([GX.ravel(),GY.ravel()],1)
def phi(p): return np.exp(-((p[...,None,:]-CEN)**2).sum(-1)/(2*0.55**2))   # frozen feature of nominal next position
D=CEN.shape[0]
class Ridge:
    def __init__(s,lam=1.0): s.A=lam*np.eye(D); s.b=np.zeros(D); s.w=np.zeros(D)
    def add(s,F,Y): s.A+=F.T@F; s.b+=F.T@Y
    def refit(s): s.w=np.linalg.solve(s.A,s.b)
    def __call__(s,F): return np.clip(F@s.w,0,1)

def rollout(gen, n, K, r, choose):
    """Run n episodes in parallel. choose(t, s, cand(n,K,2)) -> probs (n,K). Returns traj data and Y."""
    s=np.zeros((n,2))+0.1*r.standard_normal((n,2)); done=np.zeros(n,bool); Y=np.zeros(n)
    log=[]
    for t in range(T):
        S=np.c_[s,np.full(n,t/T)].astype(np.float32)
        C=gen.sample(np.repeat(S,K,0),r.integers(1<<30)).reshape(n,K,2)
        P=choose(t,s,C); j=(P.cumsum(-1)<r.random((n,1))).sum(-1).clip(0,K-1)
        a=C[np.arange(n),j]; log.append(dict(S=S,C=C,P=P,j=j,active=~done))
        s2=step(s,a); s=np.where(done[:,None],s,s2); newly=success(s)&~done; Y[newly]=1; done|=newly
    return log,Y

def run(method, eps, K, seed, rounds=6, n_sel=800, B=100, rho=0.1, lr=2.0):
    r=np.random.default_rng(seed); torch.manual_seed(seed)
    Sd,Ad=demos(1000,eps,r); gen=Diff(); gen.fit(Sd,Ad,3000,seed)
    theta=np.zeros(D); base=Ridge(); critic=Ridge(); out=[]
    uniform=lambda t,s,C: np.full(C.shape[:2],1/C.shape[1])
    soft=lambda th: (lambda t,s,C: (lambda z: np.exp(z-z.max(-1,keepdims=True))/np.exp(z-z.max(-1,keepdims=True)).sum(-1,keepdims=True))(phi(s[:,None]+C)@th))
    def argq(t,s,C):
        z=critic(phi(s[:,None]+C))+1e-9*r.random(C.shape[:2]); return (z==z.max(-1,keepdims=True)).astype(float)
    for k in range(rounds):
        Sx,Ax=[],[]
        if method in('amplify','amplify_warm','frozen_selector'):
            if method=='amplify': theta=np.zeros(D)
            for _ in range(n_sel//B):
                log,Y=rollout(gen,B,K,r,soft(theta)); g=np.zeros(D)
                for L in log:
                    m=L['active']; F=phi(L['S'][:,None,:2]+L['C']); p=L['P']; j=L['j']
                    sc=F[np.arange(B),j]-(p[...,None]*F).sum(1)
                    Fs=phi(L['S'][:,:2]); bl=base(Fs)
                    g+=((sc*(Y-bl)[:,None])[m]).sum(0)
                    Sx.append(L['S'][m]); Ax.append(L['C'][np.arange(B),j][m])
                theta+=lr*g/B
                for L in log: m=L['active']; base.add(phi(L['S'][m,:2]),Y[m])
                base.refit()
            dep=soft(theta)
        elif method=='critic_distill':
            for _ in range(n_sel//B):
                log,Y=rollout(gen,B,K,r,argq)
                for L in log:
                    m=L['active']; a=L['C'][np.arange(B),L['j']]
                    critic.add(phi(L['S'][m,:2]+a[m]),Y[m]); Sx.append(L['S'][m]); Ax.append(a[m])
                critic.refit()
            dep=argq
        elif method=='filtered_bc':
            for _ in range(n_sel//B):
                log,Y=rollout(gen,B,1,r,uniform)
                for L in log:
                    m=L['active']&(Y>0); Sx.append(L['S'][m]); Ax.append(L['C'][m,0])
            dep=uniform
        # deployable policy after this round's data
        _,Yd=rollout(gen,600,K if method!='filtered_bc' else 1,r,dep); Jsel=Yd.mean()
        if method!='frozen_selector':
            S=np.concatenate(Sx) if Sx else np.zeros((0,3),np.float32); A=np.concatenate(Ax) if Ax else np.zeros((0,2),np.float32)
            nm=max(int(rho*len(S)),200); i=r.integers(0,len(Sd),nm)
            if len(S)>=50: gen.fit(np.concatenate([S,Sd[i]]),np.concatenate([A,Ad[i]]),1500,seed+k+1)
        _,Yg=rollout(gen,600,1,r,uniform)    # generator alone (m-like quantity)
        if method=='filtered_bc': Jsel=Yg.mean()
        out.append(dict(episodes=(k+1)*n_sel,J=float(Jsel),J_gen=float(Yg.mean())))
    return out
def job(a): m,eps,K,seed=a; t=time.time(); o=run(m,eps,K,seed); return dict(method=m,eps=eps,K=K,seed=seed,curve=o,sec=time.time()-t)
if __name__=="__main__":
    if sys.argv[1:]==['smoke']:
        for m in ['amplify']:
            t=time.time(); print(m,run(m,0.1,4,0,rounds=2,n_sel=400), time.time()-t)
    else:
        grid=list(itertools.product(['amplify','frozen_selector','critic_distill','filtered_bc'],[0.1],[4],range(3)))
        from multiprocessing import Pool
        with Pool(2) as p: res=p.map(job,grid)
        json.dump(res,open('results/multistep.json','w')); print('done')
