import json, itertools, time
from multiprocessing import Pool
from ms2 import run
CFG=[]
for p in [0.01,0.02]:
    for lab,act in [('dr',True),('anchor_only',False),('dr',False)]:
        CFG.append((f'budget{p}','critic',dict(label=lab,active=act,hack=True,lam=0.1,p_anchor=p)))
for act in [False,True]:   # residual model effectively disabled (huge ridge) -> r_hat ~ 0
    CFG.append(('no_rhat','critic',dict(label='dr',active=act,hack=True,lam=0.1,rlam=1e9)))
def job(a):
    (tag,m,kw),seed=a; t=time.time(); o=run(m,seed=seed,**kw); return dict(tag=tag,method=m,kw=kw,seed=seed,curve=o,sec=time.time()-t)
if __name__=="__main__":
    with Pool(2) as p: res=p.map(job,list(itertools.product(CFG,range(3,8))))
    json.dump(res,open('results/r2b.json','w')); print('done')
