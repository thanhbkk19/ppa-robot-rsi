import json, itertools, time, sys
from multiprocessing import Pool
from ms2 import run
BEST={'pg':dict(lr=100.),'critic':dict(lam=0.1),'fbc':dict(rho=0.1),'frozen':dict(lr=100.)}
CFG=[]
for m in ['pg','critic','fbc','frozen']: CFG.append(('R1',m,dict(label='true',hack=False,**BEST[m])))
for m in ['pg','critic']:
    for lab,act in [('true',False),('self',False),('naive_mix',False),('anchor_only',False),('dr',False),('dr',True)]:
        CFG.append(('R2',m,dict(label=lab,active=act,hack=True,**BEST[m])))
CFG.append(('R2','pg',dict(label='dr',active=True,hack=True,lr=30.)))
for lab in ['self','naive_mix']: CFG.append(('R2','fbc',dict(label=lab,hack=True,**BEST['fbc'])))
def job(a):
    (tag,m,kw),seed=a; t=time.time(); o=run(m,seed=seed,**kw); return dict(tag=tag,method=m,kw=kw,seed=seed,curve=o,sec=time.time()-t)
if __name__=="__main__":
    with Pool(2) as p: res=p.map(job,list(itertools.product(CFG,range(3,8))))
    json.dump(res,open('results/r2.json','w')); print('done')
