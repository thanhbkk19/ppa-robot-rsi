import json, itertools, time
from multiprocessing import Pool
from ms2 import run
CFG=[('pg',dict(lr=v)) for v in (10.,30.,100.)]+[('critic',dict(lam=v)) for v in (0.1,1.,10.)]+[('fbc',dict(rho=v)) for v in (0.03,0.1,0.3)]+[('frozen',dict(lr=30.))]
def job(a):
    (m,kw),seed=a; t=time.time(); o=run(m,seed=seed,**kw); return dict(method=m,kw=kw,seed=seed,curve=o,sec=time.time()-t)
if __name__=="__main__":
    with Pool(2) as p: res=p.map(job,list(itertools.product(CFG,range(3))))
    json.dump(res,open('results/r1_tune.json','w')); print('done')
