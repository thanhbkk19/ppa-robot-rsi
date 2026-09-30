import json, itertools, time
from multiprocessing import Pool
from ms2 import run
CFG=[(f'{lab}_{rf}','critic',dict(label=lab,rfeat=rf,hack=True,lam=0.1,p_anchor=0.02)) for lab,rf in [('plugin','pos'),('plugin','V_only'),('dr','V_only')]]
def job(a):
    (tag,m,kw),seed=a; o=run(m,seed=seed,**kw); return dict(tag=tag,kw=kw,seed=seed,curve=o)
if __name__=="__main__":
    with Pool(2) as p: res=p.map(job,list(itertools.product(CFG,range(3,8))))
    json.dump(res,open('results/r2c.json','w')); print('done')
