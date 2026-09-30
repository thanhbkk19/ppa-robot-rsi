import json, itertools
from multiprocessing import Pool
from ms2 import run
CFG=[(f'dr_{rf}_tuned',dict(label='dr',rfeat=rf,tune_lam=True,hack=True,lam=0.1,p_anchor=0.02)) for rf in ['V_only','pos']]+[('dr_norhat_p02',dict(label='dr',rlam=1e9,hack=True,lam=0.1,p_anchor=0.02))]
def job(a):
    (tag,kw),seed=a; return dict(tag=tag,seed=seed,curve=run('critic',seed=seed,**kw))
if __name__=="__main__":
    with Pool(2) as p: res=p.map(job,list(itertools.product(CFG,range(3,8))))
    json.dump(res,open('results/r2d.json','w')); print('done')
