import numpy as np, json, itertools
from multiprocessing import Pool
from bandit import run_loop
def job(a):
    eps,K,nd,seed = a
    h = run_loop(eps,K,seed,rounds=8,n_d=nd)
    return dict(eps=eps,K=K,nd=nd,seed=seed,hist=[{k:float(v) for k,v in s.items()} for s in h])
if __name__=="__main__":
    grid = list(itertools.product([0.001,0.003,0.01,0.03],[2,4,8],[150,2000],[0,1,2]))
    with Pool(2) as p: res = p.map(job, grid)
    json.dump(res, open('results/phase.json','w'))
    print("done", len(res))
