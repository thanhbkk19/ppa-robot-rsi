import numpy as np, sys
from multiprocessing import Pool
from multistep import run
def j(a): m,lr,s=a; return a,[round(c['J'],2) for c in run(m,0.1,4,s,lr=lr)]
if __name__=="__main__":
    with Pool(2) as p:
        for a,c in p.map(j,[('amplify_warm',10.0,0),('amplify_warm',10.0,1),('amplify_warm',30.0,0),('amplify_warm',30.0,1)]): print(a,c,flush=True)
