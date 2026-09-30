from multiprocessing import Pool
from multistep import run
def j(a): m,lr,s=a; o=run(m,0.1,4,s,lr=lr); return a,[round(c['J'],2) for c in o],[round(c['J_gen'],2) for c in o]
if __name__=="__main__":
    with Pool(2) as p:
        for r in p.map(j,[('frozen_selector',30.0,0),('amplify',30.0,0),('amplify',30.0,1),('amplify_warm',30.0,2)]): print(r,flush=True)
