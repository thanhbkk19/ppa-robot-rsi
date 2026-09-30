import numpy as np, baselines
for lr in [3.0,10.0,30.0]:
    baselines.train_selector.__defaults__=(lr,50,False,None,None)
    C=np.array([baselines.run('amplify_warm',0.01,4,s) for s in range(3)])
    print('lr',lr,'J after 4k,8k,16k:',C.mean(0)[[1,3,7]].round(3), flush=True)
