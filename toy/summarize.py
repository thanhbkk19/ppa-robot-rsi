"""Print the toy tables reported in docs/PPA_SPEC.md §1 and docs/RESULTS_TOY.md from results/*.json.
Run from toy/:  python summarize.py"""
import json, numpy as np, os
def load(f): return json.load(open(os.path.join('results', f)))
def row(name, xs, key='J'):
    C = np.array([[c[key] for c in x['curve']] for x in xs])
    S = np.array([[c.get('Jself', np.nan) for c in x['curve']] for x in xs])
    print(f"  {name:44s} final {C[:,-1].mean():.2f}±{C[:,-1].std()/np.sqrt(len(C)):.2f}  AUC {C.mean():.3f}  Jself {np.nanmean(S[:,-1]):.2f}  n={len(C)}")

print("== Phase map check (bandit) ==")
P, M = [], []
for x in load('phase.json'):
    for h in x['hist']:
        if h['m'] > 0.002 and not np.isnan(h['q']): P.append(h['q']*(1-(1-h['m'])**x['K'])); M.append(h['pi'])
P, M = np.array(P), np.array(M)
print(f"  pi_pred vs pi_meas: corr {np.corrcoef(P,M)[0,1]:.3f}, median rel err {np.median(np.abs(P-M)/M):.3f}")

print("== R1/R2 held-out seeds 3-7 (multi-step, diffusion generator) ==")
R = load('r2.json')
for x in R: x['name'] = f"{x['tag']} {x['method']} {x['kw'].get('label')} active={x['kw'].get('active',False)} hp={x['kw'].get('lr', x['kw'].get('lam', x['kw'].get('rho')))}"
for n in dict.fromkeys(x['name'] for x in R): row(n, [x for x in R if x['name'] == n])
for f in ['r2b.json', 'r2c.json', 'r2d.json']:
    print(f"== {f} ==")
    R = load(f)
    for x in R: x['name'] = f"{x['tag']} {x.get('kw',{}).get('label','')} active={x.get('kw',{}).get('active',False)}"
    for n in dict.fromkeys(x['name'] for x in R): row(n, [x for x in R if x['name'] == n])
