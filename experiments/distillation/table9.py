import json,numpy as np,sys
R=json.load(open(sys.argv[1] if len(sys.argv)>1 else 'results_cr.json'))
def med(t,a,w): 
    v=[R[f"{t}|{a}|{w}|{s}"]['test'] for s in range(3) if f"{t}|{a}|{w}|{s}" in R]
    return (np.median(v),len(v)) if v else (None,0)
for t in ('untied','tied','neutral'):
    for m in [2,3,4,6,8,10,12]:
        mu,n1=med(t,'untied',m); mt,n2=med(t,'tied',m); m15=int(np.floor(1.5*m+.5)); mm,n3=med(t,'tied',m15)
        if mu is None: continue
        f=lambda x: '   -    ' if x is None else f"{x:.2e}"
        r=lambda a,b: '  -  ' if a is None or b is None else f"{a/b:.2f}"
        print(f"{t:7s} m={m:2d} untied {f(mu)} tied {f(mt)} ratio {r(mt,mu)} | tied{m15:2d} {f(mm)} ratio {r(mm,mu)}  (n={n1},{n2},{n3})")
