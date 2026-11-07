"""Camera-ready settings of the distillation study (Appendix D.12, Table 9),
run on the original harness (cpmlp.py, teachers.py).

Changes from distill.py (the first-turn version): 3 seeds, L-BFGS maxiter 6000,
widths {2,3,4,6,8,10,12}, tied arms at m and round(1.5 m), as described in the
camera-ready appendix. Resumable.
"""
import json, os, sys
import numpy as np
import cpmlp as M
import teachers as T

SEEDS = [0, 1, 2]
MS = [2, 3, 4, 6, 8, 10, 12]
MAXITER = 6000
NTRAIN = NTEST = 4000
RES = sys.argv[1] if len(sys.argv) > 1 else 'results_cr.json'


def m15(m):
    return int(np.floor(1.5 * m + 0.5))


def jobs():
    out = []
    for tname in ('untied', 'tied', 'neutral'):
        for m in MS:
            for s in SEEDS:
                out.append((tname, 'untied', m, s))
                out.append((tname, 'tied', m, s))
                if m15(m) != m:
                    out.append((tname, 'tied', m15(m), s))
    return list(dict.fromkeys(out))


def main():
    spec = json.load(open('teachers.json'))
    res = json.load(open(RES)) if os.path.exists(RES) else {}
    dseed = dict(untied=11, tied=22, neutral=33)
    data = {}
    for t in ('untied', 'tied', 'neutral'):
        Xtr = T.gauss(NTRAIN, 1000 + dseed[t]); Xte = T.gauss(NTEST, 5000 + dseed[t])
        data[t] = (Xtr, T.teacher_apply(spec[t], Xtr), Xte, T.teacher_apply(spec[t], Xte))
    for (t, arch, w, s) in jobs():
        k = f"{t}|{arch}|{w}|{s}"
        if k in res:
            continue
        Xtr, Ytr, Xte, Yte = data[t]
        tr, te = M.fit(Xtr, Ytr, w, arch == 'tied', s, maxiter=MAXITER, Xte=Xte, Yte=Yte)
        res[k] = dict(train=tr, test=te)
        print(f"{k:24s} train={tr:.3e} test={te:.3e}", flush=True)
        json.dump(res, open(RES, 'w'))
    print('DONE')


if __name__ == '__main__':
    main()
