"""Norm continuation (Appendix D.12, last paragraph), on the original harness.

Tied SwiGLU (SiLU(z) z) width 12 is fitted to the untied width-8 teacher with a
norm penalty lam * ||theta||^2 that is relaxed to zero along the path. Three
paths. Reports held-out relative MSE and squared parameter norm.

Not recovered from the rebuttal sessions: written for the public release on top
of cpmlp.py. The penalty schedule is not given in the paper.
"""
import json
import sys

import numpy as np
from scipy.optimize import minimize

import cpmlp as M
import teachers as T

LAMS = [1e-3, 3e-4, 1e-4, 3e-5, 1e-5, 3e-6, 1e-6, 3e-7, 1e-7, 0.0]
W = 12


def penalised(theta, X, Y, w, tied, Yn2, lam):
    L, g = M.loss_grad(theta, X, Y, w, tied, Yn2)
    return L + lam * float(theta @ theta), g + 2.0 * lam * theta


def main(out):
    spec = json.load(open('teachers.json'))
    Xtr = T.gauss(4000, 1011); Xte = T.gauss(4000, 5011)
    Ytr = T.teacher_apply(spec['untied'], Xtr); Yte = T.teacher_apply(spec['untied'], Xte)
    Yn2 = float(np.sum(Ytr * Ytr))
    paths = []
    for p in range(3):
        th = M.init(W, True, 900 + p)
        n0 = float(th @ th)
        trace = []
        for lam in LAMS:
            r = minimize(penalised, th, args=(Xtr, Ytr, W, True, Yn2, lam), jac=True,
                         method='L-BFGS-B',
                         options=dict(maxiter=1500, maxfun=3000, ftol=1e-16, gtol=1e-12))
            th = r.x
            R = M.forward(th, Xte, W, True) - Yte
            trace.append(dict(lam=lam, test=float(np.sum(R * R) / np.sum(Yte * Yte)),
                              sq_norm=float(th @ th)))
        paths.append(dict(initial_sq_norm=n0, trace=trace))
        print(f"path {p}: test {trace[-1]['test']:.2e}  |theta|^2 {trace[-1]['sq_norm']:.0f} "
              f"(init {n0:.0f})", flush=True)
    json.dump(paths, open(out, 'w'), indent=1)


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'norm_continuation.json')
