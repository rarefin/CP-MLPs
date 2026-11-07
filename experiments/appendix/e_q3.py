"""
Q3 — separating REPRESENTABILITY from OPTIMIZATION.

The reviewer's concern: every figure reports trained MSE, but the theoretical
quantity is atom rank, and the two are identified by reading a loss floor as a
rank threshold.  That identification assumes the optimizer reliably finds a
representable solution.

Here representability is guaranteed BY CONSTRUCTION: the target is built as an
explicit sum of exactly r atoms with random directions, so a width-r exact
solution provably exists (we also verify the witness gives ~0 loss).  Any
failure to reach it is therefore pure optimizer failure, and we measure its
rate as a function of degree k and of overparameterisation w - r.
"""
import json, os, time
import numpy as np
import torch
from cpcore import Basis, untied_vec, fit_single

OUT = "q3.json"
TOL = 1e-10
N_VARS = 3
RANK = 3
DEGREES = [3, 4, 5, 6]
SEEDS = 60


def build_target(n, k, r, tied, seed):
    """Explicit rank-r target + its exact witness parameters."""
    rng = np.random.default_rng(seed)
    U = rng.normal(size=(r, n)); V = rng.normal(size=(r, n)); C = rng.normal(size=r)
    B = Basis(n, k)
    acc = torch.zeros(B.N, dtype=torch.float64)
    for i in range(r):
        u = torch.tensor(U[i]); v = u if tied else torch.tensor(V[i])
        acc = acc + C[i] * (B.pow_vec(u) if tied else untied_vec(u, v, n, k))
    nrm = float(torch.sqrt(B.norm2(acc)))
    return (acc / nrm).numpy(), (U, V, C / nrm)


def witness_loss(target, wit, n, k, tied):
    U, V, C = wit
    B = Basis(n, k); acc = torch.zeros(B.N, dtype=torch.float64)
    for i in range(len(C)):
        u = torch.tensor(U[i]); v = u if tied else torch.tensor(V[i])
        acc = acc + C[i] * (B.pow_vec(u) if tied else untied_vec(u, v, n, k))
    t = torch.tensor(target)
    return float(B.norm2(acc - t) / B.norm2(t))


def run():
    res = json.load(open(OUT)) if os.path.exists(OUT) else {}
    for k in DEGREES:
        for tied in (True, False):
            fam = "tied" if tied else "untied"
            tgt, wit = build_target(N_VARS, k, RANK, tied, seed=300 + k)
            wl = witness_loss(tgt, wit, N_VARS, k, tied)
            for extra in (0, 1, 2):
                key = f"{fam}_k{k}_w{RANK+extra}"
                if key in res:
                    continue
                t0 = time.time()
                r = fit_single(tgt, N_VARS, k, RANK + extra, tied, seeds=SEEDS,
                               rng=np.random.default_rng(1234 + 7 * k + extra))
                ls = np.array(r["losses"])
                res[key] = dict(k=k, fam=fam, w=RANK + extra, rank=RANK,
                                witness_loss=wl, min=r["min"], median=r["median"],
                                psucc=float(np.mean(ls < TOL)), seeds=SEEDS)
                print(f"{fam:6s} k={k} w={RANK+extra} (r={RANK}) "
                      f"witness={wl:.1e} min={r['min']:.1e} med={r['median']:.1e} "
                      f"P(succ)={res[key]['psucc']:.2f} ({time.time()-t0:.0f}s)",
                      flush=True)
                json.dump(res, open(OUT, "w"), indent=1)
    print("DONE")


if __name__ == "__main__":
    run()
