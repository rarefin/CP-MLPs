"""
Q4 — synthetic-dataset training of REAL networks.

Two synthetic teachers, both genuine gated blocks on Gaussian data:
  A  role-ASYMMETRIC   y = sum_j silu(<p_j,x>) <q_j,x>,  p_j, q_j independent
  B  role-SYMMETRIC    y = sum_j silu(<p_j,x>) <p_j,x>   (a SiLU^2 teacher)

Two students, trained by Adam on minibatches from scratch:
  SwiGLU (untied)  a_i silu(<u_i,x>) <v_i,x>
  SiLU^2 (tied)    a_i silu(<u_i,x>) <u_i,x>

Students are compared at MATCHED PARAMETER COUNT, since an untied channel costs
2n+1 parameters against n+1 for a tied one.  The theory predicts untied wins on
teacher A and that the advantage vanishes or reverses on teacher B; the
reversal control is what makes this a test rather than a demonstration.
"""
import json, os, time
import numpy as np
import torch

torch.set_num_threads(1)
OUT = "q4.json"
N = 8          # input dim
S_TEACH = 4    # teacher channels
P_TRAIN, P_TEST = 20000, 5000
STEPS, BATCH, LR = 4000, 256, 3e-3
SEEDS = 4


def make_data(kind, seed):
    g = torch.Generator().manual_seed(seed)
    P = torch.randn(S_TEACH, N, generator=g)
    Q = P.clone() if kind == "sym" else torch.randn(S_TEACH, N, generator=g)
    a = torch.randn(S_TEACH, generator=g)

    def f(X):
        z = X @ P.T
        return ((z * torch.sigmoid(z)) * (X @ Q.T)) @ a
    Xtr = torch.randn(P_TRAIN, N, generator=g)
    Xte = torch.randn(P_TEST, N, generator=g)
    ytr, yte = f(Xtr), f(Xte)
    return Xtr, ytr, Xte, yte


def train(Xtr, ytr, Xte, yte, w, tied, seed):
    g = torch.Generator().manual_seed(seed)
    U = (torch.randn(w, N, generator=g) / N ** .5).requires_grad_()
    V = None if tied else (torch.randn(w, N, generator=g) / N ** .5).requires_grad_()
    a = (torch.randn(w, generator=g) / w ** .5).requires_grad_()
    ps = [U, a] + ([] if tied else [V])
    opt = torch.optim.Adam(ps, lr=LR)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, STEPS)
    yv = (ytr ** 2).mean()

    def fwd(X):
        z = X @ U.T
        val = z if tied else X @ V.T
        return ((z * torch.sigmoid(z)) * val) @ a
    for t in range(STEPS):
        idx = torch.randint(0, len(Xtr), (BATCH,), generator=g)
        loss = ((fwd(Xtr[idx]) - ytr[idx]) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step(); sch.step()
    with torch.no_grad():
        te = float(((fwd(Xte) - yte) ** 2).mean() / (yte ** 2).mean())
    return te


def nparams(w, tied):
    return w * ((N + 1) if tied else (2 * N + 1))


def run():
    res = json.load(open(OUT)) if os.path.exists(OUT) else {}
    for kind in ("asym", "sym"):
        Xtr, ytr, Xte, yte = make_data(kind, 7)
        for tied in (True, False):
            fam = "tied" if tied else "untied"
            widths = [2, 4, 6, 8, 12, 16] if tied else [1, 2, 3, 4, 6, 8]
            for w in widths:
                key = f"{kind}_{fam}_w{w}"
                if key in res:
                    continue
                t0 = time.time()
                errs = [train(Xtr, ytr, Xte, yte, w, tied, 1000 + 13 * s)
                        for s in range(SEEDS)]
                res[key] = dict(kind=kind, fam=fam, w=w, params=nparams(w, tied),
                                errs=errs, median=float(np.median(errs)),
                                min=float(np.min(errs)))
                print(f"{kind:5s} {fam:6s} w={w:2d} params={nparams(w,tied):4d} "
                      f"testMSE med={np.median(errs):.4f} min={np.min(errs):.4f} "
                      f"({time.time()-t0:.0f}s)", flush=True)
                json.dump(res, open(OUT, "w"), indent=1)
    print("DONE")


if __name__ == "__main__":
    run()
