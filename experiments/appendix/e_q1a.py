"""
Q1a — random dense degree-k forms.

Targets are drawn iid Gaussian in an apolar-orthonormal basis of Sym^k(R^n):
nothing about them is hand-picked by the theory.  The predicted exact-fit
widths are fixed BEFORE fitting:

    tied    r_t = ceil(N/n)        (+1 in the four Alexander-Hirschowitz cases)
    untied  r_u = ceil(N/(2n-1))   (+1 in the three cubic tangential cases)

We report, per (n,k) and per target:
  * the full width sweep (min/median relative loss over restarts)
  * the smallest width reaching rel. loss < 1e-10  (the empirical threshold)
  * loss at matched *effective parameter count*, since an untied atom costs
    2n-1 free parameters against n for a tied atom.
"""
import json, os, sys, time
import numpy as np
from cpcore import (Basis, Ndim, generic_tied, generic_untied, fit_single,
                    random_dense)

TOL = 1e-10
OUT = "q1a.json"

CASES = [
    # (n, k)  chosen to include both defect families and a range of ratios
    (2, 5),
    (3, 3),   # untied tangential defect
    (3, 4),   # tied AH defect
    (4, 3),   # untied tangential defect
    (2, 6),
    (4, 4),   # tied AH defect
    (5, 3),   # both defects
]
NTARGETS = 2
SEEDS_SWEEP = 12
SEEDS_REFINE = 32


def eff_params(w, n, tied):
    return w * (n if tied else 2 * n - 1)


def run():
    res = json.load(open(OUT)) if os.path.exists(OUT) else {}
    for (n, k) in CASES:
        N = Ndim(n, k)
        pt, pu = generic_tied(n, k), generic_untied(n, k)
        wmax = pt + 1
        for ti in range(NTARGETS):
            key = f"{n}_{k}_{ti}"
            if key in res and res[key].get("done"):
                continue
            tgt = random_dense(n, k, 1000 + 17 * ti + n * 31 + k)
            rec = res.get(key, dict(n=n, k=k, N=N, target=ti,
                                    pred_tied=pt, pred_untied=pu, sweep={}))
            for tied in (True, False):
                fam = "tied" if tied else "untied"
                pred = pt if tied else pu
                rec["sweep"].setdefault(fam, {})
                for w in range(1, wmax + 1):
                    if str(w) in rec["sweep"][fam]:
                        continue
                    ns = SEEDS_REFINE if abs(w - pred) <= 1 else SEEDS_SWEEP
                    t0 = time.time()
                    r = fit_single(tgt, n, k, w, tied,
                                   seeds=ns,
                                   rng=np.random.default_rng(7 * w + 3 * ti + 11))
                    rec["sweep"][fam][str(w)] = dict(
                        min=r["min"], median=r["median"], seeds=ns,
                        nsucc=int(np.sum(np.array(r["losses"]) < TOL)),
                        params=eff_params(w, n, tied),
                        secs=round(time.time() - t0, 1))
                    print(f"n={n} k={k} t{ti} {fam:6s} w={w:2d} "
                          f"min={r['min']:.2e} med={r['median']:.2e} "
                          f"succ={rec['sweep'][fam][str(w)]['nsucc']}/{ns} "
                          f"({time.time()-t0:.0f}s)", flush=True)
                    res[key] = rec
                    json.dump(res, open(OUT, "w"), indent=1)
                    if r["min"] < TOL and w >= pred:
                        break
            # empirical thresholds
            for fam in ("tied", "untied"):
                ws = sorted(int(x) for x in rec["sweep"][fam])
                thr = next((w for w in ws
                            if rec["sweep"][fam][str(w)]["min"] < TOL), None)
                rec[f"emp_{fam}"] = thr
            rec["done"] = True
            res[key] = rec
            json.dump(res, open(OUT, "w"), indent=1)
            print(f"  -> n={n} k={k} t{ti}: tied {rec['emp_tied']} (pred {pt}) | "
                  f"untied {rec['emp_untied']} (pred {pu})", flush=True)


if __name__ == "__main__":
    run()
    print("DONE")
