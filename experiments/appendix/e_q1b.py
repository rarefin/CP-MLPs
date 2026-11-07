"""
Q1b — random MIXTURES of several degrees.

Target: an independent dense random form at each degree k in K, so the target
is a genuine multi-degree polynomial with no structure the theory picked.

Model: one direction set shared across all degrees, one free output coefficient
per atom per degree (this is the "free-gate" multi-degree block; the genuine
fixed-gate SwiGLU version is Q2):

    tied    sum_i sum_k lam_{i,k} <u_i,x>^k
    untied  sum_i sum_k c_{i,k}  <u_i,x>^{k-1} <v_i,x>

Pre-registered thresholds by parameter count.  A tied atom carries n direction
parameters minus one scaling redundancy (rescaling u_i is absorbable degree by
degree into lam_{i,k}) plus |K| coefficients; an untied atom carries two
directions, two redundancies, plus |K| coefficients:

    r_tied   = ceil( sum_k N(n,k) / (n - 1 + |K|) )
    r_untied = ceil( sum_k N(n,k) / (2n - 2 + |K|) )

At |K| = 1 these reduce to ceil(N/n) and ceil(N/(2n-1)), the single-degree
formulas, so this is a strict generalisation.
"""
import json, os, time
import numpy as np

from cpcore import (Ndim, fit_multi, pred_multi_tied, pred_multi_untied,
                    random_dense)

TOL = 1e-10
OUT = "q1b.json"

CASES = [
    (3, (2, 3), 2),
    (2, (2, 3, 4, 5), 2),
    (4, (2, 3), 2),
    (3, (2, 3, 4), 2),
    (3, (3, 4), 1),
]


def run():
    res = json.load(open(OUT)) if os.path.exists(OUT) else {}
    for (n, K, ntgt) in CASES:
        K = list(K)
        pt, pu = pred_multi_tied(n, K), pred_multi_untied(n, K)
        for ti in range(ntgt):
            key = f"{n}_{'-'.join(map(str,K))}_{ti}"
            if res.get(key, {}).get("done"):
                continue
            tgt = {k: random_dense(n, k, 900 + 31 * ti + 7 * n + k) for k in K}
            rec = res.get(key, dict(n=n, degrees=K,
                                    Ntot=sum(Ndim(n, k) for k in K),
                                    target=ti, pred_tied=pt, pred_untied=pu,
                                    sweep={}))
            for tied in (True, False):
                fam = "tied" if tied else "untied"
                pred = pt if tied else pu
                rec["sweep"].setdefault(fam, {})
                for w in range(1, pt + 2):
                    if str(w) in rec["sweep"][fam]:
                        if rec["sweep"][fam][str(w)]["min"] < TOL:
                            break
                        continue
                    ns = 28 if abs(w - pred) <= 1 else 10
                    t0 = time.time()
                    r = fit_multi(tgt, n, K, w, tied, seeds=ns,
                                  rng=np.random.default_rng(19 * w + 5 * ti + 2))
                    rec["sweep"][fam][str(w)] = dict(
                        min=r["min"], median=r["median"], seeds=ns,
                        nsucc=int(np.sum(np.array(r["losses"]) < TOL)),
                        secs=round(time.time() - t0, 1))
                    print(f"n={n} K={K} t{ti} {fam:6s} w={w:2d} "
                          f"min={r['min']:.2e} med={r['median']:.2e} "
                          f"succ={rec['sweep'][fam][str(w)]['nsucc']}/{ns} "
                          f"({time.time()-t0:.0f}s)", flush=True)
                    res[key] = rec; json.dump(res, open(OUT, "w"), indent=1)
                    if r["min"] < TOL and w >= pred:
                        break
            for fam in ("tied", "untied"):
                ws = sorted(int(x) for x in rec["sweep"][fam])
                rec[f"emp_{fam}"] = next(
                    (w for w in ws if rec["sweep"][fam][str(w)]["min"] < TOL), None)
            rec["done"] = True
            res[key] = rec; json.dump(res, open(OUT, "w"), indent=1)
            print(f"  -> n={n} K={K} t{ti}: tied {rec['emp_tied']} (pred {pt}) | "
                  f"untied {rec['emp_untied']} (pred {pu})", flush=True)


if __name__ == "__main__":
    run()
    print("DONE")
