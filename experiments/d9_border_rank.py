"""Appendix D.9 - Border-rank diagnostics (Tables 4 and 5, closed-class control,
penalty-sweep slopes).

Rebuilt from the description in Appendix D.9; the original rebuttal-phase script
was not available. Unspecified choices are marked SPEC-GAP below.

Usage:  python experiments/d9_border_rank.py [--out results/d9]
"""
import argparse
import sys
import json
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cpmlp.symtensor import (Model, best_of, condition_number, fit,
                             monomial_tensor)

PENALTIES = [1e-4, 1e-6, 1e-8]     # SPEC-GAP: the paper says "three penalties" only
N_UNREG = 12
N_PER_PENALTY = 5
NEAR_BEST = 1.02


def size_diagnostic(family, d, m, T, rng):
    """27 candidates: 12 unregularised + 5 at each of three penalties.

    Penalised runs are polished without the penalty (warm start) so that every
    candidate is a fit of the unpenalised objective; their sizes come from the
    penalised basin.
    """
    M = Model(family, d, m, (T.ndim,))
    cands = []
    for _ in range(N_UNREG):
        L, th = fit(M, [T], rng)
        cands.append((L, th))
    for lam in PENALTIES:
        for _ in range(N_PER_PENALTY):
            _, th = fit(M, [T], rng, lam=lam)
            L, th2 = fit(M, [T], rng, theta0=th, maxiter=2000)
            # keep whichever of (penalised, polished) has lower data loss
            Lp = M.data_loss(th, [T])
            cands.append((L, th2) if L <= Lp else (Lp, th))
    losses = np.array([c[0] for c in cands])
    best = losses.min()
    i_best = int(losses.argmin())
    sizes = np.array([M.sizes(c[1]).max() for c in cands])
    near = np.where(losses <= NEAR_BEST * max(best, 0.0) + 1e-30)[0]
    i_near = near[np.argmin(sizes[near])]
    kappa, smin = condition_number(M, cands[i_near][1])
    return {
        "family": family, "m": m,
        "best_loss": float(best),
        "best_Amax": float(sizes[i_best]),
        "near_best_Amax": float(sizes[i_near]),
        "near_best_loss": float(losses[i_near]),
        "kappa": None if kappa is None else float(kappa),
        "sigma_min": smin,
    }


def closed_class_control(rng):
    """Tied fits to Q = diag(1,1,-1,-1): expect (4-m)/4."""
    Q = np.diag([1.0, 1.0, -1.0, -1.0])
    out = {}
    for m in (1, 2, 3):
        L, _, _ = best_of(Model("tied", 4, m, (2,)), [Q], rng, 12)
        out[m] = {"loss": L, "predicted": (4 - m) / 4}
    return out


def penalty_slopes(rng, ks=(3, 4, 5)):
    """Tied width two on x1^{k-1} x2 with a decreasing size penalty.

    Along the degeneration path the squared error should scale like A_max^{-4}.
    We record (A_max, loss) for a sweep of penalties and fit a log-log slope.
    """
    out = {}
    lams = np.logspace(-3, -9, 13)
    for k in ks:
        T = monomial_tensor(2, {(k - 1, 1): 1.0})
        M = Model("tied", 2, 2, (k,))
        # find a good starting basin
        _, th, _ = best_of(M, [T], rng, 8, lam=lams[0])
        pts = []
        for lam in lams:
            _, th = fit(M, [T], rng, theta0=th, lam=lam)
            pts.append((float(M.sizes(th).max()), M.data_loss(th, [T])))
        A = np.array([p[0] for p in pts])
        L = np.array([p[1] for p in pts])
        ok = (L > 1e-20) & (L < 1e-2)
        slope = float(np.polyfit(np.log(A[ok]), np.log(L[ok]), 1)[0]) if ok.sum() >= 3 else None
        out[k] = {"slope": slope, "points": pts}
    return out


def main(out_dir, seed):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    F3 = monomial_tensor(2, {(2, 1): 1.0})                       # x1^2 x2
    F32 = monomial_tensor(4, {(2, 1, 0, 0): 1.0, (0, 0, 2, 1): 1.0})  # x1^2x2 + x3^2x4

    table4 = [size_diagnostic("untied", 2, 1, F3, rng)] + \
             [size_diagnostic("tied", 2, m, F3, rng) for m in (1, 2, 3)]
    table5 = [size_diagnostic("untied", 4, 2, F32, rng)] + \
             [size_diagnostic("tied", 4, m, F32, rng) for m in (2, 3, 4, 5, 6)]
    ctrl = closed_class_control(rng)
    slopes = penalty_slopes(rng)

    res = {"table4_F3": table4, "table5_F32": table5,
           "closed_class_Q": ctrl, "penalty_slopes": slopes}
    (out / "d9_border_rank.json").write_text(json.dumps(res, indent=2))

    def show(name, rows):
        print(f"\n{name}")
        print(f"{'family':7s} m  {'best loss':>10s} {'best Amax':>10s} {'near Amax':>10s} {'kappa':>10s}")
        for r in rows:
            kap = "n.i." if r["kappa"] is None else f"{r['kappa']:.2g}"
            print(f"{r['family']:7s} {r['m']}  {r['best_loss']:10.2e} {r['best_Amax']:10.3g} "
                  f"{r['near_best_Amax']:10.3g} {kap:>10s}")
    show("Table 4: F3 = x1^2 x2", table4)
    show("Table 5: F32 = x1^2 x2 + x3^2 x4", table5)
    print("\nClosed-class control Q=diag(1,1,-1,-1):",
          {m: round(v["loss"], 4) for m, v in ctrl.items()})
    print("Penalty-sweep slopes:", {k: v["slope"] for k, v in slopes.items()})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/d9")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    main(a.out, a.seed)
