"""Appendix D.10, last paragraph - the F_{4,2} = x1^3 x2 + x3^3 x4 example.

F_{4,2} has exact untied rank two. The paper reports that 8 restarts all return
loss 0.25, and that a 40-restart run recovers the representation 10 times
(best 5.4e-20, median 0.25).

Rebuilt for the public release (the original snippet was not recovered).
Table 6 itself is produced by experiments/appendix/e_q3.py.

Usage:  python experiments/d10_f42_example.py [--out results/d10]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cpmlp.symtensor import Model, fit, monomial_tensor  # noqa: E402

TOL = 1e-10


def main(out_dir, seed):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    F42 = monomial_tensor(4, {(3, 1, 0, 0): 1.0, (0, 0, 3, 1): 1.0})
    M = Model("untied", 4, 2, (4,))
    l8 = [fit(M, [F42], rng, maxiter=8000)[0] for _ in range(8)]
    l40 = [fit(M, [F42], rng, maxiter=8000)[0] for _ in range(40)]
    res = {"restarts8": l8, "restarts40": l40,
           "recovered_of_40": int(np.sum(np.array(l40) < TOL)),
           "best40": float(min(l40)), "median40": float(np.median(l40))}
    print(f"8 restarts: {sorted(round(x, 4) for x in l8)}")
    print(f"40 restarts: {res['recovered_of_40']} recoveries, best {res['best40']:.1e}, "
          f"median {res['median40']:.2f}")
    (out / "d10_f42.json").write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/d10")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    main(a.out, a.seed)
