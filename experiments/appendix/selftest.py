#!/usr/bin/env python3
"""
selftest.py — validate the harness before trusting any experimental result.

Run:  python3 selftest.py
Exits nonzero on any failure.
"""
import sys

import numpy as np
import torch

from cpcore import Basis, untied_vec, fit_single, generic_tied, generic_untied, Ndim
from gated import channel_components

torch.set_default_dtype(torch.float64)
FAILED = []


def check(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}   {detail}")
    if not ok:
        FAILED.append(name)


def evaluate(m, B, X):
    """Evaluate a monomial coefficient vector at rows of X."""
    out = np.zeros(len(X))
    for c, a in zip(m, B.alphas):
        out += c * np.prod(X ** np.array(a), axis=1)
    return out


# 1. apolar norm normalises powers of linear forms
n, k = 3, 4
B = Basis(n, k)
u = torch.tensor(np.random.default_rng(1).normal(size=n))
lhs = float(torch.sqrt(B.norm2(B.pow_vec(u))))
rhs = float(u.norm()) ** k
check("apolar norm: ||<u,x>^k|| == ||u||^k", abs(lhs - rhs) < 1e-12,
      f"{lhs:.12f} vs {rhs:.12f}")

# 2. coefficient vectors match direct evaluation
X = np.random.default_rng(2).normal(size=(6, n))
e1 = np.abs(evaluate(B.pow_vec(u).numpy(), B, X) - (X @ u.numpy()) ** k).max()
check("tied atom coefficients match direct evaluation", e1 < 1e-14, f"max err {e1:.2e}")

v = torch.tensor(np.random.default_rng(3).normal(size=n))
e2 = np.abs(evaluate(untied_vec(u, v, n, k).numpy(), B, X)
            - ((X @ u.numpy()) ** (k - 1)) * (X @ v.numpy())).max()
check("untied atom coefficients match direct evaluation", e2 < 1e-13, f"max err {e2:.2e}")

# 3. an exactly-rank-2 target is recovered
# Note: this is reported as best-of-N, not median, for exactly the reason
# documented in Q3 -- individual restarts stall well above the numerical floor
# even on a target that is exactly representable at this width.
tgt = (2.0 * B.pow_vec(u) - 1.5 * B.pow_vec(v)).numpy()
r = fit_single(tgt, n, k, 2, tied=True, seeds=16)
ls = np.array(r["losses"])
check("exact rank-2 target recovered at width 2 (best of 16)", r["min"] < 1e-16,
      f"min {r['min']:.2e}, median {np.median(ls):.2e}, "
      f"{int((ls < 1e-10).sum())}/16 restarts succeeded")

# 4. gated components reproduce a genuinely evaluated SwiGLU block
n2 = 4
rng = np.random.default_rng(0)
U = torch.tensor(rng.normal(size=(2, n2)) * 0.3)
V = torch.tensor(rng.normal(size=(2, n2)) * 0.3)
a = torch.tensor(rng.normal(size=2))
degs = [2, 3, 4, 5, 6, 7]
comp = channel_components(U, V, a, n2, degs, tied=False)
Xs = rng.normal(size=(6, n2)) * 0.15
approx = sum(evaluate(comp[d].numpy(), Basis(n2, d), Xs) for d in degs)
g = Xs @ U.numpy().T
exact = ((g / (1 + np.exp(-g))) * (Xs @ V.numpy().T)) @ a.numpy()
e3 = np.abs(approx - exact).max()
check("gated components match a real SwiGLU block", e3 < 1e-13, f"max err {e3:.2e}")

# 5. generic-rank formulas reduce correctly and encode the known defects
ok = (generic_tied(3, 2) == 3 and                      # quadrics: rank = n
      generic_tied(3, 4) == 6 and                      # AH exceptional (naive 5)
      generic_untied(3, 3) == 3 and                    # tangential defect (naive 2)
      generic_untied(4, 3) == 4 and                    # tangential defect (naive 3)
      generic_tied(5, 3) == 8 and generic_untied(5, 3) == 5)
check("generic-rank formulas encode AH / tangential defects", ok,
      f"(3,4) tied={generic_tied(3,4)} (naive {-(-Ndim(3,4)//3)}), "
      f"(3,3) untied={generic_untied(3,3)} (naive {-(-Ndim(3,3)//5)})")

print()
if FAILED:
    print("FAILURES:", ", ".join(FAILED))
    sys.exit(1)
print("All self-tests passed.")
