"""Coefficient-space width sweep on F_k = x1^{k-1} x2, k = 2..5.

Paper: Section 5.2, Figure 3. Prediction: untied width 1, tied width k.

Usage:  python experiments/fig3_width_sweep.py [results directory, default: results]
"""

import json
import sys
import numpy as np
from pathlib import Path
from scipy.optimize import minimize
from math import comb
import matplotlib.pyplot as plt


# ===== coefficient-space utilities ===========================================

def monomial_coeffs(k):
    """F_k = x_1^{k-1} x_2 has coefficient 1 in basis position [1]."""
    c = np.zeros(k + 1)
    c[1] = 1.0
    return c


def linear_form_power_coeffs(a, b, k):
    return np.array([comb(k, i) * a**(k - i) * b**i for i in range(k + 1)])


def linear_form_mixed_coeffs(a, b, e, f, k):
    coeffs = np.zeros(k + 1)
    for i in range(k):
        base = comb(k - 1, i) * a**(k - 1 - i) * b**i
        coeffs[i] += e * base
        coeffs[i + 1] += f * base
    return coeffs


def tied_total_coeffs(params, m, k):
    A = params[:m]
    B = params[m:2 * m]
    C = params[2 * m:3 * m]
    total = np.zeros(k + 1)
    for a in range(m):
        total += C[a] * linear_form_power_coeffs(A[a], B[a], k)
    return total


def untied_total_coeffs(params, m, k):
    A = params[:m]
    B = params[m:2 * m]
    E = params[2 * m:3 * m]
    F = params[3 * m:4 * m]
    C = params[4 * m:5 * m]
    total = np.zeros(k + 1)
    for a in range(m):
        total += C[a] * linear_form_mixed_coeffs(A[a], B[a], E[a], F[a], k)
    return total


def fit(arch, target, m, k, rng, n_restarts=8, maxiter=2000):
    n_params = 3 * m if arch == "tied" else 5 * m

    def loss_fn(params):
        if arch == "tied":
            pred = tied_total_coeffs(params, m, k)
        else:
            pred = untied_total_coeffs(params, m, k)
        return float(np.sum((pred - target) ** 2))

    best = np.inf
    for _ in range(n_restarts):
        x0 = 0.5 * rng.standard_normal(n_params)
        res = minimize(loss_fn, x0, method="L-BFGS-B",
                      options={"maxiter": maxiter, "ftol": 1e-22, "gtol": 1e-18})
        best = min(best, res.fun)
    return best


def main(out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True, parents=True)

    ks = [2, 3, 4, 5]
    n_seeds = 2
    n_restarts = 5

    results = {}
    for k in ks:
        target = monomial_coeffs(k)
        ms = list(range(1, k + 3))  # m = 1,...,k+2
        per_arch = {"tied": {}, "untied": {}}
        for seed in range(n_seeds):
            rng = np.random.default_rng(seed * 37 + k * 11)
            for m in ms:
                lt = fit("tied", target, m, k, rng, n_restarts=n_restarts)
                lu = fit("untied", target, m, k, rng, n_restarts=n_restarts)
                per_arch["tied"].setdefault(m, []).append(lt)
                per_arch["untied"].setdefault(m, []).append(lu)
        results[str(k)] = {
            "ms": ms,
            "tied_per_seed": {str(m): per_arch["tied"][m] for m in ms},
            "untied_per_seed": {str(m): per_arch["untied"][m] for m in ms},
        }
        # print summary
        for m in ms:
            tm = np.median(per_arch["tied"][m])
            um = np.median(per_arch["untied"][m])
            print(f"k={k}, m={m}: tied={tm:.2e}, untied={um:.2e}")

    (out_dir / "results_width_sweep.json").write_text(
        json.dumps(results, indent=2))

    # ----- plot ----------------------------------------------------------------
    n_panels = len(ks)
    fig, axes = plt.subplots(1, n_panels, figsize=(2.8 * n_panels, 3.2),
                              sharey=True)

    COLOR_TIED = "#C73E3A"
    COLOR_UNTIED = "#2C5F94"

    for ax, k in zip(axes, ks):
        ms = results[str(k)]["ms"]
        tied_med = np.array([np.median(np.maximum(
            results[str(k)]["tied_per_seed"][str(m)], 1e-20)) for m in ms])
        untied_med = np.array([np.median(np.maximum(
            results[str(k)]["untied_per_seed"][str(m)], 1e-20)) for m in ms])
        tied_min = np.array([np.maximum(
            results[str(k)]["tied_per_seed"][str(m)], 1e-20).min() for m in ms])
        tied_max = np.array([np.maximum(
            results[str(k)]["tied_per_seed"][str(m)], 1e-20).max() for m in ms])
        untied_min = np.array([np.maximum(
            results[str(k)]["untied_per_seed"][str(m)], 1e-20).min() for m in ms])
        untied_max = np.array([np.maximum(
            results[str(k)]["untied_per_seed"][str(m)], 1e-20).max() for m in ms])

        ax.plot(ms, tied_med, color=COLOR_TIED, marker="o", markersize=6,
                linewidth=1.5, label="tied", zorder=3)
        ax.fill_between(ms, tied_min, tied_max, color=COLOR_TIED, alpha=0.18)
        ax.plot(ms, untied_med, color=COLOR_UNTIED, marker="s", markersize=6,
                linewidth=1.5, label="untied", zorder=3)
        ax.fill_between(ms, untied_min, untied_max, color=COLOR_UNTIED, alpha=0.18)

        # Predicted thresholds
        ax.axvline(x=k, color="gray", linestyle="--", linewidth=1.0,
                   alpha=0.6, zorder=1)
        ax.axvline(x=1, color="gray", linestyle=":", linewidth=1.0,
                   alpha=0.4, zorder=1)
        ax.text(k + 0.05, 1e-19,
                f"$m=k={k}$",
                fontsize=8, color="gray", ha="left", va="bottom")

        ax.set_xlabel("width $m$", fontsize=10)
        ax.set_yscale("log")
        ax.set_xticks(ms)
        ax.set_title(f"$k={k}$", fontsize=11)
        ax.set_ylim(1e-19, 10.0)
        ax.grid(True, which="major", alpha=0.25, linewidth=0.5)
        if k == ks[0]:
            ax.set_ylabel("median MSE\n(coefficient space, $2$ seeds)",
                          fontsize=10)
        if k == ks[-1]:
            ax.legend(fontsize=9, loc="lower left", framealpha=0.95)

    fig.suptitle(r"Coefficient-space width sweep for $F_k(x) = x_1^{k-1} x_2$:"
                 r" tied threshold at $m=k$, untied threshold at $m=1$",
                 fontsize=11, y=1.02)
    fig.tight_layout()
    fig.savefig(out_dir / "fig_width_sweep.pdf", bbox_inches="tight")
    fig.savefig(out_dir / "fig_width_sweep.png", dpi=140, bbox_inches="tight")
    plt.close(fig)

    print(f"\nWrote {out_dir}/fig_width_sweep.pdf, results_width_sweep.json")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results")
