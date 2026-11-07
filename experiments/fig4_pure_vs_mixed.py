"""Pure powers x1^k versus role-asymmetric x1^{k-1} x2 at width one.

Paper: Section 5.3, Figure 4.

Usage:  python experiments/fig4_pure_vs_mixed.py [results directory, default: results]
"""

import json
import numpy as np
from pathlib import Path
from scipy.optimize import minimize
from itertools import product
import matplotlib.pyplot as plt
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cpmlp.helpers import FEATURE_COLORS

# ----- target generation ---------------------------------------------------

def target_pure(X, d):
    """x_0^d, X is (N, n)."""
    return X[:, 0] ** d

def target_mixed(X, d):
    """x_0^{d-1} x_1, X is (N, n)."""
    return X[:, 0] ** (d - 1) * X[:, 1]

# ----- model parameterizations ---------------------------------------------

def tied_predict(params, X, m, d):
    """sum_a c_a (u_a^T x)^d"""
    n = X.shape[1]
    U = params[:m * n].reshape(m, n)
    c = params[m * n:m * n + m]
    Z = X @ U.T  # (N, m)
    return (Z ** d) @ c

def untied_predict(params, X, m, d):
    """sum_a c_a (u_a^T x)^{d-1} (v_a^T x)"""
    n = X.shape[1]
    U = params[:m * n].reshape(m, n)
    V = params[m * n:2 * m * n].reshape(m, n)
    c = params[2 * m * n:2 * m * n + m]
    Zu = X @ U.T
    Zv = X @ V.T
    return ((Zu ** (d - 1)) * Zv) @ c

# ----- single fit with restarts --------------------------------------------

def fit(arch, target_fn, m, d, n, N, rng, n_restarts=5):
    """Fit one (arch, target, m, d) combo with multiple restarts.
    Returns the best (lowest) loss."""
    X = rng.standard_normal((N, n))
    y = target_fn(X, d)

    if arch == "tied":
        n_params = m * n + m
        predict_fn = lambda p: tied_predict(p, X, m, d)
    else:
        n_params = 2 * m * n + m
        predict_fn = lambda p: untied_predict(p, X, m, d)

    def loss_fn(params):
        residual = predict_fn(params) - y
        return float(np.mean(residual ** 2))

    best_loss = np.inf
    for _ in range(n_restarts):
        x0 = 0.5 * rng.standard_normal(n_params)
        try:
            res = minimize(loss_fn, x0, method="L-BFGS-B",
                          options={"maxiter": 5000, "ftol": 1e-20, "gtol": 1e-15})
            best_loss = min(best_loss, res.fun)
        except Exception:
            pass
    return best_loss


def main(out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True, parents=True)

    n = 4
    N = 256
    n_seeds = 10
    n_restarts = 5
    ds = [2, 3, 4]
    archs = ["tied", "untied"]
    targets = [("pure", target_pure), ("mixed", target_mixed)]

    results = {}  # results[arch][target_kind][d] = list of losses across seeds

    for arch in archs:
        results[arch] = {}
        for tgt_name, _ in targets:
            results[arch][tgt_name] = {}

    for seed in range(n_seeds):
        rng = np.random.default_rng(seed)
        for arch, (tgt_name, tgt_fn), d in product(archs, targets, ds):
            loss = fit(arch, tgt_fn, m=1, d=d, n=n, N=N,
                       rng=rng, n_restarts=n_restarts)
            results[arch][tgt_name].setdefault(d, []).append(loss)
            print(f"seed {seed}: {arch}/{tgt_name}/d={d}: {loss:.3e}")

    # save raw
    json_results = {arch: {tgt: {str(d): losses for d, losses in d2losses.items()}
                            for tgt, d2losses in tgt2.items()}
                    for arch, tgt2 in results.items()}
    (out_dir / "results_pure_vs_mixed.json").write_text(json.dumps(json_results, indent=2))

    # ----- plot ---------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6), sharey=True)

    for ax, (tgt_name, _) in zip(axes, targets):
        for arch, color, marker in [("tied", FEATURE_COLORS["tied-quadratic"], "o"),
                                     ("untied", FEATURE_COLORS["untied-quadratic"], "s")]:
            losses_by_d = [np.array(results[arch][tgt_name][d]) for d in ds]
            medians = np.array([np.median(np.maximum(L, 1e-20)) for L in losses_by_d])
            mins = np.array([np.maximum(L, 1e-20).min() for L in losses_by_d])
            maxs = np.array([np.maximum(L, 1e-20).max() for L in losses_by_d])

            ax.plot(ds, medians, color=color, marker=marker, markersize=8,
                    linewidth=1.6, label=arch, zorder=3)
            ax.fill_between(ds, mins, maxs, color=color, alpha=0.18, zorder=2)
            ax.scatter(np.repeat(ds, n_seeds),
                       np.concatenate([np.maximum(L, 1e-20) for L in losses_by_d]),
                       color=color, alpha=0.4, s=14, zorder=2)

        if tgt_name == "pure":
            title = r"Pure-power target $x_0^d$"
        else:
            title = r"Role-asymmetric target $x_0^{d-1}\, x_1$"
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("polynomial degree $d$", fontsize=10)
        ax.set_yscale("log")
        ax.set_xticks(ds)
        ax.grid(True, which="both", alpha=0.25, linewidth=0.5)
        ax.set_ylim(1e-20, 10.0)
        ax.legend(fontsize=9, loc="best", framealpha=0.95)

    axes[0].set_ylabel("best-of-restarts MSE\n(width $m=1$, $10$ seeds)", fontsize=10)
    fig.suptitle(r"The tied/untied gap reverses with target structure ($m=1$ across $10$ seeds)",
                 fontsize=11, y=1.02)
    fig.tight_layout()
    fig.savefig(out_dir / "fig_pure_vs_mixed.pdf", bbox_inches="tight")
    fig.savefig(out_dir / "fig_pure_vs_mixed.png", dpi=140, bbox_inches="tight")
    plt.close(fig)

    # ----- table --------------------------------------------------------------
    def fmt(x):
        if x < 1e-15:
            return r"$<\!10^{-15}$"
        e = int(np.floor(np.log10(max(x, 1e-30))))
        m_ = x / 10 ** e
        return f"${m_:.1f}\\!\\times\\!10^{{{e}}}$"

    lines = [
        r"\begin{tabular}{@{}llccc@{}}",
        r"\toprule",
        r"Target & Architecture & $d=2$ & $d=3$ & $d=4$ \\",
        r"\midrule",
    ]
    for tgt_name, label in [("pure", r"$x_0^{d}$ (pure power)"),
                            ("mixed", r"$x_0^{d-1} x_1$ (mixed)")]:
        for arch in ["tied", "untied"]:
            row = [label if arch == "tied" else "", arch]
            for d in ds:
                losses = np.array(results[arch][tgt_name][d])
                row.append(fmt(np.median(np.maximum(losses, 1e-20))))
            lines.append(" & ".join(row) + r" \\")
        lines.append(r"\midrule")
    lines = lines[:-1] + [r"\bottomrule", r"\end{tabular}"]
    (out_dir / "table_pure_vs_mixed.tex").write_text("\n".join(lines))

    print(f"\nWrote {out_dir}/fig_pure_vs_mixed.pdf, results_pure_vs_mixed.json, "
          f"table_pure_vs_mixed.tex")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results")
