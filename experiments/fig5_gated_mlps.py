"""Gated MLPs (tied ReLU^2, tied SwiGLU, untied ReGLU, untied SwiGLU) on x1^{k-1} x2.

Paper: Section 5.4, Figure 5. Tied SwiGLU uses SiLU(z) z (v = u).

Usage:  python experiments/fig5_gated_mlps.py [results directory, default: results]
"""

import json
import numpy as np
from pathlib import Path
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cpmlp.helpers import FEATURE_COLORS, silu, relu, relu2

# ----- target ---------------------------------------------------

def target_mixed(X, d):
    return X[:, 0] ** (d - 1) * X[:, 1]

# ----- predict functions ----------------------------------------

def tied_relu2_predict(params, X, m):
    n = X.shape[1]
    U = params[:m * n].reshape(m, n)
    c = params[m * n:m * n + m]
    Z = X @ U.T
    return relu2(Z) @ c

def tied_swiglu_predict(params, X, m):
    n = X.shape[1]
    U = params[:m * n].reshape(m, n)
    c = params[m * n:m * n + m]
    Z = X @ U.T
    return (silu(Z) * Z) @ c

def untied_reglu_predict(params, X, m):
    n = X.shape[1]
    U = params[:m * n].reshape(m, n)
    V = params[m * n:2 * m * n].reshape(m, n)
    c = params[2 * m * n:2 * m * n + m]
    Zu = X @ U.T
    Zv = X @ V.T
    return (relu(Zu) * Zv) @ c

def untied_swiglu_predict(params, X, m):
    n = X.shape[1]
    U = params[:m * n].reshape(m, n)
    V = params[m * n:2 * m * n].reshape(m, n)
    c = params[2 * m * n:2 * m * n + m]
    Zu = X @ U.T
    Zv = X @ V.T
    return (silu(Zu) * Zv) @ c

ARCH_FNS = {
    "tied-ReLU2":   ("tied",   tied_relu2_predict),
    "tied-SwiGLU":   ("tied",   tied_swiglu_predict),
    "untied-ReGLU": ("untied", untied_reglu_predict),
    "untied-SwiGLU":("untied", untied_swiglu_predict),
}

def fit(arch, m, d, n, N, rng, n_restarts=5):
    X = rng.standard_normal((N, n))
    y = target_mixed(X, d)
    kind, predict_fn = ARCH_FNS[arch]
    n_params = (m * n + m) if kind == "tied" else (2 * m * n + m)

    def loss_fn(params):
        residual = predict_fn(params, X, m) - y
        return float(np.mean(residual ** 2))

    best = np.inf
    for _ in range(n_restarts):
        x0 = 0.5 * rng.standard_normal(n_params)
        try:
            res = minimize(loss_fn, x0, method="L-BFGS-B",
                          options={"maxiter": 1500, "ftol": 1e-18, "gtol": 1e-12})
            best = min(best, res.fun)
        except Exception:
            pass
    return best


def main(out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True, parents=True)

    n = 4
    N = 200
    n_seeds = 3
    n_restarts = 3
    ds = [2, 3]
    ms = [1, 2, 4]
    archs = ["tied-ReLU2", "tied-SwiGLU", "untied-ReGLU", "untied-SwiGLU"]

    results = {}  # results[arch][d][m] = list of losses

    for seed in range(n_seeds):
        rng = np.random.default_rng(seed * 31)
        for arch in archs:
            for d in ds:
                for m in ms:
                    loss = fit(arch, m=m, d=d, n=n, N=N, rng=rng,
                              n_restarts=n_restarts)
                    results.setdefault(arch, {}).setdefault(d, {}).setdefault(m, []).append(loss)
                    print(f"seed {seed}: {arch}/d={d}/m={m}: {loss:.3e}")

    # save raw
    json_results = {arch: {str(d): {str(m): losses for m, losses in m2losses.items()}
                            for d, m2losses in d2m.items()}
                    for arch, d2m in results.items()}
    (out_dir / "results_architectural.json").write_text(json.dumps(json_results, indent=2))

    # ----- plot ---------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6), sharey=True)

    # Hand-pick markers and a consistent style for the four architectures
    style = {
        "tied-ReLU2":    {"color": FEATURE_COLORS["tied-ReLU2"],    "marker": "o", "linestyle": "-",  "label": r"tied $\mathrm{ReLU}^2$"},
        "tied-SwiGLU":    {"color": FEATURE_COLORS["tied-SwiGLU"],    "marker": "^", "linestyle": "-",  "label": r"tied SwiGLU"},
        "untied-ReGLU":  {"color": FEATURE_COLORS["untied-ReGLU"],  "marker": "s", "linestyle": "--", "label": r"untied ReGLU"},
        "untied-SwiGLU": {"color": FEATURE_COLORS["untied-SwiGLU"], "marker": "D", "linestyle": "--", "label": r"untied SwiGLU"},
    }

    for ax, d in zip(axes, ds):
        for arch in archs:
            losses_by_m = [np.maximum(np.array(results[arch][d][m]), 1e-20) for m in ms]
            medians = [np.median(L) for L in losses_by_m]
            mins = [L.min() for L in losses_by_m]
            maxs = [L.max() for L in losses_by_m]
            s = style[arch]
            ax.plot(ms, medians, color=s["color"], marker=s["marker"],
                    linestyle=s["linestyle"], markersize=7, linewidth=1.6,
                    label=s["label"], zorder=3)
            ax.fill_between(ms, mins, maxs, color=s["color"], alpha=0.13, zorder=2)

        ax.set_title(rf"target $x_0^{{{d-1}}} x_1$ (degree $d={d}$)", fontsize=11)
        ax.set_xlabel("width $m$", fontsize=10)
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xticks(ms)
        ax.set_xticklabels(ms)
        ax.grid(True, which="both", alpha=0.25, linewidth=0.5)
        ax.set_ylim(1e-18, 10.0)
        # threshold reference line at the predicted tied minimum width m = d
        ax.axvline(x=d, color="gray", linestyle=":", linewidth=1, alpha=0.7)
        ax.text(d, 1.5, rf"$m=d={d}$", fontsize=8, color="gray", ha="center")

    axes[0].set_ylabel("best-of-restarts MSE\n(median over $3$ seeds)", fontsize=10)
    axes[1].legend(fontsize=8.5, loc="lower left", framealpha=0.95)
    fig.suptitle(r"Architectural lift: width$\,$$\to\,$loss for actual gates fitting $x_0^{d-1} x_1$",
                 fontsize=11, y=1.02)
    fig.tight_layout()
    fig.savefig(out_dir / "fig_architectural_lift.pdf", bbox_inches="tight")
    fig.savefig(out_dir / "fig_architectural_lift.png", dpi=140, bbox_inches="tight")
    plt.close(fig)

    # ----- table for d = 2 only (cleanest result) -----------------------------
    def fmt(x):
        if x < 1e-15:
            return r"$<\!10^{-15}$"
        e = int(np.floor(np.log10(max(x, 1e-30))))
        m_ = x / 10 ** e
        return f"${m_:.1f}\\!\\times\\!10^{{{e}}}$"

    lines = [
        r"\begin{tabular}{@{}l|ccc|ccc@{}}",
        r"\toprule",
        r" & \multicolumn{3}{c|}{$d=2$ (target $x_0 x_1$)} & \multicolumn{3}{c}{$d=3$ (target $x_0^2 x_1$)} \\",
        r"Activation & $m=1$ & $m=2$ & $m=4$ & $m=1$ & $m=2$ & $m=4$ \\",
        r"\midrule",
    ]
    for arch in archs:
        row_label = style[arch]["label"]
        cells = [row_label]
        for d in ds:
            for m in ms:
                losses = np.array(results[arch][d][m])
                cells.append(fmt(np.median(np.maximum(losses, 1e-20))))
        lines.append(" & ".join(cells) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    (out_dir / "table_architectural.tex").write_text("\n".join(lines))

    print(f"\nWrote {out_dir}/fig_architectural_lift.pdf, results_architectural.json, "
          f"table_architectural.tex")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results")
