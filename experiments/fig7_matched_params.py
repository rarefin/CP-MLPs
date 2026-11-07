"""Matched input-projection control: untied width m vs tied width 2m.

Paper: Appendix D.8, Figure 7. Tied SwiGLU uses SiLU(z) z (v = u).

Usage:  python experiments/fig7_matched_params.py [results directory, default: results]
"""

import json
import sys
import numpy as np
from pathlib import Path
from scipy.optimize import minimize
import matplotlib.pyplot as plt


# ===== activations ============================================================

def relu(z):  return np.maximum(z, 0)
def silu(z):  return z / (1 + np.exp(-z))
def relu2(z): return relu(z) ** 2


# ===== predict ================================================================

def tied_relu2(params, X, m):
    n = X.shape[1]
    U = params[:m * n].reshape(m, n)
    c = params[m * n:m * n + m]
    return relu2(X @ U.T) @ c

def tied_swiglu(params, X, m):
    n = X.shape[1]
    U = params[:m * n].reshape(m, n)
    c = params[m * n:m * n + m]
    Z = X @ U.T
    return (silu(Z) * Z) @ c

def untied_reglu(params, X, m):
    n = X.shape[1]
    U = params[:m * n].reshape(m, n)
    V = params[m * n:2 * m * n].reshape(m, n)
    c = params[2 * m * n:2 * m * n + m]
    return (relu(X @ U.T) * (X @ V.T)) @ c

def untied_swiglu(params, X, m):
    n = X.shape[1]
    U = params[:m * n].reshape(m, n)
    V = params[m * n:2 * m * n].reshape(m, n)
    c = params[2 * m * n:2 * m * n + m]
    return (silu(X @ U.T) * (X @ V.T)) @ c

ARCH_FNS = {
    "tied-ReLU2":   ("tied",   tied_relu2,   lambda m, n: m * n + m),
    "tied-SwiGLU":   ("tied",   tied_swiglu,   lambda m, n: m * n + m),
    "untied-ReGLU": ("untied", untied_reglu, lambda m, n: 2 * m * n + m),
    "untied-SwiGLU":("untied", untied_swiglu, lambda m, n: 2 * m * n + m),
}


# ===== fit ====================================================================

def target_mixed(X, k):
    return X[:, 0] ** (k - 1) * X[:, 1]

def fit(arch, m, k, n, N, rng, n_restarts=4, maxiter=1500):
    kind, predict_fn, pcount_fn = ARCH_FNS[arch]
    n_params = pcount_fn(m, n)
    X = rng.standard_normal((N, n))
    y = target_mixed(X, k)

    def loss_fn(params):
        return float(np.mean((predict_fn(params, X, m) - y) ** 2))

    best = np.inf
    for _ in range(n_restarts):
        x0 = 0.5 * rng.standard_normal(n_params)
        try:
            res = minimize(loss_fn, x0, method="L-BFGS-B",
                          options={"maxiter": maxiter, "ftol": 1e-20,
                                   "gtol": 1e-12})
            best = min(best, res.fun)
        except Exception:
            pass
    return best, n_params


# ===== main ===================================================================

def main(out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True, parents=True)

    n = 4
    N = 200
    n_seeds = 3
    n_restarts = 3
    ks = [2, 3]

    # Compare at matched parameter count:
    #   untied at m_untied has ~ 2 m_untied n input params
    #   tied   at m_tied   has ~     m_tied n input params
    # Match: m_tied = 2 m_untied.
    # We sweep m_untied in {1, 2, 4} and m_tied in {2, 4, 8} so each
    # tied/untied pair has the same total input projection size.
    pairs = [(1, 2), (2, 4), (4, 8)]  # (m_untied, m_tied_matched)

    archs = ["tied-ReLU2", "tied-SwiGLU", "untied-ReGLU", "untied-SwiGLU"]
    results = {}  # results[k][arch][m_label] = list of losses

    for k in ks:
        results[str(k)] = {arch: {} for arch in archs}
        for m_untied, m_tied in pairs:
            label = f"untied_m{m_untied}_vs_tied_m{m_tied}"
            for seed in range(n_seeds):
                rng = np.random.default_rng(seed * 19 + k)
                # Untied at m_untied
                for arch in ["untied-ReGLU", "untied-SwiGLU"]:
                    loss, npar = fit(arch, m_untied, k, n, N, rng,
                                     n_restarts=n_restarts)
                    results[str(k)][arch].setdefault(m_untied, []).append(
                        {"loss": loss, "n_params": npar})
                # Tied at m_tied
                for arch in ["tied-ReLU2", "tied-SwiGLU"]:
                    loss, npar = fit(arch, m_tied, k, n, N, rng,
                                     n_restarts=n_restarts)
                    results[str(k)][arch].setdefault(m_tied, []).append(
                        {"loss": loss, "n_params": npar})

            print(f"k={k}, untied m={m_untied} (parameters: "
                  f"~{2*m_untied*n} input) vs tied m={m_tied} "
                  f"(parameters: ~{m_tied*n} input):")
            for arch in archs:
                if arch.startswith("untied"):
                    losses = [r["loss"] for r in
                              results[str(k)][arch][m_untied][-n_seeds:]]
                else:
                    losses = [r["loss"] for r in
                              results[str(k)][arch][m_tied][-n_seeds:]]
                print(f"   {arch}: median loss {np.median(losses):.2e}")

    (out_dir / "results_matched.json").write_text(
        json.dumps(results, indent=2, default=float))

    # ----- plot ---------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6))

    style = {
        "tied-ReLU2":    {"color": "#E25C28", "marker": "o", "ls": "-",
                          "label": r"tied $\mathrm{ReLU}^2$ (matched $m$)"},
        "tied-SwiGLU":    {"color": "#F4A742", "marker": "^", "ls": "-",
                          "label": r"tied SwiGLU (matched $m$)"},
        "untied-ReGLU":  {"color": "#2A8C8B", "marker": "s", "ls": "--",
                          "label": r"untied ReGLU"},
        "untied-SwiGLU": {"color": "#3E8946", "marker": "D", "ls": "--",
                          "label": r"untied SwiGLU"},
    }

    for ax, k in zip(axes, ks):
        # x-axis: untied widths {1, 2, 4}; tied widths plotted at same x
        # because parameter count is matched
        for arch in archs:
            xs, ys, ymin, ymax = [], [], [], []
            for m_untied, m_tied in pairs:
                if arch.startswith("untied"):
                    losses = np.array([r["loss"] for r in
                                       results[str(k)][arch][m_untied]])
                else:
                    losses = np.array([r["loss"] for r in
                                       results[str(k)][arch][m_tied]])
                losses = np.maximum(losses, 1e-20)
                xs.append(m_untied)
                ys.append(np.median(losses))
                ymin.append(losses.min())
                ymax.append(losses.max())
            s = style[arch]
            ax.plot(xs, ys, color=s["color"], marker=s["marker"],
                    linestyle=s["ls"], markersize=7, linewidth=1.6,
                    label=s["label"], zorder=3)
            ax.fill_between(xs, ymin, ymax, color=s["color"], alpha=0.13)

        target_str = "x_1\\, x_2" if k == 2 else "x_1^2\\, x_2"
        ax.set_title(rf"target ${target_str}$ ($k={k}$)", fontsize=11)
        ax.set_xlabel("untied width $m$ (matched: tied at $2m$)", fontsize=10)
        ax.set_yscale("log")
        ax.set_xscale("log", base=2)
        ax.set_xticks([1, 2, 4])
        ax.set_xticklabels(["1", "2", "4"])
        ax.grid(True, which="both", alpha=0.25, linewidth=0.5)
        ax.set_ylim(1e-18, 10.0)
        ax.legend(fontsize=8.5, loc="upper right" if k == 2 else "best",
                  framealpha=0.95)

    axes[0].set_ylabel("median MSE\n($3$ seeds, parameter-matched)",
                        fontsize=10)
    fig.suptitle(r"Matched-parameter control: untied still beats tied "
                 r"on $x_1^{k-1} x_2$ at equal parameter count",
                 fontsize=11, y=1.02)
    fig.tight_layout()
    fig.savefig(out_dir / "fig_matched.pdf", bbox_inches="tight")
    fig.savefig(out_dir / "fig_matched.png", dpi=140, bbox_inches="tight")
    plt.close(fig)

    # ----- table --------------------------------------------------------------
    def fmt(x):
        if x < 1e-15:
            return r"$<\!10^{-15}$"
        e = int(np.floor(np.log10(max(x, 1e-30))))
        m_ = x / 10 ** e
        return f"${m_:.1f}\\!\\times\\!10^{{{e}}}$"

    lines = [
        r"\begin{tabular}{@{}lcc|cc|cc@{}}",
        r"\toprule",
        r" & \multicolumn{2}{c|}{$m=1$ untied / $m=2$ tied}"
        r" & \multicolumn{2}{c|}{$m=2$ untied / $m=4$ tied}"
        r" & \multicolumn{2}{c}{$m=4$ untied / $m=8$ tied} \\",
        r" & $k=2$ & $k=3$ & $k=2$ & $k=3$ & $k=2$ & $k=3$ \\",
        r"\midrule",
    ]
    label_map = {
        "tied-ReLU2": r"tied $\mathrm{ReLU}^2$",
        "tied-SwiGLU": r"tied SwiGLU",
        "untied-ReGLU": r"untied ReGLU",
        "untied-SwiGLU": r"untied SwiGLU",
    }
    for arch in archs:
        cells = [label_map[arch]]
        for m_untied, m_tied in pairs:
            for k in ks:
                if arch.startswith("untied"):
                    losses = [r["loss"] for r in
                              results[str(k)][arch][m_untied]]
                else:
                    losses = [r["loss"] for r in
                              results[str(k)][arch][m_tied]]
                cells.append(fmt(np.median(np.maximum(losses, 1e-20))))
        lines.append(" & ".join(cells) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    (out_dir / "table_matched.tex").write_text("\n".join(lines))

    print(f"\nWrote {out_dir}/fig_matched.pdf, results_matched.json, "
          f"table_matched.tex")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results")
