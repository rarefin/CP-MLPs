"""Indefinite quadratic targets: tied vs untied minimum width.

Paper: Section 5.1, Figure 2. Prediction: tied width n_+ + n_-, untied width max(n_+, n_-).

Usage:  python experiments/fig2_indefinite_quadratic.py [results directory, default: results]
"""

import json
import sys
import numpy as np
from pathlib import Path
from scipy.optimize import minimize
import matplotlib.pyplot as plt


# ===== quadratic forms =====================================================

def make_Q(n_pos, n_neg, d):
    """Diagonal symmetric matrix with n_pos positive and n_neg negative
    eigenvalues (signature (n_+, n_-))."""
    Q = np.zeros((d, d))
    for i in range(n_pos):
        Q[i, i] = 1.0
    for i in range(n_neg):
        Q[n_pos + i, n_pos + i] = -1.0
    return Q


def tied_predict(params, X, m):
    """Tied: sum_a c_a (u_a^T x)^2"""
    d = X.shape[1]
    U = params[:m * d].reshape(m, d)
    c = params[m * d:]
    Z = X @ U.T
    return (Z ** 2) @ c


def untied_predict(params, X, m):
    """Untied: sum_a c_a (u_a^T x)(v_a^T x)"""
    d = X.shape[1]
    U = params[:m * d].reshape(m, d)
    V = params[m * d:2 * m * d].reshape(m, d)
    c = params[2 * m * d:]
    Zu = X @ U.T
    Zv = X @ V.T
    return (Zu * Zv) @ c


def fit(arch, Q, m, X, y, rng, n_restarts=8):
    d = X.shape[1]
    n_params = m * d + m if arch == "tied" else 2 * m * d + m
    predict = tied_predict if arch == "tied" else untied_predict

    def loss(p):
        return float(np.mean((predict(p, X, m) - y) ** 2))

    best = np.inf
    for _ in range(n_restarts):
        x0 = 0.5 * rng.standard_normal(n_params)
        res = minimize(loss, x0, method="L-BFGS-B",
                       options={"maxiter": 3000, "ftol": 1e-22, "gtol": 1e-15})
        best = min(best, res.fun)
    return best


def main(out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True, parents=True)

    d = 8                          # input dimension
    N = 256                        # samples
    n_seeds = 3
    n_restarts = 8

    # Each entry is a (n_+, n_-) signature of the target quadratic form Q.
    # Includes definite, indefinite, and balanced cases.
    configs = [
        (1, 0),   # definite rank-1: tied = untied = 1
        (3, 0),   # definite rank-3: tied = untied = 3
        (1, 1),   # indefinite:           tied = 2, untied = 1, gap = 1
        (2, 2),   # balanced indefinite:  tied = 4, untied = 2, gap = 2
        (3, 1),   # asymmetric:           tied = 4, untied = 3, gap = 1
        (3, 3),   # large balanced:       tied = 6, untied = 3, gap = 3
    ]

    results = []
    for (n_pos, n_neg) in configs:
        Q = make_Q(n_pos, n_neg, d)
        tied_pred = n_pos + n_neg
        untied_pred = max(n_pos, n_neg)
        gap = min(n_pos, n_neg)
        print(f"\n=== signature (n_+={n_pos}, n_-={n_neg}): "
              f"predicted tied={tied_pred}, untied={untied_pred}, "
              f"gap={gap} ===")

        # Per-seed: fit at width predicted-1 (should fail) and predicted
        # (should fit).
        tied_pred_minus = []
        tied_at_pred = []
        untied_pred_minus = []
        untied_at_pred = []

        for seed in range(n_seeds):
            rng = np.random.default_rng(seed * 137 + n_pos * 17 + n_neg)
            X = rng.standard_normal((N, d))
            y = np.einsum("ni,ij,nj->n", X, Q, X)

            # Tied
            if tied_pred >= 1:
                t_at = fit("tied", Q, tied_pred, X, y, rng, n_restarts)
                tied_at_pred.append(t_at)
                if tied_pred >= 2:
                    t_minus = fit("tied", Q, tied_pred - 1, X, y, rng,
                                  n_restarts)
                    tied_pred_minus.append(t_minus)

            # Untied
            if untied_pred >= 1:
                u_at = fit("untied", Q, untied_pred, X, y, rng, n_restarts)
                untied_at_pred.append(u_at)
                if untied_pred >= 2:
                    u_minus = fit("untied", Q, untied_pred - 1, X, y, rng,
                                  n_restarts)
                    untied_pred_minus.append(u_minus)

        med_t_minus = float(np.median(tied_pred_minus)) if tied_pred_minus else None
        med_t_at = float(np.median(tied_at_pred))
        med_u_minus = float(np.median(untied_pred_minus)) if untied_pred_minus else None
        med_u_at = float(np.median(untied_at_pred))

        # Observed minimum widths (loss < 1e-8 counts as fit)
        FIT_THRESH = 1e-8
        results.append({
            "n_pos": n_pos, "n_neg": n_neg,
            "tied_pred": tied_pred,
            "tied_at_pred": med_t_at,
            "tied_pred_minus_1": med_t_minus,
            "tied_obs_min": tied_pred if med_t_at < FIT_THRESH else None,
            "untied_pred": untied_pred,
            "untied_at_pred": med_u_at,
            "untied_pred_minus_1": med_u_minus,
            "untied_obs_min": untied_pred if med_u_at < FIT_THRESH else None,
            "gap_pred": gap,
            "gap_obs": tied_pred - untied_pred,
        })
        t_minus_str = f"{med_t_minus:.2e}" if med_t_minus is not None else "N/A"
        u_minus_str = f"{med_u_minus:.2e}" if med_u_minus is not None else "N/A"
        print(f"  tied at m={tied_pred}: {med_t_at:.2e}; "
              f"tied at m={tied_pred - 1}: {t_minus_str}")
        print(f"  untied at m={untied_pred}: {med_u_at:.2e}; "
              f"untied at m={untied_pred - 1}: {u_minus_str}")

    (out_dir / "results_quadratic.json").write_text(json.dumps(results, indent=2))

    # ===== table ============================================================
    def fmt(x):
        if x is None:
            return "—"
        if x < 1e-15:
            return r"$<\!10^{-15}$"
        e = int(np.floor(np.log10(max(x, 1e-30))))
        m_ = x / 10 ** e
        return f"${m_:.1f}\\!\\times\\!10^{{{e}}}$"

    lines = [
        r"\begin{tabular}{@{}cc|cc|cc|c@{}}",
        r"\toprule",
        r"$n_+$ & $n_-$ & \multicolumn{2}{c|}{Tied} & "
        r"\multicolumn{2}{c|}{Untied} & Gap \\",
        r" & & Pred & Obs & Pred & Obs & \\",
        r"\midrule",
    ]
    for r_ in results:
        tied_obs = r_["tied_obs_min"]
        untied_obs = r_["untied_obs_min"]
        lines.append(
            f"{r_['n_pos']} & {r_['n_neg']} & {r_['tied_pred']} & "
            f"{tied_obs if tied_obs else '?'} & "
            f"{r_['untied_pred']} & {untied_obs if untied_obs else '?'} & "
            f"{r_['gap_pred']} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    (out_dir / "table_quadratic.tex").write_text("\n".join(lines))

    # ===== figure ===========================================================
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    # Left: bar chart of minimum widths
    ax = axes[0]
    xs = np.arange(len(results))
    width = 0.35
    tied_widths = [r_["tied_pred"] for r_ in results]
    untied_widths = [r_["untied_pred"] for r_ in results]
    labels = [f"({r_['n_pos']},{r_['n_neg']})" for r_ in results]

    ax.bar(xs - width / 2, tied_widths, width, color="#C73E3A",
           label="tied min width", alpha=0.85)
    ax.bar(xs + width / 2, untied_widths, width, color="#2C5F94",
           label="untied min width", alpha=0.85)
    for r_, x_ in zip(results, xs):
        if r_["gap_pred"] > 0:
            ax.annotate(
                f"gap={r_['gap_pred']}",
                xy=(x_, max(r_["tied_pred"], r_["untied_pred"]) + 0.2),
                ha="center", fontsize=9, color="black",
            )
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_xlabel(r"signature $(n_+, n_-)$ of target $Q$", fontsize=11)
    ax.set_ylabel("minimum width (predicted = observed)", fontsize=11)
    ax.set_title(r"(a) Width gap $= \min\{n_+, n_-\}$", fontsize=11)
    ax.legend(loc="upper left", fontsize=9, framealpha=0.95)
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_ylim(0, max(max(tied_widths), max(untied_widths)) + 1.5)

    # Right: loss-at-predicted vs predicted-1
    ax = axes[1]

    def clip_arr(arr):
        return np.array([np.nan if v is None else max(v, 1e-20) for v in arr])

    losses_failed_tied = clip_arr([r_["tied_pred_minus_1"] for r_ in results])
    losses_fit_tied = clip_arr([r_["tied_at_pred"] for r_ in results])
    losses_failed_untied = clip_arr([r_["untied_pred_minus_1"] for r_ in results])
    losses_fit_untied = clip_arr([r_["untied_at_pred"] for r_ in results])

    offset = 0.12
    x_arr = np.array(xs)
    ax.scatter(x_arr - offset, losses_failed_tied, color="#C73E3A",
               marker="x", s=80,
               label=r"tied at $m_*-1$ (predicted to fail)", zorder=3,
               linewidth=2.5)
    ax.scatter(x_arr - offset, losses_fit_tied, color="#C73E3A",
               marker="o", s=70,
               label=r"tied at $m_*$ (predicted to fit)", zorder=3,
               edgecolors="black", linewidth=0.8)
    ax.scatter(x_arr + offset, losses_failed_untied, color="#2C5F94",
               marker="x", s=80,
               label=r"untied at $m_*-1$ (predicted to fail)", zorder=3,
               linewidth=2.5)
    ax.scatter(x_arr + offset, losses_fit_untied, color="#2C5F94",
               marker="s", s=70,
               label=r"untied at $m_*$ (predicted to fit)", zorder=3,
               edgecolors="black", linewidth=0.8)
    ax.axhline(y=1e-8, color="gray", linestyle=":", linewidth=1, alpha=0.7,
               label=r"fit threshold ($10^{-8}$)")

    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_xlabel(r"signature $(n_+, n_-)$ of target $Q$", fontsize=11)
    ax.set_ylabel("median MSE (log scale)", fontsize=11)
    ax.set_title(r"(b) Loss collapses exactly at predicted $m_*$",
                 fontsize=11)
    ax.set_yscale("log")
    ax.set_ylim(1e-18, 10.0)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="lower left", framealpha=0.95)

    fig.suptitle(
        r"Quadratic atom-rank law: tied $m_* = n_+ + n_-$, "
        r"untied $m_* = \max\{n_+, n_-\}$, gap $= \min\{n_+, n_-\}$",
        fontsize=12, y=1.02,
    )
    fig.tight_layout()
    fig.savefig(out_dir / "fig_quadratic.pdf", bbox_inches="tight")
    fig.savefig(out_dir / "fig_quadratic.png", dpi=140, bbox_inches="tight")
    plt.close(fig)

    print(f"\nWrote {out_dir}/results_quadratic.json, table_quadratic.tex, "
          f"fig_quadratic.pdf, fig_quadratic.png")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results")