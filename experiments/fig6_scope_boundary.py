"""Scope of the role-asymmetric separation: x1^{k-1}x2 vs square-free x1 x2 x3.

Paper: Appendix D.7, Figure 6 and Table 3.

Usage:  python experiments/fig6_scope_boundary.py [results directory, default: results]
"""

import json
import sys
import numpy as np
from pathlib import Path
from scipy.optimize import minimize


# ===== Coefficient-space basis ===============================================
# We use a multi-index basis for monomials in n variables of total degree k.
# Each monomial x_1^{a_1} ... x_n^{a_n} with sum a_i = k corresponds to one
# coefficient. We enumerate all such tuples in lexicographic order.

def multi_indices(n, k):
    """All exponent tuples (a_1,...,a_n) with sum a_i = k, a_i >= 0."""
    if n == 1:
        yield (k,)
        return
    for a in range(k + 1):
        for rest in multi_indices(n - 1, k - a):
            yield (a,) + rest


def multinomial(alpha):
    """Multinomial coefficient k! / (a_1! a_2! ... a_n!) where k = sum a_i."""
    from math import factorial
    k = sum(alpha)
    out = factorial(k)
    for a in alpha:
        out //= factorial(a)
    return out


# ===== Coefficient vector representation ====================================

def make_basis(n, k):
    """Return (basis, idx_map) where basis is a list of exponent tuples
    of length D = C(n+k-1, k) and idx_map maps tuple -> index."""
    basis = list(multi_indices(n, k))
    idx_map = {alpha: i for i, alpha in enumerate(basis)}
    return basis, idx_map


def target_dv(basis, idx_map, n, k):
    """F_dv = x_1^{k-1} x_2: coefficient 1 at exponent tuple
    (k-1, 1, 0, ..., 0)."""
    D = len(basis)
    target = np.zeros(D)
    alpha = tuple([k - 1, 1] + [0] * (n - 2))
    target[idx_map[alpha]] = 1.0
    return target


def target_sf(basis, idx_map, n, k):
    """F_sf = x_1 x_2 ... x_k: coefficient 1 at exponent (1,1,...,1,0,...,0)
    with k ones and n-k zeros. Requires n >= k."""
    assert n >= k, f"need at least {k} variables for square-free degree-{k}"
    D = len(basis)
    target = np.zeros(D)
    alpha = tuple([1] * k + [0] * (n - k))
    target[idx_map[alpha]] = 1.0
    return target


# ===== Linear-form-power coefficients =======================================

def linear_power_coeffs(L, basis, idx_map, n, k):
    """Coefficients of (L^T x)^k where L = (l_1, ..., l_n), in the basis.
    By multinomial theorem:
      (sum l_i x_i)^k = sum_{|alpha|=k} multinomial(alpha) * prod l_i^{a_i} * x^alpha
    """
    D = len(basis)
    out = np.zeros(D)
    for i, alpha in enumerate(basis):
        coef = multinomial(alpha)
        for j, a in enumerate(alpha):
            coef *= L[j] ** a
        out[i] = coef
    return out


def linear_dv_coeffs(U, V, basis, idx_map, n, k):
    """Coefficients of (U^T x)^{k-1} (V^T x).
    Compute (U^T x)^{k-1} first, then multiply by (V^T x).
    """
    # Coefficients of (U^T x)^{k-1}, in degree-(k-1) basis
    basis_km1, idx_km1 = make_basis(n, k - 1)
    pow_km1 = np.zeros(len(basis_km1))
    for i, alpha in enumerate(basis_km1):
        coef = multinomial(alpha)
        for j, a in enumerate(alpha):
            coef *= U[j] ** a
        pow_km1[i] = coef

    # Multiply by (V^T x) = sum_j V_j x_j: each term increases one exponent by 1
    out = np.zeros(len(basis))
    for i_km1, alpha_km1 in enumerate(basis_km1):
        c = pow_km1[i_km1]
        for j in range(n):
            new_alpha = list(alpha_km1)
            new_alpha[j] += 1
            new_alpha = tuple(new_alpha)
            if new_alpha in idx_map:
                out[idx_map[new_alpha]] += c * V[j]
    return out


def linear_multileg_coeffs(legs, basis, idx_map, n, k):
    """Coefficients of prod_j (U_j^T x), where legs is a (k, n) matrix.
    Each column U_j has shape (n,)."""
    # Recursively multiply
    coeffs_so_far = np.zeros(len(make_basis(n, 0)[0]))
    coeffs_so_far[0] = 1.0
    basis_so_far, idx_so_far = make_basis(n, 0)

    for j in range(k):
        # current product has degree j; multiply by leg j (linear)
        new_basis, new_idx = make_basis(n, j + 1)
        new_coeffs = np.zeros(len(new_basis))
        for i_old, alpha_old in enumerate(basis_so_far):
            c = coeffs_so_far[i_old]
            if c == 0:
                continue
            for jj in range(n):
                new_alpha = list(alpha_old)
                new_alpha[jj] += 1
                new_alpha_t = tuple(new_alpha)
                new_coeffs[new_idx[new_alpha_t]] += c * legs[j, jj]
        coeffs_so_far = new_coeffs
        basis_so_far = new_basis
        idx_so_far = new_idx

    return coeffs_so_far


# ===== Loss & fit ============================================================

def fit_tied(target, basis, idx_map, n, k, m, rng, n_restarts=8, maxiter=800):
    n_params = m * n + m  # m linear forms + m coefficients

    def loss_fn(params):
        L = params[:m * n].reshape(m, n)
        c = params[m * n:]
        pred = np.zeros(len(basis))
        for a in range(m):
            pred += c[a] * linear_power_coeffs(L[a], basis, idx_map, n, k)
        return float(np.sum((pred - target) ** 2))

    best = np.inf
    for _ in range(n_restarts):
        x0 = 0.5 * rng.standard_normal(n_params)
        res = minimize(loss_fn, x0, method="L-BFGS-B",
                      options={"maxiter": maxiter, "ftol": 1e-22, "gtol": 1e-15})
        best = min(best, res.fun)
    return best


def fit_untied(target, basis, idx_map, n, k, m, rng, n_restarts=8, maxiter=800):
    n_params = 2 * m * n + m

    def loss_fn(params):
        U = params[:m * n].reshape(m, n)
        V = params[m * n:2 * m * n].reshape(m, n)
        c = params[2 * m * n:]
        pred = np.zeros(len(basis))
        for a in range(m):
            pred += c[a] * linear_dv_coeffs(U[a], V[a], basis, idx_map, n, k)
        return float(np.sum((pred - target) ** 2))

    best = np.inf
    for _ in range(n_restarts):
        x0 = 0.5 * rng.standard_normal(n_params)
        res = minimize(loss_fn, x0, method="L-BFGS-B",
                      options={"maxiter": maxiter, "ftol": 1e-22, "gtol": 1e-15})
        best = min(best, res.fun)
    return best


def fit_multileg(target, basis, idx_map, n, k, m, rng, n_restarts=8,
                 maxiter=800):
    """m channels, each is a product of k linear forms."""
    n_params = m * k * n + m

    def loss_fn(params):
        legs = params[:m * k * n].reshape(m, k, n)
        c = params[m * k * n:]
        pred = np.zeros(len(basis))
        for a in range(m):
            pred += c[a] * linear_multileg_coeffs(legs[a], basis, idx_map, n, k)
        return float(np.sum((pred - target) ** 2))

    best = np.inf
    for _ in range(n_restarts):
        x0 = 0.5 * rng.standard_normal(n_params)
        res = minimize(loss_fn, x0, method="L-BFGS-B",
                      options={"maxiter": maxiter, "ftol": 1e-22, "gtol": 1e-15})
        best = min(best, res.fun)
    return best


# ===== main ==================================================================

def main(out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True, parents=True)

    k = 3  # degree
    n = 3  # variables (need >= k for square-free)

    basis, idx_map = make_basis(n, k)
    print(f"Coefficient-space dim D = {len(basis)} for n={n}, k={k}")

    F_dv = target_dv(basis, idx_map, n, k)
    F_sf = target_sf(basis, idx_map, n, k)

    print(f"F_dv = x_1^{k-1} x_2: nonzero at index "
          f"{np.flatnonzero(F_dv)} = {basis[int(np.flatnonzero(F_dv)[0])]}")
    print(f"F_sf = x_1 x_2 x_3:    nonzero at index "
          f"{np.flatnonzero(F_sf)} = {basis[int(np.flatnonzero(F_sf)[0])]}")

    n_seeds = 2
    n_restarts = 4
    ms = [1, 2, 3, 4]

    results = {}
    for target_name, target in [("F_dv", F_dv), ("F_sf", F_sf)]:
        results[target_name] = {}
        for m in ms:
            seed_results = {"tied": [], "untied": [], "multileg": []}
            for seed in range(n_seeds):
                rng = np.random.default_rng(seed * 23 + m * 5)
                lt = fit_tied(target, basis, idx_map, n, k, m, rng,
                              n_restarts=n_restarts)
                # m=1 is the rank boundary; untied has more local minima there,
                # so use extra restarts to find the global Waring-rank-1 minimum
                untied_restarts = 12 if m == 1 else n_restarts
                lu = fit_untied(target, basis, idx_map, n, k, m, rng,
                                n_restarts=untied_restarts)
                lm = fit_multileg(target, basis, idx_map, n, k, m, rng,
                                  n_restarts=n_restarts)
                seed_results["tied"].append(lt)
                seed_results["untied"].append(lu)
                seed_results["multileg"].append(lm)
            results[target_name][m] = seed_results
            print(f"  {target_name}, m={m}: "
                  f"tied={np.median(seed_results['tied']):.2e}, "
                  f"untied={np.median(seed_results['untied']):.2e}, "
                  f"multileg={np.median(seed_results['multileg']):.2e}")

    # save
    out_results = {target: {str(m): {arch: losses for arch, losses in res.items()}
                            for m, res in target_data.items()}
                   for target, target_data in results.items()}
    (out_dir / "results_scope.json").write_text(json.dumps(out_results, indent=2))

    # ===== table ==============================================================
    def fmt(x):
        if x < 1e-15:
            return r"$<\!10^{-15}$"
        e = int(np.floor(np.log10(max(x, 1e-30))))
        m_ = x / 10 ** e
        return f"${m_:.1f}\\!\\times\\!10^{{{e}}}$"

    lines = [
        r"\begin{tabular}{@{}llcccc@{}}",
        r"\toprule",
        r"Target & Architecture & $m=1$ & $m=2$ & $m=3$ & $m=4$ \\",
        r"\midrule",
    ]
    target_labels = {"F_dv": r"$x_1^{k-1} x_2$ (role-asym.)",
                     "F_sf": r"$x_1 x_2 \cdots x_k$ (square-free)"}
    arch_labels = {"tied": "tied", "untied": "untied (det.-val.)",
                   "multileg": "multi-leg oracle"}
    for target_name in ["F_dv", "F_sf"]:
        for arch in ["tied", "untied", "multileg"]:
            row = ([target_labels[target_name] if arch == "tied" else ""]
                   + [arch_labels[arch]])
            for m in ms:
                losses = results[target_name][m][arch]
                row.append(fmt(np.median(np.maximum(losses, 1e-20))))
            lines.append(" & ".join(row) + r" \\")
        lines.append(r"\midrule")
    lines = lines[:-1] + [r"\bottomrule", r"\end{tabular}"]
    (out_dir / "table_scope.tex").write_text("\n".join(lines))

    # ===== figure =============================================================
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6), sharey=True)

    style = {
        "tied":     {"color": "#C73E3A", "marker": "o", "ls": "-",
                     "label": "tied"},
        "untied":   {"color": "#2C5F94", "marker": "s", "ls": "--",
                     "label": "untied (det.-val.)"},
        "multileg": {"color": "#3E8946", "marker": "D", "ls": ":",
                     "label": "multi-leg oracle"},
    }

    for ax, target_name, target_label in zip(
        axes,
        ["F_dv", "F_sf"],
        [r"Role-asymmetric: $F_{\mathrm{dv}} = x_1^{k-1} x_2$",
         r"Square-free: $F_{\mathrm{sf}} = x_1 x_2 \cdots x_k$"],
    ):
        target_data = results[target_name]
        sorted_ms = sorted(target_data.keys())
        for arch in ["tied", "untied", "multileg"]:
            med, lo, hi = [], [], []
            for m in sorted_ms:
                losses = np.maximum(np.array(target_data[m][arch]), 1e-20)
                med.append(np.median(losses))
                lo.append(losses.min())
                hi.append(losses.max())
            s = style[arch]
            ax.plot(sorted_ms, med, color=s["color"], marker=s["marker"],
                    linestyle=s["ls"], markersize=7, linewidth=1.6,
                    label=s["label"], zorder=3)
            ax.fill_between(sorted_ms, lo, hi, color=s["color"],
                            alpha=0.13, zorder=2)

        ax.set_xlabel("width $m$", fontsize=10)
        ax.set_yscale("log")
        ax.set_xticks(sorted_ms)
        ax.grid(True, which="both", alpha=0.25, linewidth=0.5)
        ax.set_ylim(1e-18, 10.0)
        ax.set_title(target_label, fontsize=11)
        ax.legend(fontsize=9, loc="lower left", framealpha=0.95)

    axes[0].set_ylabel("median MSE\n(coefficient space, $k=3$, $n=3$)",
                       fontsize=10)
    fig.suptitle(r"Scope boundary: untied (det.-val.) is rank-1 only when "
                 r"the target has one value role",
                 fontsize=11, y=1.02)
    fig.tight_layout()
    fig.savefig(out_dir / "fig_scope.pdf", bbox_inches="tight")
    fig.savefig(out_dir / "fig_scope.png", dpi=140, bbox_inches="tight")
    plt.close(fig)

    print(f"\nWrote {out_dir}/results_scope.json, table_scope.tex, "
          f"fig_scope.pdf, fig_scope.png")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results")