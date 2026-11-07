"""Re-plot Figures 3-7 from the JSON results with the paper's styling.

Figure 2 is drawn by fig2_indefinite_quadratic.py itself.

Usage:  python experiments/make_figures.py [results directory, default: results]
"""
import json
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator
from pathlib import Path

# ===== STYLE ===================================================================
# Set rcParams ONCE so all figures share the same style
plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.5,
    "axes.axisbelow": True,
    "lines.linewidth": 1.6,
    "lines.markersize": 6,
})

COLOR_TIED = "#C73E3A"
COLOR_UNTIED = "#2C5F94"

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "results")


def style_log_y(ax, ymin=1e-16, ymax=10):
    """Apply consistent log-y styling: ~5 major ticks, no minor labels."""
    ax.set_yscale("log")
    ax.set_ylim(ymin, ymax)
    # Major ticks every 4 decades; suppress minor ticks
    ax.yaxis.set_major_locator(LogLocator(base=10, numticks=5))
    # Suppress minor ticks by setting them to nothing (avoids divide-by-zero)
    ax.tick_params(axis="y", which="minor", left=False)


def clip_for_plot(arr, floor=1e-16):
    """Clip values to a visible floor so they appear at the bottom of log plot."""
    return np.maximum(np.asarray(arr, dtype=float), floor)


# ===== 1. Quadratic figure ===================================================
def plot_width_sweep():
    results = json.loads((OUT / "results_width_sweep.json").read_text())
    ks = sorted([int(k) for k in results.keys()])

    fig, axes = plt.subplots(1, len(ks), figsize=(11, 2.8), sharey=True)

    for ax, k in zip(axes, ks):
        kd = results[str(k)]
        ms = kd["ms"]
        tied_per_seed = kd["tied_per_seed"]
        untied_per_seed = kd["untied_per_seed"]

        tied_med, tied_lo, tied_hi = [], [], []
        untied_med, untied_lo, untied_hi = [], [], []
        for m in ms:
            t = clip_for_plot(tied_per_seed[str(m)])
            u = clip_for_plot(untied_per_seed[str(m)])
            tied_med.append(np.median(t))
            tied_lo.append(t.min()); tied_hi.append(t.max())
            untied_med.append(np.median(u))
            untied_lo.append(u.min()); untied_hi.append(u.max())

        ax.plot(ms, tied_med, color=COLOR_TIED, marker="o", label="tied")
        ax.fill_between(ms, tied_lo, tied_hi, color=COLOR_TIED, alpha=0.15)
        ax.plot(ms, untied_med, color=COLOR_UNTIED, marker="s", label="untied")
        ax.fill_between(ms, untied_lo, untied_hi, color=COLOR_UNTIED, alpha=0.15)

        ax.axvline(x=k, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)

        ax.set_xticks(ms[::1] if len(ms) <= 5 else ms[::2])
        ax.set_xlabel(r"width $m$")
        style_log_y(ax)
        # Annotation for k value on top-right
        ax.text(0.96, 0.95, f"$k = {k}$", transform=ax.transAxes,
                fontsize=11, ha="right", va="top",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          edgecolor="lightgray", alpha=0.9))

    axes[0].set_ylabel("median MSE")
    axes[-1].legend(loc="lower left", framealpha=0.95)

    fig.tight_layout()
    fig.savefig(OUT / "fig_width_sweep.pdf", bbox_inches="tight")
    fig.savefig(OUT / "fig_width_sweep.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print("  fig_width_sweep")


# ===== 3. Pure-vs-mixed =======================================================
def plot_pure_vs_mixed():
    data = json.loads((OUT / "results_pure_vs_mixed.json").read_text())
    ks = sorted([int(k) for k in data["tied"]["pure"].keys()])

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.4), sharey=True)

    for ax, tgt_name in zip(axes, ["pure", "mixed"]):
        for arch, marker, color in [("tied", "o", COLOR_TIED),
                                      ("untied", "s", COLOR_UNTIED)]:
            losses_by_k = [np.maximum(np.array(data[arch][tgt_name][str(k)]),
                                       1e-20) for k in ks]
            medians = np.array([np.median(L) for L in losses_by_k])
            mins = np.array([L.min() for L in losses_by_k])
            maxs = np.array([L.max() for L in losses_by_k])
            ax.plot(ks, medians, color=color, marker=marker, label=arch)
            ax.fill_between(ks, mins, maxs, color=color, alpha=0.15)

        if tgt_name == "pure":
            ax.set_title(r"pure power $x_1^k$", fontsize=11, pad=4)
        else:
            ax.set_title(r"role-asymmetric $x_1^{k-1} x_2$", fontsize=11, pad=4)
        ax.set_xlabel(r"polynomial degree $k$")
        ax.set_xticks(ks)
        style_log_y(ax)

    axes[0].set_ylabel("median MSE")
    axes[1].legend(loc="lower right", framealpha=0.95)

    fig.tight_layout()
    fig.savefig(OUT / "fig_pure_vs_mixed.pdf", bbox_inches="tight")
    fig.savefig(OUT / "fig_pure_vs_mixed.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print("  fig_pure_vs_mixed")


# ===== 4. Architectural lift ==================================================
def plot_architectural_lift():
    data = json.loads((OUT / "results_architectural.json").read_text())
    ks = sorted([int(k) for k in data["tied-ReLU2"].keys()])
    ms = sorted([int(k) for k in data["tied-ReLU2"][str(ks[0])].keys()])

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.4), sharey=True)

    style = {
        "tied-ReLU2":    {"color": "#E25C28", "marker": "o", "ls": "-",
                          "label": r"tied $\mathrm{ReLU}^2$"},
        "tied-SwiGLU":    {"color": "#F4A742", "marker": "^", "ls": "-",
                          "label": r"tied SwiGLU"},
        "untied-ReGLU":  {"color": "#2A8C8B", "marker": "s", "ls": "--",
                          "label": "untied ReGLU"},
        "untied-SwiGLU": {"color": "#3E8946", "marker": "D", "ls": "--",
                          "label": "untied SwiGLU"},
    }

    for ax, k in zip(axes, ks):
        for arch in ["tied-ReLU2", "tied-SwiGLU", "untied-ReGLU", "untied-SwiGLU"]:
            losses_by_m = [np.maximum(np.array(data[arch][str(k)][str(m)]),
                                       1e-20) for m in ms]
            medians = [np.median(L) for L in losses_by_m]
            mins = [L.min() for L in losses_by_m]
            maxs = [L.max() for L in losses_by_m]
            s = style[arch]
            ax.plot(ms, medians, color=s["color"], marker=s["marker"],
                    linestyle=s["ls"], label=s["label"])
            ax.fill_between(ms, mins, maxs, color=s["color"], alpha=0.13)

        target_str = (r"x_1 x_2" if k == 2 else
                      r"x_1^2 x_2" if k == 3 else
                      f"x_1^{{{k-1}}} x_2")
        ax.set_title(rf"target ${target_str}$, $k\!=\!{k}$",
                     fontsize=11, pad=4)
        ax.set_xlabel(r"width $m$")
        ax.set_xscale("log", base=2)
        ax.set_xticks(ms)
        ax.set_xticklabels(ms)
        style_log_y(ax)

    axes[0].set_ylabel("median MSE")
    axes[1].legend(loc="lower left", framealpha=0.95, ncol=1)

    fig.tight_layout()
    fig.savefig(OUT / "fig_architectural_lift.pdf", bbox_inches="tight")
    fig.savefig(OUT / "fig_architectural_lift.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print("  fig_architectural_lift")


# ===== 5. Matched parameter ==================================================
def plot_matched():
    data = json.loads((OUT / "results_matched.json").read_text())
    ks = sorted([int(k) for k in data.keys()])
    # Matched-param data has per-arch widths: tied at 2m, untied at m
    # Use untied widths as x-axis
    untied_widths = sorted([int(k) for k in data[str(ks[0])]["untied-ReGLU"].keys()])

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.4), sharey=True)

    style = {
        "tied-ReLU2":    {"color": "#E25C28", "marker": "o", "ls": "-",
                          "label": r"tied $\mathrm{ReLU}^2$ (at $2m$)"},
        "tied-SwiGLU":    {"color": "#F4A742", "marker": "^", "ls": "-",
                          "label": r"tied SwiGLU (at $2m$)"},
        "untied-ReGLU":  {"color": "#2A8C8B", "marker": "s", "ls": "--",
                          "label": r"untied ReGLU (at $m$)"},
        "untied-SwiGLU": {"color": "#3E8946", "marker": "D", "ls": "--",
                          "label": r"untied SwiGLU (at $m$)"},
    }

    def extract_losses(entries):
        """Each entry is either {loss: x, n_params: n} or just a float."""
        out = []
        for e in entries:
            if isinstance(e, dict):
                out.append(e["loss"])
            else:
                out.append(float(e))
        return clip_for_plot(out)

    for ax, k in zip(axes, ks):
        for arch in ["tied-ReLU2", "tied-SwiGLU", "untied-ReGLU", "untied-SwiGLU"]:
            arch_data = data[str(k)][arch]
            # tied keys are 2m, untied keys are m
            if arch.startswith("tied"):
                x_vals = [m for m in untied_widths if str(2*m) in arch_data]
                losses_per_x = [extract_losses(arch_data[str(2*m)])
                                for m in x_vals]
            else:
                x_vals = [m for m in untied_widths if str(m) in arch_data]
                losses_per_x = [extract_losses(arch_data[str(m)])
                                for m in x_vals]
            medians = [np.median(L) for L in losses_per_x]
            mins = [L.min() for L in losses_per_x]
            maxs = [L.max() for L in losses_per_x]
            s = style[arch]
            ax.plot(x_vals, medians, color=s["color"], marker=s["marker"],
                    linestyle=s["ls"], label=s["label"])
            ax.fill_between(x_vals, mins, maxs, color=s["color"], alpha=0.13)

        target_str = r"x_1 x_2" if k == 2 else r"x_1^2 x_2"
        ax.set_title(rf"target ${target_str}$, $k\!=\!{k}$",
                     fontsize=11, pad=4)
        ax.set_xlabel(r"untied width $m$")
        ax.set_xscale("log", base=2)
        ax.set_xticks(untied_widths)
        ax.set_xticklabels(untied_widths)
        style_log_y(ax)

    axes[0].set_ylabel("median MSE")
    axes[1].legend(loc="lower left", framealpha=0.95, ncol=1)

    fig.tight_layout()
    fig.savefig(OUT / "fig_matched.pdf", bbox_inches="tight")
    fig.savefig(OUT / "fig_matched.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print("  fig_matched")


# ===== 6. Scope boundary ======================================================
def plot_scope():
    results = json.loads((OUT / "results_scope.json").read_text())

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.4), sharey=True)

    style = {
        "tied":     {"color": COLOR_TIED, "marker": "o", "ls": "-",
                     "label": "tied"},
        "untied":   {"color": COLOR_UNTIED, "marker": "s", "ls": "--",
                     "label": "untied (det.--val.)"},
        "multileg": {"color": "#3E8946", "marker": "D", "ls": ":",
                     "label": "multi-leg oracle"},
    }

    for ax, target_key, target_label in zip(
        axes,
        ["F_dv", "F_sf"],
        [r"role-asymmetric $F_{\mathrm{dv}} = x_1^{k-1} x_2$",
         r"square-free $F_{\mathrm{sf}} = x_1 x_2 \cdots x_k$"],
    ):
        target_data = results[target_key]
        ms = sorted([int(m) for m in target_data.keys()])
        for arch in ["tied", "untied", "multileg"]:
            med, lo, hi = [], [], []
            for m in ms:
                losses = clip_for_plot(target_data[str(m)][arch])
                med.append(np.median(losses))
                lo.append(losses.min()); hi.append(losses.max())
            s = style[arch]
            ax.plot(ms, med, color=s["color"], marker=s["marker"],
                    linestyle=s["ls"], label=s["label"])
            ax.fill_between(ms, lo, hi, color=s["color"], alpha=0.13)

        ax.set_title(target_label, fontsize=11, pad=4)
        ax.set_xlabel(r"width $m$")
        ax.set_xticks(ms)
        style_log_y(ax)

    axes[0].set_ylabel("median MSE")
    axes[1].legend(loc="lower right", framealpha=0.95)

    fig.tight_layout()
    fig.savefig(OUT / "fig_scope.pdf", bbox_inches="tight")
    fig.savefig(OUT / "fig_scope.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print("  fig_scope")


if __name__ == "__main__":
    print("Re-plotting all figures with consistent paper-ready style:")
    plot_width_sweep()
    plot_pure_vs_mixed()
    plot_architectural_lift()
    plot_matched()
    plot_scope()
    print("Done.")