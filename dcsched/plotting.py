"""Static figures (PNG) for reports and the experiment pipeline.

Every figure is computed from recorded results passed in by the caller; no
value is hardcoded. Style follows the project's chart rules: thin marks, a
recessive grid, one axis per panel, text in ink colours (never series colour),
a legend whenever there are two or more series, and direct value labels so
colour is never the only carrier of identity.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from .palette import LIGHT, color_for, label_for, sort_policies, base_policy

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.edgecolor": LIGHT["grid"],
    "axes.labelcolor": LIGHT["ink_secondary"],
    "xtick.color": LIGHT["ink_secondary"],
    "ytick.color": LIGHT["ink_secondary"],
    "text.color": LIGHT["ink"],
    "axes.titlecolor": LIGHT["ink"],
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.facecolor": LIGHT["surface"],
    "axes.facecolor": LIGHT["surface"],
    "savefig.facecolor": LIGHT["surface"],
})

SEQ_CMAP = LinearSegmentedColormap.from_list("heat", ["#fdf1e7", "#f4a974", "#c2410c", "#5c1d04"])


def _wrap(label: str) -> str:
    """Break a policy label onto two lines so adjacent tick labels never collide."""
    label = label.replace(" (", "\n(")
    if "\n" not in label and " " in label:
        label = label.replace(" ", "\n", 1)
    elif "-" in label and "\n" not in label:
        label = label.replace("-", "-\n", 1)
    return label


def _style_axis(ax, ylabel=None, title=None):
    ax.grid(axis="y", color=LIGHT["grid"], linewidth=0.6)
    ax.set_axisbelow(True)
    if ylabel:
        ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title, loc="left", fontsize=10, fontweight="bold")


def _bars(ax, summary, metric, ylabel, title, fmt="{:.3f}"):
    names = sort_policies(list(summary["policy"]))
    rows = summary.set_index("policy").loc[names]
    means = rows[f"{metric}_mean"].values
    stds = rows[f"{metric}_std"].values
    x = np.arange(len(names))
    ax.bar(x, means, width=0.62, color=[color_for(n) for n in names],
           edgecolor=LIGHT["surface"], linewidth=1.5)
    ax.errorbar(x, means, yerr=stds, fmt="none", ecolor=LIGHT["ink_secondary"],
                elinewidth=0.9, capsize=3)
    span = max(means.max(), 1e-9)
    for xi, m, s in zip(x, means, stds):
        ax.text(xi, m + s + 0.02 * span, fmt.format(m), ha="center", va="bottom",
                fontsize=8, color=LIGHT["ink"])
    ax.set_xticks(x)
    ax.set_xticklabels([_wrap(label_for(n)) for n in names], fontsize=8)
    _style_axis(ax, ylabel, title)
    ax.set_ylim(0, (means + stds).max() * 1.18 + 1e-9)


def _legend(fig, names):
    from matplotlib.patches import Patch
    handles = [Patch(facecolor=color_for(n), label=label_for(n)) for n in names]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False,
               fontsize=8, bbox_to_anchor=(0.5, -0.01))


def fig_thermal_summary(summary, path, subtitle=""):
    names = sort_policies(list(summary["policy"]))
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.9))
    _bars(axes[0], summary, "hotspot_frequency", "fraction of steps", "(a) Hotspot frequency")
    _bars(axes[1], summary, "peak_temperature", "deg C", "(b) Peak server temperature", "{:.1f}")
    _bars(axes[2], summary, "mean_spatial_variance", "deg C$^2$", "(c) Spatial temperature variance", "{:.1f}")
    fig.suptitle("Thermal behaviour" + (f" - {subtitle}" if subtitle else ""),
                 x=0.01, ha="left", fontsize=11, fontweight="bold")
    _legend(fig, names)
    fig.tight_layout(rect=(0, 0.06, 1, 0.94))
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def fig_energy_work(summary, path, subtitle=""):
    names = sort_policies(list(summary["policy"]))
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.9))
    _bars(axes[0], summary, "it_energy_kwstep", "kW$\\cdot$step", "(a) IT energy", "{:.0f}")
    _bars(axes[1], summary, "cooling_energy_proxy", "kW$\\cdot$step (proxy)", "(b) Cooling-energy proxy", "{:.0f}")
    _bars(axes[2], summary, "tasks_completed", "tasks / episode", "(c) Tasks completed", "{:.0f}")
    fig.suptitle("Energy and work done" + (f" - {subtitle}" if subtitle else ""),
                 x=0.01, ha="left", fontsize=11, fontweight="bold")
    _legend(fig, names)
    fig.tight_layout(rect=(0, 0.06, 1, 0.94))
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def fig_performance(summary, path, subtitle=""):
    names = sort_policies(list(summary["policy"]))
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.9))
    _bars(axes[0], summary, "mean_wait_time", "steps", "(a) Mean wait before placement", "{:.2f}")
    _bars(axes[1], summary, "mean_response_time", "steps", "(b) Mean response time", "{:.1f}")
    _bars(axes[2], summary, "load_imbalance", "std-dev of CPU util.", "(c) Load imbalance", "{:.3f}")
    fig.suptitle("Workload performance" + (f" - {subtitle}" if subtitle else ""),
                 x=0.01, ha="left", fontsize=11, fontweight="bold")
    _legend(fig, names)
    fig.tight_layout(rect=(0, 0.06, 1, 0.94))
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def fig_temperature_traces(traces, safe_temp, path, seed_label=""):
    """Hottest-server temperature over time, one line per policy."""
    names = sort_policies(list(traces))
    fig, ax = plt.subplots(figsize=(9, 3.8))
    for n in names:
        t = traces[n]["max_temp"]
        ax.plot(np.arange(len(t)), t, color=color_for(n), linewidth=1.2,
                label=label_for(n), alpha=0.95)
    ax.axhline(safe_temp, color=LIGHT["threshold"], linewidth=1.0, linestyle="--")
    ax.text(3, safe_temp + 0.6, f"safe limit {safe_temp:.0f} C", color=LIGHT["ink_secondary"],
            va="bottom", ha="left", fontsize=8)
    _style_axis(ax, "hottest server (deg C)",
                "Peak facility temperature over one episode" + (f" ({seed_label})" if seed_label else ""))
    ax.set_xlabel("simulation step")
    ax.legend(frameon=False, fontsize=8, ncol=len(names), loc="upper center",
              bbox_to_anchor=(0.5, -0.22))
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def fig_server_heatmaps(traces, facility, safe_temp, path):
    """Mean temperature of every server (rack x slot), one panel per policy."""
    names = sort_policies(list(traces))
    n = len(names)
    spr = facility.servers_per_rack
    racks = facility.n_racks
    mats = [traces[nm]["server_temps"].mean(axis=0).reshape(racks, spr) for nm in names]
    vmin = min(m.min() for m in mats)
    vmax = max(m.max() for m in mats)
    cols = min(n, 3)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.6 * cols, 3.1 * rows), squeeze=False,
                             constrained_layout=True)
    im = None
    for k, (nm, m) in enumerate(zip(names, mats)):
        ax = axes[k // cols][k % cols]
        im = ax.imshow(m, cmap=SEQ_CMAP, vmin=vmin, vmax=vmax, aspect="auto")
        ax.set_title(label_for(nm), loc="left", fontsize=9, fontweight="bold")
        ax.set_xticks(range(spr))
        ax.set_yticks(range(racks))
        ax.set_yticklabels([f"rack {r}" for r in range(racks)], fontsize=7)
        ax.set_xticklabels([f"s{c}" for c in range(spr)], fontsize=7)
        for r in range(racks):
            for c in range(spr):
                v = m[r, c]
                ax.text(c, r, f"{v:.0f}", ha="center", va="center", fontsize=7,
                        color="#ffffff" if (v - vmin) / max(vmax - vmin, 1e-9) > 0.55 else LIGHT["ink"])
        for s in ax.spines.values():
            s.set_visible(False)
    for k in range(n, rows * cols):
        axes[k // cols][k % cols].axis("off")
    fig.suptitle("Mean temperature per server over the episode (deg C)", x=0.01, ha="left",
                 fontsize=11, fontweight="bold")
    fig.colorbar(im, ax=[a for row in axes for a in row], shrink=0.7, label="deg C")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def _rolling(y, w):
    y = np.asarray(y, dtype=float)
    if len(y) < 2:
        return y
    w = max(1, min(w, len(y)))
    k = np.ones(w) / w
    out = np.convolve(y, k, mode="valid")
    return np.concatenate([np.full(w - 1, np.nan), out])


def fig_learning_curves(curves, path, baseline_returns=None, window=10):
    """Training episode return vs timesteps. ``curves`` maps run label ->
    {"returns", "timesteps"}. ``baseline_returns`` maps policy -> mean held-out
    return (drawn as reference lines; evaluation traces differ from training
    traces so the comparison is indicative, not like-for-like)."""
    fig, ax = plt.subplots(figsize=(9, 3.8))
    for label in sorted(curves):
        c = curves[label]
        if len(c["returns"]) == 0:
            continue
        ax.plot(c["timesteps"], _rolling(c["returns"], window), color=color_for(label),
                linewidth=1.3, alpha=0.9, label=label)
    for name, val in (baseline_returns or {}).items():
        ax.axhline(val, color=color_for(name), linewidth=1.0, linestyle=":")
        ax.text(ax.get_xlim()[1], val, f" {label_for(name)} (held-out mean)", fontsize=7,
                color=LIGHT["ink_secondary"], ha="right", va="bottom")
    _style_axis(ax, "episode return", f"Training learning curves (rolling mean, window {window} episodes)")
    ax.set_xlabel("training timesteps")
    if ax.get_legend_handles_labels()[0]:
        ax.legend(frameon=False, fontsize=8, ncol=3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path
