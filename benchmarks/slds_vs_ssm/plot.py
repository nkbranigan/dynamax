"""Plot results.jsonl as three figures, one panel per dataset. Color = library.

speed.png: seconds vs timesteps, log-log. Filled circles + solid line = 100 EM iterations (execution only); hollow
circles + dashed line = compile (dynamax only); triangles + dotted line = initialization (PCA + 50 ARHMM EM
iterations).
speed_total.png: total fit time vs timesteps: ssm = initialization + EM iterations; dynamax = initialization
(including its compile) + EM iterations (including their compile).
recovery.png: recovery as scored by dynamax/slds/ssm_recovery_test.py: discrete state recovery, continuous state
recovery, then the variance explained of each parameter group. Marker = number of timesteps. Within a column each
number of timesteps has one (ssm, dynamax) pair, joined by a line. Hollow markers are the controls (random
parameters, no learning). The axis runs from 0 to 1 and every score below 0 is drawn on one line under it. The
dynamics noise covariance Q is left out of the rSLDS panel.
Usage: python plot.py
"""
import json
import os

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
rows = [json.loads(line) for line in open(os.path.join(HERE, "results.jsonl"))]
num_iters = rows[0]["num_iters"]
COLORS = {"ssm": "#eb6834", "dynamax": "#1c5cab"}
DATASETS = {"slds": "SLDS of ssm's notebook 3 (K=5, D=2, N=10)",
            "rslds": "NASCAR rSLDS of ssm's notebook 4 (K=4, D=2, N=10)"}
TOTAL = {"ssm": "initialization + EM", "dynamax": "initialization + compile + EM"}
SKIP = {"rslds": {"Q"}}
FLOOR = -0.05                                               # scores below 0 are drawn here
MARKERS = dict(zip(sorted({r["num_timesteps"] for r in rows}), "os^Dv"))   # marker = number of timesteps


def select(library, dataset):
    """The rows of one library on one dataset, by increasing number of timesteps."""
    return sorted((r for r in rows if (r["library"], r["dataset"]) == (library, dataset)),
                  key=lambda r: r["num_timesteps"])


def save(fig, name, rect):
    fig.tight_layout(rect=rect)
    fig.savefig(os.path.join(HERE, name), dpi=150)
    print(os.path.join(HERE, name))


# ---- speed.png and speed_total.png
fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
total_fig, total_axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
for ax, total_ax, (dataset, title) in zip(axes, total_axes, DATASETS.items()):
    totals = {}
    for library, color in COLORS.items():
        sel = select(library, dataset)
        T = [r["num_timesteps"] for r in sel]
        compile_s = [r.get("init_compile_s", 0) + r.get("fit_compile_s", 0) for r in sel]   # 0 for ssm
        totals[library] = [r["init_s"] + r["fit_s"] + c for r, c in zip(sel, compile_s)]
        ax.plot(T, [r["fit_s"] for r in sel], marker="o", ms=8, lw=2, color=color,
                label=f"{library}: {num_iters} EM iterations")
        if library == "dynamax":
            ax.plot(T, compile_s, marker="o", ms=8, lw=1.5, ls="--", color=color, mfc="white", mew=1.8,
                    label=f"{library}: compile")
        ax.plot(T, [r["init_s"] for r in sel], marker="^", ms=8, lw=1.5, ls=":", color=color,
                label=f"{library}: initialization")
        total_ax.plot(T, totals[library], marker="o", ms=8, lw=2, color=color, label=f"{library}: {TOTAL[library]}")
    for t, ssm_s, dynamax_s in zip(T, totals["ssm"], totals["dynamax"]):
        total_ax.annotate(f"{ssm_s / dynamax_s:.1f}×", (t, dynamax_s), xytext=(11, -3), textcoords="offset points",
                          ha="left", va="center", fontsize=9, color="#52514e")
    for a in (ax, total_ax):
        a.set_xscale("log")
        a.set_yscale("log")
        a.set_title(title, fontsize=11)
        a.set_xlabel("timesteps T")
        a.grid(True, which="major", color="#e4e2dc", lw=0.6)
        a.grid(True, which="minor", color="#f0efec", lw=0.4)
        a.set_axisbelow(True)
        a.spines[["top", "right"]].set_visible(False)
axes[0].set_ylabel("seconds (wall clock, CPU)")
total_axes[0].set_ylabel("seconds (wall clock, CPU)")
fig.legend(*axes[1].get_legend_handles_labels(), loc="lower center", ncol=3, fontsize=8.5, frameon=False,
           bbox_to_anchor=(0.5, -0.02))
total_axes[0].legend(loc="upper left", fontsize=9, frameon=False)
fig.suptitle(f"SLDS Laplace EM: dynamax vs ssm, {num_iters} iterations from each library's PCA+ARHMM "
             f"initialization (float64; single run; Apple M1 Pro, 10 cores)", fontsize=10)
total_fig.suptitle(f"Total time to fit: PCA+ARHMM initialization and {num_iters} Laplace EM iterations "
                   f"(float64; single run; Apple M1 Pro, 10 cores; labels = ssm time / dynamax time)", fontsize=10)
save(fig, "speed.png", (0, 0.14, 1, 0.96))
save(total_fig, "speed_total.png", (0, 0, 1, 0.95))

# ---- recovery.png
fig, axes = plt.subplots(2, 1, figsize=(11, 7.5))
for ax, (dataset, title) in zip(axes, DATASETS.items()):
    sel = {library: select(library, dataset) for library in COLORS}
    groups = [k for k in sel["ssm"][0]["recovery"] if k not in SKIP.get(dataset, ())]
    num_sizes = len(sel["ssm"])
    slot = 0.8 / num_sizes                                  # each group holds one (ssm, dynamax) pair per T
    for i in range(num_sizes):
        for g, group in enumerate(groups):
            center = g + (i - (num_sizes - 1) / 2) * slot
            xs = {"ssm": center - 0.22 * slot, "dynamax": center + 0.22 * slot}
            ax.plot(list(xs.values()), [sel[library][i]["recovery"][group] for library in xs], color="#b0aea8",
                    lw=1.5, zorder=1)
            for library, color in COLORS.items():
                r = sel[library][i]
                marker, control = MARKERS[r["num_timesteps"]], r["control"][group]
                ax.plot([xs[library]], [r["recovery"][group]], marker=marker, ms=7, ls="none", color=color, zorder=3)
                ax.plot([xs[library]], [control if control >= 0 else FLOOR], marker=marker, ms=11, ls="none",
                        color=color, mfc="none", mew=1.5, zorder=4)
    ax.set_xticks(range(len(groups)),
                  [k.replace(" recovery", "").replace("_", " ").replace(" ", "\n") for k in groups])
    ax.set_xlim(-0.5, len(groups) - 0.5)
    ax.set_ylim(2 * FLOOR, 1.05)
    ax.set_yticks([FLOOR, 0, 0.2, 0.4, 0.6, 0.8, 1.0], ["< 0", "0", "0.2", "0.4", "0.6", "0.8", "1.0"])
    ax.axhspan(2 * FLOOR, 0, color="#f0efec", zorder=0)
    ax.set_title(title, fontsize=11)
    ax.set_ylabel("recovery")
    ax.grid(True, axis="y", color="#e4e2dc", lw=0.6)
    for g in range(1, len(groups)):                         # a heavier line after the two state scores
        ax.axvline(g - 0.5, color="#b0aea8" if g == 2 else "#e4e2dc", lw=1.2 if g == 2 else 0.6)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
axes[1].set_xlabel("state recovery, then parameter recovery by group "
                   "(within a column: T increasing left to right, ssm then dynamax)")
handles = [Line2D([], [], marker="o", ms=7, ls="none", color=color, label=library) for library, color in COLORS.items()]
handles += [Line2D([], [], marker=marker, ms=7, ls="none", color="#52514e", label=f"T = {T:,}")
            for T, marker in MARKERS.items()]
handles += [Line2D([], [], marker="o", ms=11, ls="none", color="#52514e", mfc="none", mew=1.5,
                   label="hollow: control (random parameters, no learning)")]
fig.legend(handles=handles, loc="lower center", ncol=len(handles), fontsize=9, frameon=False, bbox_to_anchor=(0.5, 0.0))
fig.suptitle(f"Recovery of the true states and parameters after PCA+ARHMM initialization and {num_iters} Laplace EM "
             f"iterations (single run per point)", fontsize=10)
save(fig, "recovery.png", (0, 0.07, 1, 0.96))
