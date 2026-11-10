"""Plot recovery from results.jsonl, as scored by dynamax/slds/ssm_recovery_test.py, one panel per dataset: discrete
state recovery, continuous state recovery, then the variance explained of each parameter group. Colour = library, as
in plot.py; marker = number of timesteps. Within a column each number of timesteps has one (ssm, dynamax) pair,
joined by a line. Hollow rings are the controls (random parameters, no learning); with controls in the file the axis
runs from 0 to 1 and every score below 0 is drawn on one line under it. The dynamics noise covariance Q is left out.
Usage: python plot_recovery.py [results.jsonl] [out.png]
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
results = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "results.jsonl")
out = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, "recovery.png")
scored = [r for r in map(json.loads, open(results)) if "parameter_recovery" in r]
rows = [r for r in scored if not r.get("control")]
controls = [r for r in scored if r.get("control")]         # random parameters, no learning: hollow markers
num_iters = rows[0]["num_iters"]
FLOOR = -0.05                                               # with controls, scores below 0 are drawn here

SERIES = [(("ssm", "float64"), "ssm (float64)", "#eb6834"),
          (("dynamax", "float64"), "dynamax float64", "#1c5cab"),
]
DATASETS = [("slds", "SLDS of ssm's notebook 3 (K=5, D=2, N=10)"), ("rslds", "NASCAR rSLDS of ssm's notebook 4 (K=4, D=2, N=10)")]
SKIP = {"Q"}
SIZES = sorted({r["num_timesteps"] for r in rows})
MARKERS = dict(zip(SIZES, "os^Dv"))                          # marker = number of timesteps
STATES = ["discrete state", "continuous state"]


def scores(r):
    """One row's scores by column: the two state recovery scores, then parameter recovery per group."""
    return {"discrete state": r["state_accuracy"], "continuous state": r["latent_r2"], **r["parameter_recovery"]}


fig, axes = plt.subplots(2, 1, figsize=(11, 7.5))
for ax, (dataset, title) in zip(axes, DATASETS):
    sel = [r for r in rows if r["dataset"] == dataset]
    if not sel:
        ax.set_visible(False)
        continue
    sizes = sorted({r["num_timesteps"] for r in sel})
    groups = STATES + [k for k in sel[0]["parameter_recovery"] if k not in SKIP]
    slot = 0.8 / len(sizes)                                  # each group holds one (ssm, dynamax) pair per T
    for i, T in enumerate(sizes):
        pair = {r["library"]: scores(r) for r in sel if r["num_timesteps"] == T and r["dtype"] == "float64"}
        control = {r["library"]: scores(r) for r in controls if r["dataset"] == dataset and r["num_timesteps"] == T}
        for g, group in enumerate(groups):
            center = g + (i - (len(sizes) - 1) / 2) * slot
            xs = {lib: center + side * 0.22 * slot for (lib, _), side in zip([s[0] for s in SERIES], (-1, 1))}
            if len(pair) == 2:
                ax.plot([xs[lib] for lib in pair], [pair[lib][group] for lib in pair], color="#b0aea8", lw=1.5, zorder=1)
            for (lib, _), _, color in SERIES:
                if lib in pair:
                    ax.plot([xs[lib]], [pair[lib][group]], marker=MARKERS[T], ms=7, ls="none", color=color, zorder=3)
                if lib in control:
                    value = control[lib][group]
                    ax.plot([xs[lib]], [value if value >= 0 else FLOOR], marker=MARKERS[T], ms=11, ls="none", color=color,
                            mfc="none", mew=1.5, zorder=4)
    ax.set_xticks(range(len(groups)), [k.replace("_", " ").replace(" ", "\n") for k in groups])
    ax.set_xlim(-0.5, len(groups) - 0.5)
    if controls:
        ax.set_ylim(2 * FLOOR, 1.05)
        ax.set_yticks([FLOOR, 0, 0.2, 0.4, 0.6, 0.8, 1.0], ["< 0", "0", "0.2", "0.4", "0.6", "0.8", "1.0"])
        ax.axhspan(2 * FLOOR, 0, color="#f0efec", zorder=0)
    else:
        low = min(scores(r)[k] for r in sel for k in groups)
        ax.set_ylim(low - 0.1 * (1 - low), 1 + 0.1 * (1 - low))
    ax.set_title(title, fontsize=11)
    ax.set_ylabel("recovery")
    ax.grid(True, axis="y", color="#e4e2dc", lw=0.6)
    for g in range(1, len(groups)):
        ax.axvline(g - 0.5, color="#b0aea8" if g == len(STATES) else "#e4e2dc", lw=1.2 if g == len(STATES) else 0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
axes[1].set_xlabel("state recovery, then parameter recovery by group (within a column: T increasing left to right, ssm then dynamax)")
handles = [Line2D([], [], marker="o", ms=7, ls="none", color=color, label=label) for _, label, color in SERIES]
handles += [Line2D([], [], marker=MARKERS[T], ms=7, ls="none", color="#52514e", label=f"T = {T:,}") for T in SIZES]
if controls:
    handles += [Line2D([], [], marker="o", ms=11, ls="none", color="#52514e", mfc="none", mew=1.5,
                       label="hollow: control (random parameters, no learning)")]
fig.legend(handles=handles, loc="lower center", ncol=len(handles), fontsize=9, frameon=False, bbox_to_anchor=(0.5, 0.0))
fig.suptitle(f"Recovery of the true states and parameters after PCA+ARHMM initialization and {num_iters} Laplace EM iterations "
             f"(single run per point)", fontsize=10)
fig.tight_layout(rect=(0, 0.07, 1, 0.96))
fig.savefig(out, dpi=150)
print(out)
