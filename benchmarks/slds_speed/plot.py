"""Plot results.jsonl: seconds vs timesteps, one panel per dataset, log-log.

Colour = library. Filled circles + solid line = 100 EM iterations (execution only); hollow
circles + dashed line = compile (dynamax only); triangles + dotted line = initialization (PCA + 50 ARHMM EM iterations).
Usage: python plot.py [results.jsonl] [out.png]
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
results = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "results.jsonl")
out = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, "speed.png")
rows = [json.loads(l) for l in open(results)]
num_iters = rows[0]["num_iters"]

SERIES = [  # (key, label, color); colour = library and precision, marker and line style = quantity
    (("ssm", "float64"), "ssm (float64)", "#eb6834"),
    (("dynamax", "float64"), "dynamax float64", "#1c5cab"),
]
DATASETS = [("slds", "SLDS of ssm's notebook 3 (K=5, D=2, N=10)"), ("rslds", "NASCAR rSLDS of ssm's notebook 4 (K=4, D=2, N=10)")]

fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
for ax, (dataset, title) in zip(axes, DATASETS):
    for (lib, dtype), label, color in SERIES:
        sel = sorted([r for r in rows if r["library"] == lib and r["dtype"] == dtype and r["dataset"] == dataset],
                     key=lambda r: r["num_timesteps"])
        if not sel:
            continue
        T = np.array([r["num_timesteps"] for r in sel], dtype=float)
        run = np.array([r["per_iter_s"] * num_iters for r in sel])
        ax.plot(T, run, marker="o", ms=8, lw=2, color=color, label=f"{label}: {num_iters} EM iterations")
        if lib == "dynamax":
            comp = np.array([r["fit_compile_s"] for r in sel], dtype=float)
            ax.plot(T, comp, marker="o", ms=8, lw=1.5, ls="--", color=color, mfc="white", mew=1.8,
                    label=f"{label}: compile")
        init = np.array([r["init_s"] for r in sel])
        ax.plot(T, init, marker="^", ms=8, lw=1.5, ls=":", color=color, label=f"{label}: initialization")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("timesteps T")
    ax.grid(True, which="major", color="#e4e2dc", lw=0.6)
    ax.grid(True, which="minor", color="#f0efec", lw=0.4)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
axes[0].set_ylabel("seconds (wall clock, CPU)")
handles, labels = axes[1].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=8.5, frameon=False, bbox_to_anchor=(0.5, -0.02))
fig.suptitle(f"SLDS Laplace EM: dynamax vs ssm, {num_iters} iterations from each library's PCA+ARHMM start "
             f"(single run; Apple M1 Pro, 10 cores)", fontsize=10)
fig.tight_layout(rect=(0, 0.14, 1, 0.96))
fig.savefig(out, dpi=150)
print(out)
