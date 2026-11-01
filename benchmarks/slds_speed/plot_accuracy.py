"""Plot recovery vs timesteps from results.jsonl: discrete state accuracy (top row) and affine-aligned latent R^2
(bottom row), one column per dataset. Colour = library, as in plot.py.
Usage: python plot_accuracy.py [results.jsonl] [out.png]
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
out = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, "accuracy.png")
rows = [json.loads(l) for l in open(results)]
num_iters = rows[0]["num_iters"]

SERIES = [(("ssm", "float64"), "ssm (float64)", "#eb6834"),
          (("dynamax", "float64"), "dynamax float64", "#1c5cab"),
]
DATASETS = [("slds", "SLDS of ssm's notebook 3 (K=5, D=2, N=10)"), ("rslds", "NASCAR rSLDS of ssm's notebook 4 (K=4, D=2, N=10)")]
MEASURES = [("state_accuracy", "discrete state accuracy\n(most likely states, best relabelling)"),
            ("latent_r2", "latent R² (affine-aligned posterior mean)")]

fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex="col")
for col, (dataset, title) in enumerate(DATASETS):
    for row, (measure, ylabel) in enumerate(MEASURES):
        ax = axes[row, col]
        for (lib, dtype), label, color in SERIES:
            sel = sorted([r for r in rows if r["library"] == lib and r["dtype"] == dtype and r["dataset"] == dataset and measure in r],
                         key=lambda r: r["num_timesteps"])
            if not sel:
                continue
            T = np.array([r["num_timesteps"] for r in sel], dtype=float)
            ax.plot(T, [r[measure] for r in sel], marker="o", ms=8, lw=2, color=color, label=label)
        ax.set_xscale("log")
        if measure == "latent_r2":
            r2_min = min(r["latent_r2"] for r in rows if r["dataset"] == dataset)
            ax.set_ylim(r2_min - 0.1 * (1 - r2_min), 1 + 0.1 * (1 - r2_min))
        else:
            ax.set_ylim(0.6, 1.01)
        ax.grid(True, which="major", color="#e4e2dc", lw=0.6)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        if row == 0:
            ax.set_title(title, fontsize=11)
        if row == 1:
            ax.set_xlabel("timesteps T")
        if col == 0:
            ax.set_ylabel(ylabel)
handles, labels = axes[0, 1].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=9, frameon=False, bbox_to_anchor=(0.5, -0.01))
fig.suptitle(f"Recovery of the sampled states and latents after PCA+ARHMM initialization and {num_iters} Laplace EM iterations "
             f"(single run per point)", fontsize=10)
fig.tight_layout(rect=(0, 0.05, 1, 0.96))
fig.savefig(out, dpi=150)
print(out)
