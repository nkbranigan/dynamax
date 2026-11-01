"""Plot total fit time vs timesteps: ssm = initialization + EM iterations; dynamax (float64) =
initialization (including its compile) + EM iterations (including their compile).

Usage: python plot_total.py [results.jsonl] [out.png]
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
out = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, "speed_total.png")
rows = [json.loads(l) for l in open(results)]
num_iters = rows[0]["num_iters"]

DATASETS = [("slds", "SLDS of ssm's notebook 3 (K=5, D=2, N=10)"), ("rslds", "NASCAR rSLDS of ssm's notebook 4 (K=4, D=2, N=10)")]
fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
for ax, (dataset, title) in zip(axes, DATASETS):
    ssm = sorted([r for r in rows if r["library"] == "ssm" and r["dataset"] == dataset], key=lambda r: r["num_timesteps"])
    dyn = sorted([r for r in rows if r["library"] == "dynamax" and r["dtype"] == "float64" and r["dataset"] == dataset],
                 key=lambda r: r["num_timesteps"])
    T_s = np.array([r["num_timesteps"] for r in ssm], float)
    tot_s = np.array([r["init_s"] + r["fit_s"] for r in ssm])
    T_d = np.array([r["num_timesteps"] for r in dyn], float)
    # rows timed by first call vs warm call (older) or by ahead-of-time compile plus one run (newer)
    tot_d = np.array([r["init_first_s"] + r["fit_first_s"] if "fit_first_s" in r else
                      r["init_compile_s"] + r["init_s"] + r["fit_compile_s"] + r["fit_s"] for r in dyn])
    ax.plot(T_s, tot_s, marker="o", ms=8, lw=2, color="#eb6834", label="ssm: initialization + EM")
    ax.plot(T_d, tot_d, marker="o", ms=8, lw=2, color="#1c5cab", label="dynamax float64: initialization + compile + EM")
    for t, a, b in zip(T_d, tot_s, tot_d):
        ax.annotate(f"{a / b:.1f}×", (t, b), xytext=(11, -3), textcoords="offset points", ha="left", va="center",
                    fontsize=9, color="#52514e")
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
axes[0].legend(loc="upper left", fontsize=9, frameon=False)
fig.suptitle(f"Total time to fit: PCA+ARHMM initialization and {num_iters} Laplace EM iterations "
             f"(single run; Apple M1 Pro, 10 cores; labels = ssm time / dynamax time)", fontsize=10)
fig.tight_layout(rect=(0, 0, 1, 0.95))
fig.savefig(out, dpi=150)
print(out)
