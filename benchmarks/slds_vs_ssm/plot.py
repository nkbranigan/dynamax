"""Plot the results as two speed figures per CPU and two recovery figures.

For each CPU in CPUS, reads that CPU's results files, cpu_<cpu>_ssm_<model>.jsonl and
cpu_<cpu>_dynamax_<model>.jsonl, and each GPU's, gpu_<gpu>_dynamax_<model>.jsonl (the speed figures draw dynamax on
each GPU as a further series). A series is drawn at the sizes its files have, so one that stops early, or is not
there, only shortens or drops its line.

speed_<cpu>.png: seconds vs timesteps, log-log, with a row of panels per dataset and a panel per series. Each panel
has a line for the 100 EM iterations (execution only), one for compile (dynamax only), and one for initialization
(PCA + 50 ARHMM EM iterations). Color = part of the fit.
speed_total_<cpu>.png: total fit time vs timesteps, with a panel per dataset and a line per series: ssm =
initialization + EM iterations; dynamax = initialization (including its compile) + EM iterations (including their
compile). Color = series; circles are CPU runs and squares are GPU runs. The labels are ssm's time on that CPU over
the series' time, at the sizes ssm has, stacked below the points with the smallest ratio on top.
The recovery figures are drawn for one CPU only (RECOVERY_CPU): the scores are essentially the same on both.
recovery_summary_<cpu>.png: the mean of a fit's recovery scores (the ones recovery_<cpu>.png shows) vs timesteps,
with a panel per dataset and a line per series, drawn as in speed_total_<cpu>.png.
recovery_<cpu>.png: recovery as scored by dynamax/slds/ssm_recovery_test.py, with a row of panels per dataset and a
panel per score: discrete state recovery, continuous state recovery, then the parameter recovery scores. Each panel
is recovery_summary_<cpu>.png's, in small, for that one score. The dynamics noise covariance Q is left out of the
rSLDS row.
Usage: python plot.py
"""
import json
import os
import sys

import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from make_data import SEQUENCE_LENGTH  # noqa: E402

CPUS = {"epyc_7502": "AMD EPYC 7502", "epyc_8224p": "AMD EPYC 8224P"}     # --name of a run: its label; 8 cores each
DPI = 1000                                                # of the pictures
RECOVERY_CPU = "epyc_8224p"                               # the recovery figures are drawn for this CPU only
GPUS = {"rtx_2080ti": "RTX 2080 Ti", "h200": "H200"}
GPU_SERIES = [f"Dynamax on {label}" for label in GPUS.values()]
COLORS = dict(zip(["ssm", "Dynamax"] + GPU_SERIES, ["#eb6834", "#1c5cab", "#6fb1e6", "#a06cd5"]))
DATASETS = {"slds": "SLDS of ssm's notebook 3", "rslds": "rSLDS of ssm's notebook 4 (NASCAR)"}
SKIP = {"rslds": {"Q"}}
PART_COLORS = ["#3a3a3a", "#b8860b", "#2a8f6f"]           # the detailed figure's EM, compile, and initialization


def load(prefix):
    """The rows of one series, from its results files (one per model); none when the files are not there."""
    files = [os.path.join(HERE, f"{prefix}_{model}.jsonl") for model in SEQUENCE_LENGTH]
    return [json.loads(line) for file in files if os.path.exists(file) for line in open(file)]


def select(rows, dataset):
    """One series' rows on one dataset, keyed by number of timesteps, smallest first."""
    return {r["num_timesteps"]: r for r in sorted(rows, key=lambda r: r["num_timesteps"]) if r["dataset"] == dataset}


def save(fig, name, rect, **padding):
    fig.tight_layout(rect=rect, **padding)
    fig.savefig(os.path.join(HERE, name), dpi=DPI)
    plt.close(fig)
    print(os.path.join(HERE, name))


def marker(series):
    """The marker of a series where each series is one line: a square for a GPU run, a circle for a CPU run."""
    return dict(marker="s", ms=7) if series in GPU_SERIES else dict(marker="o", ms=8)


def log_axes(ax, yscale="log"):
    """The look the figures over data size share: a log x axis, the major grid only, and no top or right border."""
    ax.set_xscale("log")
    ax.set_yscale(yscale)
    ax.grid(True, which="major", color="#e4e2dc", lw=0.6)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)


def time_marks(axes):
    """Faint lines at familiar lengths of time, named at the right of the last panel."""
    low, high = axes[0].get_ylim()
    marks = {seconds: name for seconds, name in {60: "1 minute", 3600: "1 hour", 86400: "1 day"}.items()
             if low < seconds < high}
    for ax in axes:
        for seconds in marks:
            ax.axhline(seconds, color="#b0aea8", lw=0.8, ls=":", zorder=0)
    right = axes[-1].secondary_yaxis("right")
    right.set_yticks(list(marks), list(marks.values()), fontsize=8, color="#7a7873")
    right.minorticks_off()
    right.tick_params(length=0)
    right.spines["right"].set_visible(False)


def detail_figure(cpu, names, series_rows, num_iters):
    """speed_<cpu>.png: a row of panels per dataset and a panel per run, with a line for each part of the fit."""
    fig, axes = plt.subplots(len(DATASETS), len(series_rows), figsize=(12, 6.4), sharex="row", sharey=True,
                             squeeze=False)
    for row, (dataset, title) in zip(axes, DATASETS.items()):
        for ax, (series, rows) in zip(row, series_rows.items()):
            sel = select(rows, dataset)
            compiled = [r for r in sel.values() if "fit_compile_s" in r]               # none of ssm's rows
            parts = {f"{num_iters} EM iterations": [r["fit_s"] for r in sel.values()],
                     "compile": [r["init_compile_s"] + r["fit_compile_s"] for r in compiled],
                     "initialization": [r["init_s"] for r in sel.values()]}
            for (part, seconds), color in zip(parts.items(), PART_COLORS):
                if seconds:
                    ax.plot(list(sel), seconds, marker="o", ms=6, lw=2, color=color, label=part)
            log_axes(ax)
        row[0].set_ylabel(f"{title}\ntime (seconds)", fontsize=9)
    for ax, series in zip(axes[0], series_rows):
        ax.set_title(names[series], fontsize=10)
    axes[-1, 0].set_xlabel("data size")
    for row in axes:
        time_marks(row)
    legend = max((ax.get_legend_handles_labels() for ax in axes.flat), key=lambda entries: len(entries[0]))
    fig.legend(*legend, loc="lower center", ncol=len(PART_COLORS), fontsize=9, frameon=False,
               bbox_to_anchor=(0.5, -0.01))
    save(fig, f"speed_{cpu}.png", (0, 0.04, 1, 1))


def total_figure(cpu, names, series_rows):
    """speed_total_<cpu>.png: a panel per dataset and a line per run, labelled with its speedup over ssm."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.7), sharey=True)                # tall, so the stacked labels fit
    lowest = float("inf")                                   # the smallest total time in either panel
    for ax, (dataset, title) in zip(axes, DATASETS.items()):
        totals = {}                                         # series: {number of timesteps: total seconds}
        for series, rows in series_rows.items():
            sel = select(rows, dataset)
            totals[series] = {t: r["init_s"] + r["fit_s"] + r.get("init_compile_s", 0) + r.get("fit_compile_s", 0)
                              for t, r in sel.items()}      # ssm has no compile
            ax.plot(list(sel), list(totals[series].values()), **marker(series), lw=2, color=COLORS[series],
                    label=names[series])
        for t, ssm_s in totals.get("ssm", {}).items():      # labels: ssm's time over each series' time, in its color
            points = sorted(((seconds[t], series) for series, seconds in totals.items()
                             if series != "ssm" and t in seconds), reverse=True)       # this size's points, top first
            stacks = []                                     # points less than 3.5x apart share one stack of labels,
            for seconds, series in points:                  # as a stack would not fit between them
                if stacks and stacks[-1][-1][0] < 3.5 * seconds:
                    stacks[-1].append((seconds, series))
                else:
                    stacks.append([(seconds, series)])
            for stack in stacks:                            # below the stack's lowest point, the smallest ratio first
                for k, (seconds, series) in enumerate(stack):
                    ratio = ssm_s / seconds                 # fewer decimals for the larger ratios
                    text = f"{ratio:.0f}×" if ratio >= 100 else f"{ratio:.1f}×" if ratio >= 1 else f"{ratio:.2f}×"
                    ax.annotate(text, (t, stack[-1][0]), xytext=(0, -15 - 11 * k), textcoords="offset points",
                                ha="center", va="top", fontsize=9, fontweight="bold",
                                color="#52514e" if len(totals) == 2 else COLORS[series])
        lowest = min([lowest] + [min(seconds.values()) for seconds in totals.values() if seconds])
        log_axes(ax)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("data size")
        ax.margins(x=0.08)                                  # room for the labels below the first and last points
    axes[0].set_ylabel("total time to fit (seconds)")
    axes[0].set_ylim(bottom=lowest / 3)                     # room below the lowest points for their labels
    time_marks(axes)
    fig.legend(*axes[1].get_legend_handles_labels(), loc="lower center", ncol=len(series_rows), fontsize=9,
               frameon=False, bbox_to_anchor=(0.5, -0.01))
    save(fig, f"speed_total_{cpu}.png", (0, 0.05, 1, 1))


def recovery_summary_figure(cpu, names, series_rows):
    """recovery_summary_<cpu>.png: a panel per dataset and a line per series, the mean of its recovery scores."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
    for ax, (dataset, title) in zip(axes, DATASETS.items()):
        for series, rows in series_rows.items():
            sel = select(rows, dataset)
            scores = [[score for group, score in r["recovery"].items() if group not in SKIP.get(dataset, ())]
                      for r in sel.values()]                # the scores recovery_<cpu>.png shows, per size
            ax.plot(list(sel), [sum(s) / len(s) for s in scores], **marker(series), lw=2, color=COLORS[series],
                    label=names[series])
        log_axes(ax, yscale="linear")
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("data size")
    axes[0].set_ylabel("mean recovery score")
    axes[0].set_ylim(0, 1.05)                               # as in recovery_<cpu>.png
    fig.legend(*axes[1].get_legend_handles_labels(), loc="lower center", ncol=len(series_rows), fontsize=9,
               frameon=False, bbox_to_anchor=(0.5, -0.01))
    save(fig, f"recovery_summary_{cpu}.png", (0, 0.06, 1, 1))


def recovery_figure(cpu, names, series_rows):
    """recovery_<cpu>.png: recovery_summary_<cpu>.png's panels in small, one per recovery score, a row per dataset."""
    sel = {dataset: {series: select(rows, dataset) for series, rows in series_rows.items()} for dataset in DATASETS}
    scores = {}                                             # dataset: the scores to draw, in the results files' order
    for dataset in DATASETS:
        rows = [r for of_series in sel[dataset].values() for r in of_series.values()]
        scores[dataset] = [s for s in rows[0]["recovery"] if s not in SKIP.get(dataset, ())] if rows else []
    fig, axes = plt.subplots(len(DATASETS), max(map(len, scores.values())), figsize=(12, 5.8), sharex="row",
                             sharey=True, squeeze=False)
    for row, (dataset, title) in zip(axes, DATASETS.items()):
        sizes = sorted({T for of_series in sel[dataset].values() for T in of_series})
        for ax, score in zip(row, scores[dataset]):
            for series, of_series in sel[dataset].items():
                style = marker(series)
                ax.plot(list(of_series), [r["recovery"][score] for r in of_series.values()], marker=style["marker"],
                        ms=style["ms"] - 3.5, lw=1.5, color=COLORS[series], label=names[series])
            log_axes(ax, yscale="linear")
            ax.spines[["left", "bottom"]].set_linewidth(0.2)                # thin, with so many panels
            ax.set_xticks(sizes)                            # a tick at every data size, named in the first panel only
            ax.minorticks_off()
            ax.tick_params(labelbottom=False, width=0.2)
            ax.set_title(score.replace(" recovery", "").replace("_", " ").replace(" ", "\n", 1), fontsize=9)
        for ax in row[len(scores[dataset]):]:               # this dataset has fewer scores than the other
            ax.set_axis_off()
        row[0].tick_params(labelbottom=True, labelsize=7)
        plt.setp(row[0].get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        row[0].set_xlabel("data size", fontsize=9)
        row[0].set_ylabel(f"{title}\nrecovery score", fontsize=9)
    axes[0, 0].set_ylim(0, 1.05)                            # as in recovery_summary_<cpu>.png
    fig.legend(*axes[-1, 0].get_legend_handles_labels(), loc="lower center", ncol=len(series_rows), fontsize=9,
               frameon=False, bbox_to_anchor=(0.5, -0.01))
    save(fig, f"recovery_{cpu}.png", (0, 0.04, 1, 1), w_pad=0.2)   # the panels close together


for cpu, cpu_label in CPUS.items():
    series_rows = {"ssm": load(f"cpu_{cpu}_ssm"), "Dynamax": load(f"cpu_{cpu}_dynamax"),
                   **{series: load(f"gpu_{gpu}_dynamax") for gpu, series in zip(GPUS, GPU_SERIES)}}
    series_rows = {series: rows for series, rows in series_rows.items() if rows}       # the series with results
    if not {"ssm", "Dynamax"} & set(series_rows):          # no results from this CPU
        continue
    num_iters = {r["num_iters"] for rows in series_rows.values() for r in rows}
    assert len(num_iters) == 1, "the files differ in num_iters"
    names = {"ssm": f"ssm (CPU: {cpu_label})", "Dynamax": f"Dynamax (CPU: {cpu_label})",       # in titles and legends
             **{series: f"Dynamax (GPU: {gpu_label})" for series, gpu_label in zip(GPU_SERIES, GPUS.values())}}
    detail_figure(cpu, names, series_rows, *num_iters)
    total_figure(cpu, names, series_rows)
    if cpu == RECOVERY_CPU:
        recovery_summary_figure(cpu, names, series_rows)
        recovery_figure(cpu, names, series_rows)
