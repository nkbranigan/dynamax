"""Time lindermanlab/ssm's PCA+ARHMM initialization and Laplace EM on the benchmark's datasets.

Usage: <python with lindermanlab/ssm installed> bench_ssm.py [--model slds|rslds]
Fits every dataset of make_data.py, smallest first, and writes one results file per model, cpu_ssm_slds.jsonl and
cpu_ssm_rslds.jsonl, one JSON line per dataset. With --model only that model's datasets are fitted, so the two models
can run at the same time. ssm runs on the CPU in float64. A short fit before the timed ones compiles ssm's numba
kernels (about 12 s in a new environment), so no timed fit pays for it. Outside the timed region each fit is scored by
the recovery test's own function.
"""
import argparse
import json
import os
import sys
import time
import warnings

import numpy as np
import numpy.random as npr

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "..", "dynamax", "slds", "fixtures"))
from export_ssm_fixtures import NOTEBOOKS, natural_params  # noqa: E402
from make_data import SIZES, path  # noqa: E402

NUM_ITERS = 100                                           # the same as in bench_dynamax.py


def results(name):
    return os.path.join(HERE, f"cpu_ssm_{name}.jsonl")


def fit(fit_model, datas, num_iters):
    npr.seed(0)
    model = fit_model()
    start = time.perf_counter()
    model.initialize(datas, num_init_iters=50)
    init_s = time.perf_counter() - start
    start = time.perf_counter()
    _, q = model.fit(datas, method="laplace_em", variational_posterior="structured_meanfield",
                     num_iters=num_iters, initialize=False, verbose=0)
    fit_s = time.perf_counter() - start
    return init_s, fit_s, model, q


def bench(name, num_timesteps, num_iters=NUM_ITERS):
    """One row of results: the times of one fit and its recovery scores."""
    saved = np.load(path(name, num_timesteps))
    datas = list(saved[f"recovery/{name}/y"])
    _, fit_model = NOTEBOOKS[name][0]()
    init_s, fit_s, model, q = fit(fit_model, datas, num_iters)

    from dynamax.slds.ssm_recovery_test import recovery   # imports jax, so only after the first timed fit
    x_hat = q.mean_continuous_states
    z_hat = [model.most_likely_states(x, y) for x, y in zip(x_hat, datas)]
    scores = recovery(name, natural_params(model), np.concatenate(x_hat), np.concatenate(z_hat), saved)

    return {"library": "ssm", "dataset": name, "num_timesteps": num_timesteps, "num_iters": num_iters,
            "init_s": init_s, "fit_s": fit_s, "recovery": scores}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["slds", "rslds"], help="fit only this model's datasets (default: both)")
    model = parser.parse_args().model
    names = [model] if model else list(SIZES)
    grid = sorted((T, name) for name in names for T in SIZES[name])   # cheap fits first
    num_timesteps, name = grid[0]
    warm_up_data = list(np.load(path(name, num_timesteps))[f"recovery/{name}/y"])
    fit(NOTEBOOKS[name][0]()[1], warm_up_data, 2)                     # compiles numba's kernels
    for name in names:
        open(results(name), "w").close()
    for num_timesteps, name in grid:
        row = bench(name, num_timesteps)
        with open(results(name), "a") as f:
            f.write(json.dumps(row) + "\n")
        print(f"ssm {name:6s} T={num_timesteps:<8d} init {row['init_s']:7.2f}s  fit {row['fit_s']:8.2f}s  "
              f"discrete state recovery {row['recovery']['discrete state recovery']:.3f}", flush=True)
