"""Time lindermanlab/ssm's PCA+ARHMM initialization and Laplace EM on the benchmark's datasets.

Usage: <python with lindermanlab/ssm installed> bench_ssm.py
Fits every dataset of make_data.py, smallest first, and writes cpu_ssm.jsonl, one JSON line per dataset. ssm runs on
the CPU in float64. A short fit before the timed ones compiles ssm's numba kernels (about 12 s in a new
environment), so no timed fit pays for it. Outside the timed region each fit is scored by the recovery test's own
function, and so is a control: random parameters and no learning, with the states inferred by approximate_posterior
at its defaults. The control's parameters are those ssm's constructor draws at random, except the transition matrix:
ssm draws it as 0.95 I plus noise, which is also how the true one was drawn, so each of its rows is drawn from a
flat Dirichlet instead, as dynamax's prior does. The control's seed differs from the seeds that built the true
models (0 and 12345), which would reproduce some of the true parameters.
"""
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

RESULTS = os.path.join(HERE, "cpu_ssm.jsonl")
NUM_ITERS = 100                                           # the same as in bench_dynamax.py
CONTROL_SEED = 1


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
    """One row of results: the times of one fit, and the recovery scores of the fit and of the control."""
    saved = np.load(path(name, num_timesteps))
    datas = list(saved[f"recovery/{name}/y"])
    _, fit_model = NOTEBOOKS[name][0]()
    init_s, fit_s, model, q = fit(fit_model, datas, num_iters)

    npr.seed(CONTROL_SEED)
    control = fit_model()
    if hasattr(control.transitions, "log_Ps"):
        control.transitions.log_Ps = np.log(npr.dirichlet(np.ones(control.K), size=control.K))
    _, control_q = control.approximate_posterior(datas, method="laplace_em",
                                                 variational_posterior="structured_meanfield", verbose=0)

    from dynamax.slds.ssm_recovery_test import recovery   # imports jax, so only after the first timed fit

    def score(model, q):
        x_hat = q.mean_continuous_states
        z_hat = [model.most_likely_states(x, y) for x, y in zip(x_hat, datas)]
        return recovery(name, natural_params(model), np.concatenate(x_hat), np.concatenate(z_hat), saved)

    return {"library": "ssm", "dataset": name, "num_timesteps": num_timesteps, "num_iters": num_iters,
            "init_s": init_s, "fit_s": fit_s, "recovery": score(model, q), "control": score(control, control_q)}


if __name__ == "__main__":
    grid = sorted((T, name) for name in SIZES for T in SIZES[name])   # cheap fits first
    num_timesteps, name = grid[0]
    warm_up_data = list(np.load(path(name, num_timesteps))[f"recovery/{name}/y"])
    fit(NOTEBOOKS[name][0]()[1], warm_up_data, 2)                     # compiles numba's kernels
    open(RESULTS, "w").close()
    for num_timesteps, name in grid:
        row = bench(name, num_timesteps)
        with open(RESULTS, "a") as f:
            f.write(json.dumps(row) + "\n")
        print(f"ssm {name:6s} T={num_timesteps:<8d} init {row['init_s']:7.2f}s  fit {row['fit_s']:8.2f}s  "
              f"discrete state recovery {row['recovery']['discrete state recovery']:.3f}", flush=True)
