"""Time lindermanlab/ssm's PCA+ARHMM initialization and Laplace EM on a saved dataset.

Usage: <python with lindermanlab/ssm installed> bench_ssm.py <slds|rslds> <num_timesteps> <num_iters>
Prints one JSON line. numba kernels are warmed up on a tiny model first so the timed run excludes
JIT compilation; ssm has no other fixed cost to separate.
"""
import json
import os
import sys
import time
import warnings

import numpy as np
import numpy.random as npr

warnings.filterwarnings("ignore")
import ssm  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "..", "dynamax", "slds", "fixtures"))
from export_ssm_fixtures import NOTEBOOKS, natural_params  # noqa: E402

DATA = os.path.join(HERE, "..", "..", "scratch_data")


def run(fit_model, data, num_iters, num_init_iters=50):
    npr.seed(0)
    model = fit_model()
    start = time.perf_counter()
    model.initialize([data], num_init_iters=num_init_iters)
    init_s = time.perf_counter() - start
    start = time.perf_counter()
    out = model.fit([data], method="laplace_em", variational_posterior="structured_meanfield",
                    num_iters=num_iters, initialize=False, verbose=0)
    fit_s = time.perf_counter() - start
    elbos, q = out
    return init_s, fit_s, np.asarray(elbos), model, q


if __name__ == "__main__":
    name, num_timesteps, num_iters = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    saved = np.load(os.path.join(DATA, f"{name}_T{num_timesteps}.npz"))
    data = saved["emissions"]
    _, fit_model = NOTEBOOKS[name][0]()
    run(fit_model, data[:200], 2, num_init_iters=2)                 # warm up numba
    init_s, fit_s, elbos, model, q = run(fit_model, data, num_iters)
    x_hat = q.mean_continuous_states[0]
    z_hat = model.most_likely_states(x_hat, data)                   # Viterbi with x at the posterior mean
    from recovery import scores  # noqa: E402    imports jax, so only after the timed fit
    print(json.dumps({
        "library": "ssm", "dtype": "float64", "dataset": name,
        "num_timesteps": num_timesteps, "num_iters": num_iters,
        "init_s": init_s, "fit_s": fit_s, "per_iter_s": fit_s / num_iters,
        "final_objective": float(elbos[-1]),
        "objective_monotone": bool(np.all(np.diff(elbos) >= -1e-3 * abs(elbos[-1]))),
        **scores(name, saved, natural_params(model), x_hat, z_hat),
    }))
