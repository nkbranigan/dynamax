"""Time lindermanlab/ssm's PCA+ARHMM initialization and Laplace EM on a saved dataset.

Usage: ssm_venv/bin/python bench_ssm.py <slds|rslds> <num_timesteps> <num_iters>
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from accuracy import latent_r2, state_accuracy  # noqa: E402

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "scratch_data")


def build(name, N=10):
    if name == "slds":
        return ssm.SLDS(N, 5, 2, transitions="standard", dynamics="gaussian", emissions="gaussian", single_subspace=True)
    return ssm.SLDS(N, 4, 2, transitions="recurrent_only", dynamics="diagonal_gaussian", emissions="gaussian",
                    single_subspace=True)


def run(name, data, num_iters, num_init_iters=50):
    npr.seed(0)
    model = build(name, data.shape[1])
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
    run(name, data[:200], 2, num_init_iters=2)                      # warm up numba
    init_s, fit_s, elbos, model, q = run(name, data, num_iters)
    x_hat = q.mean_continuous_states[0]
    z_hat = model.most_likely_states(x_hat, data)                   # Viterbi with x at the posterior mean
    print(json.dumps({
        "library": "ssm", "dtype": "float64", "dataset": name,
        "num_timesteps": num_timesteps, "num_iters": num_iters,
        "init_s": init_s, "fit_s": fit_s, "per_iter_s": fit_s / num_iters,
        "final_objective": float(elbos[-1]),
        "objective_monotone": bool(np.all(np.diff(elbos) >= -1e-3 * abs(elbos[-1]))),
        "state_accuracy": state_accuracy(saved["states"], z_hat), "latent_r2": latent_r2(saved["latents"], x_hat),
    }))
