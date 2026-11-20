"""Time dynamax's PCA+ARHMM initialization and Laplace EM on the benchmark's datasets, in float32 (JAX's default).

Usage: python bench_dynamax.py [--device cpu|gpu] [--model slds|rslds]
Fits every dataset of make_data.py, smallest first, on the CPU (the default) or on the GPU, and writes one results
file per model, such as cpu_dynamax_slds.jsonl or gpu_dynamax_rslds.jsonl, one JSON line per dataset. With --model
only that model's datasets are fitted, so the two models can run at the same time. initialize and fit_em are wrapped
in an outer jax.jit and compiled ahead of time (jit(...).lower(...).compile(): tracing, lowering and XLA compilation,
timed as *_compile_s); the executable is then run once (timed as *_s). Outside the timed region each fit is scored by
the recovery test's own function. A short fit runs before the timed ones, because on a cluster the first dynamax fit
can pay a one-time setup price. JAX's caches are cleared before each timed fit, so each one compiles in full.
"""
import argparse
import json
import os
import sys
import time

parser = argparse.ArgumentParser()
parser.add_argument("--device", choices=["cpu", "gpu"], default="cpu")
parser.add_argument("--model", choices=["slds", "rslds"], help="fit only this model's datasets (default: both)")
ARGS = parser.parse_args()
DEVICE = ARGS.device
os.environ["JAX_PLATFORMS"] = {"cpu": "cpu", "gpu": "cuda"}[DEVICE]   # read when jax is imported

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import jax.random as jr  # noqa: E402
import numpy as np  # noqa: E402

from dynamax.slds.ssm_recovery_test import MODELS, natural_params, recovery  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from make_data import SIZES, path  # noqa: E402

NUM_ITERS = 100                                           # the same as in bench_ssm.py


def results(name):
    return os.path.join(HERE, f"{DEVICE}_dynamax_{name}.jsonl")


def timed(fn):
    start = time.perf_counter()
    out = jax.block_until_ready(fn())
    return out, time.perf_counter() - start


def bench(name, num_timesteps, num_iters=NUM_ITERS):
    """One row of results: the times of one fit and its recovery scores."""
    jax.clear_caches()   # so the fit compiles in full each time
    saved = np.load(path(name, num_timesteps))
    emissions = jnp.asarray(saved[f"recovery/{name}/y"])
    model = MODELS[name]()
    key = jr.PRNGKey(4)   # the seed of the notebooks and of the recovery test

    # The static arguments are baked into the executable and are not passed at call time.
    init = jax.jit(model.initialize, static_argnames=("method",))
    init_exe, init_compile = timed(lambda: init.lower(key, method="pca_arhmm", emissions=emissions).compile())
    (params, props), init_run = timed(lambda: init_exe(key, emissions=emissions))

    fit = jax.jit(model.fit_em, static_argnames=("num_iters", "verbose", "return_posterior"))
    fit_exe, fit_compile = timed(lambda: fit.lower(params, props, emissions, num_iters=num_iters, verbose=False,
                                                   return_posterior=True).compile())
    (fitted, _, posterior), fit_run = timed(lambda: fit_exe(params, props, emissions))

    # Recovery of the sampled states, latents, and parameters (Viterbi path with x at the posterior mean).
    z_hat = jax.vmap(lambda y, q: model.most_likely_states(fitted, y, q))(emissions, posterior)
    x_hat = np.asarray(posterior.continuous.smoothed_means)
    scores = recovery(name, natural_params(fitted), x_hat.reshape(-1, x_hat.shape[-1]),
                      np.asarray(z_hat).reshape(-1), saved)

    row = {"library": "dynamax", "dataset": name, "num_timesteps": num_timesteps, "num_iters": num_iters,
           "init_compile_s": init_compile, "init_s": init_run, "fit_compile_s": fit_compile, "fit_s": fit_run,
           "recovery": scores}
    if DEVICE == "gpu":
        row["gpu"] = jax.devices()[0].device_kind
    return row


if __name__ == "__main__":
    names = [ARGS.model] if ARGS.model else list(SIZES)
    for name in names:
        open(results(name), "w").close()
    grid = sorted((T, name) for name in names for T in SIZES[name])   # cheap fits first
    num_timesteps, name = grid[0]
    bench(name, num_timesteps, num_iters=2)                           # untimed: pays the node's one-time setup price
    for num_timesteps, name in grid:
        try:
            row = bench(name, num_timesteps)
        except Exception as error:                        # such as running out of memory: go on to the next dataset
            print(f"dynamax {name:6s} T={num_timesteps:<8d} FAILED: {type(error).__name__}: {str(error)[:300]}",
                  flush=True)
            continue
        with open(results(name), "a") as f:
            f.write(json.dumps(row) + "\n")
        print(f"dynamax {name:6s} T={num_timesteps:<8d} init {row['init_s']:7.2f}s  fit {row['fit_s']:8.2f}s  "
              f"discrete state recovery {row['recovery']['discrete state recovery']:.3f}", flush=True)
