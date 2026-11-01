"""Time dynamax's PCA+ARHMM initialization and Laplace EM on a saved dataset.

Usage: [JAX_ENABLE_X64=1] python bench_dynamax.py <slds|rslds> <num_timesteps> <num_iters>
Prints one JSON line. initialize and fit_em are wrapped in an outer jax.jit and compiled ahead of time
(jit(...).lower(...).compile(): tracing, lowering and XLA compilation, timed as *_compile_s); the executable is
then run once (timed as *_s) and per_iter = fit_s / num_iters.
"""
import json
import os
import sys
import time

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

from dynamax.slds import GaussianSLDS, RecurrentGaussianSLDS

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from accuracy import latent_r2, state_accuracy  # noqa: E402
from make_data import path  # noqa: E402


def build(name):
    if name == "slds":
        return GaussianSLDS(5, 2, 10)
    return RecurrentGaussianSLDS(4, 2, 10, recurrent_only=True, dynamics_cov_type="diag")


def timed(fn):
    start = time.perf_counter()
    out = jax.block_until_ready(fn())
    return out, time.perf_counter() - start


if __name__ == "__main__":
    name, num_timesteps, num_iters = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    x64 = jax.config.read("jax_enable_x64")
    dtype = jnp.float64 if x64 else jnp.float32
    saved = np.load(path(name, num_timesteps))
    emissions = jnp.asarray(saved["emissions"], dtype=dtype)
    model = build(name)
    key = jr.PRNGKey(0)

    # fit_em and initialize re-jit an inner function on every call (dynamax convention), so wrap each in an
    # outer jit, compile it ahead of time and run the executable once. The static arguments are baked into the
    # executable and are not passed at call time.
    init = jax.jit(model.initialize, static_argnames=("method",))
    init_exe, init_compile = timed(lambda: init.lower(key, method="pca_arhmm", emissions=emissions).compile())
    (params, props), init_run = timed(lambda: init_exe(key, emissions=emissions))

    fit = jax.jit(model.fit_em, static_argnames=("num_iters", "verbose", "return_posterior"))
    fit_exe, fit_compile = timed(lambda: fit.lower(params, props, emissions, num_iters=num_iters, verbose=False,
                                                   return_posterior=True).compile())
    (fitted, lps, posterior), fit_run = timed(lambda: fit_exe(params, props, emissions))

    # recovery of the sampled states and latents, scored outside the timed region (same definitions as ssm's
    # notebooks: Viterbi path with x at the posterior mean, best relabelling; affine-aligned latents)
    posterior = jax.tree_util.tree_map(lambda a: a[0], posterior)
    z_hat = np.asarray(model.most_likely_states(fitted, emissions, posterior))
    x_hat = np.asarray(posterior.continuous.smoothed_means)

    print(json.dumps({
        "library": "dynamax", "dtype": "float64" if x64 else "float32", "dataset": name,
        "num_timesteps": num_timesteps, "num_iters": num_iters,
        "init_compile_s": init_compile, "init_s": init_run,
        "fit_compile_s": fit_compile, "fit_s": fit_run, "per_iter_s": fit_run / num_iters,
        "final_objective": float(lps[-1]), "objective_monotone": bool(jnp.all(jnp.diff(lps) >= -1e-3 * jnp.abs(lps[-1]))),
        "state_accuracy": state_accuracy(saved["states"], z_hat), "latent_r2": latent_r2(saved["latents"], x_hat),
        "devices": [str(d) for d in jax.devices()],
    }))
