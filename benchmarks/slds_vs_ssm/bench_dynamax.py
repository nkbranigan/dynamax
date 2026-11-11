"""Time dynamax's PCA+ARHMM initialization and Laplace EM on a saved dataset, in float64.

Usage: python bench_dynamax.py <slds|rslds> <num_timesteps> <num_iters>
Prints one JSON line. initialize and fit_em are wrapped in an outer jax.jit and compiled ahead of time
(jit(...).lower(...).compile(): tracing, lowering and XLA compilation, timed as *_compile_s); the executable is
then run once (timed as *_s). Outside the timed region the fit is scored by the recovery test's own function, and so
is a control: parameters drawn from their priors by initialize(method="prior") and no learning, with the states
inferred by approximate_posterior at its defaults.
"""
import json
import os
import sys
import time

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

jax.config.update("jax_enable_x64", True)
from dynamax.slds.ssm_recovery_test import MODELS, natural_params, recovery  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_data import path  # noqa: E402


def timed(fn):
    start = time.perf_counter()
    out = jax.block_until_ready(fn())
    return out, time.perf_counter() - start


if __name__ == "__main__":
    name, num_timesteps, num_iters = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    saved = np.load(path(name, num_timesteps))
    emissions = jnp.asarray(saved[f"recovery/{name}/y"])
    model = MODELS[name]()
    key = jr.PRNGKey(0)

    # The static arguments are baked into the executable and are not passed at call time.
    init = jax.jit(model.initialize, static_argnames=("method",))
    init_exe, init_compile = timed(lambda: init.lower(key, method="pca_arhmm", emissions=emissions).compile())
    (params, props), init_run = timed(lambda: init_exe(key, emissions=emissions))

    fit = jax.jit(model.fit_em, static_argnames=("num_iters", "verbose", "return_posterior"))
    fit_exe, fit_compile = timed(lambda: fit.lower(params, props, emissions, num_iters=num_iters, verbose=False,
                                                   return_posterior=True).compile())
    (fitted, _, posterior), fit_run = timed(lambda: fit_exe(params, props, emissions))

    control, _ = model.initialize(key, method="prior")
    control_posterior, _ = model.approximate_posterior(control, emissions)

    def score(params, posterior):
        """Recovery of the sampled states, latents and parameters (Viterbi path with x at the posterior mean)."""
        z_hat = jax.vmap(lambda y, q: model.most_likely_states(params, y, q))(emissions, posterior)
        x_hat = np.asarray(posterior.continuous.smoothed_means)
        return recovery(name, natural_params(params), x_hat.reshape(-1, x_hat.shape[-1]),
                        np.asarray(z_hat).reshape(-1), saved)

    print(json.dumps({
        "library": "dynamax", "dataset": name, "num_timesteps": num_timesteps, "num_iters": num_iters,
        "init_compile_s": init_compile, "init_s": init_run, "fit_compile_s": fit_compile, "fit_s": fit_run,
        "recovery": score(fitted, posterior), "control": score(control, control_posterior),
    }))
