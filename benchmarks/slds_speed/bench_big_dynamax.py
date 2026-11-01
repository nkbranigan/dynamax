"""Time dynamax on a dataset written by make_big_data.py (float64 unless JAX_ENABLE_X64=0).

Usage: python bench_big_dynamax.py <npz> <num_iters>
Same protocol as bench_dynamax.py: initialize and fit_em wrapped in an outer jax.jit and run twice, so the
second call is the steady-state time and compile = first - second. Batched sequences (S > 1) go through
fit_em's batch axis.
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


def timed(fn):
    start = time.perf_counter()
    out = jax.block_until_ready(fn())
    return out, time.perf_counter() - start


if __name__ == "__main__":
    npz, num_iters = sys.argv[1], int(sys.argv[2])
    d = np.load(npz)
    model_name, N, D, K, T, S = str(d["model"]), int(d["N"]), int(d["D"]), int(d["K"]), int(d["T"]), int(d["S"])
    x64 = jax.config.read("jax_enable_x64")
    emissions = jnp.asarray(d["emissions"], dtype=jnp.float64 if x64 else jnp.float32)
    if S == 1:
        emissions = emissions[0]
    model = GaussianSLDS(K, D, N) if model_name == "slds" else RecurrentGaussianSLDS(K, D, N)

    init = jax.jit(model.initialize, static_argnames=("method",))
    (params, props), init_first = timed(lambda: init(jr.PRNGKey(0), method="pca_arhmm", emissions=emissions))
    _, init_second = timed(lambda: init(jr.PRNGKey(0), method="pca_arhmm", emissions=emissions))
    fit = jax.jit(model.fit_em, static_argnames=("num_iters", "verbose", "return_posterior"))
    go = lambda: fit(params, props, emissions, num_iters=num_iters, verbose=False, return_posterior=True)
    _, fit_first = timed(go)
    (fitted, lps, posterior), fit_second = timed(go)

    # recovery, scored outside the timed region (Viterbi path with x at the posterior mean, best relabelling;
    # affine-aligned latents), per sequence
    batched_emissions = emissions if S > 1 else emissions[None]
    singles = [jax.tree_util.tree_map(lambda a: a[s], posterior) for s in range(S)]
    z_hat = np.stack([np.asarray(model.most_likely_states(fitted, batched_emissions[s], singles[s])) for s in range(S)])
    x_hat = np.stack([np.asarray(p.continuous.smoothed_means) for p in singles])

    print(json.dumps({
        "library": "dynamax", "dtype": "float64" if x64 else "float32", "model": model_name,
        "N": N, "D": D, "K": K, "T": T, "S": S, "num_iters": num_iters,
        "init_first_s": init_first, "init_warm_s": init_second,
        "fit_first_s": fit_first, "fit_warm_s": fit_second,
        "compile_s": fit_first - fit_second, "per_iter_s": fit_second / num_iters,
        "final_objective": float(lps[-1]),
        "objective_monotone": bool(jnp.all(jnp.diff(lps) >= -1e-3 * jnp.abs(lps[-1]))),
        "state_accuracy": state_accuracy(d["states"], z_hat), "latent_r2": latent_r2(d["latents"], x_hat),
        "devices": [str(dev) for dev in jax.devices()],
    }))
