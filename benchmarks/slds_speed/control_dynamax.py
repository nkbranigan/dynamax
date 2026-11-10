"""Control for the recovery plot: dynamax with parameters drawn from their priors and no learning.

Usage: [JAX_ENABLE_X64=1] python control_dynamax.py <slds|rslds> <num_timesteps>
Prints one JSON line. The parameters come from initialize(method="prior"); the states are inferred with
approximate_posterior at its defaults and scored like a fit.
"""
import json
import os
import sys

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

from dynamax.slds.ssm_recovery_test import MODELS, natural_params

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_data import path  # noqa: E402
from recovery import scores  # noqa: E402

if __name__ == "__main__":
    name, num_timesteps = sys.argv[1], int(sys.argv[2])
    x64 = jax.config.read("jax_enable_x64")
    saved = np.load(path(name, num_timesteps))
    emissions = jnp.asarray(saved["emissions"], dtype=jnp.float64 if x64 else jnp.float32)
    model = MODELS[name]()
    params, _ = model.initialize(jr.PRNGKey(0), method="prior")
    posterior, _ = model.approximate_posterior(params, emissions)
    posterior = jax.tree_util.tree_map(lambda a: a[0], posterior)
    z_hat = np.asarray(model.most_likely_states(params, emissions, posterior))
    x_hat = np.asarray(posterior.continuous.smoothed_means)
    print(json.dumps({
        "library": "dynamax", "dtype": "float64" if x64 else "float32", "dataset": name, "num_timesteps": num_timesteps,
        "control": True, **scores(name, saved, natural_params(params), x_hat, z_hat),
    }))
