"""Warm fit_em time vs num_iters, and a per-sweep breakdown (q(z) update, q(x) Newton update, M-step)."""
import os
import sys
import time

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_data import path  # noqa: E402
from bench_dynamax import build  # noqa: E402
from dynamax.slds.inference import initialize_posterior, update_continuous_posterior, update_discrete_posterior  # noqa: E402


def warm_time(fn, reps=1):
    jax.block_until_ready(fn())
    start = time.perf_counter()
    for _ in range(reps):
        jax.block_until_ready(fn())
    return (time.perf_counter() - start) / reps


if __name__ == "__main__":
    name, T = sys.argv[1], int(sys.argv[2])
    emissions = jnp.asarray(np.load(path(name, T))["emissions"], dtype=jnp.float32)
    model = build(name)
    params, props = model.initialize(jr.PRNGKey(0), method="pca_arhmm", emissions=emissions)

    for n in (1, 5, 20):
        dt = warm_time(lambda: model.fit_em(params, props, emissions, num_iters=n, verbose=False)[1])
        print(f"fit_em num_iters={n:3d}: {dt:.3f}s  per-iter {dt / n:.3f}s")

    inputs = jnp.zeros((T, 0), dtype=emissions.dtype)
    post = initialize_posterior(model, params, emissions, inputs)
    xs = jnp.zeros((model.num_samples, T, model.state_dim), dtype=emissions.dtype)
    upd_z = jax.jit(lambda p, qx: update_discrete_posterior(model, p, qx, xs, emissions, inputs))
    upd_x = jax.jit(lambda p, qz, qx: update_continuous_posterior(model, p, qz, qx, emissions, inputs))
    qz = upd_z(params, post.continuous)
    print(f"q(z) update (warm): {warm_time(lambda: upd_z(params, post.continuous), 5):.4f}s")
    print(f"q(x) Newton update from the initial posterior (warm): {warm_time(lambda: upd_x(params, qz, post.continuous), 5):.4f}s")
    qx = upd_x(params, qz, post.continuous)
    print(f"q(x) Newton update from a converged posterior (warm): {warm_time(lambda: upd_x(params, qz, qx), 5):.4f}s")
    e_step = jax.jit(lambda p: model.e_step(p, emissions, inputs, posterior=post))
    print(f"e_step (num_inner_sweeps={model.num_inner_sweeps}, warm): {warm_time(lambda: e_step(params), 5):.4f}s")
    stats, _ = e_step(params)
    batch = jax.tree_util.tree_map(lambda a: a[None], stats)
    m_state = model.initialize_m_step_state(params, props)
    m_step = jax.jit(lambda p, s: model.m_step(p, props, s, m_state)[0])
    print(f"m_step (warm): {warm_time(lambda: m_step(params, batch), 5):.4f}s")
