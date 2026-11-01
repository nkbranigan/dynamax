"""Sample a stable random SLDS or rSLDS of a given size.

Usage: python make_big_data.py <slds|rslds> N D K T S [seed]
Writes scratch_data/big_<model>_N<N>_D<D>_K<K>_T<T>_S<S>.npz with emissions (S, T, N) float64, states (S, T),
latents (S, T, D) and the sizes as metadata. The true model: per-state dynamics 0.95 x a random rotation,
small random biases, process noise 0.1 I; emissions with orthonormal columns scaled to unit signal variance
per dimension, observation noise 0.1; sticky transitions (0.95 on the diagonal), with random recurrent
weights of unit scale for the rSLDS. Every sampled sequence is checked to stay bounded.
"""
import os
import sys

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

from dynamax.slds import GaussianSLDS, RecurrentGaussianSLDS


def path(model, N, D, K, T, S):
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "scratch_data", f"big_{model}_N{N}_D{D}_K{K}_T{T}_S{S}.npz")


def random_rotation(key, D):
    q, _ = jnp.linalg.qr(jr.normal(key, (D, D)))
    return q


def true_model(model, N, D, K, seed=0):
    key = jr.PRNGKey(seed)
    k_dyn, k_bias, k_emis, k_rec = jr.split(key, 4)
    m = GaussianSLDS(K, D, N) if model == "slds" else RecurrentGaussianSLDS(K, D, N)
    params, _ = m.initialize(jr.PRNGKey(seed))
    F = 0.95 * jnp.stack([random_rotation(jr.fold_in(k_dyn, k), D) for k in range(K)])
    b = 0.3 * jr.normal(k_bias, (K, D))
    H = jnp.linalg.qr(jr.normal(k_emis, (N, D)))[0] * jnp.sqrt(N / D)
    sticky = 0.95 * jnp.eye(K) + 0.05 / K
    dynamics = params.dynamics._replace(initial_mean=jnp.zeros(D), initial_cov=jnp.eye(D), weights=F, biases=b,
                                        covs=jnp.tile(0.1 * jnp.eye(D), (K, 1, 1)))
    emissions = params.emissions._replace(weights=H[None], biases=jnp.zeros((1, N)), covs=0.1 * jnp.ones((1, N)))
    if model == "slds":
        transitions = params.transitions._replace(transition_matrix=sticky)
    else:
        transitions = params.transitions._replace(logits=jnp.log(sticky),
                                                  recurrent_weights=jr.normal(k_rec, (K, D)) / jnp.sqrt(D))
    params = params._replace(initial=params.initial._replace(probs=jnp.ones(K) / K),
                             transitions=transitions, dynamics=dynamics, emissions=emissions)
    return m, params


if __name__ == "__main__":
    model, N, D, K, T, S = sys.argv[1], *[int(a) for a in sys.argv[2:7]]
    seed = int(sys.argv[7]) if len(sys.argv) > 7 else 0
    m, params = true_model(model, N, D, K, seed)
    keys = jr.split(jr.PRNGKey(seed + 1), S)
    states, latents, emissions = jax.vmap(lambda k: m.sample(params, k, T))(keys)
    assert bool(jnp.isfinite(emissions).all()), "non-finite emissions"
    assert float(jnp.abs(latents).max()) < 50, f"latents unbounded: max|x| = {float(jnp.abs(latents).max()):.1f}"
    os.makedirs(os.path.dirname(path(model, N, D, K, T, S)), exist_ok=True)
    np.savez(path(model, N, D, K, T, S), emissions=np.asarray(emissions, dtype=np.float64), states=np.asarray(states),
             latents=np.asarray(latents, dtype=np.float64), model=model, N=N, D=D, K=K, T=T, S=S)
    counts = np.bincount(np.asarray(states).ravel(), minlength=K)
    print(path(model, N, D, K, T, S), emissions.shape, f"max|x|={float(jnp.abs(latents).max()):.2f}",
          f"state occupancy min/max={counts.min()}/{counts.max()}")
