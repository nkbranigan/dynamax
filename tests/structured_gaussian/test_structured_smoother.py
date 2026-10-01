"""Check diagonal-noise smoothing against dense joint Gaussian conditioning.

The NumPy oracle constructs the entire latent trajectory from independent
innovations, then conditions it on all observations in one solve. It uses no
Kalman recursions, Dynamax inference helpers, or TFP distributions.
"""

import jax.numpy as jnp
try:
    from jax import enable_x64
except ImportError:
    from jax.experimental import enable_x64
import numpy as np

from dynamax.linear_gaussian_ssm.inference import (
    ParamsLGSSM,
    ParamsLGSSMDynamics,
    ParamsLGSSMEmissions,
    ParamsLGSSMInitial,
    lgssm_smoother,
)


def _case(ntime):
    """Use nonzero means and a nonsymmetric transition to expose transposes."""
    params = ParamsLGSSM(
        initial=ParamsLGSSMInitial(
            mean=jnp.array([0.8, -0.5]),
            cov=jnp.array([[1.2, 0.25], [0.25, 0.7]])),
        dynamics=ParamsLGSSMDynamics(
            weights=jnp.array([[0.72, 0.35], [-0.18, 0.61]]),
            bias=jnp.array([0.14, -0.09]),
            input_weights=jnp.zeros((2, 0)),
            cov=jnp.array([[0.3, 0.06], [0.06, 0.22]])),
        emissions=ParamsLGSSMEmissions(
            weights=jnp.array([[1.0, 0.2], [-0.4, 0.8], [0.3, -0.7],
                               [0.6, 0.5], [-0.2, 1.1]]),
            bias=jnp.array([0.2, -0.3, 0.1, 0.4, -0.15]),
            input_weights=jnp.zeros((5, 0)),
            cov=jnp.array([0.2, 0.45, 0.7, 0.3, 0.9])),
    )
    observations = jnp.array([
        [0.8, -0.2, 0.4, 0.7, -0.6],
        [0.5, 0.3, -0.1, 0.9, 0.2],
        [-0.2, 0.4, 0.6, 0.1, 0.7],
        [0.3, -0.5, 0.2, 0.8, -0.1],
    ])[:ntime]
    return params, observations


def _joint_conditioning_oracle(params, observations):
    """Return means, marginal covariances, and E[z_t z_(t+1).T | y]."""
    y = np.asarray(observations, dtype=np.float64)
    ntime, emission_dim = y.shape
    m0 = np.asarray(params.initial.mean, dtype=np.float64)
    state_dim = m0.size
    f = np.asarray(params.dynamics.weights, dtype=np.float64)
    b = np.asarray(params.dynamics.bias, dtype=np.float64)
    q = np.asarray(params.dynamics.cov, dtype=np.float64)
    h = np.asarray(params.emissions.weights, dtype=np.float64)
    d = np.asarray(params.emissions.bias, dtype=np.float64)
    r = np.asarray(params.emissions.cov, dtype=np.float64)

    # z = prior_mean + loading @ [initial_error, process_noise_1, ...].
    loading = np.zeros((ntime * state_dim, ntime * state_dim))
    innovation_cov = np.kron(np.eye(ntime), q)
    innovation_cov[:state_dim, :state_dim] = np.asarray(params.initial.cov)
    prior_means = np.empty((ntime, state_dim))
    prior_means[0] = m0
    for t in range(ntime):
        block = slice(t * state_dim, (t + 1) * state_dim)
        if t:
            previous = slice((t - 1) * state_dim, t * state_dim)
            loading[block] = f @ loading[previous]
            prior_means[t] = f @ prior_means[t - 1] + b
        loading[block, block] = np.eye(state_dim)
    prior_cov = loading @ innovation_cov @ loading.T

    observation_map = np.kron(np.eye(ntime), h)
    observation_cov = (observation_map @ prior_cov @ observation_map.T
                       + np.diag(np.tile(r, ntime)))
    state_observation_cov = prior_cov @ observation_map.T
    gain = np.linalg.solve(observation_cov, state_observation_cov.T).T
    residual = y.reshape(-1) - (prior_means @ h.T + d).reshape(-1)
    posterior_means = (prior_means.reshape(-1) + gain @ residual).reshape(
        ntime, state_dim)
    posterior_cov = prior_cov - gain @ state_observation_cov.T

    marginal_covs = np.empty((ntime, state_dim, state_dim))
    raw_cross_moments = np.empty((ntime - 1, state_dim, state_dim))
    for t in range(ntime):
        block = slice(t * state_dim, (t + 1) * state_dim)
        marginal_covs[t] = posterior_cov[block, block]
        if t + 1 < ntime:
            following = slice((t + 1) * state_dim, (t + 2) * state_dim)
            raw_cross_moments[t] = (
                posterior_cov[block, following]
                + np.outer(posterior_means[t], posterior_means[t + 1]))
    return posterior_means, marginal_covs, raw_cross_moments


def _assert_smoother_matches_joint_conditioning(ntime):
    with enable_x64(True):
        params, observations = _case(ntime)
        expected = _joint_conditioning_oracle(params, observations)
        posterior = lgssm_smoother(params, observations)
        actual = (posterior.smoothed_means, posterior.smoothed_covariances,
                  posterior.smoothed_cross_covariances)
        assert tuple(value.shape for value in actual) == (
            (ntime, 2), (ntime, 2, 2), (ntime - 1, 2, 2))
        for value, reference in zip(actual, expected):
            value = np.asarray(value)
            assert value.dtype == np.float64
            assert np.all(np.isfinite(value))
            # The existing backward smoother uses psd_solve's 1e-9 jitter.
            np.testing.assert_allclose(value, reference, rtol=2e-8, atol=2e-8)
        if ntime > 1:
            # Confirm the fixture distinguishes the documented t,t+1 order.
            assert np.max(np.abs(expected[2] - expected[2].swapaxes(-1, -2))) > 1e-3


def test_diagonal_smoother_single_observation_matches_joint_conditioning():
    _assert_smoother_matches_joint_conditioning(1)


def test_diagonal_smoother_four_observations_matches_joint_conditioning():
    _assert_smoother_matches_joint_conditioning(4)
