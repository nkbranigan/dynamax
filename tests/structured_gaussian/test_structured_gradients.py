"""Independent likelihood-gradient checks for the serial diagonal-R filter.

The reference constructs the *whole sequence's* joint Gaussian from its
independent state innovations. It does not call a Kalman filter, a distribution
library, or any Dynamax numerical helper. Cases stay small (at most 20 observed
coordinates); the strong-signal case has a well-conditioned dense covariance
but stresses cancellation in a subtractive Woodbury quadratic.
"""

import jax
import jax.numpy as jnp
from jax.scipy.linalg import solve_triangular
import numpy as np
import pytest

try:
    from jax import enable_x64
except ImportError:
    from jax.experimental import enable_x64

from dynamax.linear_gaussian_ssm.inference import (
    ParamsLGSSM,
    ParamsLGSSMDynamics,
    ParamsLGSSMEmissions,
    ParamsLGSSMInitial,
    lgssm_filter,
)


_GRADIENT_NAMES = (
    "emission weights H",
    "log diagonal emission covariance R",
    "initial covariance Cholesky factor",
    "process covariance Cholesky factor",
    "observations",
)


def _case(dtype, strong_signal):
    """Return differentiation arguments and fixed, nonzero model parameters."""
    array = lambda values: jnp.asarray(values, dtype=dtype)
    if strong_signal:
        # Large signal/noise ratio without an ill-conditioned dense reference:
        # all three state directions are observed strongly. The observations
        # also have a large Mahalanobis residual, so loss agreement cannot be
        # satisfied by an accurate determinant and an inaccurate quadratic.
        h = array([[1.0, 0.15, -0.1], [-0.1, 0.8, 0.2], [0.05, -0.2, 1.1]]) * 128.0
        initial_chol = array([[0.9, 0.0, 0.0], [0.1, 1.2, 0.0], [-0.08, 0.15, 0.75]])
        process_chol = array([[0.4, 0.0, 0.0], [0.04, 0.5, 0.0], [-0.03, 0.08, 0.35]])
        r = array([0.65, 1.2, 0.9])
        fixed = dict(
            f=array([[0.7, 0.05, 0.0], [-0.1, 0.65, 0.02], [0.0, -0.04, 0.8]]),
            initial_mean=array([0.2, -0.15, 0.05]),
            dynamics_bias=array([0.03, -0.02, 0.01]),
            emissions_bias=array([0.1, -0.2, 0.15]),
        )
        observations = (h @ array([30.0, -20.0, 15.0]) + fixed["emissions_bias"])[None, :]
    else:
        h = array([[0.8, -0.3], [0.15, 0.95], [-0.55, 0.4], [0.65, 0.7], [-0.35, -0.8]])
        initial_chol = array([[0.9, 0.0], [0.22, 0.7]])
        process_chol = array([[0.45, 0.0], [-0.12, 0.55]])
        r = array([0.45, 0.8, 0.6, 1.1, 0.7])
        fixed = dict(
            f=array([[0.75, 0.12], [-0.08, 0.65]]),
            initial_mean=array([0.2, -0.3]),
            dynamics_bias=array([0.08, -0.04]),
            emissions_bias=array([0.1, -0.2, 0.05, 0.15, -0.1]),
        )
        observations = array(
            [[1.3, -0.4, 0.7, 1.2, -1.0],
             [0.2, 1.1, -0.3, 0.4, -0.8],
             [-0.7, 0.6, 1.3, -0.2, 0.5],
             [0.9, -1.1, 0.2, 0.8, -0.4]]
        )
    return (h, jnp.log(r), initial_chol, process_chol, observations), fixed


def _joint_observation_gaussian(h, log_r, initial_chol, process_chol, observations, fixed):
    """Build mean/covariance via independent unit-normal state innovations."""
    num_timesteps, emission_dim = observations.shape
    state_dim = h.shape[-1]
    state_map = jnp.zeros((state_dim, num_timesteps * state_dim), dtype=h.dtype)
    state_map = state_map.at[:, :state_dim].set(jnp.tril(initial_chol))
    state_mean = fixed["initial_mean"]
    observation_maps, observation_means = [], []
    for t in range(num_timesteps):
        if t:
            state_map = fixed["f"] @ state_map
            state_map = state_map.at[:, t * state_dim:(t + 1) * state_dim].set(
                jnp.tril(process_chol)
            )
            state_mean = fixed["f"] @ state_mean + fixed["dynamics_bias"]
        observation_maps.append(h @ state_map)
        observation_means.append(h @ state_mean + fixed["emissions_bias"])
    observation_map = jnp.concatenate(observation_maps, axis=0)
    mean = jnp.concatenate(observation_means)
    covariance = observation_map @ observation_map.T
    covariance = covariance + jnp.diag(jnp.tile(jnp.exp(log_r), num_timesteps))
    assert mean.shape == (num_timesteps * emission_dim,)
    return mean, covariance


def _dense_loglik(h, log_r, initial_chol, process_chol, observations, fixed):
    mean, covariance = _joint_observation_gaussian(
        h, log_r, initial_chol, process_chol, observations, fixed
    )
    factor = jnp.linalg.cholesky(covariance)
    standardized = solve_triangular(factor, observations.reshape(-1) - mean, lower=True)
    return -0.5 * (
        observations.size * jnp.log(jnp.asarray(2.0 * np.pi, dtype=h.dtype))
        + 2.0 * jnp.log(jnp.diag(factor)).sum()
        + jnp.vdot(standardized, standardized)
    )


def _filter_loglik(h, log_r, initial_chol, process_chol, observations, fixed):
    initial_factor = jnp.tril(initial_chol)
    process_factor = jnp.tril(process_chol)
    state_dim = h.shape[-1]
    emission_dim = h.shape[-2]
    # Give every otherwise empty input array an explicit dtype: the float32
    # case must remain float32 even in an x64-capable test process.
    params = ParamsLGSSM(
        initial=ParamsLGSSMInitial(fixed["initial_mean"], initial_factor @ initial_factor.T),
        dynamics=ParamsLGSSMDynamics(
            fixed["f"], fixed["dynamics_bias"],
            jnp.zeros((state_dim, 0), dtype=h.dtype), process_factor @ process_factor.T,
        ),
        emissions=ParamsLGSSMEmissions(
            h, fixed["emissions_bias"], jnp.zeros((emission_dim, 0), dtype=h.dtype),
            jnp.exp(log_r),
        ),
    )
    return lgssm_filter(
        params, observations, inputs=jnp.zeros((observations.shape[0], 0), dtype=h.dtype)
    ).marginal_loglik


@pytest.mark.parametrize("dtype_name", ["float64", "float32"])
@pytest.mark.parametrize("strong_signal", [False, True], ids=["four_timesteps", "strong_residual"])
def test_diagonal_r_likelihood_gradients_against_joint_gaussian(dtype_name, strong_signal):
    """Check all parameter and data gradients against a separate mathematical construction."""
    with enable_x64(dtype_name == "float64"):
        dtype = jnp.dtype(dtype_name)
        arguments, fixed = _case(dtype, strong_signal)
        reference = lambda *args: _dense_loglik(*args, fixed=fixed)
        candidate = lambda *args: _filter_loglik(*args, fixed=fixed)
        argnums = tuple(range(len(arguments)))
        expected_value, expected_gradients = jax.value_and_grad(reference, argnums=argnums)(*arguments)
        actual_value, actual_gradients = jax.value_and_grad(candidate, argnums=argnums)(*arguments)

        assert actual_value.dtype == dtype
        assert expected_value.dtype == dtype
        assert np.isfinite(np.asarray(actual_value))
        assert np.isfinite(np.asarray(expected_value))
        if dtype_name == "float64":
            value_rtol, value_atol = 3e-9, 3e-9
            gradient_rtol, gradient_atol = 3e-8, 3e-9
        else:
            value_rtol, value_atol = 1e-5, 2e-3 if strong_signal else 2e-5
            gradient_rtol, gradient_atol = (1e-3, 1e-4) if strong_signal else (4e-4, 5e-5)

        np.testing.assert_allclose(
            actual_value, expected_value, rtol=value_rtol, atol=value_atol,
            err_msg="diagonal-R likelihood differs from the dense joint Gaussian",
        )
        for name, actual, expected in zip(_GRADIENT_NAMES, actual_gradients, expected_gradients):
            assert actual.dtype == dtype, name
            assert expected.dtype == dtype, name
            assert np.all(np.isfinite(np.asarray(actual))), name
            assert np.all(np.isfinite(np.asarray(expected))), name
            np.testing.assert_allclose(
                actual, expected, rtol=gradient_rtol, atol=gradient_atol,
                err_msg=f"likelihood gradient mismatch for {name}",
            )
            if not strong_signal:
                assert np.linalg.norm(np.asarray(expected)) > 1e-5, f"trivial gradient for {name}"

        if strong_signal:
            mean, covariance = _joint_observation_gaussian(*arguments, fixed=fixed)
            dense_covariance = np.asarray(covariance, dtype=np.float64)
            residual = np.asarray(arguments[-1].reshape(-1) - mean, dtype=np.float64)
            # Guard the intended regime: a large quadratic with a reliable
            # dense reference, rather than a near-singular stress test.
            assert np.linalg.cond(dense_covariance) < 20.0
            assert residual @ np.linalg.solve(dense_covariance, residual) > 500.0
