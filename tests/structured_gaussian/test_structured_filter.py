"""Public LGSSM filtering checks against an independent dense CPU oracle.

The oracle deliberately forms the observation-sized innovation covariance and
uses the Joseph covariance update. It does not call Dynamax or TFP helpers.
"""

import jax
import jax.numpy as jnp
try:
    from jax import enable_x64
except ImportError:
    from jax.experimental import enable_x64
import numpy as np
import pytest
from scipy.linalg import cho_factor, cho_solve

from dynamax.linear_gaussian_ssm.inference import (
    ParamsLGSSM,
    ParamsLGSSMDynamics,
    ParamsLGSSMEmissions,
    ParamsLGSSMInitial,
    lgssm_filter,
)


def _make_case(state_dim, emission_dim, ntime, *, dtype, time_varying,
               diagonal_noise, with_inputs=True):
    """Make nonspherical, well-conditioned parameters with distinct time slices."""
    rng = np.random.default_rng(7403 + state_dim * 19 + emission_dim * 7 + ntime)
    k, d, p = state_dim, emission_dim, 2
    prefix = (ntime,) if time_varying else ()

    def array(value):
        return jnp.asarray(value, dtype=dtype)

    def spd(dim, shape=()):
        factor = rng.normal(size=shape + (dim, dim)) / np.sqrt(dim)
        return factor @ np.swapaxes(factor, -1, -2) + 0.4 * np.eye(dim)

    noise = (0.2 + rng.uniform(size=prefix + (d,)) * 1.8
             if diagonal_noise else spd(d, prefix))
    params = ParamsLGSSM(
        initial=ParamsLGSSMInitial(
            mean=array(rng.normal(size=k)), cov=array(spd(k))),
        dynamics=ParamsLGSSMDynamics(
            weights=array(0.7 * np.eye(k) +
                          0.15 * rng.normal(size=prefix + (k, k)) / np.sqrt(k)),
            cov=array(0.2 * spd(k, prefix)),
            bias=array(0.3 * rng.normal(size=prefix + (k,))) if with_inputs else None,
            input_weights=array(0.3 * rng.normal(size=prefix + (k, p)))
            if with_inputs else None),
        emissions=ParamsLGSSMEmissions(
            weights=array(rng.normal(size=prefix + (d, k)) / np.sqrt(k)),
            cov=array(noise),
            bias=array(0.4 * rng.normal(size=prefix + (d,))) if with_inputs else None,
            input_weights=array(0.4 * rng.normal(size=prefix + (d, p)))
            if with_inputs else None),
    )
    inputs = array(rng.normal(size=(ntime, p))) if with_inputs else None
    emissions = array(rng.normal(size=(ntime, d)) + 0.15 * np.arange(ntime)[:, None])
    return params, emissions, inputs


def _dense_kalman_oracle(params, emissions, inputs, *, diagonal_noise):
    """Compute every filtered moment and total likelihood in NumPy float64."""
    observations = np.asarray(emissions, dtype=np.float64)
    ntime, d = observations.shape
    mean = np.asarray(params.initial.mean, dtype=np.float64).copy()
    cov = np.asarray(params.initial.cov, dtype=np.float64).copy()
    k = mean.size
    inputs = (np.zeros((ntime, 0), dtype=np.float64) if inputs is None
              else np.asarray(inputs, dtype=np.float64))
    p = inputs.shape[-1]

    def at_time(value, rank, t, default_shape):
        if value is None:
            return np.zeros(default_shape, dtype=np.float64)
        value = np.asarray(value, dtype=np.float64)
        return value[t] if value.ndim == rank + 1 else value

    means, covariances = [], []
    loglik = 0.0
    for t in range(ntime):
        h = at_time(params.emissions.weights, 2, t, (d, k))
        r = at_time(params.emissions.cov, 1 if diagonal_noise else 2, t,
                    (d,) if diagonal_noise else (d, d))
        r = np.diag(r) if diagonal_noise else r
        emission_bias = at_time(params.emissions.bias, 1, t, (d,))
        emission_inputs = at_time(params.emissions.input_weights, 2, t, (d, p))
        residual = observations[t] - h @ mean - emission_bias - emission_inputs @ inputs[t]
        innovation_cov = h @ cov @ h.T + r
        chol, lower = cho_factor(innovation_cov, lower=True)
        innovation_solve = cho_solve((chol, lower), residual)
        logdet = 2.0 * np.log(np.diag(chol)).sum()
        loglik -= 0.5 * (d * np.log(2.0 * np.pi) + logdet + residual @ innovation_solve)

        gain = cho_solve((chol, lower), h @ cov).T
        mean = mean + gain @ residual
        residual_map = np.eye(k) - gain @ h
        cov = residual_map @ cov @ residual_map.T + gain @ r @ gain.T
        cov = (cov + cov.T) / 2.0
        means.append(mean.copy())
        covariances.append(cov.copy())

        if t + 1 < ntime:
            f = at_time(params.dynamics.weights, 2, t, (k, k))
            q = at_time(params.dynamics.cov, 2, t, (k, k))
            dynamics_bias = at_time(params.dynamics.bias, 1, t, (k,))
            dynamics_inputs = at_time(params.dynamics.input_weights, 2, t, (k, p))
            mean = f @ mean + dynamics_bias + dynamics_inputs @ inputs[t]
            cov = f @ cov @ f.T + q
            cov = (cov + cov.T) / 2.0

    return loglik, np.stack(means), np.stack(covariances)


def _assert_matches_oracle(posterior, expected, dtype, *, tolerance=None):
    if tolerance is None:
        tolerance = 4e-5 if dtype == np.float32 else 2e-10
    for actual, reference in zip(
            (posterior.marginal_loglik, posterior.filtered_means,
             posterior.filtered_covariances), expected):
        actual = np.asarray(actual)
        assert actual.dtype == np.dtype(dtype)
        assert np.all(np.isfinite(actual))
        np.testing.assert_allclose(actual, reference, rtol=tolerance, atol=tolerance)
    covariances = np.asarray(posterior.filtered_covariances)
    np.testing.assert_allclose(covariances, np.swapaxes(covariances, -1, -2),
                               rtol=0.0, atol=tolerance)
    assert np.linalg.eigvalsh(covariances).min() >= -tolerance


@pytest.mark.parametrize("dtype", [np.float32, np.float64], ids=["float32", "float64"])
@pytest.mark.parametrize(
    "state_dim,emission_dim,ntime,time_varying,with_inputs",
    [
        pytest.param(3, 9, 7, False, False, id="wide-static-defaults"),
        pytest.param(3, 9, 7, True, True, id="wide-time-varying"),
        pytest.param(5, 2, 6, True, True, id="narrow-time-varying"),
        pytest.param(3, 1, 5, True, True, id="scalar-observation"),
        pytest.param(3, 3, 4, True, True, id="equal-dimensions"),
        pytest.param(2, 4, 1, False, True, id="one-observation-static"),
        pytest.param(2, 4, 1, True, True, id="one-observation-time-varying"),
    ],
)
def test_diagonal_filter_matches_dense_oracle(
        dtype, state_dim, emission_dim, ntime, time_varying, with_inputs):
    """Diagonal observation variances give the same posterior as dense Kalman filtering."""
    with enable_x64(dtype == np.float64):
        params, emissions, inputs = _make_case(
            state_dim, emission_dim, ntime, dtype=dtype,
            time_varying=time_varying, diagonal_noise=True, with_inputs=with_inputs)
        expected = _dense_kalman_oracle(params, emissions, inputs, diagonal_noise=True)
        posterior = lgssm_filter(params, emissions, inputs)
        _assert_matches_oracle(posterior, expected, dtype)


@pytest.mark.parametrize("dtype", [np.float32, np.float64], ids=["float32", "float64"])
@pytest.mark.parametrize("time_varying", [False, True], ids=["static", "time-varying"])
def test_full_covariance_filter_matches_dense_oracle(dtype, time_varying):
    """The existing correlated, full-observation-covariance branch stays correct."""
    with enable_x64(dtype == np.float64):
        params, emissions, inputs = _make_case(
            3, 5, 4, dtype=dtype, time_varying=time_varying, diagonal_noise=False)
        expected = _dense_kalman_oracle(params, emissions, inputs, diagonal_noise=False)
        posterior = lgssm_filter(params, emissions, inputs)
        # The unchanged dense branch adds 1e-9 jitter in psd_solve, unlike
        # the exact dense oracle; allow its accumulated float64 discrepancy.
        tolerance = 2e-8 if dtype == np.float64 else None
        _assert_matches_oracle(posterior, expected, dtype, tolerance=tolerance)


@pytest.mark.parametrize("dtype", [np.float32, np.float64], ids=["float32", "float64"])
def test_diagonal_filter_jit_vmap_matches_dense_oracle(dtype):
    """Public filtering composes with both jit and batched observation sequences."""
    with enable_x64(dtype == np.float64):
        params, emissions, inputs = _make_case(
            2, 6, 4, dtype=dtype, time_varying=True, diagonal_noise=True)
        batch = jnp.stack([emissions, emissions + dtype(0.25), emissions * dtype(-0.7)])
        batched_filter = jax.jit(jax.vmap(lgssm_filter, in_axes=(None, 0, None)))
        posteriors = batched_filter(params, batch, inputs)
        assert posteriors.filtered_means.shape == (3, 4, 2)
        assert posteriors.filtered_covariances.shape == (3, 4, 2, 2)
        assert posteriors.marginal_loglik.shape == (3,)
        for i in range(3):
            expected = _dense_kalman_oracle(params, batch[i], inputs, diagonal_noise=True)
            posterior = jax.tree_util.tree_map(lambda value: value[i], posteriors)
            _assert_matches_oracle(posterior, expected, dtype)
